"""The two-layer scoring decoder.

Per layer, three branches:

    path  DFA -> self-attn -> FFN -> norm -> score      1024 -> 128 -> 20
    vel   cross-attn(image) -> self-attn -> FFN -> score  256 ->  64 -> 20
    traj  (last layer only) outer-sum of the survivors, 20x20 = 400
          DFA -> self-attn -> FFN -> metric heads -> argmax

The ego-status encoding is added at the head of *every* layer, before the DFA.

One deviation from the reference, exact rather than approximate: the reference
gathers `traj_vocab` progressively, once per layer per axis, on a
[1024, 256, 8, 3] tensor -- 6.3M elements moved twice to end up with 20x20.
Indices compose instead, so layer 0's 128-of-1024 and layer 1's 20-of-128
become 20 absolute path indices, and the vocabulary is sliced once at the end.
`traj_mask` is dropped entirely; it only feeds the training loss.
"""

import torch
import ttnn

from .attention import TtMultiheadAttention, TtFFN, _t, to_host
from .dfa import TtDFA

V1_METRICS = ["no_at_fault_collisions", "drivable_area_compliance",
              "time_to_collision_within_bound", "ego_progress", "comfort"]
ALL_METRICS = V1_METRICS + ["driving_direction_compliance"]


def _topk(dev, row, k):
    """[1, 1, 1, T] scores -> [1, 1, 1, k] uint32 ROW_MAJOR source positions.

    ttnn.topk_select resolves ties to the lower index. torch.topk does NOT --
    measured, 64 equal values gave torch [44, 41, 42, ...] against this op's
    [0, 1, 2, ...] -- so on a tie at the k-th boundary the two can keep
    different rows. Both are valid top-k sets and this one is the reproducible
    one; what it is not is the arbitrary choice the reference PDMS was computed
    with.
    """
    val = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(ttnn.Shape([1, 1, 1, k]), ttnn.bfloat16,
                        ttnn.ROW_MAJOR_LAYOUT, ttnn.BufferType.DRAM), dev)
    idx = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(ttnn.Shape([1, 1, 1, k]), ttnn.uint32,
                        ttnn.ROW_MAJOR_LAYOUT, ttnn.BufferType.DRAM), dev)
    ttnn.topk_select(row, val, idx, k)
    ttnn.deallocate(val)
    return idx


def _like(dev, src, k):
    """A [k, width] gather destination. Allocated, not uploaded: row_gather
    overwrites every row it is given, so zeroing it meant shipping 170 KB a
    frame for nothing."""
    return ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(ttnn.Shape([k, src.shape[-1]]), src.dtype,
                        ttnn.TILE_LAYOUT, ttnn.BufferType.DRAM), dev)


def _gather2(dev, a, b, it, k):
    """out_a[r] = a[it[r]], and the same for b when given. Bit-identical rows.

    `it` is topk_select's uint32 ROW_MAJOR output, consumed directly.
    """
    oa = _like(dev, a, k)
    ob = _like(dev, b, k) if b is not None else None
    ttnn.row_gather(a, it, oa, b, ob, k)
    return oa, ob


def _ln(x_t, w, b):
    return ttnn.layer_norm(x_t, weight=w, bias=b)


class TtDecoderLayer:
    def __init__(self, sd, prefix, device, idx, is_last, num_sample_path=50,
                 num_sample_traj=8):
        self.dev, self.idx, self.is_last = device, idx, is_last
        g = lambda k: sd[prefix + k].float()
        self.p_dfa = TtDFA(sd, prefix + "p_deform_model.", device, num_sample_path)
        self.p_attn = TtMultiheadAttention(sd, prefix + "p_attention.", device)
        self.p_ffn = TtFFN(sd, prefix + "p_ffn.", device)
        self.p_mlp = TtFFN(sd, prefix + "path_mlp.", device)
        self.v_img = TtMultiheadAttention(sd, prefix + "v_img_attention.", device)
        self.v_attn = TtMultiheadAttention(sd, prefix + "v_attention.", device)
        self.v_ffn = TtFFN(sd, prefix + "v_ffn.", device)
        self.v_mlp = TtFFN(sd, prefix + "vel_mlp.", device)
        self.norms = {n: (_t(g(f"{n}.weight"), device), _t(g(f"{n}.bias"), device))
                      for n in ("p_norm1", "p_norm2", "v_norm1", "v_norm2")}
        if is_last:
            self.t_dfa = TtDFA(sd, prefix + "t_deform_model.", device, num_sample_traj)
            self.t_attn = TtMultiheadAttention(sd, prefix + "t_attention.", device)
            self.t_ffn = TtFFN(sd, prefix + "t_ffn.", device)
            self.t_mlp = TtFFN(sd, prefix + "traj_mlp.", device)
            for n in ("t_norm1", "t_norm2"):
                self.norms[n] = (_t(g(f"{n}.weight"), device), _t(g(f"{n}.bias"), device))
            self.heads = {m: TtFFN(sd, prefix + f"metric_heads.{m}.", device)
                          for m in V1_METRICS}

    def _branch(self, x, dfa, anchor, levels, proj_tt, attn, ffn, mlp, n1, n2,
                pre_attn=None, img=None, ti=None):
        """DFA -> (optional cross-attn) -> self-attn -> norm -> FFN -> norm -> score."""
        T = x.shape[0]
        xt = dfa(x, anchor, levels, proj_tt) if dfa is not None else x
        if pre_attn is not None:
            xt = ttnn.add(xt, pre_attn(xt, img, img, tq=T, tk=ti))
        xt = ttnn.add(xt, attn(xt, tq=T))
        xt = _ln(xt, *self.norms[n1])
        xt = ttnn.add(xt, ffn(xt))
        xt = _ln(xt, *self.norms[n2])
        sc = mlp(xt)
        row = ttnn.reshape(ttnn.transpose(sc, 0, 1), (1, 1, 1, T))
        ttnn.deallocate(sc)
        return xt, row

    def __call__(self, path_embed, vel_embed, path_anchor, p_abs, v_abs, status,
                 levels, img_tt, n_img, proj_tt, k_path, k_vel):
        path_embed = ttnn.add(path_embed, status)
        vel_embed = ttnn.add(vel_embed, status)
        p_out, p_scores = self._branch(
            path_embed, self.p_dfa, path_anchor, levels, proj_tt,
            self.p_attn, self.p_ffn, self.p_mlp, "p_norm1", "p_norm2")
        v_out, v_scores = self._branch(
            vel_embed, None, None, levels, proj_tt,
            self.v_attn, self.v_ffn, self.v_mlp, "v_norm1", "v_norm2",
            pre_attn=self.v_img, img=img_tt, ti=n_img)
        pi_tt = _topk(self.dev, p_scores, k_path)
        vi_tt = _topk(self.dev, v_scores, k_vel)
        p_sel, a_sel = _gather2(self.dev, p_out, path_anchor, pi_tt, k_path)
        v_sel, va_sel = _gather2(self.dev, v_out, v_abs, vi_tt, k_vel)
        pa_sel, _ = _gather2(self.dev, p_abs, None, pi_tt, k_path)
        for t in (p_scores, v_scores, pi_tt, vi_tt):
            ttnn.deallocate(t)
        return p_sel, v_sel, a_sel, pa_sel, va_sel

    def trajectory(self, p_emb, v_emb, traj_anchor, levels, proj_tt):
        """Last layer only: 20 paths x 20 velocities -> 400 candidates.

        Returns the winning candidate's index.

        The outer sum runs on device: [n_p, 1, E] + [1, n_v, E] broadcast, then
        flattened. p_emb and v_emb are what row_gather selected, so nothing has
        touched the host since the FPN.
        """
        n_p, n_v = p_emb.shape[0], v_emb.shape[0]
        E = p_emb.shape[-1]
        traj = ttnn.reshape(
            ttnn.add(ttnn.reshape(p_emb, (n_p, 1, E)), ttnn.reshape(v_emb, (1, n_v, E))),
            (n_p * n_v, E))
        T = n_p * n_v
        xt = self.t_dfa(traj, traj_anchor, levels, proj_tt)
        ttnn.deallocate(traj)
        xt = ttnn.add(xt, self.t_attn(xt, tq=T))
        xt = _ln(xt, *self.norms["t_norm1"])
        xt = ttnn.add(xt, self.t_ffn(xt))
        xt = _ln(xt, *self.norms["t_norm2"])
        cols = [ttnn.typecast(self.heads[m](xt), ttnn.float32) for m in V1_METRICS]
        sg = [ttnn.sigmoid(c) for c in cols]
        inner = ttnn.add(ttnn.add(ttnn.multiply(sg[2], 5.0), ttnn.multiply(sg[3], 5.0)),
                         ttnn.multiply(sg[4], 2.0))
        sc = ttnn.multiply(ttnn.multiply(sg[0], sg[1]), inner)
        best = ttnn.argmax(ttnn.reshape(sc, (1, T)), dim=-1)
        for t in cols + sg + [inner, sc]:
            ttnn.deallocate(t)
        return best


class TtDecoder:
    def __init__(self, sd, prefix, device, path_filter=(128, 20), vel_filter=(64, 20)):
        self.dev = device
        self.pf, self.vf = path_filter, vel_filter
        self.layers = [TtDecoderLayer(sd, prefix + f"layers.{i}.", device, i, i == 1)
                       for i in range(2)]

    def __call__(self, path_embed, vel_embed, anchor_tt, status,
                 levels, img_tt, n_img, proj_tt,
                 tv_all, tv_xy, p_abs0, v_abs0, n_vel, poses):
        """Returns the winning trajectory as a DEVICE tensor, [1, poses*3].

        Not read here: a trace capture refuses reads, so the caller does it
        after the replay, off the same buffer.

        Everything but the scores and the final answer stays on device.
        `anchor_tt` is path_vocab[..., :2] flattened, uploaded once at load --
        it is a checkpoint constant -- and each layer's anchor is the previous
        one gathered by that layer's survivors, never re-derived on the host.
        """
        anchor, p_abs, v_abs = anchor_tt, p_abs0, v_abs0
        for i, layer in enumerate(self.layers):
            path_embed, vel_embed, anchor, p_abs, v_abs = layer(
                path_embed, vel_embed, anchor, p_abs, v_abs, status, levels,
                img_tt, n_img, proj_tt, self.pf[i], self.vf[i])

        n_p, n_v = self.pf[-1], self.vf[-1]
        flat = ttnn.add(ttnn.multiply(p_abs, float(n_vel)),
                        ttnn.reshape(v_abs, (1, n_v)))
        fi = ttnn.to_layout(
            ttnn.typecast(ttnn.reshape(flat, (1, 1, 1, n_p * n_v)), ttnn.uint32),
            ttnn.ROW_MAJOR_LAYOUT)
        t_anchor = _like(self.dev, tv_xy, n_p * n_v)
        cands = _like(self.dev, tv_all, n_p * n_v)
        ttnn.row_gather(tv_xy, fi, t_anchor, tv_all, cands, n_p * n_v)
        best = self.layers[-1].trajectory(path_embed, vel_embed, t_anchor, levels,
                                          proj_tt)
        win = _like(self.dev, cands, 1)
        ttnn.row_gather(cands, best, win, None, None, 1)
        for t in (flat, fi, t_anchor, cands, best):
            ttnn.deallocate(t)
        return win
