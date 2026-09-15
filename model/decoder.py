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

    def _branch(self, x, dfa, anchor, levels, proj, proj_tt, iwh, attn, ffn, mlp, n1, n2,
                pre_attn=None, img=None, ti=None):
        """DFA -> (optional cross-attn) -> self-attn -> norm -> FFN -> norm -> score."""
        T = x.shape[0]
        # Everything stays on device. Only the SCORES come back, and only
        # because the top-k has to stay torch.topk: ttnn.topk_select picks the
        # same k values bit for bit but resolves ties to a different index, and
        # a different survivor at the boundary is a different trajectory. The
        # scores are 2 KB; the [T, 256] embedding they used to drag down with
        # them was 1.4 MB.
        xt = dfa(x, anchor, levels, proj, proj_tt, iwh) if dfa is not None else x
        if pre_attn is not None:      # velocity branch attends to image tokens first
            xt = ttnn.add(xt, pre_attn(xt, img, img, tq=T, tk=ti))
        xt = ttnn.add(xt, attn(xt, tq=T))
        xt = _ln(xt, *self.norms[n1])
        xt = ttnn.add(xt, ffn(xt))
        xt = _ln(xt, *self.norms[n2])
        # The scores stay on device: ttnn.topk_select reads them as one
        # logical row, so [T, 1] is transposed rather than downloaded.
        sc = mlp(xt)
        row = ttnn.reshape(ttnn.transpose(sc, 0, 1), (1, 1, 1, T))
        ttnn.deallocate(sc)
        return xt, row

    def __call__(self, path_embed, vel_embed, path_anchor, status, levels,
                 img_tt, n_img, proj, proj_tt, iwh, k_path, k_vel):
        path_embed = ttnn.add(path_embed, status)
        vel_embed = ttnn.add(vel_embed, status)
        p_out, p_scores = self._branch(
            path_embed, self.p_dfa, path_anchor, levels, proj, proj_tt, iwh,
            self.p_attn, self.p_ffn, self.p_mlp, "p_norm1", "p_norm2")
        v_out, v_scores = self._branch(
            vel_embed, None, None, levels, proj, proj_tt, iwh,
            self.v_attn, self.v_ffn, self.v_mlp, "v_norm1", "v_norm2",
            pre_attn=self.v_img, img=img_tt, ti=n_img)
        # top-k on device. Its uint32 ROW_MAJOR indices are exactly what
        # row_gather consumes, so the selection never touches the host: the
        # scores (4 KB at T=1024) do not come down, and the indices do not go
        # back up. The path branch gathers its anchor alongside its embedding
        # -- one row_gather carries both pairs -- which is what removes the
        # anchor's upload too: layer i+1's anchor is layer i's gathered by the
        # same indices, never re-derived from the vocabulary on the host.
        pi_tt = _topk(self.dev, p_scores, k_path)
        vi_tt = _topk(self.dev, v_scores, k_vel)
        p_sel, a_sel = _gather2(self.dev, p_out, path_anchor, pi_tt, k_path)
        v_sel, _ = _gather2(self.dev, v_out, None, vi_tt, k_vel)
        # The indices DO come back, 80 bytes of them, because the final answer
        # is a row of traj_vocab and picking it is a host index either way.
        pi = to_host(pi_tt, self.dev).reshape(-1)[:k_path].long()
        vi = to_host(vi_tt, self.dev).reshape(-1)[:k_vel].long()
        for t in (p_scores, v_scores, pi_tt, vi_tt):
            ttnn.deallocate(t)
        return p_sel, v_sel, a_sel, pi, vi

    def trajectory(self, p_emb, v_emb, traj_anchor, levels, proj, proj_tt, iwh):
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
        xt = self.t_dfa(traj, traj_anchor, levels, proj, proj_tt, iwh)
        ttnn.deallocate(traj)
        xt = ttnn.add(xt, self.t_attn(xt, tq=T))
        xt = _ln(xt, *self.norms["t_norm1"])
        xt = ttnn.add(xt, self.t_ffn(xt))
        xt = _ln(xt, *self.norms["t_norm2"])
        # The five logits come down and the score is composed on the HOST, in
        # fp32. Assembling it on device instead was tried and changed the
        # answer -- max|d| 2.3 on the trajectory, a different candidate. The
        # top two candidates sit 1.7e-06 apart in relative terms and bf16
        # carries about 4e-3, so the sigmoids and the weighted sum have to be
        # done wider than the device does them by default. Five downloads of
        # 2 KB is what that costs.
        # The PDM score is assembled on device and only the winning index
        # comes back -- four bytes.
        #
        # IN FP32, not the default. bf16 was tried and moved the trajectory by
        # 2.3, a different candidate, and the arithmetic says why: against the
        # host's fp32 the relative error is 7.6e-03 in bf16 and 8.3e-08 in
        # fp32, while the top two candidates sit 1.7e-06 apart. bf16 is 4500x
        # the margin; fp32 is a twentieth of it.
        cols = [ttnn.typecast(self.heads[m](xt), ttnn.float32) for m in V1_METRICS]
        sg = [ttnn.sigmoid(c) for c in cols]
        inner = ttnn.add(ttnn.add(ttnn.multiply(sg[2], 5.0), ttnn.multiply(sg[3], 5.0)),
                         ttnn.multiply(sg[4], 2.0))
        sc = ttnn.multiply(ttnn.multiply(sg[0], sg[1]), inner)
        best = int(to_host(ttnn.argmax(ttnn.reshape(sc, (1, T)), dim=-1), self.dev)
                   .reshape(-1)[0])
        for t in cols + sg + [inner, sc]:
            ttnn.deallocate(t)
        return best


class TtDecoder:
    def __init__(self, sd, prefix, device, path_filter=(128, 20), vel_filter=(64, 20)):
        self.dev = device
        self.pf, self.vf = path_filter, vel_filter
        self.layers = [TtDecoderLayer(sd, prefix + f"layers.{i}.", device, i, i == 1)
                       for i in range(2)]

    def __call__(self, path_embed, vel_embed, anchor_tt, traj_vocab, status,
                 levels, img_tt, n_img, proj, proj_tt, iwh):
        """Returns the selected trajectory [num_poses, 3].

        Everything but the scores and the final answer stays on device.
        `anchor_tt` is path_vocab[..., :2] flattened, uploaded once at load --
        it is a checkpoint constant -- and each layer's anchor is the previous
        one gathered by that layer's survivors, never re-derived on the host.
        """
        p_abs = torch.arange(path_embed.shape[0])
        v_abs = torch.arange(vel_embed.shape[0])
        anchor = anchor_tt
        for i, layer in enumerate(self.layers):
            path_embed, vel_embed, anchor, pi, vi = layer(
                path_embed, vel_embed, anchor, status, levels, img_tt, n_img,
                proj, proj_tt, iwh, self.pf[i], self.vf[i])
            # The absolute indices still compose on the host, for one reason:
            # the final answer is a row of traj_vocab, and picking it is a
            # host-side index either way.
            p_abs, v_abs = p_abs[pi], v_abs[vi]

        last = self.layers[-1]
        tv = traj_vocab[p_abs][:, v_abs]                    # [20, 20, poses, 3]
        t_anchor = _t(tv[..., :2].reshape(-1, tv.shape[2] * 2), self.dev,
                      dtype=ttnn.float32)                   # [400, poses*2]
        best = last.trajectory(path_embed, vel_embed, t_anchor, levels,
                               proj, proj_tt, iwh)
        return tv.reshape(-1, tv.shape[2], tv.shape[3])[best]
