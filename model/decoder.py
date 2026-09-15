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


def _idx(dev, i, k):
    """[k] host indices -> [1, 1, 1, k] uint32 ROW_MAJOR, replicated."""
    from .mesh import mesh_of
    return ttnn.from_torch(i.to(torch.int32).reshape(1, 1, 1, k).contiguous(),
                           layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                           dtype=ttnn.uint32, mesh_mapper=mesh_of(dev)[1])


def _like(dev, src, k):
    """A [k, width] gather destination. Allocated, not uploaded: row_gather
    overwrites every row it is given, so zeroing it meant shipping 170 KB a
    frame for nothing."""
    return ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(ttnn.Shape([k, src.shape[-1]]), src.dtype,
                        ttnn.TILE_LAYOUT, ttnn.BufferType.DRAM), dev)


def _gather2(dev, a, b, i, k):
    """out_a[r] = a[i[r]], and the same for b when given. Bit-identical rows."""
    it = _idx(dev, i, k)
    oa = _like(dev, a, k)
    ob = _like(dev, b, k) if b is not None else None
    ttnn.row_gather(a, it, oa, b, ob, k)
    ttnn.deallocate(it)
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
        scores = to_host(mlp(xt), self.dev).float()[:T, 0]
        return xt, scores

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
        pi = torch.topk(p_scores, k_path).indices
        vi = torch.topk(v_scores, k_vel).indices
        # The survivors are selected on device. The path branch gathers its
        # anchor alongside its embedding -- one row_gather carries both pairs --
        # which is also what removes the anchor's own upload: layer i+1's
        # anchor is layer i's gathered by the same indices, never re-derived
        # from the vocabulary on the host.
        p_sel, a_sel = _gather2(self.dev, p_out, path_anchor, pi, k_path)
        v_sel, _ = _gather2(self.dev, v_out, None, vi, k_vel)
        return p_sel, v_sel, a_sel, pi, vi

    def trajectory(self, p_emb, v_emb, traj_anchor, levels, proj, proj_tt, iwh):
        """Last layer only: 20 paths x 20 velocities -> 400 candidates.

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
        lg = {m: to_host(h(xt), self.dev).float()[:T, 0] for m, h in self.heads.items()}
        scores = (torch.sigmoid(lg["no_at_fault_collisions"])
                  * torch.sigmoid(lg["drivable_area_compliance"])) * (
            5 * torch.sigmoid(lg["time_to_collision_within_bound"])
            + 5 * torch.sigmoid(lg["ego_progress"])
            + 2 * torch.sigmoid(lg["comfort"]))
        return scores


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
        scores = last.trajectory(path_embed, vel_embed, t_anchor, levels,
                                 proj, proj_tt, iwh)
        return tv.reshape(-1, tv.shape[2], tv.shape[3])[int(scores.argmax())]
