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
        # The DFA hands back a device tensor, already gathered to every chip.
        # Only the branch without one (velocity) still starts from the host.
        if dfa is not None:
            xt = dfa(x, anchor, levels, proj, proj_tt, iwh)
        else:
            xt = _t(x, self.dev)
        if pre_attn is not None:      # velocity branch attends to image tokens first
            xt = ttnn.add(xt, pre_attn(xt, img, img, tq=T, tk=ti))
        xt = ttnn.add(xt, attn(xt, tq=T))
        xt = _ln(xt, *self.norms[n1])
        xt = ttnn.add(xt, ffn(xt))
        xt = _ln(xt, *self.norms[n2])
        scores = to_host(mlp(xt), self.dev).float()[:T, 0]
        return to_host(xt, self.dev).float()[:T], scores

    def __call__(self, path_embed, vel_embed, path_anchor, status, levels,
                 img_tt, n_img, proj, proj_tt, iwh, k_path, k_vel):
        path_embed = path_embed + status
        vel_embed = vel_embed + status
        p_out, p_scores = self._branch(
            path_embed, self.p_dfa, path_anchor, levels, proj, proj_tt, iwh,
            self.p_attn, self.p_ffn, self.p_mlp, "p_norm1", "p_norm2")
        v_out, v_scores = self._branch(
            vel_embed, None, None, levels, proj, proj_tt, iwh,
            self.v_attn, self.v_ffn, self.v_mlp, "v_norm1", "v_norm2",
            pre_attn=self.v_img, img=img_tt, ti=n_img)
        pi = torch.topk(p_scores, k_path).indices
        vi = torch.topk(v_scores, k_vel).indices
        return p_out[pi], v_out[vi], pi, vi

    def trajectory(self, p_emb, v_emb, traj_anchor, levels, proj, proj_tt, iwh):
        """Last layer only: 20 paths x 20 velocities -> 400 candidates."""
        n_p, n_v = p_emb.shape[0], v_emb.shape[0]
        traj = (p_emb.unsqueeze(1) + v_emb.unsqueeze(0)).reshape(n_p * n_v, -1)
        T = traj.shape[0]
        xt = self.t_dfa(traj, traj_anchor, levels, proj, proj_tt, iwh)
        xt = ttnn.add(xt, self.t_attn(xt, tq=T))
        xt = _ln(xt, *self.norms["t_norm1"])
        xt = ttnn.add(xt, self.t_ffn(xt))
        xt = _ln(xt, *self.norms["t_norm2"])
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

    def __call__(self, path_embed, vel_embed, path_vocab, traj_vocab, status,
                 levels, img_tt, n_img, proj, proj_tt, iwh):
        """Returns the selected trajectory [num_poses, 3].

        img_tt is the last FPN level flattened to image tokens, already on
        device: [cams*H*W, C] = [384, 256] here, and n_img is that first
        dimension.
        """
        p_abs = torch.arange(path_embed.shape[0])
        v_abs = torch.arange(vel_embed.shape[0])
        for i, layer in enumerate(self.layers):
            anchor = path_vocab[p_abs][..., :2].reshape(len(p_abs), -1)
            path_embed, vel_embed, pi, vi = layer(
                path_embed, vel_embed, anchor, status, levels, img_tt, n_img,
                proj, proj_tt, iwh, self.pf[i], self.vf[i])
            # compose rather than gather: layer i selects within layer i-1's
            # survivors, so indices into the original vocabulary chain.
            p_abs, v_abs = p_abs[pi], v_abs[vi]

        last = self.layers[-1]
        tv = traj_vocab[p_abs][:, v_abs]                    # [20, 20, poses, 3]
        anchor = tv[..., :2].reshape(-1, tv.shape[2] * 2)   # [400, poses*2]
        scores = last.trajectory(path_embed, vel_embed, anchor, levels, proj, proj_tt, iwh)
        return tv.reshape(-1, tv.shape[2], tv.shape[3])[int(scores.argmax())]
