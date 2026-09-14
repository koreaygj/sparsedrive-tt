"""DeformableFeatureAggregation on TT-NN.

Called three times per frame with three shapes, and only `num_sample` differs:

    layer 0 path   1024 anchors x 50 path points   -> 500 keypoints, clp 6000
    layer 1 path    128 anchors x 50               -> 500,            clp 6000
    layer 1 traj    400 queries x  8 poses         ->  80,            clp  960

    num_pts = num_sample * len(fix_height) * num_learnable_pts
    clp     = num_cams * num_levels * num_pts

The stage-by-stage reasoning behind each choice here is in docs/ROADMAP.md; the
short version:

  - keypoints and projection are fused and fp32. z is a constant per
    fix_height and the homogeneous 4th is 1, so nothing materialises
    [n, num_pts, 3] -- it all stays [n, num_pts], which tiles. fp32 because
    coordinates are geometry: a path point reaches 50 m where bf16 resolves to
    0.2 m, which is pixels after projection.
  - the softmax runs on the compact [n, clp*G] layout weights_fc already emits,
    with the strided per-group sum done as a 0/1 matmul. Its two matmuls need
    different fidelity: the gather is bf16 x 0/1 (fidelity irrelevant), the
    scatter takes the fp32 denominator (LoFi truncates it).
  - the mask silences a camera only where another camera sees the point, which
    is also what keeps the denominator non-zero.
  - features never exist whole -- 2.93 GiB at layer 0 -- so anchors are walked
    in blocks, and the clp reduction is split to bound the bf16 accumulator.
"""

import torch
import ttnn

FIX_HEIGHT = (0.0, -0.25, -0.5, 0.25, 0.5)
NUM_LEARNABLE = 2
LOFI = dict(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True)
HIFI = dict(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)


def legal_splits(clp, G=8):
    """clp splits the COMPACT weight layout accepts: (clp/k)*G tile-aligned."""
    return [k for k in range(1, 33) if clp % k == 0 and (clp // k) * G % 32 == 0]


class TtDFA:
    def __init__(self, sd, prefix, device, num_sample, num_cams=3, num_levels=4,
                 num_groups=8, embed_dims=256):
        self.dev, self.C, self.L, self.G, self.E = device, num_cams, num_levels, num_groups, embed_dims
        self.ns = num_sample
        self.P = num_sample * len(FIX_HEIGHT) * NUM_LEARNABLE
        self.clp = num_cams * num_levels * self.P
        self.wide = self.clp * num_groups
        self.lo = ttnn.WormholeComputeKernelConfig(**LOFI)
        self.hi = ttnn.WormholeComputeKernelConfig(**HIFI)

        g = lambda k: sd[prefix + k].float()
        T = lambda x, dt=ttnn.bfloat16: ttnn.from_torch(
            x.contiguous(), layout=ttnn.TILE_LAYOUT, device=device, dtype=dt)
        self.T, self.f32 = T, ttnn.float32

        self.Wl = T(g("kps_generator.learnable_fc.weight").t(), ttnn.float32)
        self.Bl = T(g("kps_generator.learnable_fc.bias").reshape(1, -1), ttnn.float32)
        self.Wf = T(g("weights_fc.weight").t())
        self.Bf = T(g("weights_fc.bias").reshape(1, -1))
        self.Wo = T(g("output_proj.weight").t())
        self.Bo = T(g("output_proj.bias").reshape(1, -1))
        self.ce_w = [T(g(f"camera_encoder.{i}.weight").t()) for i in (0, 3)]
        self.ce_b = [T(g(f"camera_encoder.{i}.bias").reshape(1, -1)) for i in (0, 3)]
        self.ln_w = [T(g(f"camera_encoder.{i}.weight")) for i in (2, 5)]
        self.ln_b = [T(g(f"camera_encoder.{i}.bias")) for i in (2, 5)]

        gather = torch.zeros(self.wide, num_groups)
        for q in range(num_groups):
            gather[q::num_groups, q] = 1.0
        self.gt, self.st = T(gather), T(gather.t().contiguous())

        # 0/1 selectors: offset -> x,y and the fan-out of num_sample anchor
        # points to num_pts keypoints. Exact under any fidelity.
        H, Lp, P = len(FIX_HEIGHT), NUM_LEARNABLE, self.P
        Ox, Oy = torch.zeros(P * 2, P), torch.zeros(P * 2, P)
        Ax, Ay = torch.zeros(num_sample * 2, P), torch.zeros(num_sample * 2, P)
        for p in range(P):
            Ox[2 * p, p] = Oy[2 * p + 1, p] = 1.0
            s = p // (H * Lp)
            Ax[2 * s, p] = Ay[2 * s + 1, p] = 1.0
        self.Ox, self.Oy = T(Ox, ttnn.float32), T(Oy, ttnn.float32)
        self.Ax, self.Ay = T(Ax, ttnn.float32), T(Ay, ttnn.float32)
        self.zc = torch.tensor([FIX_HEIGHT[(p // Lp) % H] for p in range(P)])

    def upload_levels(self, mc_ms_feat, shapes, starts):
        """[1, C, F, E] -> per-level NHWC on device. Once per frame."""
        return [ttnn.from_torch(
            mc_ms_feat[0, :, s:s + h * w, :].reshape(self.C, h, w, self.E).contiguous(),
            layout=ttnn.ROW_MAJOR_LAYOUT, device=self.dev, dtype=ttnn.bfloat16)
            for (h, w), s in zip(shapes, starts)]

    def _camera_embed(self, proj):
        ce = self.T(proj[:, :3].reshape(self.C, -1))
        h = ttnn.layer_norm(ttnn.relu(ttnn.linear(ce, self.ce_w[0], bias=self.ce_b[0],
                                                  compute_kernel_config=self.hi)),
                            weight=self.ln_w[0], bias=self.ln_b[0])
        return ttnn.layer_norm(ttnn.relu(ttnn.linear(h, self.ce_w[1], bias=self.ce_b[1],
                                                     compute_kernel_config=self.hi)),
                               weight=self.ln_w[1], bias=self.ln_b[1])

    def _project(self, feat32, anchor32, n, proj, iwh):
        off = ttnn.linear(feat32, self.Wl, bias=self.Bl, compute_kernel_config=self.hi)
        x = ttnn.add(ttnn.matmul(off, self.Ox, compute_kernel_config=self.hi),
                     ttnn.matmul(anchor32, self.Ax, compute_kernel_config=self.hi))
        y = ttnn.add(ttnn.matmul(off, self.Oy, compute_kernel_config=self.hi),
                     ttnn.matmul(anchor32, self.Ay, compute_kernel_config=self.hi))
        ttnn.deallocate(off)
        grids, inb = [], []
        for c in range(self.C):
            Pm = proj[c]
            cst = (Pm[:3, 2] * self.zc.unsqueeze(-1) + Pm[:3, 3]).t()
            cx, cy, cz = (self.T(cst[i].reshape(1, -1), self.f32) for i in range(3))
            X = ttnn.add(ttnn.add(ttnn.multiply(x, float(Pm[0, 0])),
                                  ttnn.multiply(y, float(Pm[0, 1]))), cx)
            Y = ttnn.add(ttnn.add(ttnn.multiply(x, float(Pm[1, 0])),
                                  ttnn.multiply(y, float(Pm[1, 1]))), cy)
            Z = ttnn.clamp(ttnn.add(ttnn.add(ttnn.multiply(x, float(Pm[2, 0])),
                                             ttnn.multiply(y, float(Pm[2, 1]))), cz),
                           1e-5, 1e30)
            u = ttnn.to_torch(ttnn.multiply(ttnn.divide(X, Z), 1.0 / float(iwh[c, 0]))).float()
            v = ttnn.to_torch(ttnn.multiply(ttnn.divide(Y, Z), 1.0 / float(iwh[c, 1]))).float()
            grids.append(torch.stack([u * 2 - 1, v * 2 - 1], -1))
            inb.append((u > 0) & (u < 1) & (v > 0) & (v < 1))
            for q in (cx, cy, cz, X, Y, Z):
                ttnn.deallocate(q)
        ttnn.deallocate(x); ttnn.deallocate(y)
        return torch.stack(grids, 0).reshape(self.C, n * self.P, 1, 2), torch.stack(inb, 1)

    def _weights(self, feat_tt, cam, n, keep):
        fx = ttnn.add(ttnn.reshape(feat_tt, (n, 1, self.E)),
                      ttnn.reshape(cam, (1, self.C, self.E)))
        wl = ttnn.linear(ttnn.reshape(fx, (n * self.C, self.E)), self.Wf, bias=self.Bf,
                         compute_kernel_config=self.hi)
        logits = ttnn.reshape(wl, (n, self.wide))
        mx = ttnn.max(logits, dim=-1, keepdim=True)
        e = ttnn.multiply(ttnn.exp(ttnn.subtract(logits, mx)), self.T(keep))
        s = ttnn.matmul(e, self.gt, compute_kernel_config=self.lo, dtype=self.f32)
        sb = ttnn.matmul(s, self.st, compute_kernel_config=self.hi, dtype=self.f32)
        ttnn.deallocate(s)
        w = ttnn.divide(e, sb)
        ttnn.deallocate(e); ttnn.deallocate(sb)
        return w

    def __call__(self, feat, anchor, levels_tt, proj, iwh, chunk=128, splits=None):
        """feat [n, E] torch, anchor [n, num_sample*2] torch. -> [n, E] torch."""
        n = feat.shape[0]
        if splits is None:
            splits = max(k for k in legal_splits(self.clp) if k <= 10)
        assert splits in legal_splits(self.clp), \
            f"clp {self.clp}: splits must be one of {legal_splits(self.clp)}"
        cam = self._camera_embed(proj)
        out = torch.zeros(n, self.E)
        for a0 in range(0, n, chunk):
            a1 = min(a0 + chunk, n)
            m = a1 - a0
            f_tt = self.T(feat[a0:a1])
            grid, inb = self._project(self.T(feat[a0:a1], self.f32),
                                      self.T(anchor[a0:a1], self.f32), m, proj, iwh)
            mp = inb[:, :, None, :, None]
            keep = (~torch.logical_and(~mp, mp.sum(1, keepdim=True) != 0)).float()
            keep = keep.expand(m, self.C, self.L, self.P, self.G).reshape(m, self.wide)
            W = self._weights(f_tt, cam, m, keep)
            g_tt = ttnn.from_torch(grid, layout=ttnn.ROW_MAJOR_LAYOUT,
                                   device=self.dev, dtype=ttnn.bfloat16)
            per = []
            for fm in levels_tt:
                s_ = ttnn.grid_sample(fm, g_tt, padding_mode="zeros", align_corners=False)
                per.append(ttnn.permute(ttnn.reshape(s_, (self.C, m, self.P, self.E)),
                                        (0, 2, 1, 3)))
            feats = ttnn.reshape(ttnn.concat(per, dim=1), (self.clp, m, self.E))
            sl = self.clp // splits
            acc = torch.zeros(m, self.E)
            for i in range(splits):
                fi = ttnn.slice(feats, [i * sl, 0, 0], [(i + 1) * sl, m, self.E])
                wi = ttnn.slice(W, [0, i * sl * self.G], [m, (i + 1) * sl * self.G])
                o = ttnn.grouped_weighted_sum(fi, wi, num_groups=self.G,
                                              group_dims=self.E // self.G)
                mp_ = ((m + 31) // 32) * 32
                acc += ttnn.to_torch(ttnn.add(ttnn.slice(o, [0, 0], [m, self.E]),
                                              ttnn.slice(o, [mp_, 0], [mp_ + m, self.E]))).float()
                ttnn.deallocate(fi); ttnn.deallocate(wi); ttnn.deallocate(o)
            proj_out = ttnn.linear(self.T(acc), self.Wo, bias=self.Bo,
                                   compute_kernel_config=self.hi)
            out[a0:a1] = ttnn.to_torch(ttnn.add(proj_out, f_tt)).float()[:m]
            for t in (f_tt, g_tt, feats, W, proj_out):
                ttnn.deallocate(t)
        return out
