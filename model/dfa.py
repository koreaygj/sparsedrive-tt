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
  - clp is ordered (camera, point, level), not the (camera, level, point) the
    checkpoint emits. The mask is constant over level and group, so that order
    puts it constant over the last L*G = 32 columns -- exactly a tile width --
    and one repeat_interleave builds it on device. See _mask_layout_perm.
  - the mask silences a camera only where another camera sees the point, which
    is also what keeps the denominator non-zero.
  - features never exist whole -- 2.93 GiB at layer 0 -- so anchors are walked
    in blocks, and the clp reduction is split to bound the bf16 accumulator.
"""

import torch
import ttnn

def _cast(x, dt):
    """Convert on the host before handing the tensor over.

    ttnn.from_torch will convert dtype itself, and it is slow at it: a 5.9 MB
    grid took 160.9 ms as fp32 -> bfloat16, against 62.5 ms when torch did the
    cast first and from_torch only copied. Same bits either way.
    """
    if dt == ttnn.bfloat16 and x.dtype != torch.bfloat16:
        return x.bfloat16()
    if dt == ttnn.float32 and x.dtype != torch.float32:
        return x.float()
    return x


FIX_HEIGHT = (0.0, -0.25, -0.5, 0.25, 0.5)
NUM_LEARNABLE = 2
LOFI = dict(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True)
HIFI = dict(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)


def _mask_layout_perm(L, P, G):
    """Reorder one camera's weights_fc outputs from (level, point, group) to
    (point, level, group).

    The mask depends on camera and point only, never on level or group. Under
    the checkpoint's (l, p, g) order it therefore repeats along two axes at
    once -- stride P*G for the level and stride 1 for the group -- and no stock
    op expands it cheaply: repeat_interleave by G=8 is sub-tile, and tiling the
    result L times concatenates 94 MB four times. Measured 80.8 ms, against
    58.6 ms to build the same thing on the host and ship it.

    Under (p, l, g) the mask is constant over the trailing L*G = 32 columns,
    which is one tile wide, so a single repeat_interleave(32) is a tile-aligned
    copy: 3.7 ms, plus 0.8 ms to upload the [n, C*P] base it expands from.

    Nothing numeric changes. This permutes the rows of weights_fc's weight and
    bias once at load, and `features` is assembled in the matching order, so
    grouped_weighted_sum sees the same (weight, feature) pairs it saw before --
    it only sums over clp, so the order within clp never mattered to it.
    """
    return torch.arange(L * P * G).reshape(L, P, G).permute(1, 0, 2).reshape(-1)


def legal_splits(clp, G=8):
    """clp splits the COMPACT weight layout accepts: (clp/k)*G tile-aligned."""
    return [k for k in range(1, 33) if clp % k == 0 and (clp // k) * G % 32 == 0]


class TtDFA:
    def __init__(self, sd, prefix, device, num_sample, num_cams=3, num_levels=4,
                 num_groups=8, embed_dims=256):
        """`device` may be a MeshDevice, in which case anchors shard across it.

        Anchors rather than cameras. The softmax normalises over clp for one
        anchor, so an anchor-sharded tensor keeps every denominator local and
        no all_reduce is needed -- which also removes the failure sparse4D-tt
        hit when it split 6 cameras 3/3 and each device normalised over its own
        half. The feature maps replicate instead; they are 8.4M elements.
        """
        self.dev, self.C, self.L, self.G, self.E = device, num_cams, num_levels, num_groups, embed_dims
        self.nd = device.get_num_devices() if hasattr(device, "get_num_devices") else 1
        self.rep = ttnn.ReplicateTensorToMesh(device) if self.nd > 1 else None
        self.sh0 = ttnn.ShardTensorToMesh(device, dim=0) if self.nd > 1 else None
        self.sh1 = ttnn.ShardTensorToMesh(device, dim=1) if self.nd > 1 else None
        self.cat0 = ttnn.ConcatMeshToTensor(device, dim=0) if self.nd > 1 else None
        self.ns = num_sample
        self.P = num_sample * len(FIX_HEIGHT) * NUM_LEARNABLE
        self.clp = num_cams * num_levels * self.P
        self.wide = self.clp * num_groups
        self.lo = ttnn.WormholeComputeKernelConfig(**LOFI)
        self.hi = ttnn.WormholeComputeKernelConfig(**HIFI)

        g = lambda k: sd[prefix + k].float()
        T = lambda x, dt=ttnn.bfloat16: ttnn.from_torch(
            _cast(x, dt).contiguous(), layout=ttnn.TILE_LAYOUT, device=device,
            dtype=dt, mesh_mapper=self.rep)
        self.T, self.f32 = T, ttnn.float32
        # anchor-sharded uploads, and the composer that puts them back together
        self.S = lambda x, dt=ttnn.bfloat16, d=0: ttnn.from_torch(
            _cast(x, dt).contiguous(), layout=ttnn.TILE_LAYOUT, device=device, dtype=dt,
            mesh_mapper=(self.sh0 if d == 0 else self.sh1) if self.nd > 1 else None)
        self.G2T = lambda t: ttnn.to_torch(t, mesh_composer=self.cat0)

        self.Wl = T(g("kps_generator.learnable_fc.weight").t(), ttnn.float32)
        self.Bl = T(g("kps_generator.learnable_fc.bias").reshape(1, -1), ttnn.float32)
        # (l, p, g) -> (p, l, g); see _mask_layout_perm.
        self.LG = num_levels * num_groups
        assert self.LG % 32 == 0, (
            f"the mask expansion wants num_levels*num_groups tile-aligned, got {self.LG}")
        wp = _mask_layout_perm(num_levels, self.P, num_groups)
        self.Wf = T(g("weights_fc.weight")[wp].t())
        self.Bf = T(g("weights_fc.bias")[wp].reshape(1, -1))
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
            _cast(mc_ms_feat[0, :, s:s + h * w, :].reshape(self.C, h, w, self.E),
                  ttnn.bfloat16).contiguous(),
            layout=ttnn.ROW_MAJOR_LAYOUT, device=self.dev, dtype=ttnn.bfloat16,
            mesh_mapper=self.rep)
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
        """-> (grid on host, in-bounds mask on host).

        The grid round-trips: u and v are computed on device, brought back,
        stacked, and uploaded again. Profiling puts that upload at 142 ms of
        DAF[0]'s 1136 (12.5%), so assembling it on device looks like free
        money. It is not -- measured, that made DAF[0] 1113 -> 1482 ms.

        The reason is the same tile arithmetic that shapes the rest of this
        port: the grid's last dimension is 2, and a TILE tensor pads its last
        dimension to 32, so building it from [1, n*P, 1, 1] pieces and
        concatenating spends 30 of every 32 columns on padding. The host
        round-trip moves less than the device assembly wastes.

        Doing this properly needs a kernel that writes the packed layout
        directly -- which is what sparse4D-tt's ttnn.grid_precompute is for.
        Left as a round-trip until there is reason to build that.
        """
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
            u = self.G2T(ttnn.multiply(ttnn.divide(X, Z), 1.0 / float(iwh[c, 0]))).float()
            v = self.G2T(ttnn.multiply(ttnn.divide(Y, Z), 1.0 / float(iwh[c, 1]))).float()
            grids.append(torch.stack([u * 2 - 1, v * 2 - 1], -1))
            inb.append((u > 0) & (u < 1) & (v > 0) & (v < 1))
            for q in (cx, cy, cz, X, Y, Z):
                ttnn.deallocate(q)
        ttnn.deallocate(x); ttnn.deallocate(y)
        return torch.stack(grids, 0).reshape(self.C, n * self.P, 1, 2), torch.stack(inb, 1)

    def _weights(self, feat_tt, cam, n, base):
        """n is the global anchor count; each device holds n // nd of them.

        `base` is the mask at [n, C*P], one value per (camera, point); the
        expansion to the full [n, clp*G] happens here, on device.
        """
        nl = n // self.nd
        fx = ttnn.add(ttnn.reshape(feat_tt, (nl, 1, self.E)),
                      ttnn.reshape(cam, (1, self.C, self.E)))
        wl = ttnn.linear(ttnn.reshape(fx, (nl * self.C, self.E)), self.Wf, bias=self.Bf,
                         compute_kernel_config=self.hi)
        logits = ttnn.reshape(wl, (nl, self.wide))
        keep = ttnn.repeat_interleave(base, self.LG, dim=-1)
        mx = ttnn.max(logits, dim=-1, keepdim=True)
        e = ttnn.multiply(ttnn.exp(ttnn.subtract(logits, mx)), keep)
        ttnn.deallocate(keep)
        s = ttnn.matmul(e, self.gt, compute_kernel_config=self.lo, dtype=self.f32)
        sb = ttnn.matmul(s, self.st, compute_kernel_config=self.hi, dtype=self.f32)
        ttnn.deallocate(s)
        w = ttnn.divide(e, sb)
        ttnn.deallocate(e); ttnn.deallocate(sb)
        return w

    def __call__(self, feat, anchor, levels_tt, proj, iwh, chunk=1024, splits=None):
        """feat [n, E] torch, anchor [n, num_sample*2] torch. -> [n, E] torch.

        Big blocks. Measured on DAF[0], PCC identical to six decimals at every
        size -- the arithmetic does not change, only how often the launch and
        the slicing are re-paid:

            chunk   single    mesh(1,2)   per-chip [clp,m,E]
              256    944.5      663.7      750 / 375 MB
              512    857.6      555.9     1500 / 750 MB
             1024    794.7      492.7     3000 / 1500 MB

        1024 is every anchor of the layer-0 path branch in one block, which
        means the [clp, m, E] buffer this file's header calls impossible to
        materialise -- 2.93 GiB -- is in fact materialised, and is faster that
        way. A 12 GB chip has room; the chunking overhead cost more than the
        bandwidth did. The header's claim was an overstatement: the tensor
        cannot fit in L1, not that it cannot exist.

        Returns are diminishing (1.10x then 1.08x single, 1.19x then 1.13x on
        mesh), so this is the end of the free lunch.
        """
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
            assert m % self.nd == 0, (
                f"chunk {m} must divide across {self.nd} devices; "
                f"pick a chunk that divides {n} into multiples of {self.nd}")
            f_tt = self.S(feat[a0:a1])
            grid, inb = self._project(self.S(feat[a0:a1], self.f32),
                                      self.S(anchor[a0:a1], self.f32), m, proj, iwh)
            # [m, C, P] -- the (camera, point) mask, unexpanded. The
            # expansion to [m, clp*G] is a device repeat_interleave inside
            # _weights: 94 MB of transfer becomes 1.5 MB. inb is [m, C, P].
            seen = inb.sum(1, keepdim=True) != 0
            base = (~torch.logical_and(~inb, seen)).float().reshape(m, self.C * self.P)
            W = self._weights(f_tt, cam, m, self.S(base))
            # grid is [C, m*P, 1, 2] with points anchor-major, so sharding dim 1
            # splits the same anchors that dim 0 of feat did.
            # Upload wide, reshape on device.
            #
            # from_torch costs per row, not per byte. The grid's natural shape
            # [C, m*P, 1, 2] is two bf16 values per row, which uploads at
            # 50 MB/s; the same bytes as [C, m*P*2/256, 256] go at 4516 MB/s.
            # Measured on the layer-0 grid, 5.9 MB: 118.0 ms direct against
            # 1.2 ms wide plus 6.5 ms for the device-side reshape. The sampled
            # output is bit-identical -- nothing changes but the row count.
            #
            # Shard on dim 1 either way: the wide shape keeps anchor-major
            # order, so half the rows is still the first half of the anchors.
            gw = _cast(grid, ttnn.bfloat16).reshape(
                self.C, m * self.P * 2 // 256, 256).contiguous()
            g_tt = ttnn.reshape(
                ttnn.from_torch(gw, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.dev,
                                dtype=ttnn.bfloat16, mesh_mapper=self.sh1),
                (self.C, (m // self.nd) * self.P, 1, 2))
            ml = m // self.nd
            per = []
            # clp is (camera, point, level): concat on the level axis with the
            # point axis already outside it. The old (camera, level, point)
            # order concatenated four [C, P, ml, E] blocks on dim 1; this
            # concatenates four [C*P, 1, ml, E] blocks on dim 1 instead.
            # Measured identical, 123.0 ms against 122.9 -- the reorder is free,
            # and it is what lets the mask expand on device.
            for fm in levels_tt:
                s_ = ttnn.grid_sample(fm, g_tt, padding_mode="zeros", align_corners=False)
                per.append(ttnn.reshape(
                    ttnn.permute(ttnn.reshape(s_, (self.C, ml, self.P, self.E)),
                                 (0, 2, 1, 3)),
                    (self.C * self.P, 1, ml, self.E)))
            feats = ttnn.reshape(ttnn.concat(per, dim=1), (self.clp, ml, self.E))
            sl = self.clp // splits
            acc = torch.zeros(m, self.E)
            for i in range(splits):
                fi = ttnn.slice(feats, [i * sl, 0, 0], [(i + 1) * sl, ml, self.E])
                wi = ttnn.slice(W, [0, i * sl * self.G], [ml, (i + 1) * sl * self.G])
                o = ttnn.grouped_weighted_sum(fi, wi, num_groups=self.G,
                                              group_dims=self.E // self.G)
                mp_ = ((ml + 31) // 32) * 32
                acc += self.G2T(ttnn.add(ttnn.slice(o, [0, 0], [ml, self.E]),
                                         ttnn.slice(o, [mp_, 0], [mp_ + ml, self.E]))).float()
                ttnn.deallocate(fi); ttnn.deallocate(wi); ttnn.deallocate(o)
            proj_out = ttnn.linear(self.S(acc), self.Wo, bias=self.Bo,
                                   compute_kernel_config=self.hi)
            out[a0:a1] = self.G2T(ttnn.add(proj_out, f_tt)).float()[:m]
            for t in (f_tt, g_tt, feats, W, proj_out):
                ttnn.deallocate(t)
        return out
