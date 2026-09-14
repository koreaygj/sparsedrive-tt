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
  - grid_sample gets a PRECOMPUTED grid. Its reader derives h0/w0 and the four
    bilinear weights in soft float on a core with no FPU, which measured 70% of
    the op; ttnn.grid_precompute does that on the Tensix engines instead. See
    _gp_consts for the constant pack it takes.
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


# grid_precompute's coords tile is 32 columns wide and the reader fills 2*K of
# them, so a row carries at most 16 points. That is a tile shape, not a model
# quantity -- a "row" here is just 16 consecutive grid points and may straddle
# anchors, which nothing downstream can see because grid_sample's K-batched
# output flattens back to point order.
GP_K = 16
Q14_SHIFT = 14                     # must match grid_precompute and grid_sample

FIX_HEIGHT = (0.0, -0.25, -0.5, 0.25, 0.5)
NUM_LEARNABLE = 2
LOFI = dict(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True)
HIFI = dict(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)


def _q14(g):
    """[-1, 1] grid -> Q14 fixed point in an int16.

    Fixed point rather than bf16 because the coordinate is clamped to [-2, 2],
    so a float's exponent buys nothing while the bilinear weight it feeds wants
    uniform absolute precision. Clamping to 2 is safe: a point at |g| = 2 lands
    at least a whole pixel outside the image on every FPN level, so it stays out
    of bounds and its four weights stay zero.
    """
    return torch.round(g.clamp(-2.0, 2.0) * (1 << Q14_SHIFT)).clamp(-32768, 32767).to(torch.int16)


def _gp_consts(shapes, K=GP_K, shift=Q14_SHIFT):
    """The f32 tile pack grid_precompute reads; ordering is its kernel's contract.

        tile      l           SCALE_l   per-column affine scale, Q14 folded in
        tile   NL+l           BIAS_l    per-column affine offset
        tile  2NL+l           C_l       per-column bound, size - 1
        tile 3NL+5j+{0..4}    SB0 SB1 SH0 SH1 SI   selectors for output tile j

    The coords tile holds the grid row as it sits in memory: column 2k is point
    k's x, column 2k+1 its y. So the even columns carry width-derived constants
    and the odd ones height-derived, and the selectors read x from 2k and y from
    2k+1.

    The kernel computes, per column, P = coords*SCALE + BIAS, F = floor(P),
    R = P - F, then the two boundary-masked factors FA0 = (1-R)*[0 <= F <= C]
    and FA1 = R*[0 <= F+1 <= C]. The four bilinear weights factorise into those,
    which is why four 0/1 selectors suffice:

        out = (FA0 @ SB0 + FA1 @ SB1) * (FA0 @ SH0 + FA1 @ SH1) + F @ SI

    Output fields are FIELD-MAJOR, the layout grid_sample's precomputed reader
    expects: [h0 x K][w0 x K][nw x K][ne x K][sw x K][se x K]. h0 is a height
    index so it comes from the y column; w0 from the x column. SI routes them
    into fields 0 and 1, where both weight factors are zero by construction.
    """
    NL, OT = len(shapes), (K * 6 + 31) // 32
    T = torch.zeros(3 * NL + 5 * OT, 32, 32, dtype=torch.float32)
    inv = 1.0 / (1 << shift)
    for l, (H, W) in enumerate(shapes):
        for c in range(32):
            S = W if c % 2 == 0 else H
            T[l, :, c] = (S / 2) * inv             # align_corners=False: g*S/2 + (S-1)/2
            T[NL + l, :, c] = (S - 1) / 2
            T[2 * NL + l, :, c] = S - 1
    for j in range(OT):
        SB0, SB1, SH0, SH1, SI = (T[3 * NL + 5 * j + i] for i in range(5))
        for o in range(32):
            og = j * 32 + o
            if og >= 6 * K:
                break
            f, k = og // K, og % K
            if f == 0:                             # h0 <- floor of the y column
                SI[2 * k + 1, o] = 1.0
            elif f == 1:                           # w0 <- floor of the x column
                SI[2 * k, o] = 1.0
            else:                                  # nw ne sw se
                (SB0 if f in (2, 4) else SB1)[2 * k, o] = 1.0
                (SH0 if f in (2, 3) else SH1)[2 * k + 1, o] = 1.0
    return T.reshape(-1, 32).contiguous()


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
        # grid_precompute's constant pack, built on first use from the level
        # shapes the feature tensors carry. It depends on nothing else, so all
        # three DFA instances would build the same one.
        self._consts, self._consts_key = None, None

    def upload_levels(self, mc_ms_feat, shapes, starts):
        """[1, C, F, E] -> per-level NHWC on device. Once per frame."""
        return [ttnn.from_torch(
            _cast(mc_ms_feat[0, :, s:s + h * w, :].reshape(self.C, h, w, self.E),
                  ttnn.bfloat16).contiguous(),
            layout=ttnn.ROW_MAJOR_LAYOUT, device=self.dev, dtype=ttnn.bfloat16,
            mesh_mapper=self.rep)
            for (h, w), s in zip(shapes, starts)]

    def _gp_pack(self, levels_tt):
        key = tuple((int(t.shape[1]), int(t.shape[2])) for t in levels_tt)
        if key != self._consts_key:
            self._consts = ttnn.from_torch(
                _gp_consts(key), layout=ttnn.TILE_LAYOUT, device=self.dev,
                dtype=ttnn.float32, mesh_mapper=self.rep)
            self._consts_key = key
        return self._consts

    def _camera_embed(self, proj):
        ce = self.T(proj[:, :3].reshape(self.C, -1))
        h = ttnn.layer_norm(ttnn.relu(ttnn.linear(ce, self.ce_w[0], bias=self.ce_b[0],
                                                  compute_kernel_config=self.hi)),
                            weight=self.ln_w[0], bias=self.ln_b[0])
        return ttnn.layer_norm(ttnn.relu(ttnn.linear(h, self.ce_w[1], bias=self.ce_b[1],
                                                     compute_kernel_config=self.hi)),
                               weight=self.ln_w[1], bias=self.ln_b[1])

    def _project(self, feat32, anchor32, n, proj, iwh):
        """-> (Q14 grid rows [C, n*P/GP_K, 1, 2*GP_K] on host, in-bounds mask).

        The grid round-trips: u and v are computed on device, brought back,
        stacked, and uploaded again. Assembling it on device instead was
        measured and lost -- the grid's last dimension is 2, and a TILE tensor
        pads its last dimension to 32, so building it from [1, n*P, 1, 1]
        pieces spends 30 of every 32 columns on padding. The round trip moves
        less than the device assembly wastes.

        What the round trip costs now is 8 ms of a 324 ms call, because the
        rows go up as Q14 int16 in groups of GP_K points: 64 bytes a row and
        half the bytes bf16 would take. Removing it needs ttnn.grid_compact,
        which produces the kept rows and their flags on device.
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
        g = torch.stack(grids, 0).reshape(self.C, n * self.P, 2)
        return (_q14(g).reshape(self.C, n * self.P // GP_K, 1, 2 * GP_K).contiguous(),
                torch.stack(inb, 1))

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
        assert (self.P * self.nd) % GP_K == 0 or (n * self.P) % (GP_K * self.nd) == 0, \
            f"P={self.P} x chunk must group into {GP_K}-point grid rows per device"
        cam = self._camera_embed(proj)
        consts = self._gp_pack(levels_tt)
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
            # grid rows are [C, m*P/GP_K, 1, 2*GP_K] Q14 with points
            # anchor-major, so sharding dim 1 splits the same anchors that dim
            # 0 of feat did. 64 bytes a row at last-dim 32; from_torch costs
            # per row, and the two-wide [C, m*P, 1, 2] shape this replaced
            # uploaded at 50 MB/s against 740 here.
            g_tt = ttnn.from_torch(grid.view(torch.uint16), layout=ttnn.ROW_MAJOR_LAYOUT,
                                   device=self.dev, dtype=ttnn.uint16,
                                   mesh_mapper=self.sh1)
            ml = m // self.nd
            # h0, w0 and the four bilinear weights, per level, on the Tensix
            # engines -- 8 ms here against the 232 it takes grid_sample's reader
            # to derive the same six values in soft float. The outputs are not
            # zeroed because grid_precompute writes every row of every one.
            rows = ml * self.P // GP_K
            spec = ttnn.TensorSpec(ttnn.Shape([self.C, rows, 1, 6 * GP_K]),
                                   ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT,
                                   ttnn.BufferType.DRAM)
            gp = [ttnn.allocate_tensor_on_device(spec, self.dev) for _ in range(4)]
            ttnn.grid_precompute(g_tt, consts, gp[0], gp[1], gp[2], gp[3], GP_K)
            per = []
            # clp is (camera, point, level): concat on the level axis with the
            # point axis already outside it. The old (camera, level, point)
            # order concatenated four [C, P, ml, E] blocks on dim 1; this
            # concatenates four [C*P, 1, ml, E] blocks on dim 1 instead.
            # Measured identical, 123.0 ms against 122.9 -- the reorder is free,
            # and it is what lets the mask expand on device.
            for fm, g in zip(levels_tt, gp):
                s_ = ttnn.grid_sample(fm, g, use_precomputed_grid=True)
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
            for t in (f_tt, g_tt, feats, W, proj_out, *gp):
                ttnn.deallocate(t)
        return out
