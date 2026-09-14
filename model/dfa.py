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
  - clp keeps the checkpoint's own (camera, level, point) order, because that
    is the one order whose level blocks are contiguous -- which is what lets
    grouped_weighted_sum run once per FPN level, straight off the grid_sample
    output, with no concatenated [clp, n, E] tensor in between.
  - the mask silences a camera only where another camera sees the point, which
    is also what keeps the denominator non-zero.
  - the clp reduction is NOT sliced. grouped_weighted_sum's own num_chunks
    splits it instead, which bounds the bf16 accumulator the same way while
    also being what fills the device -- see GWS_CHUNKS. num_chunks MUST divide
    the reduction exactly or the op hangs; the op now refuses it instead.
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

# How many partial sums grouped_weighted_sum splits the clp reduction into. It
# sets two things at once:
#
#   parallelism -- work units are ceil(anchors/32) * num_chunks, and this DFA
#     reduces 6000 clp into 512 anchors a chip, so 16 tile rows. The op's old
#     fixed 2 gave 32 work units on a 64-core device: half of it idle, which is
#     why the reduction measured 33 GiB/s against a 288 GiB/s part.
#   accuracy -- each partial accumulates in bf16 through the packer, so the
#     drift grows with its depth. This is what the `splits` slicing used to buy.
#
# The op alone, layer 0's reduction on the mesh, against an fp64 replay:
#
#   num_chunks   work units   clp/partial   gws      PCC vs fp64
#            2           32          3000   41.7 ms    0.998712
#            4           64          1500   21.0 ms    0.999262
#            8          128           750   20.9 ms    0.999605
#           16          256           375   21.0 ms    0.999858
#
# 2 -> 4 is the idle half of the device being filled; past that it is flat,
# because the work units are exact multiples of the 64 cores. So the depth is
# free, and the whole DFA against the PyTorch reference says take it:
#
#   num_chunks   DAF[0]     DAF[1]     DAF[2]
#            8   0.999671   0.998983   0.999986
#           16   0.999883   0.999704   0.999987
#           30   0.999958   0.999905   0.999988      <- 152 / 27 / 16 ms
#           40   0.999972   0.999932   0.999988
#
# 30 over 40 because DAF[1] has only 4 tile rows, so 40 is 160 work units on
# 64 cores -- 2.5 rounds -- and it measured slower there. 30 beats what the
# old splits=10 x num_chunks=2 managed (0.999919 / 0.999817 / 0.999987) on
# every call, and 6000 and 960 are both divisible by it.
GWS_CHUNKS = 30

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
        # No permutation: weights_fc already emits (level, point, group) within
        # a camera, and the camera axis comes from how the linear's rows are
        # grouped, so the full order is (camera, level, point, group).
        self.PG = self.P * num_groups
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
        # grid_precompute's constant pack, built on first use from the level
        # shapes the feature tensors carry. It depends on nothing else, so all
        # three DFA instances would build the same one.
        self._consts, self._consts_key = None, None
        # gws runs per level, so its reduction is one level's worth of clp.
        # num_chunks must divide that exactly -- the reader hands the compute
        # kernel a fixed page count per work unit, and a ragged last chunk
        # leaves it waiting for pages that never come. The op refuses a ragged
        # split rather than hanging, but pick a divisor here anyway.
        self.clp_l = num_cams * self.P
        self.nchunks = max(k for k in (GWS_CHUNKS, 25, 20, 15, 12, 10, 8, 6, 5, 4, 3, 2, 1)
                           if self.clp_l % k == 0)

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

    def _expand_mask(self, base, nl):
        """[nl, C*P] -> [nl, clp*G], the mask in (camera, level, point, group).

        It has to repeat along two axes -- level at stride P*G and group at
        stride 1 -- which is the shape that made an early attempt at this cost
        80.8 ms. It is cheap here only because both repeats land on tile
        boundaries: the group repeat is a plain interleave on the last axis,
        and a camera block is P*G = 4000 columns, 125 whole tiles, so tiling it
        L times is an aligned slice-and-concatenate.

            repeat_interleave(G) -> (c, p, g)      0.8 ms
            slice into C camera blocks             0.4 ms
            concat the L copies of each            0.9 ms
                                                   1.4 ms, exact
        """
        b8 = ttnn.repeat_interleave(base, self.G, dim=-1)
        cams = [ttnn.slice(b8, [0, c * self.PG], [nl, (c + 1) * self.PG])
                for c in range(self.C)]
        keep = ttnn.concat([cams[c] for c in range(self.C) for _ in range(self.L)], dim=-1)
        ttnn.deallocate(b8)
        for t in cams:
            ttnn.deallocate(t)
        return keep

    def _weights(self, feat_tt, cam, n, base):
        """n is the global anchor count; each device holds n // nd of them.

        `base` is the mask at [n, C*P], one value per (camera, point); the
        expansion to the full [n, clp*G] happens here, on device.

        Returns ONE TENSOR PER LEVEL, [nl, C*P*G] in (camera, point, group),
        matching what grid_sample emits for that level. The softmax still runs
        across the whole clp -- it has to, the denominator spans every camera
        and level -- and only its result is regrouped. clp is
        (camera, level, point, group), so level l is the C blocks at
        (c*L + l) * P*G, each P*G wide and so tile-aligned: 1.3 ms for all
        twelve slices and four concatenations, against the 27.0 ms concat of
        the feature tensor it replaces.
        """
        nl = n // self.nd
        fx = ttnn.add(ttnn.reshape(feat_tt, (nl, 1, self.E)),
                      ttnn.reshape(cam, (1, self.C, self.E)))
        wl = ttnn.linear(ttnn.reshape(fx, (nl * self.C, self.E)), self.Wf, bias=self.Bf,
                         compute_kernel_config=self.hi)
        logits = ttnn.reshape(wl, (nl, self.wide))
        keep = self._expand_mask(base, nl)
        mx = ttnn.max(logits, dim=-1, keepdim=True)
        e = ttnn.multiply(ttnn.exp(ttnn.subtract(logits, mx)), keep)
        ttnn.deallocate(keep)
        s = ttnn.matmul(e, self.gt, compute_kernel_config=self.lo, dtype=self.f32)
        sb = ttnn.matmul(s, self.st, compute_kernel_config=self.hi, dtype=self.f32)
        ttnn.deallocate(s)
        w = ttnn.divide(e, sb)
        ttnn.deallocate(e); ttnn.deallocate(sb)
        per = []
        for l in range(self.L):
            parts = [ttnn.slice(w, [0, (c * self.L + l) * self.PG],
                                [nl, (c * self.L + l + 1) * self.PG])
                     for c in range(self.C)]
            per.append(ttnn.concat(parts, dim=-1))
            for t in parts:
                ttnn.deallocate(t)
        ttnn.deallocate(w)
        return per

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
        assert splits in (None, 1), (
            "the clp reduction is no longer sliced here; grouped_weighted_sum's "
            "num_chunks does that, and gws runs once per level. See GWS_CHUNKS.")
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
            Wl = self._weights(f_tt, cam, m, self.S(base))
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
            # One grouped_weighted_sum per FPN level, straight off what
            # grid_sample produced for it. There is no [clp, ml, E] tensor:
            # building one cost a 1.46 GiB concatenation, 27.0 ms a chip, and
            # four calls measured 22.0 ms against one call's 22.6. The whole
            # point of keeping clp in (camera, level, point) order is that a
            # level's weights are then contiguous.
            #
            # The partial sums of all four levels add up before coming back, so
            # the host sees one [ml, E] download instead of four.
            mp_ = ((ml + 31) // 32) * 32
            nc = self.nchunks
            tot = None
            for fm, g, wi in zip(levels_tt, gp, Wl):
                s_ = ttnn.grid_sample(fm, g, use_precomputed_grid=True)
                fi = ttnn.reshape(
                    ttnn.permute(ttnn.reshape(s_, (self.C, ml, self.P, self.E)),
                                 (0, 2, 1, 3)),
                    (self.C * self.P, ml, self.E))
                o = ttnn.grouped_weighted_sum(fi, wi, num_groups=self.G,
                                              group_dims=self.E // self.G, num_chunks=nc)
                # o is [nc * mp_, E], one block per partial sum.
                so = ttnn.reshape(ttnn.sum(ttnn.reshape(o, (nc, mp_, self.E)), dim=0),
                                  (mp_, self.E))
                tot = so if tot is None else ttnn.add(tot, so)
                ttnn.deallocate(fi); ttnn.deallocate(o)
                if tot is not so:
                    ttnn.deallocate(so)
            acc = self.G2T(ttnn.slice(tot, [0, 0], [ml, self.E])).float()
            proj_out = ttnn.linear(self.S(acc), self.Wo, bias=self.Bo,
                                   compute_kernel_config=self.hi)
            out[a0:a1] = self.G2T(ttnn.add(proj_out, f_tt)).float()[:m]
            for t in (f_tt, g_tt, tot, proj_out, *gp, *Wl):
                ttnn.deallocate(t)
        return out
