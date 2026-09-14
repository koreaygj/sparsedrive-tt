"""Phase 2 gate: does grouped_weighted_sum survive SparseDriveV2's scale?

Reads the real DAF[0] call arguments captured by dump_golden.py and drives
ttnn.grouped_weighted_sum with them, chunked over anchors.

What the kernel actually does (read off grouped_weighted_sum_device_operation):

    features [clp, N, E] , weights [clp, N, G]  ->  [2*N_padded, E]

and the caller adds the two halves to get [N, E]. clp is cameras x levels x
points, so **the reduction over points is already fused** -- the output is one
row per anchor, not per anchor-point. sparse4D-tt has been doing this all
along.

So the cost is not the output, it is the *input*: features is clp*N*E.

    sparse4D v3     clp =  6*4*13  =  312 , N =  900  ->  71.9M  (144 MB bf16)
    SparseDriveV2   clp =  3*4*500 = 6000 , N = 1024  ->  1.57G  (3.1 GB bf16)

3.1 GB fits in a 12 GB chip but not in any useful cache, and writing then
reading it once is ~31 ms at 200 GB/s -- for one of three DFA calls, against
sparse4D-tt's 57 ms whole-frame budget. Anchors are independent through both
grid_sample and gws, so the fix is to never materialise it: walk anchors in
blocks. This measures whether that works and what block size costs what.

    $TT_PY tools/poc_gws_scale.py --chunks 32,64,128
"""

import argparse
import json
import os
import pathlib
import time

import torch
import torch.nn.functional as F
import ttnn

EXP = pathlib.Path(os.environ["NAVSIM_EXP_ROOT"])
GOLDEN = EXP / "golden_dfa"


def pcc(a, b):
    a = a.detach().flatten().to(torch.float64)
    b = b.detach().flatten().to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    d = a.norm() * b.norm()
    return 1.0 if d == 0 else float((a @ b) / d)


def upload_levels(device, feat, shapes, starts):
    """Feature maps go up once per frame, not once per chunk.

    NHWC is what grid_sample wants, and it is also what the FPN output already
    looks like once the cameras are folded into the batch axis, so this is a
    reshape rather than a transpose.
    """
    _, C, _, E = feat.shape
    ups = []
    for (h, w), st in zip(shapes, starts):
        lvl = feat[0, :, st:st + h * w, :].reshape(C, h, w, E).contiguous()
        ups.append(ttnn.from_torch(lvl, layout=ttnn.ROW_MAJOR_LAYOUT,
                                   device=device, dtype=ttnn.bfloat16))
    return ups


def sample_assemble_device(device, levels_tt, loc, a0, a1, pts, C, L, E):
    """Sample and rearrange to gws's [clp, N, E] without leaving the device.

    grid_sample hands back (C, n*P, 1, E) per level, points ordered
    anchor-major. gws wants clp = (cam, level, point) major with anchors down
    the middle axis, so per level:

        (C, n*P, 1, E) -> (C, n, P, E) -> (C, P, n, E)

    and then the levels concatenate along axis 1, which lands them exactly
    where clp = c*(L*P) + l*P + p expects. One final reshape folds (C, L*P)
    into clp. Nothing round-trips to the host.

    Returns the device tensor plus the in-bounds mask, which still has to be
    folded into the weights (grid_sample alone would blend the in-bounds
    corners of a point that is just outside; the CUDA kernel drops it whole).
    """
    n = a1 - a0
    p0, p1 = a0 * pts, a1 * pts
    loc_c = loc[0, p0:p1]                                     # [n*P, C, 2]
    inb = ((loc_c[..., 0] > 0) & (loc_c[..., 0] < 1)
           & (loc_c[..., 1] > 0) & (loc_c[..., 1] < 1))
    grid = (loc_c * 2 - 1).permute(1, 0, 2).unsqueeze(2).contiguous()
    g_tt = ttnn.from_torch(grid, layout=ttnn.ROW_MAJOR_LAYOUT,
                           device=device, dtype=ttnn.bfloat16)

    per_level = []
    for fm in levels_tt:
        s = ttnn.grid_sample(fm, g_tt, padding_mode="zeros", align_corners=False)
        s = ttnn.reshape(s, (C, n, pts, E))
        s = ttnn.permute(s, (0, 2, 1, 3))                     # (C, P, n, E)
        per_level.append(s)
    ttnn.deallocate(g_tt)

    cat = ttnn.concat(per_level, dim=1)                       # (C, L*P, n, E)
    for t in per_level:
        ttnn.deallocate(t)
    feats = ttnn.reshape(cat, (C * L * pts, n, E))
    return feats, inb


def weights_for_chunk(weights, inb, a0, a1, pts, C, L):
    """Mask and reorder the attention weights to [clp, n, G] on the host.

    Still host-side: these come from weights_fc, which is not ported yet.
    """
    n = a1 - a0
    w_c = weights[0, a0 * pts:a1 * pts]                       # [n*P, C, L, G]
    G = w_c.shape[-1]
    w_c = w_c.reshape(n, pts, C, L, G).permute(2, 3, 1, 0, 4)  # [C, L, P, n, G]
    m = inb.reshape(n, pts, C).permute(2, 1, 0)                # [C, P, n]
    w_c = w_c * m.unsqueeze(1).unsqueeze(-1)
    return w_c.reshape(C * L * pts, n, G)


def sample_chunk_device(device, feat, shapes, starts, loc, weights, a0, a1, pts):
    """Same as sample_chunk, but the bilinear gather runs on the device.

    ttnn.grid_sample wants NHWC input and a (N, H_out, W_out, 2) grid, so the
    three cameras ride the batch axis and every anchor-point of the chunk goes
    down H_out:

        input (C, H_l, W_l, E)   grid (C, n*pts, 1, 2)   ->  (C, n*pts, 1, E)

    The grid is bf16, and that is the whole accuracy story here. Measured
    against an fp32-grid reference on a real frame, per level:

        (64,128) 0.999801   (32,64) 0.999897
        (16,32)  0.999967   (8,16)  0.999983

    A host simulation that quantises ONLY the grid to bf16 reproduces those to
    ~5e-6, while quantising only the features scores 0.999999 -- so the loss is
    coordinate precision, not the features and not the kernel. Error grows with
    level width because a fixed coordinate epsilon is more pixels on a wider
    map. sparse4D-tt dodges this with a Q14 fixed-point grid via
    ttnn.grid_precompute; that is the upgrade path if this floor is too low.
    """
    _, C, _, E = feat.shape
    L = len(shapes)
    n = a1 - a0
    p0, p1 = a0 * pts, a1 * pts

    loc_c = loc[0, p0:p1]                                    # [n*P, C, 2]
    inb = ((loc_c[..., 0] > 0) & (loc_c[..., 0] < 1)
           & (loc_c[..., 1] > 0) & (loc_c[..., 1] < 1))
    grid = (loc_c * 2 - 1).permute(1, 0, 2).unsqueeze(2).contiguous()   # [C, n*P, 1, 2]
    g_tt = ttnn.from_torch(grid, layout=ttnn.ROW_MAJOR_LAYOUT,
                           device=device, dtype=ttnn.bfloat16)

    out = torch.empty(C, L, pts, n, E, dtype=torch.float32)
    for li, ((h, w), st) in enumerate(zip(shapes, starts)):
        lvl = feat[0, :, st:st + h * w, :].reshape(C, h, w, E).contiguous()
        fm = ttnn.from_torch(lvl, layout=ttnn.ROW_MAJOR_LAYOUT,
                             device=device, dtype=ttnn.bfloat16)
        s_tt = ttnn.grid_sample(fm, g_tt, padding_mode="zeros", align_corners=False)
        s = ttnn.to_torch(s_tt).float().squeeze(2)           # [C, n*P, E]
        out[:, li] = s.reshape(C, n, pts, E).permute(0, 2, 1, 3)
        for t in (fm, s_tt):
            ttnn.deallocate(t)
    ttnn.deallocate(g_tt)

    w_c = weights[0, p0:p1]
    G = w_c.shape[-1]
    w_c = w_c.reshape(n, pts, C, L, G).permute(2, 3, 1, 0, 4)
    m = inb.reshape(n, pts, C).permute(2, 0, 1).permute(0, 2, 1)
    w_c = w_c * m.unsqueeze(1).unsqueeze(-1)

    clp = C * L * pts
    return out.reshape(clp, n, E), w_c.reshape(clp, n, G)


def sample_chunk(feat, shapes, starts, loc, weights, a0, a1, pts, compact=False):
    """Host-side gather for anchors [a0, a1) -> (features, weights) for gws.

    features [clp, n, E] and weights [clp, n, G], clp ordered (cam, level,
    point) because the kernel's skip mode derives cam = clp / clp_per_cam.
    The strict 0<x,y<1 bound is folded into the weights here, the way the
    device path does it -- grid_sample alone would still blend the in-bounds
    corners of a point that is just outside.
    """
    _, C, _, E = feat.shape
    L = len(shapes)
    n = a1 - a0
    p0, p1 = a0 * pts, a1 * pts

    loc_c = loc[0, p0:p1]                                    # [n*P, C, 2]
    inb = ((loc_c[..., 0] > 0) & (loc_c[..., 0] < 1)
           & (loc_c[..., 1] > 0) & (loc_c[..., 1] < 1))      # [n*P, C]
    grid = (loc_c * 2 - 1).permute(1, 0, 2).reshape(C, n * pts, 1, 2)

    out = torch.empty(C, L, pts, n, E, dtype=torch.float32)
    for li, ((h, w), st) in enumerate(zip(shapes, starts)):
        level = feat[0, :, st:st + h * w, :].permute(0, 2, 1).reshape(C, E, h, w)
        s = F.grid_sample(level, grid, mode="bilinear",
                          padding_mode="zeros", align_corners=False)
        # [C, E, n*P, 1] -> [C, P, n, E]
        s = s.squeeze(-1).reshape(C, E, n, pts).permute(0, 3, 2, 1)
        out[:, li] = s

    w_c = weights[0, p0:p1]                                  # [n*P, C, L, G]
    G = w_c.shape[-1]
    w_c = w_c.reshape(n, pts, C, L, G).permute(2, 3, 1, 0, 4)   # [C, L, P, n, G]
    m = inb.reshape(n, pts, C).permute(2, 0, 1)                 # [C, n, P]
    m = m.permute(0, 2, 1).unsqueeze(1).unsqueeze(-1)           # [C, 1, P, n, 1]
    w_c = w_c * m

    clp = C * L * pts
    if compact:
        # [N, clp*G]: fills every column of a tile instead of 8 of 32. This is
        # also the order weights_fc already emits -- its 16000 outputs are
        # (L, P, G) and the camera axis sits outside, giving (C, L, P, G),
        # which is clp-major then G exactly.
        w_out = w_c.permute(3, 0, 1, 2, 4).reshape(n, clp * G)
    else:
        w_out = w_c.reshape(clp, n, G)
    return out.reshape(clp, n, E), w_out


def run(device, feats_t, wts_t, num_groups, group_dims, splits=1):
    """gws over the clp axis, optionally in `splits` pieces summed in fp32.

    The kernel's accumulator is bf16 and sequential, so its drift grows with
    the reduction length -- measured on synthetic data at N=32, E=256:

        clp   312 -> 0.999893      (this is where sparse4D v3 sits)
        clp  1024 -> 0.999658
        clp  6000 -> 0.998039      (SparseDriveV2 layer-0 path)

    Splitting clp into k pieces shortens each sequential run by k and adds the
    partials in fp32. It costs nothing measurable, because the data volume
    moved is identical either way and that is what the time is made of:
    k = 1,2,4,8 measured 58.3 / 53.9 / 53.2 / 56.2 ms for PCC
    0.998066 / 0.999013 / 0.999507 / 0.999746.
    """
    on_device = not torch.is_tensor(feats_t)
    n = feats_t.shape[1]
    E = feats_t.shape[2]
    if splits > 1:
        clp = feats_t.shape[0]
        assert clp % splits == 0, f"clp {clp} not divisible by {splits}"
        assert wts_t.dim() == 3, "clp splitting needs the 3D weight layout"
        sl = clp // splits
        acc = torch.zeros(n, E)
        for i in range(splits):
            lo, hi = i * sl, (i + 1) * sl
            # A device-resident features tensor is sliced on the device; the
            # point of splitting is to shorten the kernel's sequential bf16
            # accumulation, and copying back to the host to do it would undo
            # everything the on-device assembly just bought.
            f_i = ttnn.slice(feats_t, [lo, 0, 0], [hi, n, E]) if on_device \
                else feats_t[lo:hi]
            acc += run(device, f_i, wts_t[lo:hi], num_groups, group_dims)
            if on_device:
                ttnn.deallocate(f_i)
        return acc
    f = feats_t if on_device else ttnn.from_torch(
        feats_t, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16)
    # weights must be TILE; features may be either (validate() in the device op)
    w = ttnn.from_torch(wts_t, layout=ttnn.TILE_LAYOUT,
                        device=device, dtype=ttnn.bfloat16)
    o = ttnn.grouped_weighted_sum(f, w, num_groups=num_groups, group_dims=group_dims)
    n_pad = ((n + 31) // 32) * 32
    c0 = ttnn.slice(o, [0, 0], [n, feats_t.shape[2]])
    c1 = ttnn.slice(o, [n_pad, 0], [n_pad + n, feats_t.shape[2]])
    res = ttnn.to_torch(ttnn.add(c0, c1)).float()
    for t in ((w, o, c0, c1) if on_device else (f, w, o, c0, c1)):
        ttnn.deallocate(t)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="32,64,128")
    ap.add_argument("--call", default="DAF[0]")
    ap.add_argument("--anchors", type=int, default=0, help="0 = all")
    ap.add_argument("--device-assemble", action="store_true",
                    help="sample AND rearrange on the device; no host round-trip")
    ap.add_argument("--device-sample", action="store_true",
                    help="run the bilinear gather on the device too")
    ap.add_argument("--clp-splits", type=int, default=1,
                    help="split the clp reduction into k fp32-summed pieces")
    ap.add_argument("--compact", action="store_true",
                    help="weights as [N, clp*G] instead of [clp, N, G]")
    args = ap.parse_args()

    frame = sorted(GOLDEN.glob("*.pt"))[0]
    d = torch.load(frame, map_location="cpu", weights_only=False)
    P = args.call
    feat = d[f"{P}.mc_ms_feat"]
    loc = d[f"{P}.sampling_location"]
    wts = d[f"{P}.weights"]
    shapes = [tuple(int(v) for v in r) for r in d[f"{P}.spatial_shape"]]
    starts = [int(v) for v in d[f"{P}.scale_start_index"]]
    ref_flat = d[f"{P}.out"]

    C, L, G = wts.shape[2], wts.shape[3], wts.shape[4]
    E = feat.shape[-1]
    total_pts = loc.shape[1]
    # points-per-anchor comes from the call, not the config: layer 0 path is
    # 1024 anchors x 500, layer 1 path 128 x 500, the trajectory branch 400 x 80.
    pts = {"DAF[0]": 500, "DAF[1]": 500, "DAF[2]": 80}[P]
    N = total_pts // pts
    clp = C * L * pts

    print(f"  frame {frame.name}  {P}")
    print(f"  N={N} anchors  pts={pts}  C={C} L={L} G={G} E={E}  clp={clp}"
          f"  weights={'compact [N,clp*G]' if args.compact else '3D [clp,N,G]'}")
    print(f"  features [clp,N,E] would be {clp*N*E/1e6:.1f}M elem "
          f"= {clp*N*E*2/2**30:.2f} GiB bf16 if materialised whole")
    print()

    ref = ref_flat.reshape(1, N, pts, E).sum(2)[0]            # [N, E]
    limit = args.anchors or N

    device = ttnn.open_device(device_id=0)
    results = []
    levels_tt = None
    try:
        if args.device_assemble:
            levels_tt = upload_levels(device, feat, shapes, starts)
        for cs in [int(x) for x in args.chunks.split(",")]:
            got = torch.zeros(limit, E)
            t_host = t_dev = 0.0
            for a0 in range(0, limit, cs):
                a1 = min(a0 + cs, limit)
                t = time.time()
                if args.device_assemble:
                    f_c, inb = sample_assemble_device(device, levels_tt, loc,
                                                      a0, a1, pts, C, L, E)
                    w_c = weights_for_chunk(wts, inb, a0, a1, pts, C, L)
                elif args.device_sample:
                    f_c, w_c = sample_chunk_device(device, feat, shapes, starts,
                                                   wts if False else loc, wts,
                                                   a0, a1, pts)
                else:
                    f_c, w_c = sample_chunk(feat, shapes, starts, loc, wts,
                                            a0, a1, pts, compact=args.compact)
                t_host += time.time() - t
                t = time.time()
                got[a0:a1] = run(device, f_c, w_c, G, E // G,
                                 splits=args.clp_splits)
                ttnn.synchronize_device(device)
                t_dev += time.time() - t
            p = pcc(got, ref[:limit])
            per_chunk_mb = clp * cs * E * 2 / 2**20
            tag = ("dev-assemble" if args.device_assemble
                   else "dev-sample" if args.device_sample else "host-sample")
            print(f"  [{tag}] chunk {cs:4d} x{args.clp_splits:<2d}  PCC {p:.6f}  "
                  f"device {t_dev*1e3:8.1f} ms  host {t_host*1e3:8.1f} ms  "
                  f"buf/chunk {per_chunk_mb:7.1f} MB")
            results.append({"chunk": cs, "pcc": p, "device_ms": t_dev * 1e3,
                            "host_ms": t_host * 1e3, "buf_mb": per_chunk_mb})
    finally:
        if levels_tt:
            for t in levels_tt:
                ttnn.deallocate(t)
        ttnn.close_device(device)

    best = max(results, key=lambda r: r["pcc"])
    ok = best["pcc"] >= 0.999
    print()
    print(f"  best PCC {best['pcc']:.6f} at chunk {best['chunk']}  "
          f"(floor 0.999)")
    print("PASS" if ok else "FAIL")
    (EXP / "poc_gws.json").write_text(json.dumps(
        {"call": P, "N": N, "pts": pts, "clp": clp, "results": results}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
