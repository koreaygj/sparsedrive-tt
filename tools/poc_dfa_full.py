"""Phase 2 milestone: the whole DeformableFeatureAggregation on the device.

Every stage, end to end, graded against golden p_deform_model.out -- the
module's actual output on a real navtest frame.

    camera_encoder      [3,12] -> [3,256]        Linear/ReLU/LN x2
    kps + project       fp32, fused              -> sampling_location
    weights_fc          [n*3,256] -> [n*3,16000]
    mask + softmax      strided over clp = 6000
    grid_sample         per level, bf16 grid
    assemble            -> [clp, n, E], no host round-trip
    gws                 -> [n, E]
    output_proj         Linear(256,256) + residual

Layout note: the softmax output is already gws's COMPACT weight layout. Both
index a row as clp*G + g with clp = c*(L*P) + l*P + p, because weights_fc emits
(L, P, G) with the camera axis outside. Nothing is rearranged between them --
and a clp split is then a contiguous column slice, [lo*G, hi*G).

    $TT_PY tools/poc_dfa_full.py --anchors 256 --clp-splits 8
"""

import argparse
import os
import pathlib
import time

import torch
import ttnn

EXP = pathlib.Path(os.environ["NAVSIM_EXP_ROOT"])
GOLDEN = EXP / "golden_dfa"
ROOT = pathlib.Path(__file__).resolve().parents[1]
CKPT = ROOT / "ckpt" / "sparsedrive_navsimv1.ckpt"
PRE = "agent._sparsedrive_model._trajectory_head.decoder.layers.0.p_deform_model."

NUM_SAMPLE, FIX_HEIGHT, NUM_LEARNABLE = 50, (0.0, -0.25, -0.5, 0.25, 0.5), 2
NUM_PTS = NUM_SAMPLE * len(FIX_HEIGHT) * NUM_LEARNABLE          # 500
G, E, L = 8, 256, 4

LOFI = dict(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True)
HIFI = dict(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)


def pcc(a, b):
    a = a.detach().flatten().to(torch.float64); b = b.detach().flatten().to(torch.float64)
    a = a - a.mean(); b = b - b.mean()
    d = a.norm() * b.norm()
    return 1.0 if d == 0 else float((a @ b) / d)


def selectors():
    H, Lp = len(FIX_HEIGHT), NUM_LEARNABLE
    Ox = torch.zeros(NUM_PTS * 2, NUM_PTS); Oy = torch.zeros(NUM_PTS * 2, NUM_PTS)
    Ax = torch.zeros(NUM_SAMPLE * 2, NUM_PTS); Ay = torch.zeros(NUM_SAMPLE * 2, NUM_PTS)
    for p in range(NUM_PTS):
        Ox[2 * p, p] = 1.0; Oy[2 * p + 1, p] = 1.0
        s = p // (H * Lp); Ax[2 * s, p] = 1.0; Ay[2 * s + 1, p] = 1.0
    z = torch.tensor([FIX_HEIGHT[(p // Lp) % H] for p in range(NUM_PTS)])
    return Ox, Oy, Ax, Ay, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", type=int, default=256)
    # The COMPACT weight layout needs clp_slice * G to be a multiple of the
    # tile width, so clp_slice must be divisible by 4 (G is 8). clp = 6000, so
    # k = 8 gives 750 and is rejected; 10 gives 600 and is not. The 3D layout
    # has no such constraint, which is why the earlier gws PoC ran k = 8.
    ap.add_argument("--clp-splits", type=int, default=10)
    args = ap.parse_args()

    frame = sorted(GOLDEN.glob("*.pt"))[0]
    d = torch.load(frame, map_location="cpu", weights_only=False)
    K = "_trajectory_head.decoder.layers.0.p_deform_model."
    anchor = d[K + "kps_generator.in[0]"]; feat_in = d[K + "in[0]"]
    proj = d[K + "in[4].projection_mat"][0]; iwh = d[K + "in[4].image_wh"][0]
    mc = d["DAF[0].mc_ms_feat"]
    shapes = [tuple(int(v) for v in r) for r in d["DAF[0].spatial_shape"]]
    starts = [int(v) for v in d["DAF[0].scale_start_index"]]
    ref_out = d[K + "out"]

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
    g = lambda k: sd[PRE + k].float()

    n, C = args.anchors, proj.shape[0]
    clp, wide = C * L * NUM_PTS, C * L * NUM_PTS * G
    k = args.clp_splits
    if clp % k or (clp // k) * G % 32:
        raise SystemExit(
            f"  clp-splits={k}: clp {clp} / {k} = {clp / k}, and compact weights "
            f"need (clp/k)*G divisible by 32. Try one of "
            f"{[j for j in range(1, 33) if clp % j == 0 and (clp // j) * G % 32 == 0]}")
    Ox, Oy, Ax, Ay, zc = selectors()
    print(f"  frame {frame.name}   anchors={n}  C={C} L={L} P={NUM_PTS} G={G}")
    print(f"  clp={clp}  compact weights row = {wide}   clp-splits={args.clp_splits}")
    print()

    dev = ttnn.open_device(device_id=0)
    try:
        lo = ttnn.WormholeComputeKernelConfig(**LOFI)
        hi = ttnn.WormholeComputeKernelConfig(**HIFI)
        T = lambda x, dt=ttnn.bfloat16, ly=ttnn.TILE_LAYOUT: ttnn.from_torch(
            x.contiguous(), layout=ly, device=dev, dtype=dt)

        # --- static uploads -------------------------------------------------
        f32 = ttnn.float32
        feat_tt = T(feat_in[0, :n]); anc32 = T(anchor[0, :n], f32)
        Wl, Bl = T(g("kps_generator.learnable_fc.weight").t(), f32), \
                 T(g("kps_generator.learnable_fc.bias").reshape(1, -1), f32)
        Ox_t, Oy_t, Ax_t, Ay_t = (T(v, f32) for v in (Ox, Oy, Ax, Ay))
        Wf, Bf = T(g("weights_fc.weight").t()), T(g("weights_fc.bias").reshape(1, -1))
        Wo, Bo = T(g("output_proj.weight").t()), T(g("output_proj.bias").reshape(1, -1))
        gather = torch.zeros(wide, G)
        for q in range(G):
            gather[q::G, q] = 1.0
        gt, st = T(gather), T(gather.t().contiguous())
        levels = [T(mc[0, :, s:s + h * w, :].reshape(C, h, w, E), ly=ttnn.ROW_MAJOR_LAYOUT)
                  for (h, w), s in zip(shapes, starts)]
        cam_in = torch.cat([proj[:, :3].reshape(C, -1)], -1)          # [C, 12]
        ce = T(cam_in)
        ce_w = [T(g(f"camera_encoder.{i}.weight").t()) for i in (0, 3)]
        ce_b = [T(g(f"camera_encoder.{i}.bias").reshape(1, -1)) for i in (0, 3)]
        ln_w = [T(g(f"camera_encoder.{i}.weight")) for i in (2, 5)]
        ln_b = [T(g(f"camera_encoder.{i}.bias")) for i in (2, 5)]

        def run():
            # 1. camera_encoder --------------------------------------------
            h1 = ttnn.layer_norm(ttnn.relu(ttnn.linear(ce, ce_w[0], bias=ce_b[0],
                                                       compute_kernel_config=hi)),
                                 weight=ln_w[0], bias=ln_b[0])
            cam = ttnn.layer_norm(ttnn.relu(ttnn.linear(h1, ce_w[1], bias=ce_b[1],
                                                        compute_kernel_config=hi)),
                                  weight=ln_w[1], bias=ln_b[1])            # [C, 256]
            # 2. weights_fc over (anchor, camera) ---------------------------
            fx = ttnn.add(ttnn.reshape(feat_tt, (n, 1, E)), ttnn.reshape(cam, (1, C, E)))
            wl = ttnn.linear(ttnn.reshape(fx, (n * C, E)), Wf, bias=Bf,
                             compute_kernel_config=hi)
            logits = ttnn.reshape(wl, (n, wide))
            # 3. keypoints + projection, fp32 -------------------------------
            off = ttnn.linear(T(feat_in[0, :n], f32), Wl, bias=Bl, compute_kernel_config=hi)
            x = ttnn.add(ttnn.matmul(off, Ox_t, compute_kernel_config=hi),
                         ttnn.matmul(anc32, Ax_t, compute_kernel_config=hi))
            y = ttnn.add(ttnn.matmul(off, Oy_t, compute_kernel_config=hi),
                         ttnn.matmul(anc32, Ay_t, compute_kernel_config=hi))
            grids, inb = [], []
            for c in range(C):
                Pm = proj[c]
                cst = (Pm[:3, 2] * zc.unsqueeze(-1) + Pm[:3, 3]).t()
                cx, cy, cz = (T(cst[i].reshape(1, -1), f32) for i in range(3))
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
            gr = torch.stack(grids, 0).reshape(C, n * NUM_PTS, 1, 2)
            m = torch.stack(inb, 1)                                        # [n, C, P]
            mp = m[:, :, None, :, None]
            keep = (~torch.logical_and(~mp, mp.sum(1, keepdim=True) != 0)).float()
            keep = keep.expand(n, C, L, NUM_PTS, G).reshape(n, wide)
            # 4. mask + softmax --------------------------------------------
            mx = ttnn.max(logits, dim=-1, keepdim=True)
            e = ttnn.multiply(ttnn.exp(ttnn.subtract(logits, mx)), T(keep))
            sm = ttnn.matmul(e, gt, compute_kernel_config=lo, dtype=f32)
            sb = ttnn.matmul(sm, st, compute_kernel_config=hi, dtype=f32)
            W = ttnn.divide(e, sb)                                         # [n, wide] compact
            # 5. grid_sample + assemble ------------------------------------
            g_tt = T(gr, ly=ttnn.ROW_MAJOR_LAYOUT)
            per = []
            for fm in levels:
                s_ = ttnn.grid_sample(fm, g_tt, padding_mode="zeros", align_corners=False)
                s_ = ttnn.permute(ttnn.reshape(s_, (C, n, NUM_PTS, E)), (0, 2, 1, 3))
                per.append(s_)
            feats = ttnn.reshape(ttnn.concat(per, dim=1), (clp, n, E))
            # 6. gws, split over clp ---------------------------------------
            sl = clp // k
            acc = torch.zeros(n, E)
            for i in range(k):
                fi = ttnn.slice(feats, [i * sl, 0, 0], [(i + 1) * sl, n, E])
                wi = ttnn.slice(W, [0, i * sl * G], [n, (i + 1) * sl * G])
                o = ttnn.grouped_weighted_sum(fi, wi, num_groups=G, group_dims=E // G)
                np_ = ((n + 31) // 32) * 32
                acc += ttnn.to_torch(ttnn.add(ttnn.slice(o, [0, 0], [n, E]),
                                              ttnn.slice(o, [np_, 0], [np_ + n, E]))).float()
            # 7. output_proj + residual ------------------------------------
            out = ttnn.linear(T(acc), Wo, bias=Bo, compute_kernel_config=hi)
            return ttnn.to_torch(ttnn.add(out, feat_tt)).float()

        run()
        ttnn.synchronize_device(dev); t0 = time.time()
        got = run()
        ttnn.synchronize_device(dev); dt = (time.time() - t0) * 1e3

        want = ref_out[0, :n]
        p = pcc(got, want)
        print(f"  [DFA 전체] {dt:7.1f} ms (웜업 후, anchor {n})")
        print(f"      p_deform_model.out   PCC {p:.6f}   max|d| {(got - want).abs().max():.3e}")
        print(f"      스케일  got {got.abs().mean():.4f}  vs  want {want.abs().mean():.4f}")
        ok = p >= 0.999
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    raise SystemExit(main())
