"""Phase 2: keypoint generation and projection on the device.

The DFA's front end, graded against golden kps_generator.out and
DAF[0].sampling_location.

Fused, because the naive shape is hostile to tiles. Written out, a keypoint is

    kp[n, s, h, l] = ( anchor[n,s,0] + off[n,s,h,l,0],
                       anchor[n,s,1] + off[n,s,h,l,1],
                       fix_height[h] )

-- the z is a constant that depends only on h, and the 4th homogeneous
coordinate is 1. So projecting by P never needs the [n, 500, 3] tensor:

    X = x*P00 + y*P01 + (P02*z + P03)
    Y = x*P10 + y*P11 + (P12*z + P13)
    Z = x*P20 + y*P21 + (P22*z + P23)

with the bracketed terms constant per (camera, h). Everything stays [n, 500],
which tiles cleanly, instead of a last dimension of 3 that wastes 29 of every
32 columns. This is the same fusion sparse4D-tt's kps_project_fused kernel does
for boxes, reached here with stock ops because the path case is affine in xy.

The two gathers -- pulling x and y out of the interleaved offset, and fanning
the 50 path points out to 500 keypoints -- are 0/1 matmuls, which are exact.

**The coordinate chain is fp32, not bf16.** Coordinates are geometry, not
activations: a path point reaches 50 m ahead, where bf16 resolves to 0.2 m, and
projection turns that into pixels. Measured on a real frame, 256 anchors:

                        bf16        fp32
    time                2.4 ms      1.7 ms
    key_points max|d|   0.271 m     0.041 m
    in-bounds flips     0.42%       0.03%     of visible points
    u error p99         4.22 px     0.33 px
    u error max         8.56 px     1.35 px

fp32 wins on every axis and is not slower; the tensors are [n, 500]. The grid
handed to grid_sample is still bf16 -- that contract does not change, and its
own ~0.5 px floor stays -- but the 13x of error stacked on top of it goes away.
The same dtype question answered the opposite way for the features, which lose
nothing at bf16 (0.999999).

Scoring note: PCC over all sampling locations is meaningless here. A point
behind the camera has Z clamped to 1e-5, so X/Z is ~1e6, and a wobble in Z
moves it by millions -- which a correlation counts and a sampler never sees.
Grade the in/out decision and the accuracy of the survivors.

    $TT_PY tools/poc_kps_project.py --anchors 256
"""

import argparse
import os
import pathlib
import time

import torch
import ttnn

EXP = pathlib.Path(os.environ["NAVSIM_EXP_ROOT"])
GOLDEN = EXP / "golden_dfa"
CKPT = pathlib.Path(__file__).resolve().parents[1] / "ckpt" / "sparsedrive_navsimv1.ckpt"
PRE = "agent._sparsedrive_model._trajectory_head.decoder.layers.0.p_deform_model."

NUM_SAMPLE = 50
FIX_HEIGHT = (0.0, -0.25, -0.5, 0.25, 0.5)
NUM_LEARNABLE = 2
NUM_PTS = NUM_SAMPLE * len(FIX_HEIGHT) * NUM_LEARNABLE      # 500


def pcc(a, b):
    a = a.detach().flatten().to(torch.float64)
    b = b.detach().flatten().to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    d = a.norm() * b.norm()
    return 1.0 if d == 0 else float((a @ b) / d)


def selectors():
    """0/1 matrices for the two gathers. Exact under any fidelity."""
    H, Lp = len(FIX_HEIGHT), NUM_LEARNABLE
    Ox = torch.zeros(NUM_PTS * 2, NUM_PTS)      # offset -> x
    Oy = torch.zeros(NUM_PTS * 2, NUM_PTS)      # offset -> y
    Ax = torch.zeros(NUM_SAMPLE * 2, NUM_PTS)   # anchor -> x, fanned 50 -> 500
    Ay = torch.zeros(NUM_SAMPLE * 2, NUM_PTS)
    for p in range(NUM_PTS):
        Ox[2 * p, p] = 1.0
        Oy[2 * p + 1, p] = 1.0
        s = p // (H * Lp)
        Ax[2 * s, p] = 1.0
        Ay[2 * s + 1, p] = 1.0
    z = torch.tensor([FIX_HEIGHT[(p // Lp) % H] for p in range(NUM_PTS)])
    return Ox, Oy, Ax, Ay, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", type=int, default=256)
    ap.add_argument("--coord-dtype", default="float32",
                    choices=["bfloat16", "float32"],
                    help="dtype of the keypoint/projection chain; the grid "
                         "handed to grid_sample is bf16 either way")
    args = ap.parse_args()

    frame = sorted(GOLDEN.glob("*.pt"))[0]
    d = torch.load(frame, map_location="cpu", weights_only=False)
    K = "_trajectory_head.decoder.layers.0.p_deform_model."
    anchor = d[K + "kps_generator.in[0]"]                 # [1, 1024, 100]
    feat = d[K + "kps_generator.in[1]"]                   # [1, 1024, 256]
    ref_kp = d[K + "kps_generator.out"]                   # [1, 1024, 500, 3]
    proj = d[K + "in[4].projection_mat"][0]               # [3, 4, 4]
    iwh = d[K + "in[4].image_wh"][0]                      # [3, 2]
    ref_loc = d["DAF[0].sampling_location"]               # [1, 512000, 3, 2]

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
    Wl = sd[PRE + "kps_generator.learnable_fc.weight"].float()   # [1000, 256]
    Bl = sd[PRE + "kps_generator.learnable_fc.bias"].float()

    n = args.anchors
    C = proj.shape[0]
    Ox, Oy, Ax, Ay, zc = selectors()

    print(f"  frame {frame.name}   anchors={n}  pts={NUM_PTS}  cams={C}")
    print(f"  fix_height {FIX_HEIGHT}  num_learnable_pts {NUM_LEARNABLE}")
    print(f"  좌표 체인 dtype: {args.coord_dtype}")
    print()

    dev = ttnn.open_device(device_id=0)
    try:
        cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True, packer_l1_acc=True)
        CD = ttnn.bfloat16 if args.coord_dtype == "bfloat16" else ttnn.float32
        t = lambda x, dt=None: ttnn.from_torch(
            x.contiguous(), layout=ttnn.TILE_LAYOUT, device=dev,
            dtype=CD if dt is None else dt)

        f_tt = t(feat[0, :n])                              # [n, 256]
        w_tt = t(Wl.t())
        b_tt = t(Bl.reshape(1, -1))
        a_tt = t(anchor[0, :n])                            # [n, 100]
        Ox_t, Oy_t, Ax_t, Ay_t = t(Ox), t(Oy), t(Ax), t(Ay)

        def run():
            off = ttnn.linear(f_tt, w_tt, bias=b_tt, compute_kernel_config=cfg)
            x = ttnn.add(ttnn.matmul(off, Ox_t, compute_kernel_config=cfg),
                         ttnn.matmul(a_tt, Ax_t, compute_kernel_config=cfg))
            y = ttnn.add(ttnn.matmul(off, Oy_t, compute_kernel_config=cfg),
                         ttnn.matmul(a_tt, Ay_t, compute_kernel_config=cfg))
            ttnn.deallocate(off)
            locs = []
            for c in range(C):
                P = proj[c]
                const = (P[:3, 2] * zc.unsqueeze(-1) + P[:3, 3]).t()    # [3, 500]
                cx, cy, cz = (t(const[i].reshape(1, -1)) for i in range(3))
                X = ttnn.add(ttnn.add(ttnn.multiply(x, float(P[0, 0])),
                                      ttnn.multiply(y, float(P[0, 1]))), cx)
                Y = ttnn.add(ttnn.add(ttnn.multiply(x, float(P[1, 0])),
                                      ttnn.multiply(y, float(P[1, 1]))), cy)
                Z = ttnn.add(ttnn.add(ttnn.multiply(x, float(P[2, 0])),
                                      ttnn.multiply(y, float(P[2, 1]))), cz)
                Zc = ttnn.clamp(Z, 1e-5, 1e30)
                u = ttnn.multiply(ttnn.divide(X, Zc), 1.0 / float(iwh[c, 0]))
                v = ttnn.multiply(ttnn.divide(Y, Zc), 1.0 / float(iwh[c, 1]))
                locs.append((u, v))
                for q in (cx, cy, cz, X, Y, Z, Zc):
                    ttnn.deallocate(q)
            return x, y, locs

        run()                                              # warm
        ttnn.synchronize_device(dev); t0 = time.time()
        x, y, locs = run()
        ttnn.synchronize_device(dev); dt = (time.time() - t0) * 1e3

        xh = ttnn.to_torch(x).float()
        yh = ttnn.to_torch(y).float()
        kp = torch.stack([xh, yh, zc.expand(n, NUM_PTS)], dim=-1)
        print(f"  [kps_generator] {dt:6.1f} ms (fused with projection, "
              f"{args.coord_dtype})")
        print(f"      key_points  PCC {pcc(kp, ref_kp[0, :n]):.6f}"
              f"   max|d| {(kp - ref_kp[0, :n]).abs().max():.3e}")

        got = torch.stack([torch.stack([ttnn.to_torch(u).float(),
                                        ttnn.to_torch(v).float()], -1)
                           for u, v in locs], dim=2)       # [n, 500, C, 2]
        want = ref_loc[0, :n * NUM_PTS].reshape(n, NUM_PTS, C, 2)

        # PCC over every point is the wrong number here. A point behind the
        # camera has Z clamped to 1e-5, so X/Z is ~1e6 and a bf16 wobble in Z
        # moves it by millions -- which a correlation notices and a sampler
        # never does, because the point was never in frame. What matters is
        # whether the in/out decision flips, and how accurate the survivors are.
        inb_g = ((got[..., 0] > 0) & (got[..., 0] < 1)
                 & (got[..., 1] > 0) & (got[..., 1] < 1))
        inb_w = ((want[..., 0] > 0) & (want[..., 0] < 1)
                 & (want[..., 1] > 0) & (want[..., 1] < 1))
        flips = int((inb_g != inb_w).sum())
        print(f"      sampling_location  (전체 PCC {pcc(got, want):.6f}"
              f", max|d| {(got - want).abs().max():.2e} -- 클램프 폭발 포함, 무의미)")
        print(f"      in-bounds 판정 일치율 {(inb_g == inb_w).float().mean():.6f}"
              f"  (뒤집힘 {flips} / {inb_g.numel()}, 보이는 점의 "
              f"{flips / max(int(inb_w.sum()), 1) * 100:.2f}%)")

        keep = inb_w & inb_g
        gk, wk = got[keep], want[keep]
        px = float((gk[..., 0] - wk[..., 0]).abs().max())
        py = float((gk[..., 1] - wk[..., 1]).abs().max())
        qx = float((gk[..., 0] - wk[..., 0]).abs().quantile(0.99))
        qy = float((gk[..., 1] - wk[..., 1]).abs().quantile(0.99))
        print(f"      보이는 점만: PCC {pcc(gk, wk):.6f}")
        print(f"        최대 오차  u {px*512:6.2f} px, v {py*256:6.2f} px")
        print(f"        p99 오차   u {qx*512:6.2f} px, v {qy*256:6.2f} px")

        ok = pcc(kp, ref_kp[0, :n]) >= 0.999 and pcc(gk, wk) >= 0.999
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    raise SystemExit(main())
