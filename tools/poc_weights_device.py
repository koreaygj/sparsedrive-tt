"""Phase 2: weights_fc and the 6000-wide softmax on the device.

Risks 3 and 4 from docs/ROADMAP.md, and the last piece of the DFA still
running on the host.

    weights_fc   Linear(256, 16000) over 1024 anchors x 3 cameras
                 -> [1, 1024, 3, 16000] = 49.15M elements, 94 MB in bf16
    softmax      over clp = C*L*P = 6000, for each (anchor, group)

The softmax axis is the problem. Laid out flat the row is
clp*G = 48000 wide and the entries being summed are strided by G = 8, which no
stock reduction op walks. sparse4D-tt's answer, reused here: exponentiate,
then take the per-group sums with a 0/1 matmul that picks every G-th column,
scatter them back with its transpose, and divide. Everything stays on the
compact layout, which is also exactly what weights_fc already emits -- its
16000 outputs are (L, P, G) with the camera axis outside, i.e. (C, L, P, G),
i.e. clp-major then G.

    $TT_PY tools/poc_weights_device.py --anchors 128
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
PREFIX = "agent._sparsedrive_model._trajectory_head.decoder.layers.0.p_deform_model."


def pcc(a, b):
    a = a.detach().flatten().to(torch.float64)
    b = b.detach().flatten().to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    d = a.norm() * b.norm()
    return 1.0 if d == 0 else float((a @ b) / d)


def host_softmax_strided(logits, G):
    """Reference: softmax over the strided clp axis, done the obvious way."""
    n, wide = logits.shape
    clp = wide // G
    return logits.reshape(n, clp, G).softmax(dim=1).reshape(n, wide)


def device_softmax_strided(device, logits_tt, G, clp, gather_tt, scatter_tt, cfg):
    """exp -> per-group sums by 0/1 matmul -> scatter back -> divide.

    The shift is the row maximum, not the per-group maximum: any constant
    stabilises the exponential, and the per-group one would need the very
    strided reduction this is working around.

    Two settings on the denominator matter, measured on a real frame by summing
    2000 fixed bf16 values and comparing to their fp64 sum:

        LoFi                            2.830e-01
        HiFi4                           2.830e-01   <- fidelity is irrelevant
        HiFi4 + fp32_dest_acc_en        6.176e-02
        HiFi4 + fp32_dest + packer_l1   4.664e-03
        ... and with an fp32 output     9.428e-04

    math_fidelity does nothing because the multiplicand is a 0/1 matrix -- there
    is no mantissa to truncate, the error is all accumulation. packer_l1_acc is
    worth 13x. And the floor at ~4.7e-3 was never the accumulator: it is bf16's
    2^-9 storing the result. Asking for an fp32 output buys another 5x, for
    nothing -- the tensor is [n, G]. Widening the *inputs* to fp32 changes the
    answer not at all (9.428e-04 either way), which is what says the remaining
    error is accumulation rather than the compute path.

    It matters because w = e/sb: if sb were exact the row would sum to 1 by
    construction, so every bit of row-sum error is the denominator's.
    """
    m = ttnn.max(logits_tt, dim=-1, keepdim=True)
    shifted = ttnn.subtract(logits_tt, m)
    ttnn.deallocate(m)
    e = ttnn.exp(shifted)
    ttnn.deallocate(shifted)
    s = ttnn.matmul(e, gather_tt, compute_kernel_config=cfg,
                    dtype=ttnn.float32)                           # [n, G] fp32
    sb = ttnn.matmul(s, scatter_tt, compute_kernel_config=cfg,
                     dtype=ttnn.float32)                          # [n, clp*G] fp32
    ttnn.deallocate(s)
    w = ttnn.divide(e, sb)
    ttnn.deallocate(e)
    ttnn.deallocate(sb)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", type=int, default=128)
    args = ap.parse_args()

    frame = sorted(GOLDEN.glob("*.pt"))[0]
    d = torch.load(frame, map_location="cpu", weights_only=False)
    K = "_trajectory_head.decoder.layers.0.p_deform_model."
    x = d[K + "weights_fc.in[0]"]                    # [1, 1024, 3, 256]
    ref_fc = d[K + "weights_fc.out"]                 # [1, 1024, 3, 16000]

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
    W = sd[PREFIX + "weights_fc.weight"].float()     # [16000, 256]
    B = sd[PREFIX + "weights_fc.bias"].float()       # [16000]

    n = args.anchors
    C, E = x.shape[2], x.shape[3]
    G = 8
    wide = W.shape[0]                                # 16000 = L*P*G
    clp = wide // G * C                              # 6000 across all cameras

    xin = x[0, :n].reshape(n * C, E).contiguous()    # [n*C, 256]
    ref = ref_fc[0, :n].reshape(n * C, wide)         # [n*C, 16000]

    print(f"  frame {frame.name}   anchors={n}  C={C}  E={E}")
    print(f"  weights_fc: [{n*C}, {E}] x [{E}, {wide}] -> [{n*C}, {wide}]"
          f"  = {n*C*wide/1e6:.1f}M elem, {n*C*wide*2/2**20:.0f} MB bf16")
    print(f"  softmax axis: clp = {clp}  (per camera {wide//G}), G = {G}")
    print()

    device = ttnn.open_device(device_id=0)
    try:
        cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        x_tt = ttnn.from_torch(xin, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        w_tt = ttnn.from_torch(W.t().contiguous(), layout=ttnn.TILE_LAYOUT,
                               device=device, dtype=ttnn.bfloat16)
        b_tt = ttnn.from_torch(B.reshape(1, -1), layout=ttnn.TILE_LAYOUT,
                               device=device, dtype=ttnn.bfloat16)

        ttnn.synchronize_device(device); t0 = time.time()
        y_tt = ttnn.linear(x_tt, w_tt, bias=b_tt, compute_kernel_config=cfg)
        ttnn.synchronize_device(device); t_fc = (time.time() - t0) * 1e3
        y = ttnn.to_torch(y_tt).float()
        print(f"  [weights_fc]  {t_fc:7.1f} ms   PCC {pcc(y, ref):.6f}"
              f"   max|d| {(y - ref).abs().max():.3e}")

        # softmax over the per-camera clp slice (1 camera's 2000 entries here;
        # the real axis spans all 3 cameras and is stitched after the port of
        # the camera-major layout -- measured separately below)
        sub_clp = wide // G
        gather = torch.zeros(wide, G)
        for g in range(G):
            gather[g::G, g] = 1.0
        scatter = gather.t().contiguous()
        g_tt = ttnn.from_torch(gather, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        s_tt = ttnn.from_torch(scatter, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

        ttnn.synchronize_device(device); t0 = time.time()
        w_out = device_softmax_strided(device, y_tt, G, sub_clp, g_tt, s_tt, cfg)
        ttnn.synchronize_device(device); t_sm = (time.time() - t0) * 1e3
        got = ttnn.to_torch(w_out).float()
        want = host_softmax_strided(ref, G)
        print(f"  [softmax {sub_clp:>4d}] {t_sm:7.1f} ms   PCC {pcc(got, want):.6f}"
              f"   row-sum err {(got.reshape(-1, sub_clp, G).sum(1) - 1).abs().max():.3e}")
        print()
        print(f"  total {t_fc + t_sm:.1f} ms  (replaces the weights preparation that was left on the host)")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
