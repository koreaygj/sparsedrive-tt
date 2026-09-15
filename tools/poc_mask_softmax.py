"""Phase 2: the out-of-bounds mask, and the softmax that has to respect it.

Graded against golden DAF[0].weights -- the post-mask, post-softmax tensor the
CUDA op actually consumed on a real frame, not a reconstruction.

The rule, from blocks.py _get_weights:

    mask                     (bs, C, na, P)   from project_points
    -> permute + unsqueeze   (bs, na, C, 1, P, 1)   broadcast over L and G
    masked_fill(-inf) where  ~mask AND mask.sum(over C) != 0

so a camera that cannot see a point is silenced only when some *other* camera
can. A point no camera sees is left alone and competes normally -- which is
also what keeps the denominator non-zero.

Applied here as a 0/1 multiply after exp rather than -inf before it. The two
are identical -- exp(-inf) = 0 -- but the multiply cannot produce inf-inf if
the row maximum happens to sit on a masked entry, and the row maximum is taken
over the unmasked row anyway, since any constant stabilises the exponential.

    $TT_PY tools/poc_mask_softmax.py --anchors 128
"""

import argparse
import os
import pathlib
import time

import torch
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


def true_mask(kps, proj, image_wh):
    """project_points' mask, recomputed exactly. -> (bs, C, na, P) bool."""
    pts = torch.cat([kps, torch.ones_like(kps[..., :1])], dim=-1)
    p2d = torch.matmul(proj[:, :, None, None], pts[:, None, ..., None]).squeeze(-1)
    depth = p2d[..., 2]
    m = depth > 1e-5
    p2d = p2d[..., :2] / torch.clamp(p2d[..., 2:3], min=1e-5)
    m = m & (p2d[..., 0] > 0) & (p2d[..., 1] > 0)
    p2d = p2d / image_wh[:, :, None, None]
    m = m & (p2d[..., 0] < 1) & (p2d[..., 1] < 1)
    return m, p2d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", type=int, default=128)
    args = ap.parse_args()

    frame = sorted(GOLDEN.glob("*.pt"))[0]
    d = torch.load(frame, map_location="cpu", weights_only=False)
    K = "_trajectory_head.decoder.layers.0.p_deform_model."

    kps = d[K + "kps_generator.out"]                     # [1, 1024, 500, 3]
    proj = d[K + "in[4].projection_mat"]                 # [1, 3, 4, 4]
    iwh = d[K + "in[4].image_wh"]                        # [1, 3, 2]
    logits = d[K + "weights_fc.out"]                     # [1, 1024, 3, 16000]
    gold_w = d["DAF[0].weights"]                         # [1, 512000, 3, 4, 8]

    n = args.anchors
    C, P, G = proj.shape[1], kps.shape[2], 8
    L = 4
    wide = C * L * P * G                                 # 48000

    m_true, p2d = true_mask(kps[:, :n], proj, iwh)       # (1, C, n, P)
    loc = d["DAF[0].sampling_location"][0, :n * P]       # [n*P, C, 2]
    m_bounds = ((loc[..., 0] > 0) & (loc[..., 0] < 1)
                & (loc[..., 1] > 0) & (loc[..., 1] < 1))
    m_bounds = m_bounds.reshape(n, P, C).permute(2, 0, 1).unsqueeze(0)   # (1,C,n,P)

    same = (m_true == m_bounds).float().mean()
    print(f"  frame {frame.name}   anchors={n}  C={C} L={L} P={P} G={G}")
    print(f"  visible fraction: {m_true.float().mean():.4f}")
    print(f"  bounds-only approximation agreement: {same:.6f}"
          f"  (mismatched {int((m_true != m_bounds).sum())} / {m_true.numel()})")
    print()

    # silenced: this camera cannot see the point but another one can
    mp = m_true.permute(0, 2, 1, 3)[..., None, :, None]  # (1, n, C, 1, P, 1)
    silence = torch.logical_and(~mp, mp.sum(dim=2, keepdim=True) != 0)
    keep01 = (~silence).float().expand(1, n, C, L, P, G).reshape(n, wide).contiguous()
    print(f"  silenced fraction: {silence.float().mean():.4f}")

    # host reference: -inf + softmax, exactly as the model does it
    lg = logits[0, :n].reshape(n, C, 1, 1, 1) if False else \
        logits[0, :n].reshape(n, C, L, P, G)
    ref = lg.masked_fill(silence[0], float("-inf")).reshape(n, -1, G).softmax(dim=1)
    ref = ref.reshape(n, C, L, P, G)

    # against the golden: [n*P, C, L, G] -> [n, C, L, P, G]
    gw = gold_w[0, :n * P].reshape(n, P, C, L, G).permute(0, 2, 3, 1, 4)
    print(f"  host reference vs golden DAF[0].weights   PCC {pcc(ref, gw):.6f}"
          f"   max|d| {(ref - gw).abs().max():.3e}")
    print()

    flat = logits[0, :n].reshape(n, wide).contiguous()

    device = ttnn.open_device(device_id=0)
    try:
        # Two configs, because the two matmuls are not alike.
        #
        # gather is bf16 x a 0/1 matrix: there is no mantissa to truncate, so
        # fidelity buys nothing (LoFi and HiFi4 measured identical to every
        # printed digit). scatter takes the fp32 denominator as an operand, and
        # fp32 matmul on Wormhole is emulated -- LoFi truncates it in one pass,
        # HiFi4 keeps it across three. Measured on a real frame, row-sum error:
        #
        #     scatter LoFi    1.959e-02   2.9 ms
        #     scatter HiFi4   5.554e-03   2.8 ms
        #     no scatter      5.438e-03  11.7 ms   (reshape + broadcast divide)
        #
        # HiFi4 is free here because the scatter is [n,8] x [8,48000] -- small
        # enough that the three passes do not show. Dropping the scatter for a
        # broadcast divide matches its accuracy and costs 4x, because G = 8 in a
        # 32-wide tile wastes 24 of every 32 columns; that is the same trade
        # sparse4D-tt made when it kept the compact layout.
        #
        # What remains (5.5e-3 against a 9.0e-4 bf16 floor) is the gather, whose
        # error grows with the reduction length: 9.4e-4 over 2000, 4.2e-3 over
        # the 6000 this axis actually is.
        cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True, packer_l1_acc=True)
        cfg_scatter = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True, packer_l1_acc=True)
        gather = torch.zeros(wide, G)
        for g in range(G):
            gather[g::G, g] = 1.0
        g_tt = ttnn.from_torch(gather, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        s_tt = ttnn.from_torch(gather.t().contiguous(), layout=ttnn.TILE_LAYOUT,
                               device=device, dtype=ttnn.bfloat16)
        x_tt = ttnn.from_torch(flat, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        k_tt = ttnn.from_torch(keep01, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

        def run():
            m = ttnn.max(x_tt, dim=-1, keepdim=True)
            sh = ttnn.subtract(x_tt, m); ttnn.deallocate(m)
            e = ttnn.exp(sh); ttnn.deallocate(sh)
            em = ttnn.multiply(e, k_tt); ttnn.deallocate(e)
            s = ttnn.matmul(em, g_tt, compute_kernel_config=cfg, dtype=ttnn.float32)
            sb = ttnn.matmul(s, s_tt, compute_kernel_config=cfg_scatter,
                             dtype=ttnn.float32)
            ttnn.deallocate(s)
            w = ttnn.divide(em, sb)
            ttnn.deallocate(em); ttnn.deallocate(sb)
            return w

        run()                                            # JIT warm-up
        ttnn.synchronize_device(device); t0 = time.time()
        w_tt = run()
        ttnn.synchronize_device(device); dt = (time.time() - t0) * 1e3
        got = ttnn.to_torch(w_tt).float().reshape(n, C, L, P, G)

        print(f"  [device mask+softmax] {dt:7.1f} ms (warm)")
        print(f"      vs golden DAF[0].weights      PCC {pcc(got, gw):.6f}"
              f"   max|d| {(got - gw).abs().max():.3e}")
        print(f"      row-sum error               "
              f"{(got.reshape(n, -1, G).sum(1) - 1).abs().max():.3e}")
        ok = pcc(got, gw) >= 0.999
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    raise SystemExit(main())
