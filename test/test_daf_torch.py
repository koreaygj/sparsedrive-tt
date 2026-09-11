"""Check daf_torch against a literal transcription of the CUDA kernel.

There is no GPU here to diff against, so the reference is the kernel source:
deformable_aggregation_cuda.cu, transcribed index-for-index at sizes small
enough to loop in Python. If the vectorised version and the transcription
agree, the reading of the kernel is right.
"""

import math
import sys
import pathlib

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from reference.daf_torch import deformable_aggregation_forward  # noqa: E402


def naive(mc_ms_feat, spatial_shape, scale_start_index, loc, weights):
    """Transcription of deformable_aggregation_kernel + bilinear_sampling."""
    B, C, _, E = mc_ms_feat.shape
    S = spatial_shape.shape[0]
    P = loc.shape[1]
    G = weights.shape[-1]
    gd = E // G
    out = torch.zeros(B, P, E, dtype=mc_ms_feat.dtype)

    for b in range(B):
        for p in range(P):
            for e in range(E):
                g = e // gd
                acc = 0.0
                for c in range(C):
                    lw = loc[b, p, c, 0].item()
                    lh = loc[b, p, c, 1].item()
                    if not (0 < lw < 1 and 0 < lh < 1):
                        continue
                    for s in range(S):
                        h = int(spatial_shape[s, 0])
                        w = int(spatial_shape[s, 1])
                        st = int(scale_start_index[s])
                        h_im = lh * h - 0.5
                        w_im = lw * w - 0.5
                        h_lo, w_lo = math.floor(h_im), math.floor(w_im)
                        h_hi, w_hi = h_lo + 1, w_lo + 1
                        dh, dw = h_im - h_lo, w_im - w_lo

                        def val(y, x):
                            if 0 <= y < h and 0 <= x < w:
                                return mc_ms_feat[b, c, st + y * w + x, e].item()
                            return 0.0

                        v = (
                            (1 - dh) * (1 - dw) * val(h_lo, w_lo)
                            + (1 - dh) * dw * val(h_lo, w_hi)
                            + dh * (1 - dw) * val(h_hi, w_lo)
                            + dh * dw * val(h_hi, w_hi)
                        )
                        acc += v * weights[b, p, c, s, g].item()
                out[b, p, e] = acc
    return out


def main():
    torch.manual_seed(0)
    B, C, S, G, gd = 1, 3, 2, 2, 4
    E = G * gd
    shapes = torch.tensor([[4, 6], [2, 3]], dtype=torch.int64)
    starts = torch.tensor([0, 4 * 6], dtype=torch.int64)
    F_total = int((shapes[:, 0] * shapes[:, 1]).sum())
    P = 24

    feat = torch.randn(B, C, F_total, E, dtype=torch.float64)
    # Spread points over, on, and outside the [0,1] box so the strict bound and
    # the zero padding both get exercised.
    loc = torch.rand(B, P, C, 2, dtype=torch.float64) * 1.4 - 0.2
    loc[0, 0, 0] = torch.tensor([0.0, 0.5])     # exactly on the edge -> skipped
    loc[0, 1, 0] = torch.tensor([1.0, 0.5])     # exactly on the edge -> skipped
    loc[0, 2, 0] = torch.tensor([0.01, 0.01])   # inside, but padding territory
    w = torch.rand(B, P, C, S, G, dtype=torch.float64)

    got = deformable_aggregation_forward(feat, shapes, starts, loc, w)
    want = naive(feat, shapes, starts, loc, w)

    err = (got - want).abs().max().item()
    denom = want.abs().max().item()
    print(f"  shapes    {tuple(got.shape)} vs {tuple(want.shape)}")
    print(f"  max |err| {err:.3e}   (ref max |v| {denom:.3f})")

    # Chunking must not change the answer.
    chunked = deformable_aggregation_forward(feat, shapes, starts, loc, w, chunk=5)
    cerr = (chunked - got).abs().max().item()
    print(f"  chunk=5 vs chunk=all  max |err| {cerr:.3e}")

    ok = err < 1e-9 and cerr == 0.0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
