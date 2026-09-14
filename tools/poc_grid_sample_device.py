"""Phase 2: can ttnn.grid_sample produce the DFA's sampled features at our scale?

The gws PoC (poc_gws_scale.py) left sampling on the host -- 438.9 ms of the
1711 ms it measured. This moves it to the device and grades it against the
PyTorch reference on the same real frame.

ttnn.grid_sample contract, read off grid_sample.hpp / the device op:

    input  (N, H, W, C)          ROW_MAJOR, NHWC -- not the (N,C,H,W) torch uses
    grid   (N, H_out, W_out, 2)  ROW_MAJOR, coords in [-1, 1]
    out    (N, H_out, W_out, C)

So one call per FPN level with the 3 cameras as the batch axis, and every
anchor-point of the chunk laid out down H_out:

    input  (3, H_l, W_l, 256)
    grid   (3, n*500, 1, 2)
    out    (3, n*500, 1, 256)

The grid dtype is the thing to watch. sparse4D-tt feeds grid_sample a Q14
fixed-point grid rather than bf16, and the reason shows up here: a bf16
coordinate in [-1,1] has ~8 bits of mantissa, which on a 128-wide level is
half a pixel of positional error. This sweeps the dtypes to measure it.

    $TT_PY tools/poc_grid_sample_device.py --anchors 64
"""

import argparse
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


def host_reference(feat, shapes, starts, grid, C, E):
    """torch grid_sample, fp32 -- the number the device has to match."""
    out = []
    for (h, w), st in zip(shapes, starts):
        level = feat[0, :, st:st + h * w, :].permute(0, 2, 1).reshape(C, E, h, w)
        s = F.grid_sample(level, grid, mode="bilinear",
                          padding_mode="zeros", align_corners=False)
        out.append(s.squeeze(-1).permute(0, 2, 1))      # [C, npts, E]
    return out


def device_sample(device, feat, shapes, starts, grid, C, E, dtype):
    """ttnn.grid_sample per level. Returns list of [C, npts, E] torch tensors."""
    out = []
    for (h, w), st in zip(shapes, starts):
        # (C, F_level, E) -> NHWC (C, h, w, E)
        lvl = feat[0, :, st:st + h * w, :].reshape(C, h, w, E).contiguous()
        fm = ttnn.from_torch(lvl, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=dtype)
        g = ttnn.from_torch(grid.unsqueeze(2).contiguous(),   # (C, npts, 1, 2)
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=dtype)
        s = ttnn.grid_sample(fm, g, padding_mode="zeros", align_corners=False)
        r = ttnn.to_torch(s).float().squeeze(2)               # (C, npts, E)
        for t in (fm, g, s):
            ttnn.deallocate(t)
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", type=int, default=64)
    ap.add_argument("--call", default="DAF[0]")
    args = ap.parse_args()

    frame = sorted(GOLDEN.glob("*.pt"))[0]
    d = torch.load(frame, map_location="cpu", weights_only=False)
    P = args.call
    feat = d[f"{P}.mc_ms_feat"]                       # [1, C, F, E]
    loc = d[f"{P}.sampling_location"]                 # [1, npts_total, C, 2]
    shapes = [tuple(int(v) for v in r) for r in d[f"{P}.spatial_shape"]]
    starts = [int(v) for v in d[f"{P}.scale_start_index"]]
    pts = {"DAF[0]": 500, "DAF[1]": 500, "DAF[2]": 80}[P]

    C, E = feat.shape[1], feat.shape[3]
    n = args.anchors
    npts = n * pts
    loc_c = loc[0, :npts]                              # [npts, C, 2]
    grid = (loc_c * 2 - 1).permute(1, 0, 2).contiguous()   # [C, npts, 2]

    print(f"  frame {frame.name}  {P}   anchors={n}  pts/anchor={pts}")
    print(f"  levels {shapes}   C={C} E={E}   grid points/cam = {npts}")
    print()

    ref = host_reference(feat, shapes, starts, grid.unsqueeze(2), C, E)

    device = ttnn.open_device(device_id=0)
    try:
        for name, dt in (("bfloat16", ttnn.bfloat16), ("float32", ttnn.float32)):
            try:
                device_sample(device, feat, shapes, starts, grid, C, E, dt)  # warm
                ttnn.synchronize_device(device)
                t0 = time.time()
                got = device_sample(device, feat, shapes, starts, grid, C, E, dt)
                ttnn.synchronize_device(device)
                dt_ms = (time.time() - t0) * 1e3
                print(f"  [{name}]  {dt_ms:7.1f} ms")
                for i, ((h, w), g, r) in enumerate(zip(shapes, got, ref)):
                    print(f"      level {i} {str((h,w)):10s}  PCC {pcc(g, r):.6f}"
                          f"   max|d| {(g - r).abs().max():.3e}")
            except Exception as e:
                print(f"  [{name}]  FAILED: {type(e).__name__}: {str(e)[:150]}")
            print()
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
