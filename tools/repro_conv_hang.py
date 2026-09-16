"""Minimal reproducer: run one conv2d in a loop until the device wedges.

The full model needs about 2000 ops and 1500 frames to hang, which makes every
experiment a 17 minute round trip. Every triage dump of that hang named the
same op -- the FPN's first output conv, 3x3, 256 -> 256 over 3x64x128 -- so
this runs that one conv and nothing else.

The hang itself is not conv-specific: one dump caught a matmul instead. What is
specific is the compute grid. On this n300 the failure needs the full 8x8 = 64
core grid; at 56 cores, in either shape, 28000 iterations ran clean. So the
grid is the knob this script exists to bisect, and --grid drives it by writing
the eth-dispatch core descriptor before the device opens.

    $TT_PY tools/repro_conv_hang.py --iters 20000
    $TT_PY tools/repro_conv_hang.py --iters 20000 --mix 4
    $TT_PY tools/repro_conv_hang.py --iters 20000 --single

--mix N runs N convs of different shapes round-robin instead of one shape over
and over. The single-shape loop hits the program cache every iteration, so the
cores never swap kernels; 40000 of those ran clean, which is 26 times the
exposure the model needs to hang. The model instead pushes about 2000 distinct
programs per frame, and every callstack of the real hang sat in the launch
handshake -- brisc in wait_ncrisc_trisc, ncrisc in wait_for_brisc_notification,
the triscs still at GO -- so kernel switching, not conv arithmetic, is the next
thing to bisect.

A hang leaves a triage dump at /tmp/triage_hang.csv, with the stuck cores, the
resolved callstacks of all five RISCs, and the per-RISC sync state.
"""

import argparse
import os
import pathlib
import sys
import time

import torch

os.environ.setdefault("TT_METAL_OPERATION_TIMEOUT_SECONDS", "50")
os.environ.setdefault(
    "TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE",
    str(pathlib.Path(os.environ["TT_METAL_HOME"]) / "tools" / "tt-triage.py")
    + " -v --llm-output-path=/tmp/triage_hang.csv")
os.environ["PATH"] = f"{pathlib.Path(sys.executable).parent}:{os.environ['PATH']}"

import ttnn  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.mesh import enable_fabric  # noqa: E402

BATCH, H, W, CH = 3, 64, 128, 256


def build_conv(device, out_ch=CH):
    """The FPN layer_block[0] conv, with the config the model gives it."""
    torch.manual_seed(out_ch)
    w = torch.randn(out_ch, CH, 3, 3, dtype=torch.float32) * 0.02
    b = torch.zeros(1, 1, 1, out_ch, dtype=torch.float32)
    weight = ttnn.from_torch(w, dtype=ttnn.bfloat16)
    bias = ttnn.from_torch(b, dtype=ttnn.bfloat16)
    cfg = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        deallocate_activation=False,
        reshard_if_not_optimal=True,
        activation=None)
    compute = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=True, packer_l1_acc=False, math_approx_mode=False)
    return weight, bias, cfg, compute


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--mix", type=int, default=1,
                    help="number of distinct conv shapes to round-robin")
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--fabric", action="store_true", default=True)
    ap.add_argument("--no-fabric", dest="fabric", action="store_false")
    args = ap.parse_args()

    if args.fabric and not args.single:
        enable_fabric()
    dev = (ttnn.open_device(device_id=0, l1_small_size=24576)
           if args.single else
           ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576))
    g = dev.compute_with_storage_grid_size()
    print(f"  grid {g.x} x {g.y} = {g.x * g.y} cores   "
          f"single={args.single} fabric={args.fabric} mix={args.mix}", flush=True)

    variants = [build_conv(dev, CH - 8 * k) for k in range(args.mix)]
    mapper = None if args.single else ttnn.ReplicateTensorToMesh(dev)
    act = ttnn.from_torch(
        torch.randn(1, 1, BATCH * H * W, CH, dtype=torch.float32) * 0.1,
        dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
        mesh_mapper=mapper)

    i = 0
    t0 = time.time()
    try:
        for i in range(1, args.iters + 1):
            v = (i - 1) % len(variants)
            weight, bias, cfg, compute = variants[v]
            out, _, [weight, bias] = ttnn.conv2d(
                input_tensor=act, weight_tensor=weight, bias_tensor=bias,
                device=dev, in_channels=CH, out_channels=CH - 8 * v,
                input_height=H, input_width=W, batch_size=BATCH,
                kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1,
                conv_config=cfg, compute_config=compute,
                return_output_dim=True, return_weights_and_bias=True)
            variants[v] = (weight, bias, cfg, compute)
            ttnn.deallocate(out)
            if i % 1000 == 0:
                el = time.time() - t0
                print(f"    {i}/{args.iters}  {i / el:.1f} conv/s  "
                      f"{el / i * 1e3:.2f} ms", flush=True)
        print(f"  survived {args.iters} convs", flush=True)
    except BaseException as e:
        print(f"  HUNG at conv {i}: {type(e).__name__}", flush=True)
        print(f"  {e}"[:600], flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    (ttnn.close_device if args.single else ttnn.close_mesh_device)(dev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
