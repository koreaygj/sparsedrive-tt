"""Hammer one frame to find which part of it wedges the device.

Every configuration tried so far hangs at a random frame -- 675, 1250, 2268,
5225, 7700 -- with fabric on and off, with and without the trace, with and
without the device-side shard. Resuming at the exact token that hung runs
clean, so nothing about the data causes it: the per-frame probability is about
1/3000 and the question is which part of a frame carries it.

A frame is three things: six host uploads, the device graph, one host download.
This replays a single captured frame thousands of times with those parts
switched off one at a time.

  device   execute_trace only, no host traffic at all
  upload   uploads plus the replay, no download
  full     uploads, replay, download -- the real loop, as a control
  eager    no trace at all: the model's own per-frame path, which lets the
           anchor shard and the gather go through the host instead of
           mesh_partition and all_gather. That is the only way to run the mesh
           with no CCL op in the graph, because a trace cannot hold a host
           round trip -- and separating "mesh" from "CCL" is what is left after
           64 cores on one chip ran 8000 clean while 64 cores on the mesh hangs
           every 1518.

A hang under "device" puts it in the ops or in trace dispatch. A hang that
needs "upload" or "full" puts it in host traffic racing the device graph.
"""

import argparse
import gzip
import os
import pathlib
import pickle
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

from model.features import build            # noqa: E402
from model.sparsedrive import TtSparseDrive  # noqa: E402
from model.mesh import enable_fabric         # noqa: E402

EXP = pathlib.Path(os.environ["NAVSIM_EXP_ROOT"])
CKPT = ROOT / "ckpt" / "sparsedrive_navsimv1.ckpt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("device", "upload", "full", "eager"),
                    default="device")
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--token", type=int, default=0)
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--fabric", action="store_true")
    args = ap.parse_args()

    files = sorted((EXP / "data_cache_navtest").glob("*/*/sparsedrive_feature.gz"))
    f = files[args.token]
    with gzip.open(f, "rb") as fh:
        fargs = build(pickle.load(fh))
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]

    if args.mode != "eager":
        os.environ["TT_MESH_PARTITION"] = "1"
    if args.fabric and not args.single:
        enable_fabric()
    trs = 200 * 1024 * 1024
    dev = (ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=trs)
           if args.single else
           ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576,
                                 trace_region_size=trs))
    print(f"  mode={args.mode} iters={args.iters} token={f.parent.name} "
          f"fabric={args.fabric} single={args.single}", flush=True)
    net = TtSparseDrive(sd, dev)
    if args.mode == "eager":
        net(*fargs)
        print("  warmed (no trace)", flush=True)
    else:
        net.capture(*fargs)
        ttnn.synchronize_device(dev)
        print("  captured", flush=True)

    t0 = time.time()
    i = 0
    try:
        for i in range(1, args.iters + 1):
            if args.mode == "eager":
                net(*fargs)
            else:
                if args.mode != "device":
                    net.prepare(*fargs)
                ttnn.execute_trace(dev, net.trace_id, cq_id=0, blocking=True)
                if args.mode == "full":
                    net.read(net.trace_out)
            if i % 250 == 0:
                el = time.time() - t0
                print(f"    {i}/{args.iters}  {i/el:.2f} it/s  "
                      f"{el/i*1e3:.0f} ms", flush=True)
        print(f"  survived {args.iters} iterations of {args.mode}", flush=True)
    except BaseException as e:
        print(f"  HUNG at iteration {i} of {args.mode}: {type(e).__name__}", flush=True)
        print(f"  {e}"[:2000], flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    (ttnn.close_device if args.single else ttnn.close_mesh_device)(dev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
