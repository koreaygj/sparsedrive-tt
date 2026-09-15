"""navtest inference on device: cache -> images -> model -> {token: trajectory}.

Runs entirely in $TT_PY. The feature cache holds image paths and calibration as
plain pickle, and model/features.py reproduces navsim's test-mode image
pipeline bit for bit (tools/poc_features.py), so nothing crosses the
two-interpreter seam except the trajectories this writes.

    $TT_PY scripts/run_navtest_tt.py --limit 200 --out exp/tt_traj_200.pt
    $NAVSIM_PY scripts/score_navtest_tt.py --traj exp/tt_traj_200.pt

--prefetch runs the image pipeline a few tokens ahead on a loader thread.
model/features.py costs 38 ms a token -- 9.6 of JPEG decode and 22 of resizing
1920x1080 down to 512x288, three cameras' worth -- against 120 ms on device,
and in series that is 160 ms a frame with each half idling while the other
works. Prefetching hides the host work behind the device, 6.2 -> 8.2 fps, and
the data is untouched: the same build() on the same bytes, only earlier.
Threads suffice because PIL's decode and resize and numpy all drop the GIL.

--fabric turns on FABRIC_1D so the DFA gathers its anchor-sharded output with
ttnn.all_gather rather than through the host, worth about 12 ms a frame. It
NEEDS TT_MESH_GRAPH_DESC_PATH set:

    MGD=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors
    export TT_MESH_GRAPH_DESC_PATH=$MGD/n300_mesh_graph_descriptor.textproto

Without it the run dies after a few hundred to a few thousand frames with

    Failed to discover available ethernet links; falling back to 1 link
    Read unexpected run_mailbox value: 0x40 ... from core 25-17
    TIMEOUT: device timeout in fetch queue wait, potential hang detected

and cores 25-16/25-17 are ethernet cores, so what goes bad is exactly what the
fabric programs. The cause is in metal_env.cpp: tt-metal only looks up the
stock descriptor when there is more than one host rank, so a single process
takes auto-discovery instead, auto-discovery does not recognise this
motherboard, and the E/W routing planes are never registered -- which is what
get_num_usable_routing_planes reads. model/mesh.py's enable_fabric refuses
without the variable rather than let that happen.
"""

import argparse
import collections
import concurrent.futures
import gzip
import os
import pathlib
import pickle
import sys
import time

import torch
import ttnn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.features import build            # noqa: E402
from model.sparsedrive import TtSparseDrive  # noqa: E402
from model.mesh import enable_fabric      # noqa: E402

EXP = pathlib.Path(os.environ["NAVSIM_EXP_ROOT"])
CKPT = ROOT / "ckpt" / "sparsedrive_navsimv1.ckpt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = every token")
    ap.add_argument("--out", default=str(EXP / "tt_traj.pt"))
    ap.add_argument("--single", action="store_true", help="one chip instead of the mesh")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fabric", action="store_true",
                    help="enable FABRIC_1D so the DFA gathers its output with "
                         "ttnn.all_gather; see the note above -- broken on this "
                         "host and off by default")
    ap.add_argument("--prefetch", action="store_true",
                    help="loader thread runs ahead; see the note above -- this "
                         "has hung the device twice and is off by default")
    ap.add_argument("--depth", type=int, default=3,
                    help="tokens to keep preprocessed ahead (about 5 MB each)")
    args = ap.parse_args()

    files = sorted((EXP / "data_cache_navtest").glob("*/*/sparsedrive_feature.gz"))
    if args.limit:
        files = files[:args.limit]
    out = pathlib.Path(args.out)
    done = {}
    if args.resume and out.exists():
        done = torch.load(out, map_location="cpu", weights_only=False)
        files = [f for f in files if f.parent.name not in done]
    print(f"  {len(files)} tokens  ({len(done)} already done, skipped)", flush=True)

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
    if args.fabric:
        enable_fabric()
    dev = (ttnn.open_device(device_id=0, l1_small_size=24576) if args.single
           else ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576))
    try:
        net = TtSparseDrive(sd, dev)

        def load(f):
            with gzip.open(f, "rb") as fh:
                return f.parent.name, build(pickle.load(fh))

        t_feat = t_run = 0.0
        t0 = time.time()

        def report(i):
            if (i + 1) % 25 and i + 1 != len(files):
                return
            el = time.time() - t0
            rate = (i + 1) / el
            print(f"    {i+1}/{len(files)}  {rate:.2f} fps  "
                  f"feat {t_feat/(i+1)*1e3:.0f} ms  model {t_run/(i+1)*1e3:.0f} ms  "
                  f"eta {(len(files)-i-1)/rate/60:.0f} min", flush=True)
            torch.save(done, out)

        if not args.prefetch:
            for i, f in enumerate(files):
                a = time.time()
                tok, fargs = load(f)
                b = time.time()
                done[tok] = net(*fargs).cpu()
                c = time.time()
                t_feat += b - a
                t_run += c - b
                report(i)
        else:
            depth = max(1, args.depth)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                q = collections.deque(pool.submit(load, f) for f in files[:depth])
                for i in range(len(files)):
                    a = time.time()
                    tok, fargs = q.popleft().result()
                    b = time.time()
                    if i + depth < len(files):
                        q.append(pool.submit(load, files[i + depth]))
                    done[tok] = net(*fargs).cpu()
                    c = time.time()
                    t_feat += b - a
                    t_run += c - b
                    report(i)
        torch.save(done, out)
        print(f"  saved {len(done)} -> {out}", flush=True)
    finally:
        (ttnn.close_device if args.single else ttnn.close_mesh_device)(dev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
