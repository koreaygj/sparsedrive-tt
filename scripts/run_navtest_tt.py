"""navtest inference on device: cache -> images -> model -> {token: trajectory}.

Runs entirely in $TT_PY. The feature cache holds image paths and calibration as
plain pickle, and model/features.py reproduces navsim's test-mode image
pipeline bit for bit (tools/poc_features.py), so nothing crosses the
two-interpreter seam except the trajectories this writes.

    $TT_PY scripts/run_navtest_tt.py --limit 200 --out exp/tt_traj_200.pt
    $NAVSIM_PY scripts/score_navtest_tt.py --traj exp/tt_traj_200.pt

The image pipeline is PREFETCHED. model/features.py costs 38 ms a token --
9.6 of JPEG decode and 22 of resizing 1920x1080 down to 512x288, three
cameras' worth -- against 122 ms on device, and run in series that is
160 ms a frame where the two halves take turns idling. A loader thread runs a
few tokens ahead instead, so the host work hides entirely behind the device:
measured 6.2 -> 8.2 fps.

Nothing about the data changes -- the same build() on the same bytes, only
earlier -- so the trajectories stay bit-identical. It works with threads and
needs no processes because PIL's decode and resize and numpy all drop the GIL.
--sequential restores the old behaviour for an A/B.
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
    ap.add_argument("--sequential", action="store_true",
                    help="no loader thread; preprocess in line with the model")
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
    print(f"  토큰 {len(files)}개  (완료 {len(done)}개 건너뜀)", flush=True)

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
    if not args.single:
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
            # `feat` is what the model WAITED for the loader, not what the
            # loader spent: with prefetch on it should fall to roughly zero.
            print(f"    {i+1}/{len(files)}  {rate:.2f} fps  "
                  f"feat {t_feat/(i+1)*1e3:.0f} ms  model {t_run/(i+1)*1e3:.0f} ms  "
                  f"남은 {(len(files)-i-1)/rate/60:.0f}분", flush=True)
            torch.save(done, out)

        if args.sequential:
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
            # Two workers, not one: one is enough on the arithmetic (38 ms of
            # work per 122 ms of device time) but leaves nothing spare when a
            # read comes off cold cache.
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                q = collections.deque(pool.submit(load, f) for f in files[:depth])
                for i in range(len(files)):
                    a = time.time()
                    tok, fargs = q.popleft().result()
                    b = time.time()
                    # Queue the next one BEFORE the model runs, so the loader
                    # works through the device's 122 ms rather than after it.
                    if i + depth < len(files):
                        q.append(pool.submit(load, files[i + depth]))
                    done[tok] = net(*fargs).cpu()
                    c = time.time()
                    t_feat += b - a
                    t_run += c - b
                    report(i)
        torch.save(done, out)
        print(f"  저장 {len(done)}개 -> {out}", flush=True)
    finally:
        (ttnn.close_device if args.single else ttnn.close_mesh_device)(dev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
