"""navtest inference on device: cache -> images -> model -> {token: trajectory}.

Runs entirely in $TT_PY. The feature cache holds image paths and calibration as
plain pickle, and model/features.py reproduces navsim's test-mode image
pipeline bit for bit (tools/poc_features.py), so nothing crosses the
two-interpreter seam except the trajectories this writes.

    $TT_PY scripts/run_navtest_tt.py --limit 200 --out exp/tt_traj_200.pt
    $NAVSIM_PY scripts/score_navtest_tt.py --traj exp/tt_traj_200.pt
"""

import argparse
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

EXP = pathlib.Path(os.environ["NAVSIM_EXP_ROOT"])
CKPT = ROOT / "ckpt" / "sparsedrive_navsimv1.ckpt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = every token")
    ap.add_argument("--out", default=str(EXP / "tt_traj.pt"))
    ap.add_argument("--single", action="store_true", help="one chip instead of the mesh")
    ap.add_argument("--resume", action="store_true")
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
    dev = (ttnn.open_device(device_id=0, l1_small_size=24576) if args.single
           else ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576))
    try:
        net = TtSparseDrive(sd, dev)
        t_feat = t_run = 0.0
        t0 = time.time()
        for i, f in enumerate(files):
            tok = f.parent.name
            a = time.time()
            with gzip.open(f, "rb") as fh:
                imgs, status, proj, iwh = build(pickle.load(fh))
            b = time.time()
            done[tok] = net(imgs, status, proj, iwh).cpu()
            c = time.time()
            t_feat += b - a
            t_run += c - b
            if (i + 1) % 25 == 0 or i + 1 == len(files):
                el = time.time() - t0
                rate = (i + 1) / el
                print(f"    {i+1}/{len(files)}  {rate:.2f} fps  "
                      f"feat {t_feat/(i+1)*1e3:.0f} ms  model {t_run/(i+1)*1e3:.0f} ms  "
                      f"남은 {(len(files)-i-1)/rate/60:.0f}분", flush=True)
                torch.save(done, out)
        torch.save(done, out)
        print(f"  저장 {len(done)}개 -> {out}", flush=True)
    finally:
        (ttnn.close_device if args.single else ttnn.close_mesh_device)(dev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
