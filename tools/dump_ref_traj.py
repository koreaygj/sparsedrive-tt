"""Run the PyTorch model and save its trajectories, to compare against TT.

The port is graded per module by PCC against exp/golden, but a picture of the
planned path needs the end of the model, not its middle, and in the same
{token: [poses, 3]} shape run_navtest_tt.py writes. This produces that file so
tools/plot_traj.py can draw the two on one frame.

Runs under $NAVSIM_PY -- it loads scenes and the reference model, which pull in
nuplan.

    $NAVSIM_PY tools/dump_ref_traj.py --frames 8
    $NAVSIM_PY tools/dump_ref_traj.py --frames 8 --device cpu --gt exp/gt_traj.pt

Output: exp/ref_traj.pt, and with --gt the human driver's future as well.

The human path is what the log recorded, not what the model is graded on:
navsim scores a prediction by simulating it (collision, drivable area, time to
collision, progress, comfort), on the argument that L2 to one human path treats
every other safe path as wrong. Expect the model to differ from it and still
score well.
"""

import argparse
import os
import pathlib
import sys
import time

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test"))

import reference                      # noqa: E402
reference.install()

from test_forward_smoke import build_config   # noqa: E402
from test_real_frame import SENSOR_CONFIG, FILTER_YAML, batchify, load_model  # noqa: E402

from navsim.agents.sparsedrive.sparsedrive_features import SparseDriveFeatureBuilder  # noqa: E402
from navsim.common.dataloader import SceneLoader  # noqa: E402

DATA = pathlib.Path(os.environ["OPENSCENE_DATA_ROOT"])
EXP = pathlib.Path(os.environ["NAVSIM_EXP_ROOT"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(EXP / "ref_traj.pt"))
    ap.add_argument("--gt", default="", help="also write the human future here")
    args = ap.parse_args()

    scene_filter = instantiate(OmegaConf.load(FILTER_YAML))
    scene_filter.max_scenes = args.frames
    loader = SceneLoader(
        data_path=DATA / "navsim_logs" / "test",
        original_sensor_path=DATA / "sensor_blobs" / "test",
        scene_filter=scene_filter,
        sensor_config=SENSOR_CONFIG,
    )
    tokens = loader.tokens[: args.frames]
    cfg = build_config()
    builder = SparseDriveFeatureBuilder(cfg)
    model = load_model(cfg, args.device)
    print(f"  {len(tokens)} tokens on {args.device}", flush=True)

    out, gt = {}, {}
    for i, token in enumerate(tokens):
        t0 = time.time()
        agent_input = loader.get_agent_input_from_token(token)
        feats = builder.compute_features(agent_input)
        feats, _, _ = builder.pipeline(feats, {}, token, test_mode=True)
        with torch.no_grad():
            pred, _ = model(batchify(feats, args.device), {})
        out[token] = pred["trajectory"][0].float().cpu()
        if args.gt:
            human = loader.get_scene_from_token(token).get_future_trajectory(
                len(out[token]))
            gt[token] = torch.as_tensor(human.poses).float()
        print(f"  [{i}] {token}  {tuple(out[token].shape)}  {time.time() - t0:.1f}s",
              flush=True)

    torch.save(out, args.out)
    print(f"  wrote {len(out)} -> {args.out}", flush=True)
    if args.gt:
        torch.save(gt, args.gt)
        print(f"  wrote {len(gt)} -> {args.gt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
