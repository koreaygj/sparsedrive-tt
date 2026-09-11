"""Run one real navtest frame through the PyTorch reference.

Phase 1 step 0. Before committing hours to metric/dataset caching, prove the
data path works end to end: logs -> sensor blobs -> feature pipeline -> model.
Synthetic inputs (test_forward_smoke.py) cannot catch a broken symlink, a
missing camera, or a projection matrix that puts every keypoint out of frame.
"""

import argparse
import os
import pathlib
import sys
import time

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test"))

import reference                      # noqa: E402
reference.install()

from test_forward_smoke import build_config, CKPT, PREFIX   # noqa: E402

from navsim.agents.sparsedrive.sparsedrive_features import SparseDriveFeatureBuilder  # noqa: E402
from navsim.agents.sparsedrive.sparsedrive_model import SparseDriveModel  # noqa: E402
from navsim.common.dataclasses import SensorConfig  # noqa: E402
from navsim.common.dataloader import SceneLoader  # noqa: E402

DEVKIT = pathlib.Path(os.environ["NAVSIM_DEVKIT_ROOT"])
DATA = pathlib.Path(os.environ["OPENSCENE_DATA_ROOT"])
FILTER_YAML = (DEVKIT / "navsim/planning/script/config/common"
               / "train_test_split/scene_filter/navtest.yaml")

# SparseDriveAgent.get_sensor_config(): 8 cameras, 4 history frames, no lidar.
SENSOR_CONFIG = SensorConfig(
    cam_f0=[0, 1, 2, 3], cam_l0=[0, 1, 2, 3], cam_l1=[0, 1, 2, 3],
    cam_l2=[0, 1, 2, 3], cam_r0=[0, 1, 2, 3], cam_r1=[0, 1, 2, 3],
    cam_r2=[0, 1, 2, 3], cam_b0=[0, 1, 2, 3], lidar_pc=[],
)


def load_model(cfg, device):
    model = SparseDriveModel(cfg)
    raw = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
    sd = {k[len(PREFIX):]: v for k, v in raw.items() if k.startswith(PREFIX)}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    return model.eval().to(device)


def batchify(features, device):
    """The builder returns one unbatched frame; the model wants a batch axis."""
    cam = features["camera_feature"]
    out = {}
    for k, v in cam.items():
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v.copy())
        if torch.is_tensor(v):
            out[k] = v.unsqueeze(0).to(device)
        else:
            out[k] = v
    return {
        "camera_feature": out,
        "status_feature": features["status_feature"].unsqueeze(0).to(device),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-scenes", type=int, default=2)
    ap.add_argument("--num-frames", type=int, default=3)
    args = ap.parse_args()

    scene_filter = instantiate(OmegaConf.load(FILTER_YAML))
    scene_filter.max_scenes = args.num_scenes

    t0 = time.time()
    loader = SceneLoader(
        data_path=DATA / "navsim_logs" / "test",
        original_sensor_path=DATA / "sensor_blobs" / "test",
        scene_filter=scene_filter,
        sensor_config=SENSOR_CONFIG,
    )
    tokens = loader.tokens[: args.num_frames]
    print(f"  scene loader          {len(loader.tokens)} tokens "
          f"({time.time() - t0:.1f}s), running {len(tokens)}")

    cfg = build_config()
    builder = SparseDriveFeatureBuilder(cfg)
    model = load_model(cfg, args.device)

    ok = True
    for i, token in enumerate(tokens):
        t0 = time.time()
        agent_input = loader.get_agent_input_from_token(token)
        feats = builder.compute_features(agent_input)
        feats, _, _ = builder.pipeline(feats, {}, token, test_mode=True)
        t_data = time.time() - t0

        batch = batchify(feats, args.device)
        imgs = batch["camera_feature"]["imgs"]

        t0 = time.time()
        with torch.no_grad():
            out, _ = model(batch, {})
        if args.device == "cuda":
            torch.cuda.synchronize()
        t_fwd = time.time() - t0

        traj = out["trajectory"][0].cpu()
        finite = bool(torch.isfinite(traj).all())
        moved = float(traj[..., :2].abs().max())
        print(f"  [{i}] {token}  data {t_data:5.2f}s  fwd {t_fwd:5.2f}s")
        print(f"      imgs {tuple(imgs.shape)} range "
              f"[{imgs.min():.2f}, {imgs.max():.2f}]  "
              f"traj {tuple(traj.shape)} finite={finite} max|xy|={moved:.2f}m")
        # A trajectory pinned at the origin means the vocabulary lookup never
        # moved -- almost always a dead camera or a bad projection.
        if not finite or moved < 1e-3:
            ok = False
            print("      !! degenerate trajectory")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
