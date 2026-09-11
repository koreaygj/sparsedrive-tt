"""Build SparseDriveModel with real weights and run one forward.

Synthetic inputs of the right shape; this proves the graph is wired and the
checkpoint fits, not that the numbers are right. Real numbers come in Phase 1,
from navsim frames.

    $NAVSIM_PY test/test_forward_smoke.py               # cpu
    $NAVSIM_PY test/test_forward_smoke.py --device cuda
    $NAVSIM_PY test/test_forward_smoke.py --parity      # both, and compare

--parity is the useful one: the CPU path is the transcription-checked
reference, so agreement there is what licenses using the GPU for Phase 1
golden-tensor dumps.
"""

import argparse
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import reference                      # noqa: E402
reference.install()                   # must precede any navsim.agents import

from navsim.agents.sparsedrive.sparsedrive_config import SparseDriveConfig      # noqa: E402
from navsim.agents.sparsedrive.sparsedrive_model import SparseDriveModel        # noqa: E402

CKPT = ROOT / "ckpt" / "sparsedrive_navsimv1.ckpt"
PREFIX = "agent._sparsedrive_model."


def build_config():
    return SparseDriveConfig(
        bkb_path=str(ROOT / "ckpt" / "resnet34.bin"),
        path_anchor=str(ROOT / "ckpt" / "kmeans" / "path_1024.npy"),
        velocity_anchor=str(ROOT / "ckpt" / "kmeans" / "velocity_256.npy"),
        trajectory_anchor=str(ROOT / "ckpt" / "kmeans" / "trajectory_1024_256.npz"),
        dataset_version="v1",
        metrics=("no_at_fault_collisions", "drivable_area_compliance",
                 "driving_direction_compliance", "time_to_collision_within_bound",
                 "comfort", "ego_progress"),
        velocity_filter_num=(64, 20),   # scripts/evaluation/run_pdm_score_navtest_v1.sh
    )


def build_inputs(cfg, batch=1):
    """Synthetic frame. The projection pushes keypoints into frame; with an
    identity matrix every point lands at the origin and the strict 0<x,y<1
    bound rejects all of them, which would exercise nothing."""
    torch.manual_seed(0)
    C = len(cfg.cams)
    H, W = cfg.final_dim
    proj = torch.eye(4).repeat(batch, C, 1, 1)
    proj[:, :, 0, 3] = 100.0
    proj[:, :, 1, 3] = 100.0
    proj[:, :, 2, 3] = 10.0
    return {
        "camera_feature": {
            "imgs": torch.randn(batch, C, 3, H, W),
            "projection_mat": proj,
            "image_wh": torch.tensor([[float(W), float(H)]]).repeat(batch, C, 1),
        },
        "status_feature": torch.randn(batch, 8),
    }


def to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    return obj


def run(cfg, device, report=True):
    model = SparseDriveModel(cfg)
    raw = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
    sd = {k[len(PREFIX):]: v for k, v in raw.items() if k.startswith(PREFIX)}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.eval().to(device)

    features = to_device(build_inputs(cfg), device)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        out, _ = model(features, {})
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

    traj = out["trajectory"]
    if report:
        mem = ""
        if device == "cuda":
            mem = f"  peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
        print(f"  [{device:4s}] {len(sd)} tensors, {len(missing)} missing, "
              f"{len(unexpected)} unexpected | forward {dt:6.2f}s{mem}")
        print(f"         trajectory {tuple(traj.shape)} "
              f"finite={bool(torch.isfinite(traj).all())}")
    return traj.cpu(), missing, unexpected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--parity", action="store_true",
                    help="run both devices and compare")
    args = ap.parse_args()

    cfg = build_config()
    expected = (1, cfg.trajectory_sampling.num_poses, 3)
    ok = True

    if args.parity:
        if not torch.cuda.is_available():
            print("  cuda unavailable"); return 1
        gpu, m, u = run(cfg, "cuda")
        cpu, _, _ = run(cfg, "cpu")
        err = (gpu - cpu).abs().max().item()
        scale = cpu.abs().max().item()
        print(f"  parity  max |gpu - cpu| = {err:.3e}   (ref max |v| {scale:.3f})")
        # fp32 over a ~50-layer graph plus a 6000-wide softmax; cuDNN and the
        # CPU kernels do not associate the same way. Tolerance is on the
        # trajectory in metres.
        ok = err < 1e-3 and not m and not u
        traj = gpu
    else:
        traj, m, u = run(cfg, args.device)
        ok = not m and not u

    ok = ok and tuple(traj.shape) == expected and bool(torch.isfinite(traj).all())
    if tuple(traj.shape) != expected:
        print(f"  !! expected trajectory {expected}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
