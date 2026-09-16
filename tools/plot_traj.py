"""Draw a frame's planned trajectory on the three cameras and in BEV.

Reads the trajectories two runs produced for the same token and puts them on
the same picture: the left, front and right camera the model actually saw, plus
a bird's-eye view. Two trajectories overlap almost everywhere, so the pair is
drawn in different styles rather than different panels, and the header carries
the largest distance between them.

The camera panels come from the feature cache, not from a fresh NAVSIM load, so
they use the same resize and crop the model was given and the same lidar2img
matrix it projected its anchors with. That makes the drawing exact rather than
approximate, and it runs under $TT_PY with no nuplan import.

    $TT_PY tools/plot_traj.py --traj exp/tt_traj.pt --ref exp/ref_traj.pt --n 4
    $TT_PY tools/plot_traj.py --traj exp/tt_traj.pt --tokens 05d0a1a763fc5334
    $TT_PY tools/plot_traj.py --traj exp/tt_traj.pt --ref exp/ref_traj.pt \
           --n 200 --video exp/traj.mp4

Output: exp/traj_plots/<token>.png, and with --video an H.264 mp4.
"""

import argparse
import gzip
import os
import pathlib
import pickle
import shutil
import subprocess
import sys
import tempfile

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.features import CAMS, camera_params, _aug  # noqa: E402

EXP = pathlib.Path(os.environ["NAVSIM_EXP_ROOT"])
CAM_TITLE = {
    "cam_f0": "front", "cam_l0": "front-left", "cam_r0": "front-right",
    "cam_l1": "left", "cam_r1": "right",
    "cam_l2": "rear-left", "cam_r2": "rear-right", "cam_b0": "rear",
}
RING = [
    ("cam_l0", 0, 0), ("cam_f0", 0, 1), ("cam_r0", 0, 2),
    ("cam_l1", 1, 0),                   ("cam_r1", 1, 2),
    ("cam_l2", 2, 0), ("cam_b0", 2, 1), ("cam_r2", 2, 2),
]
STYLE = {
    "tt": dict(color="#e8392f", lw=2.4, ls="-", label="TT-NN"),
    "ref": dict(color="#2f6fe8", lw=2.0, ls="--", label="PyTorch"),
    "gt": dict(color="#12a150", lw=1.8, ls=":", label="human"),
}


def cache_files():
    """token -> path of the cached feature the runner reads."""
    return {f.parent.name: f
            for f in (EXP / "data_cache_navtest").glob("*/*/sparsedrive_feature.gz")}


def load_frame(path):
    """Cached entry -> (current frame dict, lidar2img per camera)."""
    with gzip.open(path, "rb") as fh:
        feature = pickle.load(fh)
    frame = feature["camera_feature"][-1]
    resize, resize_dims, crop = _aug()
    mat = np.eye(3)
    mat[:2, :2] *= resize
    mat[:2, 2] -= np.array(crop[:2])
    ext = np.eye(4)
    ext[:3, :3] = mat
    l2i = [ext @ m for m in camera_params(frame)]
    return frame, l2i, resize_dims, crop


def raw_l2i(info):
    """One camera's calibration -> lidar2img at the sensor's own resolution.

    The five cameras the model never sees have no resize or crop to compose,
    so they get the plain matrix; this is camera_params for a single entry.
    """
    r = np.linalg.inv(info["sensor2lidar_rotation"])
    l2c = np.eye(4)
    l2c[:3, :3] = r.T
    l2c[3, :3] = -(info["sensor2lidar_translation"] @ r.T)
    viewpad = np.eye(4)
    k = info["intrinsics"]
    viewpad[:k.shape[0], :k.shape[1]] = k
    return viewpad @ l2c.T


def densify(poses, per_segment=24):
    """[n, 2] waypoints -> a polyline dense enough to survive perspective."""
    pts = np.concatenate([np.zeros((1, 2), dtype=np.float64), poses[:, :2]])
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        t = np.linspace(0.0, 1.0, per_segment, endpoint=False)[:, None]
        out.append(a[None, :] + (b - a)[None, :] * t)
    out.append(pts[-1:][:, :2])
    return np.concatenate(out)


def project(points_xy, l2i, w, h, z=0.0):
    """Ground-plane points in the lidar frame -> pixel coordinates and a mask.

    z stays at the road because that is where the model's own path anchors sit;
    it projects them with this very matrix, offset only by fix_height.
    """
    n = len(points_xy)
    pts = np.concatenate([points_xy, np.full((n, 1), z), np.ones((n, 1))], axis=1)
    img = (l2i @ pts.T).T
    depth = img[:, 2]
    ok = depth > 1e-3
    uv = np.zeros((n, 2))
    uv[ok] = img[ok, :2] / depth[ok, None]
    ok &= (uv[:, 0] > -w) & (uv[:, 0] < 2 * w) & (uv[:, 1] > -h) & (uv[:, 1] < 2 * h)
    return uv, ok


def draw_on_camera(ax, uv, ok, style):
    """Plot only the runs of consecutive in-front-of-camera points."""
    start = None
    for i in range(len(ok) + 1):
        inside = i < len(ok) and ok[i]
        if inside and start is None:
            start = i
        elif not inside and start is not None:
            if i - start > 1:
                ax.plot(uv[start:i, 0], uv[start:i, 1], **style)
            start = None


def camera_ax(ax, frame, cam, l2i, trajs, size=None, mark=False):
    """One camera image with the trajectories drawn where the lens can see them."""
    from PIL import Image
    im = Image.open(str(frame[cam]["image_path"]))
    if size is not None:
        resize_dims, crop = size
        im = im.resize(resize_dims).crop(crop)
    a = np.array(im)
    h, w = a.shape[:2]
    ax.imshow(a)
    for key, poses in trajs.items():
        uv, ok = project(densify(poses), l2i, w, h)
        draw_on_camera(ax, uv, ok, dict(STYLE[key], label=None))
    title = CAM_TITLE[cam]
    ax.set_title(title + ("  (model input)" if mark else ""), fontsize=9,
                 color="#111111" if mark else "#888888")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")


def panel(fig, gs, frame, l2i, resize_dims, crop, trajs):
    """Three camera axes across the top of the figure."""
    for c, cam in enumerate(CAMS):
        camera_ax(fig.add_subplot(gs[0, c]), frame, cam, l2i[c], trajs,
                  size=(resize_dims, crop))


def bev(fig, gs, trajs, span=40.0, cell=None):
    """Ego-frame bird's-eye view, x forward, y left, ego box at the origin."""
    ax = fig.add_subplot(gs[1, :] if cell is None else cell)
    ax.add_patch(plt.Rectangle((-1.15, -1.5), 2.3, 4.6, facecolor="#444444",
                               edgecolor="none", zorder=3))
    for key, poses in trajs.items():
        p = np.concatenate([np.zeros((1, 2)), poses[:, :2]])
        ax.plot(-p[:, 1], p[:, 0], marker="o", markersize=3.2, zorder=4,
                **STYLE[key])
    ax.set_xlim(-span / 2, span / 2)
    ax.set_ylim(-5.0, span)
    ax.set_aspect("equal")
    ax.grid(True, color="#dddddd", lw=0.6)
    ax.set_xlabel("y  [m]  (left positive)", fontsize=8, labelpad=1)
    ax.set_ylabel("x  [m]  (forward)", fontsize=8, labelpad=1)
    ax.tick_params(labelsize=7)
    ax.legend(loc="upper right", fontsize=8)


def plot_token(token, path, trajs, out, tight=True, caption="", ring=False):
    """One figure: the cameras and one BEV.

    ring=False shows the three cameras the model is given, cropped exactly as
    it received them. ring=True shows all eight the log carries, at their own
    resolution, with the BEV in the middle -- useful for seeing the scene, but
    five of those views are not model input and are labelled as such.

    tight=False keeps every frame the same pixel size, which an encoder needs
    and a trimmed figure cannot promise.
    """
    frame, l2i, resize_dims, crop = load_frame(path)
    if ring:
        fig = plt.figure(figsize=(15.0, 9.6))
        gs = fig.add_gridspec(3, 3, hspace=0.26, wspace=0.03,
                              left=0.03, right=0.98, top=0.93, bottom=0.02)
        for cam, r, c in RING:
            camera_ax(fig.add_subplot(gs[r, c]), frame, cam,
                      raw_l2i(frame[cam]), trajs, mark=cam in CAMS)
        bev(fig, gs, trajs, cell=gs[1, 1])
    else:
        fig = plt.figure(figsize=(12.5, 8.4))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 2.3], hspace=0.12,
                              wspace=0.04, left=0.06, right=0.97, top=0.9,
                              bottom=0.07)
        panel(fig, gs, frame, l2i, resize_dims, crop, trajs)
        bev(fig, gs, trajs)
    head = f"{caption}{token}" if caption else token
    if "tt" in trajs and "ref" in trajs:
        a, b = trajs["tt"][:, :2], trajs["ref"][:, :2]
        head += f"    max |TT - PyTorch| = {np.abs(a - b).max():.4f} m"
    fig.suptitle(head, fontsize=10)
    fig.savefig(out, dpi=110, **({"bbox_inches": "tight"} if tight else {}))
    plt.close(fig)
    return out


def encode(png_dir, out, fps):
    """PNG sequence -> H.264 mp4 that plays anywhere.

    yuv420p and the even-dimension pad are what quicktime, browsers and phones
    insist on; without them the file exists and only ffmpeg will open it.
    """
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", str(png_dir / "%05d.png"),
        "-c:v", "libx264", "-preset", "slow", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-movflags", "+faststart",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default=str(EXP / "tt_traj.pt"))
    ap.add_argument("--ref", default="", help="PyTorch trajectories, optional")
    ap.add_argument("--gt", default="", help="human driver trajectories, optional")
    ap.add_argument("--tokens", default="", help="comma separated; default first --n")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default=str(EXP / "traj_plots"))
    ap.add_argument("--video", default="", help="also encode the frames to this mp4")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--ring", action="store_true",
                    help="all eight cameras around the BEV, not just the three "
                         "the model reads")
    args = ap.parse_args()

    tt = torch.load(args.traj, map_location="cpu", weights_only=False)
    ref = (torch.load(args.ref, map_location="cpu", weights_only=False)
           if args.ref else {})
    gt = (torch.load(args.gt, map_location="cpu", weights_only=False)
          if args.gt else {})
    files = cache_files()

    if args.tokens:
        tokens = [t for t in args.tokens.split(",") if t]
    elif ref:
        tokens = [t for t in ref if t in tt and t in files][: args.n]
    else:
        tokens = sorted(t for t in tt if t in files)[: args.n]
    if not tokens:
        print("  no token has both a trajectory and a cached frame", flush=True)
        return 1

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seq = pathlib.Path(tempfile.mkdtemp(prefix="traj_seq_")) if args.video else None
    n = 0
    for i, token in enumerate(tokens):
        if token not in files:
            print(f"  {token}: no cached frame, skipped", flush=True)
            continue
        trajs = {"tt": np.asarray(tt[token], dtype=np.float64)}
        if token in ref:
            trajs["ref"] = np.asarray(ref[token], dtype=np.float64)
        if token in gt:
            trajs["gt"] = np.asarray(gt[token], dtype=np.float64)
        if seq is not None:
            plot_token(token, files[token], trajs, seq / f"{n:05d}.png",
                       tight=False, caption=f"[{i + 1}/{len(tokens)}]  ",
                       ring=args.ring)
            if (n + 1) % 25 == 0:
                print(f"    rendered {n + 1}/{len(tokens)}", flush=True)
        else:
            print(f"  {plot_token(token, files[token], trajs, out_dir / f'{token}.png', ring=args.ring)}",
                  flush=True)
        n += 1

    if seq is not None:
        try:
            print(f"  {encode(seq, pathlib.Path(args.video), args.fps)}  "
                  f"({n} frames, {args.fps} fps)", flush=True)
        finally:
            shutil.rmtree(seq, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
