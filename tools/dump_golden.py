"""Dump PyTorch reference tensors for the TT-NN port to check against.

Every module ported to TT-NN gets graded by PCC against the tensors this
writes. They come from real navtest frames, not synthetic input, because the
things that break a port -- anchors projecting out of frame, a level whose
feature map is mostly zeros, a softmax row with no in-bounds camera -- only
appear on real geometry.

    $NAVSIM_PY tools/dump_golden.py --frames 8
    $NAVSIM_PY tools/dump_golden.py --frames 2 --with-dfa-internals

Output: exp/golden/<token>.pt, each a flat {name: cpu fp32 tensor} dict, plus
manifest.json listing tokens, shapes and config. Module *inputs* are saved
alongside outputs so a module can be tested standalone without replaying
everything upstream of it.
"""

import argparse
import json
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
from test_real_frame import SENSOR_CONFIG, FILTER_YAML, batchify, load_model  # noqa: E402

from navsim.agents.sparsedrive.sparsedrive_features import SparseDriveFeatureBuilder  # noqa: E402
from navsim.common.dataloader import SceneLoader  # noqa: E402

DATA = pathlib.Path(os.environ["OPENSCENE_DATA_ROOT"])
OUT = pathlib.Path(os.environ["NAVSIM_EXP_ROOT"]) / "golden"

# Ported bottom-up, so the dump is ordered that way too. Each name is an
# nn.Module path; inputs and outputs of each are saved.
MODULES = [
    "_backbone.img_backbone",
    "_backbone.img_neck",
    "_backbone",
    "_status_encoding",
    "_trajectory_head.path_pos_embed",
    "_trajectory_head.vel_pos_embed",
]
for L in (0, 1):
    MODULES += [
        f"_trajectory_head.decoder.layers.{L}.p_deform_model.kps_generator",
        f"_trajectory_head.decoder.layers.{L}.p_deform_model.camera_encoder",
        f"_trajectory_head.decoder.layers.{L}.p_deform_model.weights_fc",
        f"_trajectory_head.decoder.layers.{L}.p_deform_model.output_proj",
        f"_trajectory_head.decoder.layers.{L}.p_deform_model",
        f"_trajectory_head.decoder.layers.{L}.p_attention",
        f"_trajectory_head.decoder.layers.{L}.p_ffn",
        f"_trajectory_head.decoder.layers.{L}.path_mlp",
        f"_trajectory_head.decoder.layers.{L}.v_img_attention",
        f"_trajectory_head.decoder.layers.{L}.v_attention",
        f"_trajectory_head.decoder.layers.{L}.v_ffn",
        f"_trajectory_head.decoder.layers.{L}.vel_mlp",
    ]
MODULES += [
    "_trajectory_head.decoder.layers.1.t_deform_model",
    "_trajectory_head.decoder.layers.1.t_attention",
    "_trajectory_head.decoder.layers.1.t_ffn",
    "_trajectory_head.decoder.layers.1.traj_mlp",
]


def _store(bag, key, value):
    """Flatten whatever a module handed us into named cpu fp32 tensors."""
    if torch.is_tensor(value):
        bag[key] = value.detach().float().cpu()
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _store(bag, f"{key}[{i}]", v)
    elif isinstance(value, dict):
        for k, v in value.items():
            _store(bag, f"{key}.{k}", v)


def attach(model, bag):
    handles = []
    named = dict(model.named_modules())
    for name in MODULES:
        if name not in named:
            print(f"  ! no module {name}")
            continue

        def mk(n):
            def fn(_m, inp, out):
                _store(bag, f"{n}.in", inp)
                _store(bag, f"{n}.out", out)
            return fn

        handles.append(named[name].register_forward_hook(mk(name)))
    return handles


def wrap_daf(bag):
    """Capture the deformable-aggregation call arguments.

    These are the tensors the custom kernel will consume, and the ones that
    decide whether the port is feasible: `weights` alone is 49.2M elements on
    the layer-0 path branch. Off by default for that reason.
    """
    from navsim.agents.sparsedrive import blocks
    original = blocks.DAF
    counter = {"n": 0}

    def wrapped(mc_ms_feat, spatial_shape, scale_start_index,
                sampling_location, weights, *a, **k):
        i = counter["n"]
        counter["n"] += 1
        for nm, t in (("mc_ms_feat", mc_ms_feat), ("spatial_shape", spatial_shape),
                      ("scale_start_index", scale_start_index),
                      ("sampling_location", sampling_location), ("weights", weights)):
            _store(bag, f"DAF[{i}].{nm}", t)
        out = original(mc_ms_feat, spatial_shape, scale_start_index,
                       sampling_location, weights, *a, **k)
        _store(bag, f"DAF[{i}].out", out)
        return out

    blocks.DAF = wrapped
    return lambda: setattr(blocks, "DAF", original)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--with-dfa-internals", action="store_true",
                    help="also save DAF call arguments (~800 MB per frame)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    manifest = {
        "tokens": [],
        "device": args.device,
        "torch": torch.__version__,
        "checkpoint": CKPT.name,
        "dfa_internals": args.with_dfa_internals,
        "config": {
            "cams": list(cfg.cams), "final_dim": list(cfg.final_dim),
            "num_levels": cfg.num_levels, "mode_path": cfg.mode_path,
            "mode_vel": cfg.mode_vel, "len_path": cfg.len_path,
            "path_filter_num": list(cfg.path_filter_num),
            "velocity_filter_num": list(cfg.velocity_filter_num),
            "num_learnable_pts": cfg.num_learnable_pts,
            "fix_height": list(cfg.fix_height),
        },
        "shapes": {},
    }

    for token in tokens:
        bag = {}
        restore = wrap_daf(bag) if args.with_dfa_internals else None
        handles = attach(model, bag)

        agent_input = loader.get_agent_input_from_token(token)
        feats = builder.compute_features(agent_input)
        feats, _, _ = builder.pipeline(feats, {}, token, test_mode=True)
        batch = batchify(feats, args.device)

        t0 = time.time()
        with torch.no_grad():
            out, _ = model(batch, {})
        if args.device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0

        for h in handles:
            h.remove()
        if restore:
            restore()

        _store(bag, "INPUT.imgs", batch["camera_feature"]["imgs"])
        _store(bag, "INPUT.projection_mat", batch["camera_feature"]["projection_mat"])
        _store(bag, "INPUT.image_wh", batch["camera_feature"]["image_wh"])
        _store(bag, "INPUT.status_feature", batch["status_feature"])
        _store(bag, "OUTPUT.trajectory", out["trajectory"])

        path = out_dir / f"{token}.pt"
        torch.save(bag, path)
        mb = path.stat().st_size / 2**20
        print(f"  {token}  {len(bag):3d} tensors  {mb:8.1f} MB  fwd {dt:.2f}s")

        manifest["tokens"].append(token)
        if not manifest["shapes"]:
            manifest["shapes"] = {k: list(v.shape) for k, v in sorted(bag.items())}

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    total = sum(p.stat().st_size for p in out_dir.glob("*.pt")) / 2**30
    print(f"  wrote {len(tokens)} frames to {out_dir}  ({total:.2f} GiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
