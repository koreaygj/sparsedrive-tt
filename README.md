# SparseDrive-tt

> A [SparseDriveV2](https://github.com/swc-17/SparseDriveV2) implementation running on Tenstorrent devices with TT-NN.

Based on [SparseDriveV2: Scoring is All You Need for End-to-End Autonomous Driving](https://arxiv.org/abs/2603.29163) (Sun et al., 2026), with pretrained weights from [wenchaosun/SparseDriveV2](https://huggingface.co/wenchaosun/SparseDriveV2) on Hugging Face.

Starting point is [sparse4D-tt](../../sparse4D-tt), a finished Sparse4D v3 port
(57.4 ms / 17.41 FPS on N300, mAP 0.4515). SparseDriveV2 shares Sparse4D's sparse
perception core, so the backbone, FPN, attention, and deformable-aggregation work
carries over. The scale does not — see [Deformable aggregation](#deformable-aggregation-the-hard-part).

## Status

Phases 0 (environment) and 1 (PyTorch reference) complete. Nothing runs on
device yet.

| | |
|---|---|
| Environment | `source env.sh` — 2 interpreters, both verified |
| Custom kernels | 9/9 present in the installed `ttnn` |
| Assets | backbone + 3 anchor vocabularies downloaded |
| PyTorch reference | 400/400 checkpoint tensors matched, 0.239 s/frame on GPU |
| CPU/GPU parity | worst PCC 0.999997 across 11 intermediate tensors |
| **navtest PDMS** | **0.9222 — matches the published 92.22, 12146/12146 valid** |
| Golden tensors | 8 frames x 156 tensors, plus 2 frames of DAF kernel arguments |
| TT-NN model | not started |

### Reference baseline — NAVSIM v1, navtest, `sparsedrive_navsimv1_92p2.ckpt`

| metric | value |
|---|---:|
| **score (PDMS)** | **0.9222** |
| no_at_fault_collisions | 0.9869 |
| drivable_area_compliance | 0.9844 |
| time_to_collision_within_bound | 0.9529 |
| ego_progress | 0.8865 |
| comfort | 0.9998 |
| driving_direction_compliance | 0.9611 |

12146 scenes, 0 failed, 53 min end to end. This is the number the port is
graded against. Reproduce with `scripts/eval_navtest_v1.sh`.

## Setup

```bash
source env.sh
```

It resolves its own location, so the repo can be cloned anywhere. Everything is
scoped to this file: no NAVSIM or nuplan variable is exported from `~/.zshrc`.

### Two interpreters

They cannot be merged, so `env.sh` exports both rather than picking one:

| | `$NAVSIM_PY` | `$TT_PY` |
|---|---|---|
| Path | `~/miniconda3/envs/navsim-sm120` | `~/.tenstorrent-venv` |
| Python | 3.9 | 3.12 |
| torch | 2.8.0+cu128 (sm_120) | 2.11 CPU |
| Has | navsim, nuplan-devkit | ttnn + 9 custom kernels |
| Runs | PyTorch reference, caching, PDM scoring | TT-NN model, PCC tests, device runs |

nuplan-devkit 1.2.0 pins py3.9-era dependencies, and `_ttnn.so` is a py3.12
nanobind build. **The seam between them is the feature cache**: `$NAVSIM_PY`
writes it with `scripts/cache/run_dataset_caching_navtest.sh`, `$TT_PY` reads
plain tensors and never imports nuplan.

Two traps worth writing down:

- `PYTHONPATH` must contain `$TT_METAL_HOME/ttnn`, not `$TT_METAL_HOME`. With the
  latter, `import ttnn` silently resolves to an empty namespace package — it
  succeeds, and every op is missing.
- The `sparse` env has an editable `navsim` install pointing at a different
  (April) SparseDriveV2 checkout under `~/project/turbo-plan`. `env.sh` puts our
  `$NAVSIM_DEVKIT_ROOT` ahead of it on `PYTHONPATH`, which wins because
  `PYTHONPATH` is searched before site-packages `.pth` entries.

### The GPU, and why there is no CUDA extension

This box has an RTX 5060 Ti (sm_120, 7.5 GiB). The stock `sparse` env carries
torch 2.0.1+cu117, whose fat binary stops at sm_86 — `torch.cuda.is_available()`
returns `True` and then every kernel dies with `no kernel image is available`.

`navsim-sm120` is a clone of that env with torch swapped to **2.8.0+cu128**, the
oldest cu128 build still shipping cp39 wheels, which matters because
nuplan-devkit's dependency set (`numpy==1.23.4`, `hydra-core==1.2.0`) is
py3.9-shaped. `numpy` stays at 1.23.4; navsim and nuplan import cleanly.

The deformable-aggregation CUDA extension under
`navsim/agents/sparsedrive/ops/src` is **not built and not needed**.
`reference/daf_torch.py` implements the op in pure PyTorch on top of
`grid_sample`, so it runs on either device, and `reference.install()` registers
it as a fake extension module — which must happen before anything imports
`navsim.agents.sparsedrive`, since that package imports the extension at module
scope. Semantics are read off `deformable_aggregation_cuda.cu`;
`test/test_daf_torch.py` checks them against an index-for-index transcription of
that kernel (agrees to 1.3e-15). It doubles as the executable spec for the port.

Peak VRAM for one frame is 1.78 GiB, comfortably inside 7.5.

### Top-k order is not stable across backends

Worth knowing before any of it reaches TT-NN. Layer 0 scores 1024 paths and
keeps the top 128; `torch.gather` then reorders the survivors by rank. Between
CPU and GPU the selected *set* was identical (128/128) but the *order* was not:
scores differing by ~6e-2 — ordinary fp32 reassociation — swapped neighbouring
ranks around a boundary gap of 2.8e-2. Layer 1 then saw the same rows in a
different order, and an elementwise comparison read 0.9979 when nothing was
wrong. After permutation alignment it is 1.000000.

bf16 on device will widen that score gap, not narrow it. So: compare
post-filter tensors by aligned order, and do not let anything downstream depend
on rank order per se.

### Data

`dataset/` holds symlinks into `/mnt/data2/navsim-test`, renamed the way
`navsim/planning/script/config/common/default_dataset_paths.yaml` expects:

```
dataset/navsim_logs/test   -> test_navsim_logs/test    (2.0G)
dataset/sensor_blobs/test  -> test_sensor_blobs/test   (219G)
dataset/navhard_two_stage  -> navhard_two_stage        (31G)
dataset/maps               -> maps                     (1.4G)
```

Only the `test` split is present, so evaluation and caching work; training needs
`trainval` from `SparseDriveV2/download/download_navtrain_hf.sh` linked in the
same shape.

### Assets

`ckpt/` (gitignored) is populated from Hugging Face:

```
ckpt/resnet34.bin                     timm/resnet34.a1_in1k pytorch_model.bin
ckpt/kmeans/path_1024.npy             1024 path anchors, 50 points each
ckpt/kmeans/velocity_256.npy          256 velocity profiles, 8 steps
ckpt/kmeans/trajectory_1024_256.npz   composed trajectories + mask
ckpt/sparsedrive_navsimv{1,2}.ckpt    -> ../checkpoints/
```

The vocabularies are also inside the checkpoint as buffers, but the config reads
the `.npy`/`.npz` files at construction time, so they must exist regardless.

## Caches

Both are required by the evaluation entrypoint — `CacheOnlyDataset` reads the
first, `MetricCacheLoader` the second.

```bash
source env.sh
$NAVSIM_PY $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py \
    agent=sparsedrive_agent experiment_name=cache_navtest \
    train_test_split=navtest cache_path=$NAVSIM_EXP_ROOT/data_cache_navtest

$NAVSIM_PY $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching_v1.py \
    train_test_split=navtest \
    cache.cache_path=$NAVSIM_EXP_ROOT/metric_cache_navtestv1
```

| cache | scenes | size | wall |
|---|---:|---:|---:|
| `data_cache_navtest` | 12,146 | 144 MB | 15 min |
| `metric_cache_navtestv1` | 12,146 | 3.1 GB | 15 min |

The dataset cache holds image *paths* and calibration, not pixels; the feature
pipeline loads and normalises images at `__getitem__` time.

`exp/` is a symlink to `/mnt/data2`, which has the room.

Ray's `Failed to establish connection to the metrics exporter agent` is
telemetry noise — the run that printed it 32 times cached all 12,146 scenes.

## Deformable aggregation, the hard part

Derived from `weights_fc` shapes in the checkpoint, confirmed against
`SparseDriveConfig`:

```
num_pts = len_path(50) x len(fix_height)(5) x num_learnable_pts(2) = 500
```

| | Sparse4D v3 | SparseDriveV2 (L0 path) | ratio |
|---|---:|---:|---:|
| cameras | 6 | 3 | 0.5x |
| anchors | 900 | 1024 | 1.1x |
| keypoints / anchor | 13 | **500** | 38x |
| bilinear lookups / frame | 281K | **6.14M** | 22x |
| `weights` tensor | 2.2M elem | **49.2M elem** | 22x |
| sampled features, unfused | 71.9M elem | **1.57G elem (3.1 GB)** | 22x |
| softmax axis (C x L x P) | 312 | **6,000** | 19x |

`grouped_weighted_sum` currently emits `(B, A*P, E)`; SparseDrive folds `P` away
immediately afterwards. Folding that reduction into the kernel takes the output
from 131M elements to 262K. This is the first thing to prove, before anything
else is built.

One design change from sparse4D-tt: **split the mesh by anchor, not by camera.**
sparse4D-tt gave each of 2 chips 3 of 6 cameras, and that is what produced the
softmax bug documented in its README (each device normalised over its own half).
Three cameras do not halve. Anchors do, softmax is per-anchor so no `all_reduce`
is needed, and the feature maps are only 8.4M elements (16.8 MB bf16) to
replicate.

## Layout

```
env.sh                  environment, both interpreters
reference/daf_torch.py    pure-PyTorch deformable aggregation (CPU reference)
reference/__init__.py   install() -- fake extension modules, call before navsim imports
reference/pcc.py        PCC helper, the accuracy gate used throughout
shim/sitecustomize.py   installs the DAF stand-in into ray workers too
tools/dump_golden.py    golden tensor dump from real navtest frames
scripts/eval_navtest_v1.sh  full navtest PDMS run of the reference
test/test_daf_torch.py  daf_torch vs transcription of the CUDA kernel
test/test_forward_smoke.py  full model + real weights, one forward (--device, --parity)
test/test_gpu_parity.py     cpu vs cuda on intermediate tensors, permutation-aware
ckpt/                   gitignored, see Assets
dataset/                gitignored, symlinks, see Data
exp/                    gitignored, caches and run output
```

## Tests

```bash
source env.sh
$NAVSIM_PY test/test_daf_torch.py                     # op vs CUDA-kernel transcription
$NAVSIM_PY test/test_forward_smoke.py --device cuda   # full model, real weights
$NAVSIM_PY test/test_forward_smoke.py --parity        # cpu vs cuda, final output
$NAVSIM_PY test/test_real_frame.py                    # real navtest frames end to end
cd test && $NAVSIM_PY test_gpu_parity.py              # cpu vs cuda, intermediates
```

## Golden tensors

```bash
source env.sh
$NAVSIM_PY tools/dump_golden.py --frames 8                      # -> exp/golden
$NAVSIM_PY tools/dump_golden.py --frames 2 --with-dfa-internals # -> DAF kernel args
```

Each frame is a flat `{name: cpu fp32 tensor}` dict holding both the **input
and the output** of every module on the porting list, so a TT-NN module can be
graded standalone without replaying everything upstream of it. Real navtest
frames, not synthetic — anchors projecting out of frame and softmax rows with
no in-bounds camera only occur on real geometry.

`--with-dfa-internals` additionally captures the arguments of all three
deformable-aggregation calls, measured here:

| call | points | `weights` | `out` |
|---|---:|---:|---:|
| `DAF[0]` layer 0, path | 512,000 | 49.15M | **131.07M (524 MB)** |
| `DAF[1]` layer 1, path | 64,000 | 6.14M | 16.38M |
| `DAF[2]` layer 1, trajectory | 32,000 | 3.07M | 8.19M |

`DAF[0].out` is summed over the P axis immediately on return, collapsing
131.07M elements to 262K. Folding that reduction into `grouped_weighted_sum`
is the first thing to prove in Phase 2.
