# SparseDrive-tt

> A [SparseDriveV2](https://github.com/swc-17/SparseDriveV2) implementation running on Tenstorrent devices with TT-NN.

Based on [SparseDriveV2: Scoring is All You Need for End-to-End Autonomous Driving](https://arxiv.org/abs/2603.29163) (Sun et al., 2026), with pretrained weights from [wenchaosun/SparseDriveV2](https://huggingface.co/wenchaosun/SparseDriveV2) on Hugging Face.

Starting point is [sparse4D-tt](https://github/koreaygj/sparse4D-tt), a finished Sparse4D v3 port
(57.4 ms / 17.41 FPS on N300, mAP 0.4515). SparseDriveV2 shares Sparse4D's sparse
perception core, so the backbone, FPN, attention, and deformable-aggregation work
carries over. The scale does not — see [Deformable aggregation](#deformable-aggregation-the-hard-part).

## Performance

N300, both chips, one frame split across them. Steady-state frame time; frame 0
additionally pays the kernel JIT and, under `--trace`, the capture, which is
70-80 s and is excluded below.

| | frame | rate | vs GPU |
| --- | --- | --- | --- |
| PyTorch reference, RTX 5060 Ti | 239 ms | 4.2 fps | 1.00x |
| TT-NN, 64-core compute grid | **128 ms** | **7.8 fps** | **1.87x** |
| TT-NN, 56-core compute grid | 140 ms | 7.1 fps | 1.71x |
| TT-NN, one chip only | 213 ms | 4.7 fps | 1.12x |

The 56-core grid is the stable configuration: the full 8x8 grid wedges the
device once per ~1518 frames, and 56 cores in either shape ran 28146 stress
iterations and a 12146-frame navtest with no hang. `scripts/run_until_done.sh`
finishes a run either way by resetting and resuming.

Per frame the host moves 3.04 MB in 7 transfers — six uploads (images, status,
projection) and one download (the trajectory). Everything else stays on device.

### Accuracy

navtest, 12146/12146 tokens valid, scored with `scripts/score_navtest_tt.py`.

| | PDMS |
| --- | --- |
| Published SparseDriveV2 | 0.9222 |
| PyTorch reference, this checkpoint | 0.9222134 |
| **TT-NN, 64-core grid** | **0.9220551** (-0.017%) |
| TT-NN, 56-core grid | 0.9218368 (-0.041%) |

Sub-scores for the 64-core run:

| metric | score |
| --- | --- |
| no_at_fault_collisions | 0.9867 |
| drivable_area_compliance | 0.9845 |
| ego_progress | 0.8866 |
| time_to_collision_within_bound | 0.9523 |
| comfort | 0.9998 |
| driving_direction_compliance | 0.9608 |

The gap to the reference is trajectory-vocabulary ties, not numerical drift.
Over 300 frames the TT and PyTorch paths are bit-identical on 247 tokens; where
they differ, two of the 1024 candidate trajectories score within bf16 epsilon
of each other and the argmax picks the other one. The 56-core grid shifts 9% of
tokens the same way, 536 of them up and 585 down, because a different shard
split changes the accumulation order.

## Demo Video

## Setup

```bash
source env.sh
```

Paths live in `env.yaml`, so a different machine edits that and leaves the
script alone. Relative paths resolve against the repo root and a leading `~`
expands:

```yaml
devkit_root: ../SparseDriveV2                          # the SparseDriveV2 checkout
data_root: dataset                                     # symlinks into the real dataset
exp_root: exp                                          # caches, logs, trajectories
map_version: nuplan-maps-v1.0

tt_metal_home: ~/project/tenstorrent/tt-metal          # the tt-metal carrying this port's ops
arch_name: wormhole_b0

navsim_py: ~/miniconda3/envs/navsim-sm120/bin/python
tt_py: ~/.tenstorrent-venv/bin/python
```

Flat `key: value` only — `env.sh` parses it with `sed`, not a YAML library,
because it is the file that decides which interpreter exists. An already
exported variable always wins, so a one-off override needs no edit:

```bash
TT_PY=/other/python source env.sh          # this run only
SPARSEDRIVE_ENV_YAML=~/other.yaml source env.sh
```

`source env.sh` prints every resolved path with `ok` or `MISSING`, which is the
quickest check that a new machine is wired up.

The `tt_metal_home` it points at needs this port's kernels in it. `tt-metal/`
holds them — eight ops to copy in and six patches to apply — with
[tt-metal/README.md](tt-metal/README.md) covering what each one is for.

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

| cache                    | scenes |   size |   wall |
| ------------------------ | -----: | -----: | -----: |
| `data_cache_navtest`     | 12,146 | 144 MB | 15 min |
| `metric_cache_navtestv1` | 12,146 | 3.1 GB | 15 min |

The dataset cache holds image _paths_ and calibration, not pixels; the feature
pipeline loads and normalises images at `__getitem__` time.

`exp/` is a symlink to `/mnt/data2`, which has the room.

Ray's `Failed to establish connection to the metrics exporter agent` is
telemetry noise — the run that printed it 32 times cached all 12,146 scenes.

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

| call                         |  points | `weights` |                `out` |
| ---------------------------- | ------: | --------: | -------------------: |
| `DAF[0]` layer 0, path       | 512,000 |    49.15M | **131.07M (524 MB)** |
| `DAF[1]` layer 1, path       |  64,000 |     6.14M |               16.38M |
| `DAF[2]` layer 1, trajectory |  32,000 |     3.07M |                8.19M |

`DAF[0].out` is summed over the P axis immediately on return, collapsing
131.07M elements to 262K. Folding that reduction into `grouped_weighted_sum`
is the first thing to prove in Phase 2.
