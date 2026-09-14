# Porting roadmap

SparseDriveV2 (PyTorch, NAVSIM) -> TT-NN on Wormhole, starting from
[sparse4D-tt](../../sparse4D-tt), a finished Sparse4D v3 port.

Status is kept current. Corrections are kept rather than edited away — a wrong
estimate that was measured is worth more than a clean one that was not.

| phase | | |
|---|---|---|
| 0 | environment | **done** |
| 1 | PyTorch reference fixed, golden tensors | **done** — PDMS 92.22 reproduced |
| 2 | module-by-module PCC port, bottom-up | in progress — DFA middle verified on device |
| 3 | assembly + navtest evaluation | |
| 4 | performance | |

## Why start from sparse4D-tt

SparseDriveV2 shares Sparse4D's sparse perception core.

| sparse4D-tt module | SparseDriveV2 counterpart | reuse |
|---|---|---|
| `deformable_feature_aggregation.py` | `blocks.DeformableFeatureAggregation` | high — swap `_kps_generator` box→path |
| `fpn.py` | torchvision FPN, 4 levels | high — channels only |
| `multihead_attention.py` | `p/v/t_attention`, `v_img_attention` | high |
| kernels `topk_select`, `row_gather` | path/velocity top-k + gather | exact fit |
| `asymmetric_ffn.py` | `p/v/t_ffn` | partial — simpler here |
| `resnet_bottleneck.py` | ResNet-34 = BasicBlock | partial — new block, same conv/shard patterns |
| `sparse4d_head.py` | `custom_decoder.py` | partial — rewrite, different structure |
| `instance_bank.py`, `refinement_module.py`, `sparse_box3d_encoder.py` | none | drop — no temporal, no box refinement |

Inference never enters the `self.training` branch, so the PDM scorer and every
loss stay out of the port.

## Phase 2 order

1. ResNet-34 + FPN — safest start, floor PCC 0.999
2. MHA / FFN / LayerNorm — port existing modules, only token counts differ
3. **DFA** — the body of the project
4. top-k filter + gather — reuse `topk_select` / `row_gather`
5. metric heads, score combination, argmax — trivial

## DFA on device — where it stands

Graded against `exp/golden_dfa`, the arguments and outputs the CUDA op actually
saw on a real navtest frame.

| stage | | PCC |
|---|---|---|
| kps_generator | not started | |
| project_points | not started | |
| `weights_fc` | **done** | 0.999998 |
| mask + softmax | **done** | 0.999954 |
| `grid_sample` | **done** | 0.9998 per level |
| assemble to `[clp, N, E]` | **done**, no host round-trip | |
| `grouped_weighted_sum` | **done** | 0.999822 |
| `output_proj` | not started | |

Moving the assembly onto the device took the chunk from 1272 ms to 366 ms, and
the host side from 439 ms to 12 ms, at the same accuracy.

### Settings that turned out to matter

- **The two softmax matmuls need different fidelity.** gather is bf16 x a 0/1
  matrix, so fidelity buys nothing — LoFi and HiFi4 measure identical. scatter
  takes the fp32 denominator as an operand, and fp32 matmul is emulated on
  Wormhole, so LoFi truncates what the fp32 denominator was for. Row-sum error
  1.959e-02 -> 5.554e-03 by changing that one config, at no time cost.
  Carrying the gather's conclusion over to the scatter is what caused it.
- **`packer_l1_acc` is worth 13x** on the gather (6.176e-02 -> 4.664e-03);
  `fp32_dest_acc_en` alone only gets 2.830e-01 -> 6.176e-02.
- **An fp32 output on the denominator matmul is worth another 5x** and costs
  nothing — the tensor is [n, G]. Widening the *inputs* to fp32 changes nothing.
- **bf16 grids are the whole of grid_sample's error.** Quantising only the grid
  reproduces the device numbers to ~5e-6; quantising only the features scores
  0.999999. Error grows with level width, so the 64x128 level is worst.
  `ttnn.grid_precompute`'s Q14 grid is the upgrade path if that floor bites.
- **fp32 grids make grid_sample return NaN.** It is Q14 or bf16.

### Measurement discipline

Three times this session a device time was really JIT compile: 1272 ms that was
366, 339 ms that was 3.8, 292 ms that was 2.8. Warm every variant before
timing it, and never compare a fresh config against a cached one.

Likewise, three hypotheses about the row-sum error (axis length, `ttnn.divide`,
the mask itself) were each wrong, and each took a run to disprove. Pulling out
every intermediate and diffing stage by stage found it immediately. Decompose
first.

## Risks

Ordered by severity, revised as they get measured.

### 1. DFA scale

Derived from `weights_fc` shapes, confirmed against real frames:

```
num_pts = len_path(50) x len(fix_height)(5) x num_learnable_pts(2) = 500
```

| | Sparse4D v3 | SparseDriveV2 layer-0 path | ratio |
|---|---:|---:|---:|
| cameras | 6 | 3 | 0.5x |
| anchors | 900 | 1024 | 1.1x |
| keypoints / anchor | 13 | **500** | 38x |
| bilinear lookups / frame | 281K | 6.14M | 22x |
| `weights` | 2.2M elem | 49.15M elem | 22x |
| gws input `features [clp,N,E]` | 71.9M (144 MB) | **1.57G (2.93 GiB)** | 22x |

**Correction.** The original plan said `grouped_weighted_sum` emits
`[B, A*P, E]` and that folding the P-reduction into the kernel was the first
thing to prove. That was wrong — read off
`grouped_weighted_sum_device_operation.cpp`, `features` is `[clp, N, E]` with
clp = cameras x levels x points and the output is `[N, E]`. The reduction over
points has always been fused; sparse4D-tt calls it with
`clp_per_cam = num_levels * num_pts`. The stale header comment
(`// [n, clp, embed_dims]`) is what misled the estimate.

The real cost is the **input**. 2.93 GiB fits a 12 GB chip but writing and
reading it once is ~31 ms at 200 GB/s, for one of three DFA calls, against
sparse4D-tt's 57 ms whole-frame budget. Anchors are independent through both
grid_sample and gws, so the answer is to never materialise it: walk anchors in
blocks. Measured at 32 anchors/block the buffer is 93.8 MB.

### 2. bf16 accumulator drift scales with the reduction length — *new, found in Phase 2*

Not in the original plan. The gws accumulator is bf16 and sequential, so its
error grows with the reduction length. Synthetic, N=32, E=256:

| clp | 32 | 312 | 1024 | 2048 | 4000 | 6000 |
|---|---|---|---|---|---|---|
| PCC | 0.999984 | 0.999893 | 0.999658 | 0.999338 | 0.998690 | **0.998039** |

sparse4D v3 sits at clp=312. SparseDriveV2's layer-0 path sits at 6000.
Torch's `.sum()` hides this — it reduces pairwise, and scores 0.999998 on the
same data.

**Mitigation, no kernel change.** Split clp into k pieces, sum the partials in
fp32:

| splits | slice clp | PCC | ms |
|---:|---:|---:|---:|
| 1 | 6000 | 0.998066 | 58.3 |
| 2 | 3000 | 0.999013 | 53.9 |
| 4 | 1500 | 0.999507 | 53.2 |
| 8 | 750 | **0.999746** | 56.2 |

The time was flat in that synthetic measurement because host transfers
dominated it. **Once the assembly moved onto the device and those transfers
went away, splitting cost 2.4x** — 150 ms at k=1 against 366 ms at k=8, for
PCC 0.998428 against 0.999822. So it is not free; it is 216 ms for four nines,
and k=4 or k=2 is worth revisiting in the performance phase.

The same drift sets the floor on the softmax denominator, where the reduction
is the same 6000: 9.4e-4 over a 2000-long axis, 4.2e-3 over 6000.

### 3. `weights_fc` intermediate

`[1, 1024, 3, 16000]` = 49.15M elements. Confirmed on real frames.

### 4. 6,000-wide softmax

sparse4D reduces over 312.

### 5. Split the mesh by anchor, not by camera

sparse4D-tt gave each of 2 chips 3 of 6 cameras, which is what produced the
softmax bug in its README — each device normalised over its own half. Three
cameras do not halve. Anchors do; softmax is per-anchor so no `all_reduce` is
needed, and the feature maps are only 8.4M elements (16.8 MB bf16) to
replicate.

### 6. Top-k order is not stable across backends — *new, found in Phase 1*

Layer 0 keeps the top 128 of 1024 paths and `torch.gather` reorders survivors
by rank. Between CPU and GPU the selected *set* was identical (128/128) but the
order was not: scores differing by ~6e-2 against a boundary gap of 2.8e-2
swapped neighbouring ranks. Downstream tensors were then the same rows in a
different order, reading as 0.9979 when nothing was wrong.

bf16 will widen that score gap. Compare post-filter tensors by aligned order,
and let nothing downstream depend on rank order itself.

## Corrections log

- **`DAF[2]` is 32,000 points, not 16,000.** The plan assumed the config
  default `velocity_filter_num=(64,10)`; the v1 evaluation script overrides it
  to `[64,20]`, making 20x20 = 400 queries x 80 points.
- **The CUDA extension is neither built nor needed.** `reference/daf_torch.py`
  is `grid_sample`-based and runs on either device.
- **Phase 0 grew a step the plan did not have**: this box's GPU is sm_120 and
  the stock env's torch stops at sm_86, so `navsim-sm120` had to be built
  before the reference could run at all.

## Operational notes

- Killing a TT process mid-run (a timeout, Ctrl-C) leaves the device stuck on
  an ETH heartbeat. `tt-smi -r` recovers it. Give PoCs generous timeouts.
- Python buffers stdout to a file; long device runs need `-u` or the output is
  lost when the process is killed.
