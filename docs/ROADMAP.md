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

1. ResNet-34 + FPN — **done**, 0.9993-0.9998 across the four FPN levels
2. MHA / FFN / LayerNorm — **done**, 0.99991-0.999998
3. **DFA** — the body of the project
4. top-k filter + gather — **done**, 0.999999 / 0.999998 after order alignment
5. metric heads, score combination, argmax — **done**, 0.99988-0.99999

## DFA on device — where it stands

Graded against `exp/golden_dfa`, the arguments and outputs the CUDA op actually
saw on a real navtest frame.

| stage | | PCC |
|---|---|---|
| kps_generator | **done**, fused with projection | 1.000000 |
| project_points | **done** | 1.000000 on visible points |
| `weights_fc` | **done** | 0.999998 |
| mask + softmax | **done** | 0.999954 |
| `grid_sample` | **done** | 0.9998 per level |
| assemble to `[clp, N, E]` | **done**, no host round-trip | |
| `grouped_weighted_sum` | **done** | 0.999822 |
| `output_proj` | **done** | |
| **whole module, end to end** | **done** | **0.999869** |

Lifted out of the PoC into `model/dfa.py`, parameterised by `num_sample`, and
checked against all three calls a frame makes rather than just the first:

| call | anchors | num_pts | clp | | PCC |
|---|---:|---:|---:|---:|---:|
| layer 0 path | 1024 | 500 | 6000 | 1113.5 ms | 0.999868 |
| layer 1 path | 128 | 500 | 6000 | 141.5 ms | 0.999549 |
| layer 1 traj | 400 | 80 | **960** | 97.7 ms | 0.999988 |

The trajectory branch had never been run before this -- clp 960 rather than
6000 -- and the compact layout's split constraint holds there too.

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
- **Coordinates want fp32; features do not.** The keypoint/projection chain in
  bf16 costs 4.22 px of p99 error and flips the visibility of 0.42% of visible
  points; in fp32 those are 0.33 px and 0.03%, and it runs no slower. A path
  point reaches 50 m, where bf16 resolves to 0.2 m. The features quantise to
  bf16 for 0.999999. Same question, opposite answers — decide it per tensor,
  by dynamic range against required precision, not by a blanket default.
- **kps_generator needs no custom kernel.** sparse4D-tt wrote
  `kps_project_fused` in Metalium for box keypoints; the path case is affine in
  xy, so z is a per-h constant and the homogeneous 4th is 1, which lets stock
  ops do the same fusion. Nothing ever materialises [n, 500, 3] — everything
  stays [n, 500], which tiles without wasting 29 of every 32 columns.

### COMPACT weights constrain the clp split

gws takes weights either as 3D `[clp, N, G]` or COMPACT `[N, clp*G]`, and the
compact form needs `clp*G` to be a multiple of the tile width. With G = 8 that
means the clp slice must be divisible by 4, so clp = 6000 accepts k = 10 (600)
and rejects k = 8 (750). The 3D layout has no such rule, which is why the
earlier gws PoC ran k = 8 happily.

Worth paying: the softmax already emits the compact layout, so nothing is
rearranged between softmax and gws, and compact moves a quarter of the traffic
the 3D form does. `poc_dfa_full.py` validates k up front and names the legal
values rather than letting TT_FATAL surface from inside the kernel.

### head_dim 32 is the one free alignment in this model

Attention splits 256 channels across 8 heads, so head_dim is 32 -- exactly the
tile width. The head split is a reshape with no padding.

That is the exception. Everywhere else the natural layout fights the tile grid:
G = 8 in the DFA wastes 24 of every 32 columns in the [N, CLP, G] form (which
is why the compact layout exists), and a keypoint's last dimension of 3 would
waste 29, which is why the projection is fused to keep everything [n, 500].

Attention also scores lower than the FFNs beside it -- 0.99991-0.99995 against
0.999997 -- for the reason every softmax in this port does: a reduction whose
length sets the accumulator drift. At 1024 and 384 tokens it is mild; the DFA's
6000-wide one was not.

### Dumping module inputs *and* outputs pays for itself

The six decoder LayerNorms were never hooked in `dump_golden.py`, and did not
need a re-dump. Every one sits between two modules that were hooked, so its
input is a sum of tensors already on disk:

    p_norm1( p_deform_model.out + p_attention.out[0] )  ==  p_ffn.in[0]
    p_norm2( p_ffn.in[0]        + p_ffn.out         )  ==  path_mlp.in[0]

That is a better check than the one a re-dump would have given: it grades the
residual add along with the norm against a real boundary value, rather than
grading `ttnn.layer_norm` against torch on synthetic input. Dropout is identity
at eval, so it falls out of the identity.

The same property caught a wrong reference earlier -- `row_gather` takes two
sources, and the one that stayed at 0.999998 while the other sat at 0.33 said
the gather was fine. Prefer references with a built-in control.

### Conv is a layout problem, not an arithmetic one

Nothing in the backbone needed a precision decision. Sixteen BasicBlocks and
eight FPN convs run at 0.9993-0.9998 with bf16 weights and activations
throughout, and BatchNorm folds into the preceding conv exactly (36 ops gone,
no accuracy cost). Every obstacle was memory layout or allocation:

| symptom | cause | fix |
|---|---|---|
| `L1_SMALL buffer ... bank size is 0 B` | `open_device` defaults `l1_small_size=0`; conv2d allocates there | pass 24576 |
| `Conv2d supports Height/Block/Width ... got INTERLEAVED` | `shard_layout=None` copied from sparse4D-tt | HEIGHT_SHARDED |
| `Tiled input must be tile-aligned` | a sharded TILE conv output fed to `ttnn.upsample` | via ROW_MAJOR in DRAM |
| circular buffers 1.9 MB at ResNet layer4 | 8x16 spatial, 512 channels: few rows per core carrying every channel's weights | BLOCK_SHARDED for that stage |
| circular buffers 1.7 MB at FPN layer0 | **L1 slicing**, see below | `slice_l1=False` |

**`Conv2dL1FullSliceConfig` is not a safety net.** It reads like one -- slice
if it does not fit -- and sparse4D-tt leaves it on, but on FPN layer0 (3x3,
256->256, 3x64x128 rows) enabling it is what overflows L1. Measured over 24
combinations, `slice_l1=False` passes all of them and is the only variable that
decides; block sharding and `act_block_h_override` only matter while slicing is
on. Sharding alone never fixes that conv -- it fails under Height, Block and
Width equally.

Two attempts went in before that was known (block sharding everywhere, then
capping the activation block), and both left the error size unchanged to the
byte. An unchanged error size is the signal that the knob being turned is not
attached to anything: stop and decompose. An 8x3 conv-by-sharding matrix found
the culprit in one run, and a 24-point sweep of that one conv found the knob.

### A golden tensor at a module boundary includes what happened between modules

Scoring the top-k gather took three tries, and all three were the reference,
not the kernel:

    0.326  gathered p_ffn.out -- the FFN branch, before its residual and norm
    0.881  gathered p_norm2's output, right, but scored against layer 1's DFA
           input, which is that plus the ego-status encoding layer 1 adds
    0.999999  status subtracted back out

The tell was there the whole time: `row_gather` takes two sources, and the
vocabulary gathered by the same indices in the same call held 0.999998 while
the embedding did not. Two tensors gathered by one index list cannot disagree
because of the gather. Prefer a reference with a built-in control like that.

PCC has a shape worth reading, too. Wrong tensor entirely reads 0.3; right
tensor off by one constant row reads 0.88; right tensor reads four nines. 0.88
means structure is right and an offset is missing, not that precision is poor.

### Measurement discipline

Four times this session a device time was really JIT compile: 1272 ms that was
366, 339 ms that was 3.8, 292 ms that was 2.8, and 5613 ms that was 97.7.

**Warm at the shape you will time, not the function.** The last one warmed the
same DFA call with chunk=32 and timed it at chunk=128; a different chunk is a
different tensor shape is a different kernel, so the "warm" run compiled
nothing that the timed run used. It only got caught because the answer was
physically impossible -- the trajectory branch does a sixteenth of layer 0's
work and read 1.45x its time. A plausible wrong number would have survived.

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

### 6. The final argmax flips on near-ties — *new, found in Phase 2*

The sharpest form of risk 7 below, and the one that reaches the output. Five
metric heads score 400 trajectory candidates and argmax picks the one the model
emits. Measured across the 8 golden frames:

| frame | argmax | 1-2 gap / range |
|---|---|---|
| 05d0a1a7 | 3 = 3 | 3.10e-05 |
| 2930485d | 1 = 1 | 2.56e-03 |
| 482989b8 | 44 = 44 | 7.95e-05 |
| 9d626ce2 | 2 = 2 | 1.18e-03 |
| **a6c24c9c** | **180 vs 20** | **2.25e-06** |
| bca43253 | 1 = 1 | 2.39e-04 |
| e5dc48dd | 12 = 12 | 2.10e-03 |
| f68aaf95 | 52 = 52 | 1.28e-04 |

7 of 8 agree, and the one that does not is exactly the frame with the smallest
margin. Its head PCC is 0.999916 — unremarkable. It did not flip because the
arithmetic was poor; it flipped because there was no gap to hold.

This is the model, not the port: 400 candidates drawn from one vocabulary score
very close together, median relative margin 1.28e-04. fp32 on different
hardware would flip too, the same way Phase 1's top-k ranking differed between
CPU and GPU.

A flip is also not obviously a wrong answer — candidates 180 and 20 are
equivalent by the model's own metric. What it costs is a PDMS question, and
**Phase 3 should expect end-to-end PDMS to differ from PyTorch for this reason
rather than from accumulated error.** Measure it before explaining it.

### 7. Top-k order is not stable across backends — *new, found in Phase 1*

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
