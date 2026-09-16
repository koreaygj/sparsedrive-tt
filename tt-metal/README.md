# tt-metal side of the port

Everything this port needs from tt-metal that is not in tt-metal. Two kinds:
whole ops that get dropped in, and patches against files that already exist.

Baseline commit: `3352b30745a` (tenstorrent#41631, "Route uint8 through bf16
last-dim transpose path"). The patches apply against that tree.

## ops/

Eight ops, none of them upstream. Copy each directory to
`$TT_METAL_HOME/ttnn/cpp/ttnn/operations/pool/<name>/`, then apply
`patches/02-build-register-custom-ops.patch` so cmake compiles them and
nanobind exposes them.

| op | what it is for |
| --- | --- |
| `grid_precompute` | turns Q14 fixed-point sample coordinates into the 6-field FIELD-MAJOR grid `grid_sample(use_precomputed_grid=True)` reads |
| `grid_compact` | drops the samples whose camera never sees them, and emits the permutation |
| `grouped_weighted_sum` | the deformable-aggregation reduction, `num_chunks` wide |
| `kps_project_fused` | keypoint generation and projection in one kernel |
| `row_gather` | gathers rows of two tensors by one index tensor |
| `topk_select` | top-k over a row, returning indices |
| `anchor_bucket` | buckets anchors for the compaction above |
| `transposed_s2i` | sharded-to-interleaved with the transpose folded in |

`grouped_weighted_sum` asserts that `num_chunks` divides the clp count. A
ragged division there does not fail, it hangs the device, which is how the
assert got written.

`grid_precompute` loops its kernels over tiles per core. The first version
wrote one tile and silently capped at 2048 rows.

## patches/

```bash
cd "$TT_METAL_HOME"
for p in "$SPARSEDRIVE_TT_ROOT"/tt-metal/patches/*.patch; do git apply "$p"; done
```

| patch | why |
| --- | --- |
| `01-grid_sample-precomputed-grid` | adds `use_precomputed_grid`, the Q14 int16 coordinate path, and the bounds check that makes an out-of-range sample skip a read instead of issuing one |
| `02-build-register-custom-ops` | cmake sources and the nanobind registrations for `ops/` |
| `03-conv2d-pass-dst_full_sync_en` | conv2d unpacked `dst_full_sync_en` from the compute config and dropped it, so `DstSync` was always `SyncHalf` whatever the caller asked for |
| `04-firmware-reread-go_message_index` | subordinate firmware latched `go_message_index` once outside its poll loop, unlike BRISC; equivalent to tt-metal PR #52764, which closed unmerged |
| `05-log-fabric-tensix-config` | prints the resolved `FabricTensixConfig` at fabric init. Without it there is no way to tell from the host whether tensix fabric is on |
| `06-compute-grid-56-cores` | shrinks `nebula_x2`'s compute grid from 8x8 to 8x7 |

Patches 3, 4 and 5 are upstream bugs or gaps, not choices this port made. 3 and
4 are worth submitting.

Patch 6 is the hang workaround, and it is the one to drop first if the
underlying bug gets fixed. On this N300 the full 8x8 compute grid wedges the
device once per ~1518 frames when the mesh's CCL path is in use; 56 cores in
either shape ran 28146 stress iterations and a full 12146-frame navtest with no
hang, at 140 ms a frame instead of 128. One chip at 64 cores is also clean, so
the failure needs both the full grid and the mesh. See `docs/` for the
measurements and the nine mechanisms that were ruled out.
