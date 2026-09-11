#!/usr/bin/env bash
# Full navtest PDMS evaluation of the PyTorch reference (NAVSIM v1).
# Target: 92.22 PDMS, the number SparseDriveV2's README reports for this ckpt.
#
#   source env.sh && scripts/eval_navtest_v1.sh
#
# Differences from SparseDriveV2/scripts/evaluation/run_pdm_score_navtest_v1.sh:
#   - absolute paths, so it does not depend on cwd
#   - batch_size 2, not 8. DAF[0] emits [B, 512000, 256] before the P-axis is
#     summed away: 524 MB per sample at fp32. Eight would need ~14 GiB on a
#     7.5 GiB card. Per-frame time is flat at 0.239 s from batch 1 to 4, so
#     there is nothing to win by going wider -- it is compute-bound. 2 keeps
#     peak at 3.4 GiB and still gives the dataloader 2 workers, since the
#     entrypoint hardcodes num_workers = batch_size.
set -euo pipefail

: "${SPARSEDRIVE_TT_ROOT:?source env.sh first}"
CKPT_DIR="$SPARSEDRIVE_TT_ROOT/ckpt"
export HYDRA_FULL_ERROR=1

exec "$NAVSIM_PY" "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_navtest_v1_fast.py" \
    train_test_split=navtest \
    agent=sparsedrive_agent \
    agent.checkpoint_path="$CKPT_DIR/sparsedrive_navsimv1.ckpt" \
    experiment_name=sparsedrive_navtest_v1 \
    metric_cache_path="$NAVSIM_EXP_ROOT/metric_cache_navtestv1" \
    +test_cache_path="$NAVSIM_EXP_ROOT/data_cache_navtest" \
    dataloader.params.batch_size=2 \
    +agent.config.dataset_version=v1 \
    +agent.config.metrics='["no_at_fault_collisions","drivable_area_compliance","driving_direction_compliance","time_to_collision_within_bound","comfort","ego_progress"]' \
    +agent.config.velocity_filter_num='[64,20]' \
    +agent.config.bkb_path="$CKPT_DIR/resnet34.bin" \
    +agent.config.path_anchor="$CKPT_DIR/kmeans/path_1024.npy" \
    +agent.config.velocity_anchor="$CKPT_DIR/kmeans/velocity_256.npy" \
    +agent.config.trajectory_anchor="$CKPT_DIR/kmeans/trajectory_1024_256.npz" \
    "$@"
