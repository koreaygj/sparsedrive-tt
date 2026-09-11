#!/usr/bin/env bash
# SparseDrive-tt environment.
#
#   source env.sh
#
# Scoped to this repo only: nothing is exported from ~/.zshrc, so every shell
# that wants to run NAVSIM or TT-NN code must source this file first.
# Works under bash and zsh.
#
# TWO INTERPRETERS. They cannot be merged:
#   $NAVSIM_PY  py3.9  torch 2.8.0+cu128  navsim + nuplan-devkit, sm_120 CUDA
#               -> PyTorch reference, dataset/metric caching, PDM scoring
#   $TT_PY      py3.12 torch 2.11 CPU     ttnn + 9 custom kernels
#               -> TT-NN model, PCC tests, on-device runs
# nuplan-devkit 1.2.0 pins py3.9-era deps and ttnn's _ttnn.so is a py3.12
# nanobind build, so one env cannot hold both. The seam is the feature cache:
# $NAVSIM_PY writes it, $TT_PY reads tensors only and never imports nuplan.

# --- resolve this repo's root, wherever it was cloned -------------------------
_sd_src="${BASH_SOURCE[0]:-$0}"
export SPARSEDRIVE_TT_ROOT="$(cd "$(dirname "$_sd_src")" && pwd)"
unset _sd_src

# --- NAVSIM devkit ------------------------------------------------------------
# The upstream SparseDriveV2 checkout, used by scripts/**/*.sh as
# "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/...". Set unconditionally: a stale
# value inherited from another project's shell must not win. To point somewhere
# else, set SPARSEDRIVE_DEVKIT_ROOT before sourcing.
export NAVSIM_DEVKIT_ROOT="${SPARSEDRIVE_DEVKIT_ROOT:-$(dirname "$SPARSEDRIVE_TT_ROOT")/SparseDriveV2}"

# --- data ---------------------------------------------------------------------
# $SPARSEDRIVE_TT_ROOT/dataset holds symlinks into the real dataset on disk,
# named the way navsim/planning/script/config/common/default_dataset_paths.yaml
# expects:
#   dataset/navsim_logs/<split>       <- test_navsim_logs/<split>
#   dataset/sensor_blobs/<split>      <- test_sensor_blobs/<split>
#   dataset/navhard_two_stage
#   dataset/maps
export OPENSCENE_DATA_ROOT="$SPARSEDRIVE_TT_ROOT/dataset"
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"

# --- experiment output (caches, logs, checkpoints written by runs) -------------
export NAVSIM_EXP_ROOT="$SPARSEDRIVE_TT_ROOT/exp"
mkdir -p "$NAVSIM_EXP_ROOT"

# --- tenstorrent --------------------------------------------------------------
export TT_METAL_HOME="${TT_METAL_HOME_OVERRIDE:-$HOME/project/tenstorrent/tt-metal}"
export ARCH_NAME="${ARCH_NAME:-wormhole_b0}"

# --- interpreters -------------------------------------------------------------
# navsim-sm120 = a clone of the `sparse` env with torch swapped to 2.8.0+cu128,
# the oldest cu128 build that still ships cp39 wheels. The original `sparse`
# env carries torch 2.0.1+cu117, whose fat binary stops at sm_86, so every CUDA
# kernel died on this box's RTX 5060 Ti (sm_120). numpy stays pinned at 1.23.4
# as navsim requires. Point NAVSIM_PY elsewhere to override.
export NAVSIM_PY="${NAVSIM_PY:-$HOME/miniconda3/envs/navsim-sm120/bin/python}"
export TT_PY="${TT_PY:-$HOME/.tenstorrent-venv/bin/python}"

# --- python path --------------------------------------------------------------
# Our SparseDriveV2 must precede the `sparse` env's editable navsim install,
# which points at a different (April) checkout under ~/project/turbo-plan.
# PYTHONPATH is searched before site-packages .pth entries, so this wins.
# ttnn lives under $TT_METAL_HOME/ttnn (NOT $TT_METAL_HOME -- that path makes
# `import ttnn` resolve to an empty namespace package that silently has no ops).
_sd_pp="$NAVSIM_DEVKIT_ROOT:$TT_METAL_HOME/ttnn:$TT_METAL_HOME:$TT_METAL_HOME/tools"
case ":${PYTHONPATH}:" in
  *":$NAVSIM_DEVKIT_ROOT:"*) ;;
  *) export PYTHONPATH="$_sd_pp${PYTHONPATH:+:$PYTHONPATH}" ;;
esac
unset _sd_pp

# --- report -------------------------------------------------------------------
echo "SparseDrive-tt env:"
for _sd_v in SPARSEDRIVE_TT_ROOT NAVSIM_DEVKIT_ROOT OPENSCENE_DATA_ROOT NUPLAN_MAPS_ROOT \
             NAVSIM_EXP_ROOT TT_METAL_HOME NAVSIM_PY TT_PY; do
  eval "_sd_p=\$$_sd_v"
  if [ -e "$_sd_p" ]; then _sd_mark="  ok"; else _sd_mark="  MISSING"; fi
  printf '  %-20s %s%s\n' "$_sd_v" "$_sd_p" "$_sd_mark"
done
unset _sd_v _sd_p _sd_mark
