#!/usr/bin/env bash
# Run navtest to completion across device hangs.
#
# The dispatch pipeline on this N300 wedges every few thousand frames: chip 1
# is reachable only over ethernet, so cq_prefetch and cq_dispatch both live on
# erisc cores and a stall there takes the whole queue down with it. It is a
# tt-metal bug (#52650, #54541), it survives --trace, and nothing in this model
# can avoid it. What the model can do is not lose the frames already scored.
#
# run_navtest_tt.py saves what it has on any failure and exits hard so the PCIe
# lock is released; --resume skips the tokens already in the output file. This
# loop ties the two together: reset the boards, resume, repeat until the run
# reports no tokens left. An attempt that scores nothing new stops the loop
# rather than spinning on a device that is genuinely broken.
#
# The 45 second wait after each reset is not padding. tt-smi -r returns before
# the chips are ready, and opening the device too early fails as
# "SIGBUS: Non-existant physical address" or "Query mappings failed on device 0".
#
# usage: scripts/run_until_done.sh exp/tt_traj_trace.pt [extra args...]
set -u
OUT=${1:?usage: run_until_done.sh OUT [args...]}; shift
cd "$(dirname "$0")/.."
# env.sh reads TT_METAL_HOME before it sets it, which set -u calls fatal.
set +u
source ./env.sh >/dev/null
set -u
LOG=${OUT%.pt}.log
TT_SMI=$(dirname "$TT_PY")/tt-smi

prev=-1
for attempt in $(seq 1 40); do
    echo "=== attempt $attempt  $(date +%H:%M:%S) ===" | tee -a "$LOG"
    $TT_PY -u scripts/run_navtest_tt.py --resume --out "$OUT" "$@" 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    have=$($TT_PY -c "import torch,sys
try: print(len(torch.load(sys.argv[1],map_location='cpu',weights_only=False)))
except Exception: print(0)" "$OUT")
    echo "=== attempt $attempt rc=$rc  $have tokens scored ===" | tee -a "$LOG"
    [ "$rc" = 0 ] && break
    if [ "$have" = "$prev" ]; then
        echo "no progress this attempt; stopping" | tee -a "$LOG"
        exit 1
    fi
    prev=$have
    "$TT_SMI" -r 2>&1 | tail -2 | tee -a "$LOG"
    sleep 45
done
echo "done: $have tokens -> $OUT" | tee -a "$LOG"
