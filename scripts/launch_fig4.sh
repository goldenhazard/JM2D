#!/bin/bash
# Launch (or resume) the re-run of the Fig. 4 DP/RAIL curves with the released code, detached from the terminal.
#
#   scripts/launch_fig4.sh [OUT] [WORKERS] [GPU_BUDGET_MIB]      defaults: results/fig4_rerun 4 7900
#
# Writes OUT/queue.json on the first call (60 jobs = {rail, dp} x 5 checkpoints x 6 pads, 100 env seeds each,
# longest first; see scripts/sweep_fig4.py), then starts `scripts/sweep_fig4.py work` with setsid nohup.
# Re-running this script resumes: finished jobs are skipped, an interrupted job continues after its last finished
# seed. Monitor: `python scripts/sweep_fig4.py status --out OUT`; per-job logs OUT/logs/<planner>_<ckpt>_pad<www>.log,
# master log OUT/logs/sweep_master.log. Stop: `touch OUT/STOP` (graceful: running jobs finish) or
# `kill $(cat OUT/sweep_master.pid)` (terminates running jobs; they resume on the next launch).
#
# Environment: PYTHON (default: python on PATH) must be the jm2d_maze environment (README "Installation");
# checkpoints in checkpoints/ (scripts/download_checkpoints.py --all). One thread per job process (THREADS).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$REPO/results/fig4_rerun}"
WORKERS="${2:-4}"
BUDGET="${3:-7900}"
THREADS="${THREADS:-1}"
PYTHON="${PYTHON:-python}"
mkdir -p "$OUT/logs"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 MPLBACKEND=Agg
if [ ! -f "$OUT/queue.json" ]; then
  "$PYTHON" "$REPO/scripts/sweep_fig4.py" plan --out "$OUT"
fi
rm -f "$OUT/STOP"
cd "$REPO"
setsid nohup "$PYTHON" "$REPO/scripts/sweep_fig4.py" work --out "$OUT" --workers "$WORKERS" \
  --gpu-budget-mib "$BUDGET" --threads "$THREADS" >> "$OUT/logs/sweep_master.log" 2>&1 < /dev/null &
echo "sweep master pid $! (log $OUT/logs/sweep_master.log)"
