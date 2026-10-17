#!/bin/bash
# Launch (or resume) the D3IL re-run sweep detached from the terminal.
#
#   scripts/launch_sweep.sh [OUT] [WORKERS] [GPU_BUDGET_MIB]      defaults: results/rerun 20 20000
# (20 workers ~ 17 GB GPU / 20 cores: measured best total throughput on one RTX 4090 + 32 CPUs; use fewer on a shared GPU)
# (jobs start only while the GPU keeps >= 2.5 GB free after the new job: the GPU may be shared)
#
# Writes OUT/queue.json on the first call (groups in order: Table 2 SafeDiffuser ReS/TVS + MPD, DPCC reproduction
# check, Table 3 JM2D u-steps, Ours reproduction check; see scripts/sweep.py),
# then starts `scripts/sweep.py work` with nohup + setsid. Re-running this script resumes: finished jobs are skipped,
# unfinished jobs continue from their trials file. Monitor with `python scripts/sweep.py status --out OUT`,
# per-job logs in OUT/logs/<label>/, master log OUT/logs/sweep_master.log. Stop: `touch OUT/STOP` (graceful) or
# `kill $(cat OUT/sweep_master.pid)` (terminates running jobs; they resume on the next launch).
#
# Environment: PYTHON (default: python on PATH) must be the D3IL environment. PYTHONNOUSERSITE=1 keeps ~/.local
# packages out; EXTRA_PYTHONPATH is prepended to PYTHONPATH (e.g. a directory with missing pure-python packages).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$REPO/results/rerun}"
WORKERS="${2:-20}"
BUDGET="${3:-20000}"
PYTHON="${PYTHON:-python}"
mkdir -p "$OUT/logs"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${EXTRA_PYTHONPATH:+$EXTRA_PYTHONPATH:}${PYTHONPATH:-}"
export MPLBACKEND=Agg
if [ ! -f "$OUT/queue.json" ]; then
  "$PYTHON" "$REPO/scripts/sweep.py" plan --out "$OUT" ${COST_JSON:+--cost "$COST_JSON"}
fi
rm -f "$OUT/STOP"
cd "$REPO"
setsid nohup "$PYTHON" "$REPO/scripts/sweep.py" work --out "$OUT" --workers "$WORKERS" --gpu-budget-mib "$BUDGET" \
  >> "$OUT/logs/sweep_master.log" 2>&1 < /dev/null &
echo "sweep master pid $! (log $OUT/logs/sweep_master.log)"
