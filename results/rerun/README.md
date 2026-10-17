# Re-runs (Table 2 SafeDiffuser / MPD rows, Table 3, DPCC / Ours checks)

Output of the sweep (`scripts/launch_sweep.sh`, job queue `scripts/sweep.py`), one directory per label:

| Group | Labels | Purpose |
|---|---|---|
| `table2` | `safediffuser-res`, `safediffuser-tvs`, `mpd` | Table 2 rows without stored per-seed results |
| `table2_dpcc_repro` | `dpcc` | reproduction check of the stored Table 2 DPCC row |
| `table3` | `jm2d-u0`, `-u1`, `-u2`, `-u5`, `-u10` | Table 3 |
| `table2_ours_repro` | `jm2d` (u = full) | reproduction check of the stored Table 2 Ours row |

Table 3 and the DPCC / Ours checks use 5 seeds × 20 trials per scenario (`queue.json`, written by
`scripts/sweep.py plan --groups table2_dpcc_repro table3 table2_ours_repro --n-trials 20`); the full protocol, 5 seeds ×
100 trials for every group, is `queue_full_n100.json`. The SafeDiffuser / MPD `seed<k>.trials.jsonl` files are tracked
because the Table 2 SafeDiffuser / MPD rows are built from them (`--include-partial`).

Each job writes `<label>/<scenario>/seed<k>.json` (summary: per-trial arrays, settings, checkpoint md5, versions) and
`seed<k>.trials.jsonl` (one line per trial). Tables, from the repository root (= `results/rerun_tables.md` / `.json`):

```bash
python scripts/summarize.py --results results/rerun --figure table2 table3 --check-paper \
    --include-partial --min-partial-trials 10 --out results/rerun_tables.md --json results/rerun_tables.json
```
