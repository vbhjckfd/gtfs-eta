# Model-search journal — run 17 (archived in run 20)

## 2026-09-28 — run 17 (history-table lookback: 21 / 7 days)

Lock taken at 23:16 UTC (09-27). No housekeeping due (journal 359 lines; run 15 did it, next at run 20).
Main was already merged. Fresh box: `pip install -e . tzdata`.
Tested run-16 next step 3: the lookback of the static history tables that the leader uses (hist link × hour,
weekday/weekend table, location dwell), which is set by `MS_HIST_DAYS` (default 14).

Rebuild: `pipeline_lite.py --parallel 4 --days 2026-08-17..2026-09-21` (23:17–~03:00), then
`sh research/model-search/run17.sh`:
prep 08-17..09-21 into `ms_features_v10` (14 d) → refit baseline + leader on `ds1v10f`; then, for HD in 21 and 7,
re-prep 09-07..09-21 into `ms_features_v10h$HD` with `MS_HIST_DAYS=$HD` (the prior days' `links_` / `runs_`
files are symlinked from v10, since they don't depend on the lookback) and fit the leader on tag `ds1h$HD`.
Two preps (08-30, 09-12) were OOM-killed next to the pipeline again. The driver was rerun in a retry loop, and cached days were skipped.
The ds1h21 / ds1h7 baseline rows are copies of the ds1v10f baseline: the rows are the same and it has no history cols.

### Results (ds1, seed 42, train 2.18M, test 1.41M rows; Sat 422k / Sun 404k / Mon 589k)
| arm | lookback | MAE / median / p90 / bias | Δbase | Δ vs 14 d | sa1 / sa5 / sa10 | Sat / Sun / Mon |
|---|---|---|---|---|---|---|
| baseline | — | 110.6 / 56.0 / 230.9 / −28.2 | | | 72.6 / 112.0 / 153.5 | 100.2 / 107.1 / 120.5 |
| sc_big_dwell | 14 d (ds1v10f, reproduced exactly) | 87.0 / 43.2 / 178.2 / −12.1 | −21.3% | | 60.6 / 87.0 / 122.0 | 81.4 / 85.0 / 92.4 |
| sc_big_dwell | 21 d (ds1h21) | 87.2 / 43.1 / 178.3 / −11.8 | −21.2% | +0.2% | 60.7 / 87.3 / 122.2 | 81.4 / 85.1 / 92.7 |
| sc_big_dwell | 7 d (ds1h7) | 87.8 / 43.5 / 178.8 / −14.2 | −20.6% | +0.9% | 61.6 / 87.8 / 123.0 | 82.8 / 85.8 / 92.9 |

- 21 d is noise-level worse across the board. 7 d is worse in every bucket (sa1 +1.7%) and every day, most on Saturday
  (+1.7%). A 7-day window holds only 2 weekend days for the weekday/weekend table.
- Route 122 moves the opposite way to the total (14 d 340, 21 d 336, 7 d 321). Route 133 improves with the shorter
  lookback (206 → 200 at 7 d; 208 at 21 d). Both are sparse or drifting routes, where a recent table matters more
  than a large one, but each is a single route on one split, so this is no claim.
- Worst routes (21 d): 122 336, 133 208, 106 167, 125 159, 129 132, 111 131, 881 129, 137 126, 113 125, 131 125.
  (7 d): 122 321, 133 200, 106 164, 125 160, 881 134, 137 133, 111 131, 113 130, 129 127, 131 123.

**Conclusion: no new leader. The 14-day lookback production would use for the static tables is the right choice;
21 d adds nothing and 7 d costs 0.9%. The shift split was not run (neither arm is a candidate).**

### Next steps
1. The model-side search is saturated: every family tried in runs 13–17 lands within ±0.3%, except the dwell table (−0.8%)
   and a shorter lookback (+0.9%, worse). Further hourly runs are unlikely to find a ≥3% win over sc_big_dwell. The useful
   next work is hand-off (a serving prototype of sc_big_dwell), and that is the owner's decision.
2. If the search continues without a hand-off, the only untested lever is recency weighting inside the tables (for
   example a 14-day table with the last 3 days weighted up, aimed at drifting routes such as 133 and 122). Expect ±0.3%.
