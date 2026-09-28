# Model-search journal — runs 17–18 (archived in run 20)

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

## 2026-09-28 — run 18 (housekeeping; monotone constraints, tree capacity)

Lock taken at 04:17 UTC. Housekeeping was due (journal 401 lines), commit "model-search: housekeeping": runs 13–16
moved to `archive/JOURNAL-runs-13-16.md`; LEADERBOARD.md shows the current tags only (old tag definitions now in
`archive/LEADERBOARD-old.md`, written by report.py); run13/13b/14/14b/15b.sh deleted; arms that lost on both splits
removed (sc_big_net, sc_big_v9, sc_big_lap6, sc_big_v8, sc_lap_127, sc_big_nxt, path_own_s_drop(_cold)), plus the
lap6_* and V9 cols in features_live.py. The leader's cols are untouched, and it reproduced exactly afterwards.

Rebuild: `pipeline_lite.py --parallel 4 --days 2026-08-31..2026-09-21` (04:18–~06:40), then `sh research/model-search/run18.sh`
(prep chain into `ms_features_v10`, tag `v18`; the 09-06 prep was OOM-killed next to the pipeline, and the retry
wrapper resumed with cached days skipped). Follow-up fits with
`ARMS_A=... ARMS_B=... PREP_DIR= FEAT_DIR=ms_features_v10 RUN=18 TAG=v18 sh research/model-search/run.sh`.

Arms (arms.py, no new cols; all = sc_big_dwell's features):
- `sc_big_mono`: `monotonic_cst` +1 on remaining_dist_m, stops_ahead, speed_eta_warm, hist_travel_time_est
  (the never-missing ones; the −1 sentinel rules out the live/hist path cols).
- `sc_big_511`: 511 leaves, min_samples_leaf 100. `sc_big_it2k`: max_iter 2400. `sc_big_511_it2k`: both.

### Results (MAE / median / p90 / bias; Δ vs the leader sc_big_dwell; rows train 2.18M / 1.98M, test 1.41M / 1.42M)
| arm | ds1v18 (s42) | ΔL | sa1 / sa5 / sa10 | ds1shiftv18 (s7) | ΔL | sa1 / sa5 / sa10 | iters, fit |
|---|---|---|---|---|---|---|---|
| baseline | 110.6 / 56.0 / 230.9 / −28.2 | | 72.6 / 112.0 / 153.5 | 113.0 / 59.5 / 241.1 / −14.2 | | 71.2 / 114.3 / 159.7 | 1200, 7 min |
| sc_big_dwell | 87.0 / 43.1 / 178.2 / −12.6 | | 60.7 / 87.1 / 121.6 | 87.0 / 43.9 / 179.5 / −14.6 | | 59.7 / 87.1 / 122.3 | 1200, 12 min |
| sc_big_mono | 87.2 / 43.2 / 178.1 / −13.2 | +0.24% | 61.0 / 87.1 / 121.7 | — | | | 1200, 12 min |
| sc_big_511 | 86.7 / 43.1 / 177.4 / −12.7 | −0.28% | 60.3 / 86.8 / 121.4 | 86.5 / 43.8 / 178.7 / −14.4 | −0.51% | 59.1 / 86.7 / 122.0 | 1200, 14 min |
| sc_big_it2k | 86.7 / 43.0 / 177.9 / −11.9 | −0.32% | 60.3 / 86.8 / 121.3 | 86.6 / 43.9 / 179.2 / −13.9 | −0.37% | 59.3 / 86.8 / 121.9 | 2400, 21 min |
| sc_big_511_it2k | 86.5 / 43.1 / 177.3 / −11.6 | −0.51% | 59.9 / 86.6 / 121.2 | 86.4 / 43.8 / 178.8 / −13.8 | −0.62% | 59.0 / 86.6 / 121.8 | 2400, 25 min |

- Baseline and leader reproduced exactly (110.6 / 113.0; 87.0 / 87.0).
- Monotone constraints lose a little (sa1 +0.6%); the trees already learn the monotone shape, the constraint only
  blocks useful local splits.
- Capacity helps a little and consistently: every bucket (−0.3% to −1.3%), every test day (ds1 Sat 81.5→81.1,
  Sun 84.9→84.5, Mon 92.3→91.8; shift Fri 91.9→91.2, Sat 81.8→81.2, Sun 85.1→84.8), p90 not worse. The 2400 cap
  binds again, so the recipe is still under-fit at lr 0.05 on 2.2M rows. vs baseline: −21.8% / −23.5%.
- Worst routes (511_it2k): ds1 122 327, 133 210, 106 162, 125 155, 129 137, 137 129, 881 128, 111 127, 131 124,
  113 124; shift 122 241, 106 164, 125 142, 98 137, 133 137, 137 132, 111 132, 2302 129, 129 128, 88 120.

**Conclusion: no new leader under the protocol (needs ≥ 3% vs the current best to be worth a notification; this is
−0.5 to −0.6%). `sc_big_511_it2k` is recorded as a capacity option: the same serving work as sc_big_dwell plus ~2x
trees and ~2x leaves (≈ 3 s per 3000-row batch single-threaded from the run-11 timing, still inside the 10 s cycle
but with less headroom; export ≈ 4x, ~120 MB). Monotone constraints are ruled out.**

### Next steps
1. Search is saturated on features, lookbacks and now capacity. The useful next step is hand-off of sc_big_dwell,
   which is the owner's call; if headroom allows, serve the 511/2400 variant.
2. If the search continues: time the 511/2400 export with `timing.py` (the run-11 script) to replace the 2x
   estimate with a measurement; and a 3x-rows check of sc_big_511_it2k (capacity may pay more with more rows).
