# Model-search journal — archived runs 19–20

## 2026-09-28 — run 19 (recency-weighted history tables; measured serving cost of the capacity variant)

Lock taken at 09:17 UTC. No housekeeping due (journal 197 lines; next at run 20). Main was already merged.
Fresh box: `pip install -e . tzdata`. The pipeline was slow for the first round (≈ 50 min/day/worker, R2 latency?), then
≈ 26 min: `pipeline_lite.py --parallel 4 --days 2026-08-31..2026-09-21` ran 09:18–~12:10.

**A. Recency weighting (run-17 next step 2).** New env in harness.py prep: `MS_HIST_RECENT=R MS_HIST_RECENT_REP=K` adds
K−1 extra copies of the last R prior days' `links_` / `runs_` frames before the hist link × hour, weekday/weekend and
dwell tables are built (a weighted median; the 14-day lookback is unchanged). Side effect: the count thresholds
(HIST_MIN_N, DWELL_MIN_N) and the `dwell_n` feature see the duplicated counts. `sh research/model-search/run19.sh`:
prep v10 → ds1v19 baseline + leader, then re-prep 09-07..09-21 per variant (prior-day files symlinked from v10) and
fit the leader on `ds1r32` (R=3, K=2) and `ds1r72` (R=7, K=2).

**B. Timing (run-18 next step 2).** `MS_SAVE_MODEL=<pkl>` is back in harness.py (it saves the last `fit_hgbt` pipe,
its columns and 15k training rows for `timing.py`). `sh research/model-search/run19b.sh` refits baseline / sc_big_dwell /
sc_big_511_it2k on ds1 (tag `ds1v19t`, identical to ds1v19) and times them.

### Results (ds1, seed 42, train 2.18M, test 1.41M rows; Sat 09-19 / Sun 09-20 / Mon 09-21)
| arm | MAE / median / p90 / bias | Δbase | Δ leader | sa1 / sa5 / sa10 | Sat / Sun / Mon |
|---|---|---|---|---|---|
| baseline (ds1v19, reproduced exactly) | 110.6 / 56.0 / 230.9 / −28.2 | | | 72.6 / 112.0 / 153.5 | 100.2 / 107.1 / 120.5 |
| sc_big_dwell (ds1v19, reproduced exactly) | 87.0 / 43.1 / 178.2 / −12.6 | −21.3% | | 60.7 / 87.1 / 121.6 | 81.5 / 84.9 / 92.3 |
| sc_big_dwell, last 3 d ×2 (ds1r32) | 87.1 / 43.1 / 178.1 / −13.2 | −21.2% | +0.14% | 60.9 / 87.0 / 122.0 | 81.4 / 85.0 / 92.6 |
| sc_big_dwell, last 7 d ×2 (ds1r72) | 87.1 / 43.1 / 177.7 / −13.5 | −21.3% | +0.11% | 60.7 / 87.0 / 121.8 | 81.5 / 84.9 / 92.6 |
| sc_big_511_it2k (ds1v19t) | 86.5 / — / 177.3 | −21.7% | −0.51% | (= run 18) | |

- Recency weighting is noise-level: Monday gets slightly worse (+0.3%), p90 is flat or slightly better (r72 177.7).
  Route 122 improves with r72 (332 → 319), as it did with the 7-day lookback in run 17, but 133 does not (208 → 212).
  Worst routes (r32): 122 331, 133 206, 106 163, 125 157, 129 131, 111 130, 881 129, 137 129, 113 126, 131 121.
  (r72): 122 319, 133 212, 106 164, 125 157, 129 133, 111 129, 881 128, 137 127, 113 126, 131 123.
- Timing, `timing.py` (single-threaded, 3000-row push batches, parity with sklearn ≤ 2e-11 for all three):

| model | trees | nodes | export (float64) | categorical nodes | 3000 rows (median) |
|---|---|---|---|---|---|
| baseline | 1200 | 303,600 | ~15 MB | 0 | 0.87 s |
| sc_big_dwell | 1200 | 610,800 | ~29 MB | 109,601 | 1.90 s |
| sc_big_511_it2k | 2400 | 2,450,400 | ~118 MB | 358,709 | 5.01 s |

**Conclusion: no new leader. Recency weighting is ruled out (±0.1%). The capacity variant costs 2.6x the leader's
serving time (5 s of a 10 s cycle on this box) for −0.5%, so it is not recommended; sc_big_dwell (1.9 s, 29 MB) remains
the hand-off candidate.** No notification (no protocol win).

### Next steps
1. The search is saturated on features, table lookback/recency and capacity. Remaining value is the sc_big_dwell hand-off
   (owner's call). A cheaper serving path (vectorised numpy walk over all trees at once, or treelite-style compiled
   code) would make the 511/2400 variant affordable, but only for −0.5%.
2. Run 20 is a housekeeping run (every 5th): archive runs 17–18, drop run16/17.sh (keep run18/19/19b.sh), consider
   folding run19.sh's re-prep loop into run.sh as a `VARIANT_ENV` option (run 17 and run 19 repeat the same pattern).
3. If hourly runs continue without a hand-off, consider pausing the schedule: each run costs ~5 h and the last five
   runs moved the leader by 0%.

