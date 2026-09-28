# Model-search journal

Research-only search for a better ETA model. Branch `claude/model-search`.
Harness: `harness.py` (prep/run), arms in `arms.py`, live features in
`features_live.py` (+ add-on scripts `addon_v*.py` that append cols to an existing
feature dir). Results JSON per arm in `results/`; `report.py` regenerates
LEADERBOARD.md (current tags) and archive/LEADERBOARD-old.md (superseded tags).
Full run entries are archived in `archive/JOURNAL-runs-1-12.md` and `archive/JOURNAL-runs-13-16.md`.

## State of the search (updated run 18)

### Protocol set-up (fixed)
- Fresh box: `pip install -e . tzdata` (without tzdata every pipeline day fails).
- Rebuild: `python research/model-search/pipeline_lite.py --parallel 4 --days 2026-08-31..2026-09-21`
  (≈ 1.5–2.25 h; 10% of snapshots + full-day `.cross` / `.pos` side tables; peak ~11 GB).
  Never run a prep or smoke test next to it by hand (OOM); `run.sh` waits for free memory.
  `run_pipeline.py` itself OOMs at `--parallel 2+` on this 4-core / 15.7 GB box.
- Prep: `harness.py prep --days D` per day, sequential (history lookups), ≈ 2.5 min/day,
  into `data/$MS_FEAT_DIR`. The prep chain + fits are driven by `run.sh` (parameterised).
- Splits: `ds1` = train 09-07..09-18, test 09-19 Sat / 09-20 Sun / 09-21 Mon, seed 42;
  `ds1shift` = train 09-07..09-17, test 09-18 Fri / 09-19 / 09-20, seed 7. Train 1% of
  snapshots/day (≈2.18M / 1.98M rows), test 3% (≈1.41M / 1.42M rows).
  08-31..09-06 only feed the 14-day history tables. Baseline = `src/train.py` recipe:
  110.6 / p90 230.9 (ds1), 113.0 / p90 241.1 (shift); reproduces bit-for-bit every run.
- 3x-rows check: `--keep-pct 3` with `MS_INPLACE=1` and `MS_DROP_COLS=<cols the arm doesn't use>`
  (≈ 8 GB peak; see archive run 15).
- Fits: ~5 min baseline, ~10–15 min for 255-leaf leaders; never fit while the pipeline runs.
  Background loop refreshes LOCK every 50 min (runs last > 3 h).

### Current leader: `sc_big_dwell` (run 14) — WIN on both splits/seeds
ds1 110.6 → 87.0 (−21.3%), p90 230.9 → 178.2; shift 113.0 → 87.0 (−23.1%), p90 241.1 → 179.5;
every stops_ahead bucket better (ds1 sa1 72.6→60.7, sa10 153.5→121.6). Confirmed at 3x train data
(run 15, `ds1v12k3`): 108.1 → 84.7 (−21.7%), p90 226.9 → 173.3.
Recipe (arms.py `sc_big_dwell`): production FEATURE_COLS − {month, day_of_week, stop_sequence},
route_id and target-stop code (top-254 stops) as native categoricals, 255 leaves /
min_samples_leaf 50, lr 0.05, 1200 iters (cap binds), sentinel −1 for missing live values, plus:
1. live link store: per stop-pair link the last 5 traversals (≤ 30 min, 30 s detect lag) →
   path sums with median-of-3 / median-of-5, coverage, age, speed (`_STACK`);
2. own speed over 60/180/300 s (5-min position ring);
3. historical link × hour median tables (all days + weekday/weekend) from the prior 14 days
   (`_DT`), built at export like the priors;
4. current-link live m5 / hist / fraction left (`_CUR`);
5. own previous-lap link times, 3 h window (`_LAP`);
6. vehicle-level own/hist link-time ratio over the last hour, across trips (`_VEH`);
7. conditional remaining dwell at the current location (last passed stop, 50 m bin) given
   stationary_sec, from the prior 14 days' stationary runs (`_DWELL`).
**Serving status: needs serving work, ≈ 3 days** (live link store + position ring + per-vehicle
crossing log persisted in `feed/tracker_state.json` across the 5-min push-feed restarts;
export of categorical bitsets / missing_go_to_left; static hist + dwell tables at export).
Timing (run 11): 255 leaves + bitsets = 1.55 s per 3000-row batch single-threaded (baseline
0.75 s), export ~29 MB — fits the 10 s cycle. Productionising is the owner's call.

### What won (in order; each Δ is vs the previous leader, both splits)
- live link path times (latest traversal per link): −7 to −9% vs baseline (run 1).
- + own speed, − headway / since_last_at_target: −8.3% / −9.0% (run 2).
- median of last 3/5 traversals, drop calendar cols (`stack`): −12.4% / −14.5% (run 3).
- historical link × hour table (`stack_hist`): ≈ −2% (run 4).
- weekday/weekend table + route_id categorical (`stack_hist_dt_cat`): ≈ −2.3% (run 6).
- target-stop categorical + 255 leaves (`dtcat_stopcat_big`): −1.6% (run 7).
- current-link cols (`sc_big_cur`): −0.6 to −1.1% (run 8).
- own previous lap (`sc_big_lap`): −1.2 to −1.5% (run 11).
- vehicle-level own/hist ratio (`sc_big_veh`): −1.0 to −1.2% (run 12).
- location dwell table (`sc_big_dwell`): −0.8% (run 14).

### Ruled out (one line each)
- Segment-additive link-time model: loses to direct HGBT at every horizon (repo history).
- Lookup-table bias correction: hurts MAE (repo history).
- Headway via since_last_at_target (any route): noise (run 1–2).
- NaN for missing live values: worse than sentinel −1 (exported trees route NaN right) (run 1).
- Cold-start dropout: costs warm accuracy; persist the live state instead; gate not needed (runs 2–3).
- lr 0.1 / lr 0.08, min_samples_leaf 100, l2 1, leaf100: neutral (runs 2, 6, 8, 11).
- No hour sample weights: neutral (run 3). Loss: prod already uses absolute_error (run 11).
- m7, EWMA per link, p75 table, live/hist ratio, fill path: ≤ 0.3% (runs 4, 6).
- Own-vs-hist ratio keyed by trip, recency weighting, pruning 3 cols: noise/loss (run 7).
- Residual target (base ETA + tree) and log-ratio target: no gain (run 7).
- Next-stop categorical: noise, p90 worse (run 8).
- 127 leaves for the lap model: loses most of the lap gain (run 11).
- 6 h lap window, 20 min / clipped vehicle ratio, network-wide 15-min ratio: ±0.2% (runs 12–13).
- Dwell table keyed by 3-hour band, live same-location last-3 dwell: +0.1–0.2% (run 14).
- Route 122 (230–330 s): data-sparse (2 vehicles, 43% live coverage, long links); untouched by
  every feature family; more training data helps it most (3x: 325→296). Treat as data issue.

- Same-route headway to the leader (at the last stop / target, leader's own gap): +0.3% (run 15).
- Stop-keyed dwell fallback for sparse 50 m bins: +0.1% (run 15).

- Longer train window (19 vs 12 days) is a data lever, not a model one: at equal rows both the baseline and
  the leader gain ≈ 0.8–1.5%, so the leader's relative gain is unchanged (−20.7% vs −21.3%) (run 16).
- History-table lookback (hist link × hour + dwell tables): 14 d is best; 21 d +0.2% (noise), 7 d +0.9% (run 17).

- Code pruned in run 18 (lost on both splits): arms sc_big_net, sc_big_v9, sc_big_lap6, sc_big_v8,
  sc_lap_127, sc_big_nxt, path_own_s_drop(_cold); the lap6_* and V9 (net / ratio20 / ratio60c) cols in
  features_live.py. Their results JSON stay; the code is in git history (before run 18).

### Open next steps
1. Longer window / more rows help both models equally (run 16), and production already trains on all days.
   The model-side search has saturated. The remaining open item is hand-off.
2. Route 133 trip-structure diagnostic (headway cols moved only this route).
3. Hand-off of sc_big_dwell is the owner's call. Keep the 14-day lookback for its static tables (run 17).

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
