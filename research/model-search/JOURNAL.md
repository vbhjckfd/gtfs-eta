# Model-search journal

Research-only search for a better ETA model. Branch `claude/model-search`.
Harness: `harness.py` (prep/run), arms in `arms.py`, live features in
`features_live.py` (+ add-on scripts `addon_v*.py` that append cols to an existing
feature dir). Results: one compact JSON line per fit in `results.jsonl` (keyed by `name` = arm_tag_sSEED; written by `harness.save_result`, run 20 replaced the per-fit `results/*.json` files); `report.py` regenerates
LEADERBOARD.md (current tags) and archive/LEADERBOARD-old.md (superseded tags).
Full run entries are archived in `archive/JOURNAL-runs-1-12.md` and `archive/JOURNAL-runs-13-16.md`, `archive/JOURNAL-runs-17-18.md`, `archive/JOURNAL-runs-19-20.md`, `archive/JOURNAL-runs-21-23.md`.

## State of the search (updated run 25)

### Protocol set-up (fixed)
- Results live in `results.jsonl` (owner asked for this on 2026-09-28, run 20). Where the routine prompt says "keep results/*.json", read it
  as "keep results.jsonl". Never recreate per-fit JSON files; `harness.py run` appends through `save_result`.
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
**Later day set (runs 21–22, ds3 = train 09-14..09-25, test 09-26..09-28; ds3shift = train 09-14..09-24, test 09-25..09-27):**
109.4 → 86.9 (−20.6%), p90 227.4 → 175.7; shift 111.4 → 88.4 (−20.7%), p90 233.3 → 180.9; every bucket better. The gain holds
on unseen later weeks.
**Serving status: needs serving work, ≈ 3 days** (live link store + position ring + per-vehicle
crossing log persisted in `feed/tracker_state.json` across the 5-min push-feed restarts;
export of categorical bitsets / missing_go_to_left; static hist + dwell tables at export).
Timing (run 19 box): 255 leaves + bitsets = 1.90 s per 3000-row batch single-threaded (baseline
0.87 s; run 11 measured 1.55 / 0.75 s), export ~29 MB — fits the 10 s cycle. Productionising is the owner's call.

**Capacity variant (run 18, needs serving work):** `sc_big_511_it2k` = the same features with 511 leaves /
min_samples_leaf 100 / up to 2400 iterations: −0.51% (ds1) / −0.62% (shift) vs sc_big_dwell, every bucket
better. Not a protocol win over the leader. Measured serving cost (run 19): 2400 trees, 2.45M nodes,
~118 MB export, 5.0 s per 3000-row batch single-threaded (2.6x the leader, 5.8x the baseline): half the 10 s cycle,
so only viable with vectorised/compiled serving or smaller batches. Not recommended over sc_big_dwell.

**Free tweak (run 23, ds3 only): `sc_big_mf5`** = sc_big_dwell with `max_features=0.5` (per-split column subsampling):
ds3 86.87 → 86.43 (−0.50%), p90 175.7 → 174.8; ds3shift 88.42 → 88.03 (−0.45%), p90 180.9 → 180.3; every bucket better.
Training-only, export and serving unchanged, so if sc_big_dwell is handed off, ship it with max_features 0.5.
**Run 24 sweep (ds3, both splits):** max_features 0.3 (`sc_big_mf3`) is the best setting: ds3 86.87 → 86.30 (−0.65%), p90 175.7 → 174.7;
ds3shift 88.42 → 87.85 (−0.64%), p90 180.9 → 180.4; every bucket better. 0.4 and 0.2 land between 0.5 and 0.3, and min_samples_leaf 20 is neutral.
**Hand-off recipe: sc_big_dwell with max_features 0.3.**
**Run 25, newest day set ds4 (train 09-15..09-26, test 09-27 Sun / 09-28 Mon / 09-29 Tue):** baseline 113.9 → leader 88.1 (−22.7%) → mf3 87.8
(−23.0%, p90 241.6 → 180.8); ds4shift (seed 11) baseline 110.0 → mf3 86.7 (−21.1%). mf3 beats the leader on a third split (−0.41%), every bucket.
Cheaper serving options (ds4): `sc_big_mf3_127` (127-leaf trees, ~half the export) gives 88.3 (+0.6% vs mf3, about the same as the leader). `sc_big_mf3_fast`
(600 trees at lr 0.1, ~half the tree walks) gives 88.7 / 87.3 (+1.1% / +0.7% vs mf3). Both still beat the baseline by more than 20%, so either is a
safe fallback if serving time binds.
Bagging two max_features=0.7 fits (`sc_big_bag2`) gains −0.76% / −0.98% vs the leader, but it doubles the trees (~3.8 s per 3000 rows). Optional.

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

- Monotone +1 constraints on remaining_dist / stops_ahead / prior ETAs: +0.24% (run 18). Live/hist
  path cols can't be constrained (−1 sentinel).

- Recency weighting inside the 14-day tables (last 3 or 7 prior days counted twice): +0.14% / +0.11% (run 19).
- Dropping the production hour sample weights from the leader (`sc_big_nw`): −0.07% / −0.11% (run 20). Neutral, so
  production can keep or drop them.
- Trip-keyed holds (`sc_big_hold`, V13: trip × location remaining dwell + prior days' holds between here and the target):
  −0.02% (ds1) / −0.52% (shift) (run 20). Helps only where it applies (6–11% of rows, −0.7 to −1.4% there). Routes 133
  and 122 barely move: their timetabled breaks are real, but the recurring trip ids are too new (133 since 09-16).

- Trip-keyed holds again on ds3, where the route-133 trip ids have 10+ days of history (`sc_big_hold`): +0.15% (ds3) /
  −0.17% (ds3shift), p90 worse on both, and route 133 doesn't move (258 → 260 / 342 → 338) (run 22). Ruled out twice; its code
  (addon_v13.py, arm sc_big_hold) was removed in run 23 (git history has it; run22.sh still names it).

- Route 137 (run 23 diagnostic, ds3): the same class as 133. 4 vehicles; 28% of rows have actual > hist + 300 s and carry
  66% of its AE (bias −473 s); stationary > 900 s rows MAE 340. These are driver holds, not traffic.

### Open next steps
0. (run 25) mf3 is confirmed on ds4/ds4shift, so the ds1 check is moot. Open: time the serving cost of sc_big_mf3_127 / _fast (timing.py, MS_SAVE_MODEL) on the GH-runner-sized box.
   Old note: the max_features sweep (run 24) left a ds1 check open, which needed
   the 08-31..09-21 rebuild. It is optional; the setting can't hurt serving.
1. The model-side search has saturated: runs 16–20 all land within ±0.6% of sc_big_dwell (capacity, lookback,
   recency, weights, trip-keyed holds). The open item is hand-off of sc_big_dwell, which is the owner's call. Keep
   the 14-day lookback for its static tables (run 17).
2. Routes 133 / 122 (run 20 diagnostic, `diag_route.py` + `diag_holds.py`): 1–2 vehicles each. 62% / 45% of their
   error sits in rows where the actual time exceeds the historical path by > 300 s. These are driver breaks
   at a fixed place and clock time on weekdays (133: 400 m into the trip at ~10:03 / ~12:15 / ~15:45 since
   09-16; 122: ~12:45 at 2.7 km, ~09:30 at 900 m). A table keyed by trip id needs weeks of history for
   these trips. If `sc_big_hold` is ever retried, use a day set ending ≥ 3 weeks after 09-16. Better still,
   get the break times from the operator.

## 2026-09-30 — run 24 (max_features sweep: 0.4 / 0.3 / 0.2, min_samples_leaf 20)

Lock taken at 12:19 UTC. Main was already merged. No housekeeping (journal 220 lines; the 5th-run housekeeping falls on run 25).
Rebuild: `pipeline_lite.py --parallel 4 --days 2026-09-07..2026-09-28` (12:20–14:39, ≈ 25 min/day/worker). This time `run24.sh` was
started by a waiter after pipeline_lite exited, and prep had no OOM. Prep ran 14:40–15:40, then fits (`sh research/model-search/run24.sh`,
features ms_features_v10, tag ds3v13 / ds3shiftv13). Baseline (109.3996) and mf5 (86.427) reproduce run 23 bit-for-bit.
Pitfall: the tool shell that launched pipeline_lite with `setsid nohup ... &` stays alive, and its command line contains
"pipeline_lite.py", so run.sh's `pgrep -f pipeline_lite.py` would wait forever. Kill that shell by pid (or use `pgrep -x -f` with the full command).
`pkill -f <script>` from a tool shell kills that shell too (known pitfall). Use pids.

New arms (arms.py, no new cols): `sc_big_mf4` / `sc_big_mf3` / `sc_big_mf2` = the leader with max_features 0.4 / 0.3 / 0.2;
`sc_big_mf5_msl20` = mf5 with min_samples_leaf 20.

### Results (MAE / median / p90 / bias; Δ vs the leader sc_big_dwell; ds3 s42 train 2.25M / test 1.42M; ds3shift s7 train 2.04M / test 1.44M)
| arm | ds3 | ΔL | sa1 / sa5 / sa10 | Sat / Sun / Mon | ds3shift | ΔL | sa1 / sa5 / sa10 | Fri / Sat / Sun | fit s |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 109.40 / 55.1 / 227.4 / −28.0 | | 73.0 / 111.1 / 152.0 | 103.4 / 106.6 / 115.4 | 111.44 / 57.0 / 233.3 / −27.5 | | 71.5 / 113.8 / 154.6 | 119.4 / 103.9 / 107.0 | |
| sc_big_dwell (run 23) | 86.87 / 43.0 / 175.7 / −15.9 | | 61.6 / 87.5 / 120.6 | 85.1 / 85.1 / 89.2 | 88.42 / 43.7 / 180.9 / −18.8 | | 59.9 / 89.2 / 123.6 | 92.4 / 85.3 / 85.5 | 747 / 684 |
| sc_big_mf5 | 86.43 / 42.8 / 174.8 / −15.8 | −0.50% | 61.1 / 87.0 / 120.0 | 84.5 / 84.6 / 88.9 | 88.03 / 43.6 / 180.3 / −18.4 | −0.45% | 59.6 / 88.5 / 123.4 | 92.1 / 85.0 / 84.8 | 860 / 751 |
| sc_big_mf4 | 86.48 / 42.7 / 175.0 / −15.6 | −0.45% | 61.2 / 87.0 / 120.0 | 84.7 / 84.5 / 89.0 | 87.96 / 43.5 / 180.6 / −18.7 | −0.52% | 59.5 / 88.6 / 123.3 | 92.0 / 84.9 / 84.9 | 894 / 764 |
| **sc_big_mf3** | 86.30 / 42.7 / 174.7 / −15.3 | **−0.65%** | 61.0 / 86.9 / 119.7 | 84.6 / 84.5 / 88.7 | 87.85 / 43.5 / 180.4 / −18.5 | **−0.64%** | 59.5 / 88.6 / 122.9 | 91.9 / 84.8 / 84.8 | 850 / 746 |
| sc_big_mf2 | 86.49 / 42.7 / 174.7 / −15.9 | −0.43% | 61.2 / 87.1 / 119.7 | 84.6 / 84.7 / 89.0 | 87.92 / 43.6 / 180.2 / −18.7 | −0.56% | 59.7 / 88.5 / 123.1 | 92.0 / 84.6 / 85.1 | 808 / 769 |
| sc_big_mf5_msl20 | 86.46 / 42.8 / 175.4 / −15.1 | −0.47% | 61.1 / 87.1 / 120.1 | 84.5 / 84.8 / 88.9 | 88.18 / 43.5 / 180.5 / −18.9 | −0.27% | 59.7 / 88.8 / 123.5 | 92.2 / 85.2 / 85.1 | 844 / 757 |

vs baseline, sc_big_mf3 is −21.1% (ds3) / −21.2% (ds3shift), p90 227.4 → 174.7 / 233.3 → 180.4, and every bucket is better (the protocol win carries over from the leader).
Worst routes, mf3 ds3: 133 254.5, 137 215.3, 122 175.3, 143 153.9, 106 151.6, 2460 131.2, 1630 128.2, 111 123.1, 2302 122.9, 125 116.4.
mf3 ds3shift: 133 334.1, 137 203.1, 143 200.5, 106 165.8, 2460 146.9, 129 146.3, 2302 137.0, 122 136.0, 1630 125.1, 1062 119.5.

**Conclusion: no protocol win vs the leader (< 3%) and no notification.** max_features 0.3 is the best column-subsampling setting,
−0.65% on both splits with every bucket and p90 better and the same serving cost. 0.2–0.5 all fall within 0.2% of each other, so the exact
value matters little. Finer leaves (min_samples_leaf 20) add nothing.

### Next steps
1. Run 25 = housekeeping: archive runs 21–23 and remove the mf/msl arms that only lost (mf7, mf4, mf2, mf5_msl20 once they're marked twice).
   Delete run22.sh / run23.sh.
2. Hand-off of sc_big_dwell + max_features 0.3 (owner's call). This hyper-parameter family is now saturated too.
3. Recommend pausing the schedule: runs 16–24 moved the leader by < 1%. The remaining large errors (133 / 137 / 122 driver holds) need
   operator break data. A later day set (ending ≥ 10-07) would allow a proper re-test of trip-keyed holds.

## 2026-09-30/10-01 — run 25 (housekeeping; newest day set ds4; cheaper-to-serve mf3 variants)

Lock taken at 20:13 UTC. Main (aeb652a) was already merged. Housekeeping commit 7b94c24: runs 21–23 went to `archive/JOURNAL-runs-21-23.md`,
the ds1v18 / ds1v20 tags moved to the old leaderboard, the arms mf7 / mf4 / mf2 / mf5_msl20 (dominated by mf3 on both ds3 splits) were removed,
and run22/23.sh were deleted.
Rebuild: `pipeline_lite.py --parallel 4 --days 2026-09-08..2026-09-29`. The container restarted at about 22:10 with 14 days done (≈ 32 min/day/worker),
which killed every background job. A re-run skipped the finished days and finished at 22:59. Then `sh research/model-search/run25.sh` ran
(prep into ms_features_v10, fits tagged ds4v25 / ds4shiftv25), chained after the pipeline in one script so they can't overlap.
New arms (no new cols): `sc_big_mf3_fast` = mf3 with lr 0.1 and a 600-tree cap; `sc_big_mf3_127` = mf3 with 127 leaves.
Note: ds4's train days 09-15..09-21 have < 14 prior days for the history tables (data starts 09-08). This is the same for every arm.

### Results (MAE / median / p90 / bias; Δ vs the baseline of the same split)
| arm | ds4 (s42; train 2.21M, test 1.61M) | Δ | sa1 / sa5 / sa10 | Sun / Mon / Tue | ds4shift (s11; train 2.06M, test 1.42M) | Δ | sa1 / sa5 / sa10 | Sat / Sun / Mon | fit s |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 113.95 / 60.4 / 241.6 / −19.3 | | 73.4 / 116.2 / 157.8 | 104.5 / 115.5 / 118.5 | 109.95 / 55.9 / 228.7 / −23.5 | | 73.3 / 111.8 / 152.7 | 103.6 / 106.8 / 116.5 | 509 / 418 |
| sc_big_dwell | 88.13 / 44.9 / 181.3 / −12.4 | −22.7% | 61.1 / 89.2 / 122.7 | 82.9 / 88.9 / 90.8 | — | | | | 819 |
| **sc_big_mf3** | 87.77 / 44.7 / 180.8 / −12.7 | −23.0% | 61.0 / 88.6 / 122.3 | 82.4 / 88.4 / 90.6 | 86.71 / 42.9 / 175.5 / −14.9 | −21.1% | 61.4 / 87.2 / 120.4 | 84.7 / 84.6 / 89.5 | 781 / 727 |
| sc_big_mf3_127 | 88.30 / 44.8 / 181.4 / −13.7 | −22.5% | 61.4 / 89.1 / 122.9 | 83.0 / 89.0 / 91.1 | — | | | | 668 |
| sc_big_mf3_fast | 88.71 / 45.0 / 182.6 / −13.1 | −22.1% | 61.7 / 89.5 / 123.5 | 83.8 / 89.3 / 91.4 | 87.34 / 43.2 / 177.0 / −14.9 | −20.6% | 61.7 / 88.0 / 121.2 | 85.7 / 85.2 / 89.9 | 449 / 401 |

Every arm wins the protocol vs the baseline on its split (every bucket better by ≥ 15.8%, p90 better by ≥ 22%).
Worst routes, ds4:
- baseline: 137 339, 133 248, 2302 199, 106 197, 122 194, 131 188, 143 179, 2460 175, 111 167, 118 164.
- mf3: 137 242, 133 200, 122 169, 118 165, 106 144, 131 135, 2302 134, 2460 133, 111 130, 143 130.
- mf3_127: 137 250, 133 194, 122 170, 118 166, 106 150.

Worst routes, ds4shift:
- baseline: 133 300, 137 288, 106 204, 2302 193, 122 193.
- mf3: 133 252, 137 223, 122 173, 143 156, 106 156, 2460 133, 2302 131, 1630 128, 111 124, 125 119.

Route 118 joins the worst list on ds4 (~165 for every arm). The 09-29 test day is new there, which is worth a look if it persists.

**Conclusion: no new protocol win vs the leader and no notification.**
- On the newest days the leader family holds (−22.7% / −23.0%), and max_features 0.3 helps on a third split (−0.41% vs the leader). The hand-off recipe stays sc_big_dwell + max_features 0.3.
- Two cheaper-to-serve versions cost little:
  - 127-leaf trees: +0.6% vs mf3, about the same as the unsubsampled leader. It is the production per-tree size, so the export stays ~15 MB rather than ~29 MB.
  - 600 trees at lr 0.1: +0.7 to +1.1%, with about half the tree walks per prediction.

### Next steps
1. Hand-off of sc_big_dwell + max_features 0.3 (owner's call). If the serving budget is tight, 127 leaves is the first thing to give up (≈ +0.6%).
2. Optional: measure the serving time of mf3_127 / mf3_fast with timing.py (needs `MS_SAVE_MODEL` refits).
3. Still recommend pausing the schedule. The model-side search is saturated; runs 16–25 moved the leader by < 1%.
