# Model-search journal

Research-only search for a better ETA model. Branch `claude/model-search`.
Harness: `harness.py` (prep/run), arms in `arms.py`, live features in
`features_live.py` (+ add-on scripts `addon_v*.py` that append cols to an existing
feature dir). Results: one compact JSON line per fit in `results.jsonl` (keyed by `name` = arm_tag_sSEED; written by `harness.save_result`, run 20 replaced the per-fit `results/*.json` files); `report.py` regenerates
LEADERBOARD.md (current tags) and archive/LEADERBOARD-old.md (superseded tags).
Full run entries are archived in `archive/JOURNAL-runs-1-12.md` and `archive/JOURNAL-runs-13-16.md`, `archive/JOURNAL-runs-17-18.md`.

## State of the search (updated run 22)

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
  (addon_v13.py, arm sc_big_hold) can go at the next housekeeping.

### Open next steps
1. The model-side search has saturated: runs 16–20 all land within ±0.6% of sc_big_dwell (capacity, lookback,
   recency, weights, trip-keyed holds). The open item is hand-off of sc_big_dwell, which is the owner's call. Keep
   the 14-day lookback for its static tables (run 17).
2. Routes 133 / 122 (run 20 diagnostic, `diag_route.py` + `diag_holds.py`): 1–2 vehicles each. 62% / 45% of their
   error sits in rows where the actual time exceeds the historical path by > 300 s. These are driver breaks
   at a fixed place and clock time on weekdays (133: 400 m into the trip at ~10:03 / ~12:15 / ~15:45 since
   09-16; 122: ~12:45 at 2.7 km, ~09:30 at 900 m). A table keyed by trip id needs weeks of history for
   these trips. If `sc_big_hold` is ever retried, use a day set ending ≥ 3 weeks after 09-16. Better still,
   get the break times from the operator.

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

## 2026-09-28 — run 20 (housekeeping; results.jsonl; no hour weights; route 133/122 holds; trip-keyed holds)

Lock taken at 16:18 UTC. Housekeeping was due (every 5th run), commit "model-search: housekeeping":
- Run 17 was moved to `archive/JOURNAL-runs-17-18.md`, and run 18 joined it at the end of this run.
- The ds1h21 / ds1h7 / ds1r32 / ds1r72 tags moved to the old leaderboard.
- run16.sh and run17.sh were deleted.
- Arms that lost on both splits were removed: sc_big_v6, sc_big_dwell2.
- run.sh gained `VARIANTS=` (the re-prep loop of run17/19.sh).

Owner request mid-run: the 176 per-fit `results/*.json` files are now one compact line each in `results.jsonl`
(harness `save_result` / `load_results`; report.py and cmp.py read it). Both leaderboards regenerated byte-identical,
and the branch diff vs main went from +19.8k to +3.9k lines. The owner also asked to move the routine to every 4th hour.
That has to be changed in the routine settings on claude.ai; this session can't do it.

Rebuild: `pipeline_lite.py --parallel 4 --days 2026-08-31..2026-09-21` (16:19–18:43; the first round took ≈ 42 min/day/worker,
later rounds were faster). `sh research/model-search/run20.sh`: prep into v10 → `ds1v20` / `ds1shiftv20`, with baseline,
sc_big_nw and sc_big_dwell. The ds1 fits save per-row predictions (`MS_SAVE_PRED`, new in harness.py). No OOM this time.

**A. `sc_big_nw`** = the leader without the production hour sample weights (`_build_sample_weights`).
**B. Route diagnostic** on the ds1 leader predictions (`diag_route.py pred 133 122`, `diag_holds.py 2026-09-07..2026-09-21 133 122`):
- 133: MAE 208, bias −119 s. One vehicle (2528) has 80% of the rows. The error sits in a few local hours (11h MAE 449,
  18h MAE 412) and in stationary > 900 s rows. 26% of rows have actual > hist path + 300 s; they carry 64% of the
  route's AE, with bias −501 s.
- `diag_holds.py`: vehicle 2528 holds 12–46 min at 400 m into the trip at ~10:03, ~12:15 and ~15:45 on every weekday
  since its trip ids changed to block 29328 on 09-16.
- 122 (MAE 332) is the same kind of case: vehicle 2258 holds ~22–25 min at 2.7 km at ~12:45 on weekdays; vehicle 901
  holds 39–55 min at 900 m at ~09:30.
- These are driver breaks, timetabled per trip id, not traffic.

**C. `sc_big_hold`** (V13, `addon_v13.py` → `ms_features_v13`, `sh research/model-search/run20b.sh`). New cols:
- trip_id × (last stop, 50 m bin) conditional remaining dwell: `dwell_t_rem_med` / `dwell_t_p_more120` / `dwell_t_n`, min 5 runs;
- `hold_ahead_med`: median over prior days of the trip's summed stops ≥ 120 s between the current position and the target;
- `hold_ahead_days`.

Coverage (test days): trip ids seen before 100%, hold_ahead > 0 on 4–7% of rows, trip dwell on 12–18% of weekday stopped
rows and ~1% on weekends. Serving: two static tables at export (~+0.5 day).

### Results (MAE / median / p90 / bias; Δ vs the leader; train 2.18M / 1.98M, test 1.41M / 1.42M rows)
| arm | ds1 (s42) | ΔL | sa1 / sa5 / sa10 | Sat / Sun / Mon | shift (s7) | ΔL | sa1 / sa5 / sa10 | Fri / Sat / Sun |
|---|---|---|---|---|---|---|---|---|
| baseline (v20 = v13, reproduced exactly) | 110.6 / 56.0 / 230.9 / −28.2 | | 72.6 / 112.0 / 153.5 | 100.2 / 107.1 / 120.5 | 113.0 / 59.5 / 241.1 / −14.2 | | 71.2 / 114.3 / 159.7 | 126.8 / 99.6 / 106.6 |
| sc_big_dwell (reproduced exactly) | 87.0 / 43.1 / 178.2 / −12.6 | | 60.7 / 87.1 / 121.6 | 81.5 / 84.9 / 92.3 | 87.0 / 43.9 / 179.5 / −14.6 | | 59.7 / 87.1 / 122.3 | 91.9 / 81.8 / 85.1 |
| sc_big_nw | 86.9 / 43.1 / 177.8 / −12.4 | −0.07% | 60.7 / 86.9 / 121.6 | 81.4 / 84.8 / 92.4 | 86.9 / 43.8 / 179.6 / −14.7 | −0.11% | 59.7 / 86.9 / 122.2 | 91.7 / 81.8 / 85.0 |
| sc_big_hold | 87.0 / — / 178.6 / −11.6 | −0.02% | 60.4 / 86.8 / 122.0 | 81.7 / 85.1 / 92.0 | 86.5 / 43.9 / 179.4 / −13.7 | −0.52% | 59.0 / 86.7 / 122.0 | 91.2 / 81.4 / 84.8 |

- sc_big_hold on the ds1 test rows: rows with hold_ahead > 0 (5.9%) MAE 198.6 → 197.3; rows with a trip dwell
  (10.7%) 90.2 → 88.9; the rest 79.6 → 79.7. The weekday test days gain (Mon −0.3%, Fri −0.8%), the weekend days don't:
  there is not enough weekend history per trip.
- Routes: 133 208 → 206 (ds1), 122 332 → 332 (ds1), 239 → 231 (shift). Worst routes (hold, ds1) 122 332, 133 206, 106 164,
  125 157, 129 141, 881 129, 137 129, 111 128, 113 126, 131 123; (hold, shift) 122 231, 106 164, 125 142, 133 140,
  98 137, 137 133, 111 132, 2302 126, 129 125, 88 121.
- Worst routes (nw, ds1): 122 319, 133 201, 106 162, 125 158, 129 130, 111 129, 137 129, 881 128, 131 124, 113 124.

**Conclusion: no new leader and no notification.**
- The hour weights are neutral on the leader.
- Trip-keyed holds are real but sparse. They gain 0–0.5%, and less where history is short. The breaks behind routes
  133 / 122 are identified: they are recurring per-trip driver breaks.

### Next steps
See "Open next steps" above. The search is saturated; the useful work is hand-off. At a 4-hour cadence with ~5 h runs,
most firings will find the lock and exit.

## 2026-09-29/30 — runs 21–22 (later day set ds3; trip-keyed holds with longer history)

Run 21 (lock 00:13 UTC on 09-29) parameterised the split days in run.sh (TRAIN_A/TEST_A/PFX_A, TRAIN_B/TEST_B/PFX_B) and wrote run21.sh
(new day set ds3, data 09-07..09-28). It pushed the ds3 baseline and leader fits, then the session died (no journal entry).
Run 22 (lock 20:13 UTC) finished it with `run22.sh`. It rebuilt `pipeline_lite.py --parallel 4 --days 2026-09-07..2026-09-28`
(≈ 30 min/day/worker, 20:14–23:10), then prep, the V13 add-on and the fits.
Pitfalls hit this run (the ds3 baseline still reproduced run 21 bit-for-bit, 109.3996):
- A prep started next to the running pipeline was OOM-killed, so start run22.sh only after the pipeline has finished.
- A waiter or pkill using `-f <pattern>` matches its own shell's command line (a pkill killed its own shell); use pids instead.
- A worker restart killed every background job. Start long jobs with `setsid nohup ... < /dev/null`. Prep skips finished days.

Note: ds3's train days 09-14..09-20 have < 14 prior days for the history tables (data starts 09-07). This is the same for every arm.

### Results (MAE / median / p90 / bias; Δ vs the leader)
| arm | ds3 (s42; train 2.25M, test 1.42M) | ΔL | sa1 / sa5 / sa10 | Sat / Sun / Mon | ds3shift (s7; train 2.04M, test 1.44M) | ΔL | sa1 / sa5 / sa10 | Fri / Sat / Sun |
|---|---|---|---|---|---|---|---|---|
| baseline | 109.4 / 55.1 / 227.4 / −28.0 | | 73.0 / 111.1 / 152.0 | 103.4 / 106.6 / 115.4 | 111.4 / 57.0 / 233.3 / −27.5 | | 71.5 / 113.8 / 154.6 | 119.4 / 103.9 / 107.0 |
| sc_big_dwell | 86.9 / 43.0 / 175.7 / −15.9 (−20.6% vs base) | | 61.6 / 87.5 / 120.6 | 85.1 / 85.1 / 89.2 | 88.4 / 43.7 / 180.9 / −18.8 (−20.7%) | | 59.9 / 89.2 / 123.6 | 92.4 / 85.3 / 85.5 |
| sc_big_hold | 87.0 / 43.1 / 176.6 / −14.4 | +0.15% | 61.4 / 87.8 / 120.9 | 85.3 / 85.0 / 89.5 | 88.3 / 43.8 / 181.2 / −17.5 | −0.17% | 59.7 / 88.9 / 123.7 | 91.9 / 85.5 / 85.5 |

Worst routes:
- baseline ds3: 133 300, 137 290, 106 206, 122 193, 2302 179, 143 178, 131 169, 2460 169, 94 167, 1630 167.
- Leader ds3: 133 258, 137 218, 122 179, 143 163, 106 158, 1630 131, 2460 130, 2302 128, 111 125, 125 116.
- Leader ds3shift: 133 342, 137 203, 143 201, 106 168, 2460 153, 129 147, 122 137, 2302 135, 1630 131, 1062 121.
- Hold ds3: 133 260, 137 231, 122 181, 143 158, 106 157. Hold ds3shift: 133 338, 137 204, 143 196, 106 169, 2460 157.

**Conclusion: no new leader and no notification.**
- sc_big_dwell keeps its −21% on two later splits it was never tuned on, which answers the drift question.
- Trip-keyed holds don't help even with 10+ days of trip history. Route 133 is now the worst route by far (258–342 s), and its
  error is the driver breaks. Only operator break times or a real-time "vehicle is on break" signal can fix it; more history
  of the same kind can't.

### Next steps
1. Hand-off of sc_big_dwell (owner's call). ds3 confirms it on newer data.
2. Next housekeeping (run 25, or sooner): archive runs 19–22, delete addon_v13.py / arm sc_big_hold / run19*.sh / run20*.sh / run21.sh,
   and add the ds1*v13 tags to the old leaderboard.
3. Pausing the schedule is still the recommendation: runs 16–22 moved the leader by 0%.
