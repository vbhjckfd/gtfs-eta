# Model-search journal

Research-only search for a better ETA model. Branch `claude/model-search`.
Harness: `harness.py` (prep/run), arms in `arms.py`, live features in
`features_live.py` (+ add-on scripts `addon_v*.py` that append cols to an existing
feature dir). Results: one compact JSON line per fit in `results.jsonl` (keyed by `name` = arm_tag_sSEED; written by `harness.save_result`); `report.py` regenerates
LEADERBOARD.md (current tags) and archive/LEADERBOARD-old.md (superseded tags).
Full run entries are archived in `archive/JOURNAL-runs-1-12.md`, `-13-16.md`, `-17-18.md`, `-19-20.md`, `-21-23.md`, `-24-27.md`, `-28-32.md`, `-33-37.md`.

## State of the search (updated run 40)

### Protocol set-up (fixed)
- Results live in `results.jsonl` (owner asked for this on 2026-09-28, run 20). Where the routine prompt says "keep results/*.json", read it
  as "keep results.jsonl". Never recreate per-fit JSON files; `harness.py run` appends through `save_result`.
- Fresh box: `pip install -e . tzdata` (without tzdata every pipeline day fails).
- One runner since run 35: `sh research/model-search/ds.sh` = pipeline_lite `--parallel 3` (≈ 2–2.5 h) in the background, feed-clock prep into
  ms_features_v10 as days land (≈ 2.5 min/day), the addon chain v26, v28–v32 → ms_features_v32, then fits (ARMS_A on ds4, ARMS_B on ds4shift; defaults
  = baseline + fo_long / fo_pa / fo_duty). New day set: override DAYS / ADDON_DAYS / TRAIN_* / TEST_* / PFX_* (see its header). run.sh does the work.
  The box memcg limit is 14 GB: pipeline_lite OOMs at `--parallel 4`. Never run a prep or smoke test next to it by hand.
- Splits: ds4 = train 09-15..09-26, test 09-27 Sun / 09-28 Mon / 09-29 Tue, seed 42 (train 2.21M / test 1.61M rows);
  ds4shift = train 09-15..09-25, test 09-26 Sat / 09-27 / 09-28, seed 11 (2.06M / 1.42M). Train 1% of snapshots/day, test 3%.
  09-08..09-14 only feed the 14-day history tables. Reproduction controls: ds4 baseline 113.946, sc_big_fo_long 85.65, sc_big_fo_pa 85.197, sc_big_fo_duty 84.770.
- Fits: ~5 min baseline, ~12–15 min for 255-leaf arms. Background loop refreshes LOCK every 50 min (runs last > 3 h).
- `sh` reads scripts incrementally: never edit a runner while it runs. run.sh exits on PREP FAIL but ds.sh still prints "DS ALL DONE".

### Current hand-off recipe: `sc_big_fo_long` (run 29)
ds4 113.95 → 85.65 (−24.8%), p90 241.6 → 177.1; ds4shift 109.95 → 84.20 (−23.4%), p90 228.7 → 170.6; every stops_ahead bucket and every test day better.
It is sc_big_dwell (the protocol winner, run 14, confirmed on ds1 / ds1shift / ds3 / ds3shift / ds4 and at 3x rows) plus small additive gains:
- Base (arms.py `sc_big_dwell`): production FEATURE_COLS − {month, day_of_week, stop_sequence}; route_id and target-stop code (top-254 stops) as native
  categoricals; 255 leaves / min_samples_leaf 50, lr 0.05, 1200 iters; sentinel −1 for missing live values; plus
  1. live link store: per stop-pair link the last 5 traversals (≤ 30 min, 30 s detect lag) → path sums median-of-3/5, coverage, age, speed (`_STACK`);
  2. own speed over 60/180/300 s (5-min position ring);
  3. historical link × hour median tables (all days + weekday/weekend), prior 14 days (`_DT`), built at export like the priors;
  4. current-link live m5 / hist / fraction left (`_CUR`); 5. own previous-lap link times, 3 h window (`_LAP`);
  6. vehicle-level own/hist link-time ratio over the last hour (`_VEH`);
  7. conditional remaining dwell at (last passed stop, 50 m bin) given stationary_sec, prior 14 days (`_DWELL`).
- + max_features 0.3 (runs 23–25: −0.4 to −0.65% on 3 splits; training-only).
- + GPS-fix age cols (run 26, addon_v26: pos_age_s [+ 600 s median, × speed]; −0.3 to −0.6%). pos_age_s alone is enough (run 28 ablation).
- + feed-reported speed / odometer (run 28, addon_v28: fs_now, fs_mean60, odo_spd_60/180, odo_m_300; −1.1 / −1.6%).
- + long odometer windows (run 29, addon_v29 `_ODOL`: odo_spd_600/1200, fs_zero_frac600; −0.7 / −1.0%).
**Serving status: needs serving work, ≈ 3 days** (live link store + position ring + per-vehicle crossing log persisted in `feed/tracker_state.json`
across the 5-min push-feed restarts; export of categorical bitsets / missing_go_to_left; static hist + dwell tables at export; a per-vehicle 1200 s ring
of (feed ts, vehicle ts, speed, odometer)). Timing (run 19): 255 leaves + bitsets = 1.90 s per 3000-row batch single-threaded (baseline 0.87 s),
export ~29 MB, fits the 10 s cycle. Cheaper fallbacks (ds4, mf3 base): 127 leaves +0.6%, 600 trees at lr 0.1 +1.1% — both still > 20% better than baseline.
Productionising is the owner's call.
**Run 30 optional add-on: `sc_big_fo_pa`** = fo_long + other vehicles on the path links right now (addon_v30): ds4 85.65 → 85.20 (−0.53%), ds4shift 84.20 → 83.95 (−0.30%),
every bucket better; cheap to serve (one per-cycle link → vehicles dict). Ablation: dropping `_LAP` + `_VEH` costs +1.7 / +2.0%, so the crossing log stays.
**Run 32 optional add-on: `sc_big_fo_duty`** = fo_pa + shift time / odometer since the vehicle's last ≥ 30 min feed gap (addon_v32 `_DUTY`): ds4 85.20 → 84.77 (−0.50%),
ds4shift 83.95 → 83.62 (−0.39%), every bucket and p90 better on both splits; trivial to serve (one shift-start (t, odometer) per vehicle).

### What won (in order; each Δ is vs the previous leader)
- live link path times: −7 to −9% vs baseline (run 1). + own speed, − headway: −8.3 / −9.0% (run 2).
- median of last 3/5 traversals, drop calendar cols (`stack`): −12.4 / −14.5% (run 3).
- historical link × hour table: ≈ −2% (run 4). weekday/weekend table + route_id categorical: ≈ −2.3% (run 6).
- target-stop categorical + 255 leaves: −1.6% (run 7). current-link cols: −0.6 to −1.1% (run 8).
- own previous lap: −1.2 to −1.5% (run 11). vehicle own/hist ratio: −1.0 to −1.2% (run 12). location dwell table: −0.8% (run 14).
- max_features 0.3: −0.4 to −0.65% (runs 23–25). fix-age cols: −0.3 to −0.6% (run 26).
- feed speed + odometer: −1.1 / −1.6% (run 28). odometer 600/1200 s windows: −0.7 / −1.0% (run 29). vehicles on the path now: −0.5 / −0.3% (run 30).
- shift duty (time / odometer since shift start): −0.50 / −0.39% (run 32).

### Ruled out (one line each)
- Segment-additive link-time model: loses to direct HGBT at every horizon (repo history). Lookup-table bias correction: hurts MAE.
- Headway via since_last_at_target; same-route headway to the leader (+0.3%, run 15): noise.
- NaN for missing live values: worse than sentinel −1 (exported trees route NaN right) (run 1).
- Cold-start dropout: costs warm accuracy; persist the live state instead (runs 2–3).
- lr 0.1 / 0.08, min_samples_leaf 20 / 100, l2 1, leaf100, no hour weights: neutral (runs 2–24). Loss: prod already uses absolute_error.
- m7, EWMA per link, p75 table, live/hist ratio, fill path: ≤ 0.3% (runs 4, 6).
- Own-vs-hist ratio keyed by trip, recency weighting, pruning 3 cols: noise/loss (run 7).
- Residual target (base ETA + tree) and log-ratio target: no gain (run 7). Next-stop categorical: noise (run 8).
- 127 leaves for the lap model; 6 h lap window, 20 min / clipped vehicle ratio, network-wide 15-min ratio: loss / ±0.2% (runs 11–13).
- Dwell table by 3-hour band, live same-location last-3 dwell, stop-keyed dwell fallback: +0.1–0.2% (runs 14–15).
- Longer train window (19 vs 12 days): data lever, relative gain unchanged (run 16). History lookback: 14 d best (run 17).
- Monotone constraints on distance / stops_ahead / prior ETAs: +0.24% (run 18).
- 511 leaves / 2400 iters: −0.5 to −0.6% but 2.6x serving time (runs 18–19). Bagging 2 fits: −0.8 to −1.0%, 2x trees (run 24). Optional only.
- Recency weighting inside the 14-day tables: +0.1% (run 19).
- Trip-keyed holds (`sc_big_hold`): ±0.5%, p90 worse; code removed in run 23. Routes 133 / 137 / 122 errors are driver breaks at fixed places / clock times;
  need operator break data (runs 20–23 diagnostics: `diag_route.py`, `diag_holds.py`).
- Fix-clock link timing (MS_AGE_CORR=1, run 27): −0.38% alone but does not stack with the age cols.
- Reported-speed trajectory (V28b: lag, accel, max, time since moving, 30 s odometer speed): −0.19 / −0.07%, sa1 worse (run 28).
- Odometer-vs-shape progress gap (V29 `_ODOG`): −0.29% alone, doesn't stack with `_ODOL` (run 29).
- Dropping the per-vehicle crossing log (`_LAP` / `_VEH`) now that the odometer exists: +0.6 / +1.1%, both +1.7 / +2.0% (run 30). Keep them.
- Occupied-link "now" path time + slow-vehicle gap (V31 `_PAN`): −0.13 / −0.07% vs fo_pa (run 31). 40/60 min odometer (`_ODOX`): −0.19 / −0.49%, sa1 +0.27% on ds4 (run 31).
- Own break history (V32 `_BRK`: time / odometer since the last ≥ 3 / ≥ 10 min stop, its length, 3 h count): +0.27 / −0.06% (run 32). Does not touch 133 / 137.
- Pace target (s per metre of remaining_dist_m, weight × distance so the loss stays L1 in seconds; `sc_big_duty_pace`): +4.1 / +4.7%, sa1 +21% (run 33). Short-horizon dwell / stop time blows up per-metre targets.
- Stacking 40/60 min odometer onto duty (`sc_big_fo_all`): −0.08 / −0.34%, noise (run 33). The optional add-ons have stopped stacking.
- Arms removed in run 35 (lost on both splits / superseded): sc_big_mf3_age_fo2 + addon_v28b.py, sc_big_fo_gap / fo_v29, sc_big_fo_pan / fo_v31, sc_big_fo_brk / fo_v32, sc_big_duty_pace.
- Odometer-based crossing times in the link store: reasoned out (run 31). Snapshots are ~10 s apart and both odometer and dist_along interpolate linearly in between, so crossing times barely move.

### Production status (run 41)
- **Main shipped the hand-off as `live_v2`** (commits 6fa9bd6..35c8927, 2026-10-04): `src/live_features.py` (one `LiveState` engine for training replay and
  serving), `src/train_live.py` (255 leaves, route / target-stop categoricals, max_features 0.3, absolute error), `predict_live` in src/inference.py,
  state carried across restarts in `feed/live_state.pkl.gz`, weekly retrain. Its 36 live cols = fo_long (no `_PA` / `_DUTY`).
  README held-out: train 09-19..09-30, test 10-01..10-03, legacy 119.3 → live_v2 92.8 (−22.2%), p90 256.1 → 192.3.
- **New protocol baseline = `python -m src.train_live`** on the same days (not `src.train`). Future arms should be built on src/live_features.py +
  src/train_live.py (behind flags), not on this branch's harness/addon chain, so that a winner is already in serving form.
- Against that baseline, the known optional add-ons (`_PA` −0.3/−0.5%, `_DUTY` −0.4/−0.5%) cannot reach the 3% win bar even stacked.

### Open next steps
1. Hand-off of `sc_big_fo_long` (+ optional V30 `sc_big_fo_pa`, + optional V32 duty `sc_big_fo_duty`) is the owner's call. **Strongly recommend pausing the schedule**: run 33 found nothing; runs 16–32 moved the leader by ≈ 4.5% in total,
   and runs 31–32 each added ≤ 0.5%. Live-feature ideas from the feed (positions, odometer, speed, other vehicles) look exhausted.
2. What is left needs new data, not features: operator break / layover schedules (routes 133 / 137 / 122 dominate the tail), or more training days (data lever, run 16).
3. Fresh-day check: once ≥ 10-12 is in R2, build ds5 with ds.sh (e.g. DAYS 09-22..10-11, train 09-29..10-08, test 10-09 Fri / 10-10 Sat / 10-11 Sun;
   shift split train ..10-07, test 10-08..10-10) and re-check baseline vs fo_long / fo_pa / fo_duty on unseen days. Until then runs exit early.
4. Cleanup debt: addon_v29 / v31 / v32 still compute the pruned cols (_ODOG, _PAN, _BRK). Strip them on the next full rebuild, when the controls
   above can confirm the kept cols are bit-identical (not done blind in run 35).

## 2026-10-04 — run 38 (early exit, no fresh day set)

Lock taken at 12:14 UTC. Main had not moved. R2 credentials present. Newest possible complete day is 10-03; a fresh split needs ≥ 10-12 (state step 3),
so no experiment, no result rows, no notification. No housekeeping due (last done run 35).

### Next steps
1. Unchanged: pause the schedule (owner's call), or keep exiting early until ≥ 10-12 is in R2 (earliest useful run: 2026-10-12 UTC); then ds5 via ds.sh.

## 2026-10-04 — run 39 (early exit, no fresh day set)

Lock taken at 16:14 UTC. Main had not moved. R2 credentials present. Newest possible complete day is 10-03; a fresh split needs ≥ 10-12 (state step 3),
so no experiment, no result rows, no notification. Housekeeping due next run (run 40, 5th since run 35).

### Next steps
1. Unchanged: pause the schedule (owner's call), or keep exiting early until ≥ 10-12 is in R2 (earliest useful run: 2026-10-12 UTC); then ds5 via ds.sh.

## 2026-10-04 — run 40 (housekeeping; early exit, no fresh day set)

Lock taken at 20:13 UTC. Main had not moved. R2 credentials present. Newest possible complete day is 10-03; a fresh split needs ≥ 10-12 (state step 3),
so no experiment, no result rows, no notification.
**Housekeeping** (5th run since run 35, own commit): runs 33–37 → `archive/JOURNAL-runs-33-37.md`; arms.py drops sc_big_fo_novl and sc_big_fo_pax
(each lost on ds4 and ds4shift; nothing imports them; _ODOX kept for sc_big_fo_all; report.py keeps their descriptions for the archived rows).
Only three .sh files remain (ds.sh, run.sh, queue.sh), so no runner pruning. Checks: `python report.py` regenerates LEADERBOARD.md unchanged;
baseline and sc_big_fo_duty fit and predict on a synthetic 3000-row frame (no data on this box).

### Next steps
1. Unchanged: pause the schedule (owner's call), or keep exiting early until ≥ 10-12 is in R2 (earliest useful run: 2026-10-12 UTC); then ds5 via ds.sh.
2. Next housekeeping due run 45.

## 2026-10-05 — run 41 (early exit; hand-off landed on main)

Lock taken at 00:14 UTC. R2 credentials present. Main moved: it now ships `live_v2` = this branch's `sc_big_fo_long` (see "Production status").
Merged origin/main into the branch (clean). No experiment: the newest complete day is 10-04, and every fresh-split idea now has to beat live_v2 by ≥ 3%,
which no recorded candidate does. No result rows, no notification.

### Next steps
1. **Recommend pausing the schedule** (owner's call): the search's hand-off is in production, and the remaining ideas need new data (operator break
   schedules for 133 / 137 / 122) rather than features.
2. If it continues: from ≥ 10-12, build a fresh day set with the production pipeline (`make pipeline`; `python -m src.train_live --train … --test …`)
   to measure live_v2 on unseen days, then try `_PA` + `_DUTY` ported into src/live_features.py behind a flag (expected ≈ −1%, below the win bar;
   only worth it as a cheap optional add-on).
3. Next housekeeping due run 45.
