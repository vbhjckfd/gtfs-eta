# Model-search journal

Research-only search for a better ETA model. Branch `claude/model-search`.
Harness: `harness.py` (prep/run), arms in `arms.py`, live features in
`features_live.py` (+ add-on scripts `addon_v*.py` that append cols to an existing
feature dir). Results: one compact JSON line per fit in `results.jsonl` (keyed by `name` = arm_tag_sSEED; written by `harness.save_result`); `report.py` regenerates
LEADERBOARD.md (current tags) and archive/LEADERBOARD-old.md (superseded tags).
Full run entries are archived in `archive/JOURNAL-runs-1-12.md`, `-13-16.md`, `-17-18.md`, `-19-20.md`, `-21-23.md`, `-24-27.md`, `-28-32.md`.

## State of the search (updated run 35)

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

### Open next steps
1. Hand-off of `sc_big_fo_long` (+ optional V30 `sc_big_fo_pa`, + optional V32 duty `sc_big_fo_duty`) is the owner's call. **Strongly recommend pausing the schedule**: run 33 found nothing; runs 16–32 moved the leader by ≈ 4.5% in total,
   and runs 31–32 each added ≤ 0.5%. Live-feature ideas from the feed (positions, odometer, speed, other vehicles) look exhausted.
2. What is left needs new data, not features: operator break / layover schedules (routes 133 / 137 / 122 dominate the tail), or more training days (data lever, run 16).
3. Fresh-day check: once ≥ 10-12 is in R2, build ds5 with ds.sh (e.g. DAYS 09-22..10-11, train 09-29..10-08, test 10-09 Fri / 10-10 Sat / 10-11 Sun;
   shift split train ..10-07, test 10-08..10-10) and re-check baseline vs fo_long / fo_pa / fo_duty on unseen days. Until then runs exit early.
4. Cleanup debt: addon_v29 / v31 / v32 still compute the pruned cols (_ODOG, _PAN, _BRK). Strip them on the next full rebuild, when the controls
   above can confirm the kept cols are bit-identical (not done blind in run 35).

## 2026-10-03 — run 33 (stack the optional add-ons; pace target)

Lock taken at 12:14 UTC. Main was already merged. No housekeeping (journal 334 lines; due run 35).
No new day set yet: data ends 10-02 and a fresh split needs a new weekend in its test days (≥ 10-05; ideally ≥ 10-11 per run 32), so this run stays on ds4 / ds4shift.
**Ideas** (both training-only, zero serving cost beyond sc_big_fo_duty; arms in arms.py, feature dir ms_features_v32, tag v33):
- `sc_big_fo_all` = sc_big_fo_duty + _ODOX (40 / 60 min odometer, run 31). Do the optional add-ons stack?
- `sc_big_duty_pace` = sc_big_fo_duty with target y / max(remaining_dist_m, 50) (s per metre), sample weight = prod hour weight × that distance
  (so the absolute_error objective, early-stopping score included, is still L1 in seconds), prediction = pace × distance. Untested target transform from the routine's list
  (log and residual / log-ratio targets were tried in runs 1 and 7).

Commands: `python research/model-search/pipeline_lite.py --parallel 3 --days 2026-09-08..2026-09-29` (12:20–≈14:30) next to `sh research/model-search/run33.sh`
(prep v10, addons v26, v28–v32, fits ds4 / ds4shift; fits 15:00–16:25). No container restarts this time.
Reproduction: ds4v33 baseline 113.946 and sc_big_fo_duty 84.770 equal ds4v32; ds4shiftv33 sc_big_fo_duty 83.623 equals ds4shiftv32. ds4shift baseline = ds4shiftv25 (same days, same code).

### Results (MAE / median / p90 / bias; ds4 s42 train 2.21M / test 1.61M; ds4shift s11 train 2.06M / test 1.42M)
| arm | ds4 | Δ vs duty | sa1 / sa5 / sa10 | Sun / Mon / Tue | ds4shift | Δ vs duty | sa1 / sa5 / sa10 | Sat / Sun / Mon |
|---|---|---|---|---|---|---|---|---|
| baseline | 113.95 / 60.4 / 241.6 / −19.3 | | 73.4 / 116.2 / 157.8 | 104.5 / 115.5 / 118.5 | 109.95 / 55.9 / 228.7 / −23.5 (v25) | | 73.3 / 111.8 / 152.7 | 103.6 / 106.8 / 116.5 |
| sc_big_fo_duty | 84.77 / 43.2 / 174.6 / −13.7 | | 58.4 / 85.7 / 119.0 | 78.9 / 85.7 / 87.7 | 83.62 / 41.5 / 168.7 / −16.0 | | 58.6 / 84.3 / 117.0 | 81.8 / 80.9 / 86.7 |
| sc_big_duty_pace | 88.24 / 44.1 / 177.8 / −17.9 | +4.09% | 70.6 / 87.5 / 119.7 | 83.3 / 88.8 / 90.9 | 87.55 / 42.5 / 172.7 / −20.5 | +4.69% | 71.2 / 86.6 / 118.4 | 85.4 / 85.8 / 90.2 |
| sc_big_fo_all | 84.70 / 43.2 / 174.3 / −13.7 | −0.08% | 58.2 / 85.7 / 119.0 | 78.7 / 85.7 / 87.7 | 83.34 / 41.6 / 168.5 / −15.1 | −0.34% | 58.3 / 83.9 / 116.7 | 81.3 / 80.5 / 86.6 |

Bucket Δ (sa1..10) vs duty:
- pace ds4: +20.96 +9.07 +5.55 +3.53 +2.06 +1.58 +1.11 +0.81 +0.60 +0.59%. ds4shift: +21.43 +10.01 +6.28 +4.14 +2.77 +1.93 +1.54 +1.34 +0.89 +1.22%.
- all ds4: −0.34 −0.22 −0.09 −0.13 −0.02 +0.16 −0.10 −0.07 −0.06 −0.05%. ds4shift: −0.56 −0.59 −0.40 −0.34 −0.46 −0.39 −0.27 −0.01 −0.28 −0.19%.

Worst routes:
- duty ds4: 137 223, 133 188, 118 167, 122 160, 106 141, 2302 132, 2460 130, 111 125, 143 124, 131 124. ds4shift: 133 253, 137 209, 122 165, 106 152, 143 150, 2460 129, 1630 124, 111 120, 2302 118, 125 112.
- all ds4: 137 223, 133 185, 122 162, 118 159, 106 140, 2460 130, 131 126, 111 125, 143 124, 2302 124. ds4shift: 133 250, 137 209, 122 170, 143 150, 106 143, 2460 129, 1630 126, 111 121, 2302 116, 125 114.
- pace ds4: 137 249, 133 196, 118 187, 122 162, 106 151, 2460 133, 111 133, 131 132, 2302 130, 143 122.

**Conclusion: no protocol win, so no notification.**
- The pace target loses clearly (+4.1 / +4.7%), and the loss is concentrated at short horizons (sa1 +21%, sa10 +0.6 / +1.2%) with a more negative bias.
  Near the target stop, the remaining time is mostly dwell / stopped time, so seconds per metre explodes and the trees can't tile it. Direct seconds stays.
  (A hybrid, pace only for remaining_dist_m > ~1 km, would at best recover the far buckets' ≤ 1%; not worth it.)
- Stacking the 40/60 min odometer onto duty: −0.08 / −0.34%, within noise. The optional add-ons no longer stack.
- Hand-off recipe unchanged: sc_big_fo_long, optionally + V30 (`sc_big_fo_pa`) + V32 duty (`sc_big_fo_duty`).

### Next steps
1. **Pause the schedule** (owner's call). Feature, target and loss levers on ds4 are exhausted; runs 30–33 each added ≤ 0.5% or lost.
2. If it continues: wait for a fresh day set (≥ 10-11, two new weekends) to re-check the hand-off recipe on new days; until then runs should exit early
   rather than rebuild ds4 for more ≤ 0.5% ideas.
3. Run 35 = housekeeping. Prune candidates: sc_big_duty_pace (lost on both splits), _BRK / sc_big_fo_brk / sc_big_fo_v32, sc_big_fo_pan / _PAN,
   addon_v28b / sc_big_mf3_age_fo2, sc_big_fo_gap / fo_v29 / _ODOG; run28b.sh, run29.sh, run30.sh, run31.sh.

## 2026-10-03 — run 34 (early exit: waiting for a fresh day set)

Lock taken at 20:12 UTC. Main was already merged. R2 credentials present. No housekeeping (journal 381 lines; due run 35).
Per run 33 next step 2: the newest complete day is 10-02, and a fresh split needs ≥ 10-11 (two new weekends among the held-out days). Rebuilding ds4
(≈ 2.5 h pipeline) for another ≤ 0.5% idea isn't worth it, so no experiment this run. No result rows, no notification.

### Next steps
1. Unchanged: pause the schedule (owner's call), or keep exiting early until ≥ 10-12 is in R2.
2. Then build ds5 (e.g. train 09-22..10-07, test 10-09..10-11 + shift split) and re-check baseline vs sc_big_fo_long / fo_pa / fo_duty on fresh days.
3. Run 35 = housekeeping (prune list in run 33 step 3).

## 2026-10-04 — run 35 (housekeeping; early exit, no fresh day set)

Lock taken at 00:14 UTC. Main had not moved. R2 credentials present. Newest complete day is 10-03, so no fresh split yet (needs ≥ 10-12); no experiment, no notification.
**Housekeeping** (5th run since run 30, own commit):
- Runs 28–32 → `archive/JOURNAL-runs-28-32.md`; state section updated (runner, controls, removed arms, next steps 3–4).
- run28.sh, run28b.sh, run29–33.sh replaced by one parameterised `ds.sh` (defaults = ds4 / ds4shift, feature chain → ms_features_v32, hand-off arms).
- arms.py: removed sc_big_mf3_age_fo2, sc_big_fo_gap, sc_big_fo_v29, sc_big_fo_pan, sc_big_fo_v31, sc_big_fo_brk, sc_big_fo_v32, sc_big_duty_pace
  (and _FS2 / _ODOG / _PAN / _BRK / pace helpers); deleted addon_v28b.py (not in the feature chain). Their results stay in results.jsonl.
- report.py: ds4v28 / ds4v31 tags → archive/LEADERBOARD-old.md; added tag definitions for v31–v33; fixed missing baseline aliases, so ds4shiftv31–v33 rows now show Δ vs ds4shiftv25.
- Checks: `python research/model-search/report.py` regenerates both tables; the baseline and sc_big_fo_duty arms fit and predict on a synthetic 3000-row frame
  (no data on this box). `sh -n ds.sh` OK.

### Next steps
1. Unchanged: pause the schedule (owner's call), or keep exiting early until ≥ 10-12 is in R2; then ds5 via ds.sh (state step 3).
