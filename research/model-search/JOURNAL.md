# Model-search journal

Research-only search for a better ETA model. Branch `claude/model-search`.
Harness: `harness.py` (prep/run), arms in `arms.py`, live features in
`features_live.py` (+ add-on scripts `addon_v*.py` that append cols to an existing
feature dir). Results: one compact JSON line per fit in `results.jsonl` (keyed by `name` = arm_tag_sSEED; written by `harness.save_result`); `report.py` regenerates
LEADERBOARD.md (current tags) and archive/LEADERBOARD-old.md (superseded tags).
Full run entries are archived in `archive/JOURNAL-runs-1-12.md`, `-13-16.md`, `-17-18.md`, `-19-20.md`, `-21-23.md`, `-24-27.md`, `-28-32.md`, `-33-37.md`, `-38-43.md`.

## State of the search (updated run 45)

### Protocol set-up (since run 45: production trainer only)
- **Baseline = `python -m src.train_live`** (live_v2, the production recipe) on the same rows. The harness/addon chain of runs 1–43
  (harness.py, arms.py, features_live.py, addon_v*.py, pipeline_lite.py, ds.sh, run.sh, …) moved to `archive/harness-v1/` in run 45:
  it imported `src.train`, which main removed with the legacy model, so it no longer runs. Its results stay in `results.jsonl`
  (`report.py` reads them through `resultslib.py`); production-trainer fits go to `results_tl.jsonl` via `tl.sh`.
- Fresh box: `python -m pip install -e . tzdata` (without tzdata every pipeline day fails; plain `pip` may be another interpreter).
- Data: `python scripts/run_pipeline.py --all --parallel 3 --keep-pct 10 --since A --until B` (≈ 8 min/day wall at 3 workers; 14 GB memcg,
  `--parallel 4` OOMs). Need 14 prior days before the first train day for the history tables.
- Arms: flags in src/live_features.py / src/train_live.py (`GTFS_ETA_LIVE_EXTRA`, `GTFS_ETA_LIVE_DROP`); one fit per `tl.sh` call
  (≈ 34 min at 5M train rows). Use one feature cache for all arms so rows are identical.
- Harness-era splits and controls (ds1..ds4, runs 1–43) are in the archived journals; they cannot be re-run without the archived code.

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
- Residual feedback (run 45, `rfb.py`): the model's own matured errors per vehicle (600 / 1800 s) and route (600 s) as features: +0.08% MAE, +0.2% p90 on split C. Live state already explains them.
- Odometer-based crossing times in the link store: reasoned out (run 31). Snapshots are ~10 s apart and both odometer and dist_along interpolate linearly in between, so crossing times barely move.

### Production status (run 41)
- **Main shipped the hand-off as `live_v2`** (commits 6fa9bd6..35c8927, 2026-10-04): `src/live_features.py` (one `LiveState` engine for training replay and
  serving), `src/train_live.py` (255 leaves, route / target-stop categoricals, max_features 0.3, absolute error), `predict_live` in src/inference.py,
  state carried across restarts in `feed/live_state.pkl.gz`, weekly retrain. Its 36 live cols = fo_long (no `_PA` / `_DUTY`).
  README held-out: train 09-19..09-30, test 10-01..10-03, legacy 119.3 → live_v2 92.8 (−22.2%), p90 256.1 → 192.3.
- **New protocol baseline = `python -m src.train_live`** on the same days (not `src.train`). Future arms should be built on src/live_features.py +
  src/train_live.py (behind flags), not on this branch's harness/addon chain, so that a winner is already in serving form.
- Against that baseline, the known optional add-ons (`_PA` −0.3/−0.5%, `_DUTY` −0.4/−0.5%) cannot reach the 3% win bar even stacked.

### Run 44 (production trainer, fresh days 10-01..10-13)
- live_v2 holds on unseen days: split A (test 10-10..10-13) legacy 126.7 → live_v2 99.0 (−22%), p90 270.2 → 201.4.
- `_DUTY` + `_PA` ported into src/live_features.py behind GTFS_ETA_LIVE_EXTRA (off by default): −1.1% on A and B, p90 −1.5 / −1.2%;
  `_PA` alone −0.9%, `_DUTY` alone −0.2%. Below the 3% bar → optional only (needs: duty fields in the persisted state, per-cycle occupancy dict).

### Data-quality flag (run 45) — for the owner
- **Duplicated raw snapshots 10-12 (from ≈ 09 UTC), 10-13, 10-14**: R2 `raw/` holds ≈ 720 objects/h instead of ≈ 360 (11193 / 14336 / 13833 per day
  vs ≈ 7190), as if a second collector ran. The pipeline then emits each vehicle-snapshot twice for ≈ 19% of 10-12's rows (1.22M of 3.14M); the
  second copy has `progress_speed_mps = -1` (zero dt), the other columns are identical. That is the "row-count jump" run 44 saw. Back to normal from 10-15.
  The weekly retrain over 10-12..10-14 trains on these (over-weighted days + fake "speed unknown" rows). Suggest de-duplicating snapshots by feed
  header timestamp in src/snapshots.py, and checking what wrote the extra objects. Not fixed here (research branch, no production changes).

### Open next steps
1. Hand-off of `sc_big_fo_long` (+ optional V30 `sc_big_fo_pa`, + optional V32 duty `sc_big_fo_duty`) is the owner's call. **Strongly recommend pausing the schedule**: run 33 found nothing; runs 16–32 moved the leader by ≈ 4.5% in total,
   and runs 31–32 each added ≤ 0.5%. Live-feature ideas from the feed (positions, odometer, speed, other vehicles) look exhausted.
2. What is left needs new data, not features: operator break / layover schedules (routes 133 / 137 / 122 dominate the tail), or more training days (data lever, run 16).
3. Fresh-day check: once ≥ 10-12 is in R2, build ds5 with ds.sh (e.g. DAYS 09-22..10-11, train 09-29..10-08, test 10-09 Fri / 10-10 Sat / 10-11 Sun;
   shift split train ..10-07, test 10-08..10-10) and re-check baseline vs fo_long / fo_pa / fo_duty on unseen days. Until then runs exit early.
4. Cleanup debt: addon_v29 / v31 / v32 still compute the pruned cols (_ODOG, _PAN, _BRK). Strip them on the next full rebuild, when the controls
   above can confirm the kept cols are bit-identical (not done blind in run 35).

## 2026-10-14 — run 44 (fresh days on the production trainer; no winner)

Lock taken 13:01 UTC (refreshed every 50 min by a background loop). Main had moved (worker-only commits #80 / #81); merged cleanly.
R2 credentials present. Fresh box: `python -m pip install -e . tzdata` (plain `pip` points at another interpreter here).

**Data.** `python scripts/run_pipeline.py --all --parallel 3 --keep-pct 10 --since 2026-09-17 --until 2026-10-13` — 27 days in 3 h 40 min
(16–38 min per day per worker; peak ≈ 9 GB used, no OOM). Drift: 10-12 / 10-13 have 3.1M / 3.9M pipeline rows vs ≈ 2.0M on earlier weekdays
(test rows on those days are 2–3x the others; worth a look at the feed — more vehicles or a faster snapshot cadence?).

**Code** (flag-gated; production unchanged with the flag off):
- src/live_features.py: `GTFS_ETA_LIVE_EXTRA=1` appends `EXTRA_COLS` = duty_s, duty_odo (feed t / odometer since the last ≥ 30 min feed gap)
  and pa_n, pa_spd_mean, pa_spd_min, pa_stop_frac, pa_gap_m, pa_gap_spd, pa_cov (other vehicles whose current link is on the path at this
  snapshot, 120 s odometer speed). Ports of addon_v32 `_DUTY` / addon_v30 `_PA` onto LiveState; dump/load do not carry the duty fields yet.
- src/train_live.py: `GTFS_ETA_LIVE_DROP=a,b` leaves live cols out of the model (MODEL_LIVE_COLS).
- tl.sh (one fit on src.train_live → results_tl.jsonl; all arms read one EXTRA=1 feature cache data/features_live_x so rows are identical),
  run44.sh (this run's six fits); report.py now also renders a "Production trainer" section from results_tl.jsonl.

**Splits.** A = train 10-01..10-09, test 10-10 Sat / 10-11 Sun / 10-12 Mon / 10-13 Tue, seed 42 (train 4.94M rows, test 2.90M).
B = train 10-01..10-08, test 10-09 Fri / 10-10 / 10-11, seed 11 (4.34M / 1.38M). Baseline = production `live_v2` (`lv2`) on the same rows.
Fits ≈ 34 min each (5M rows, 1200 trees, 4 cores); legacy ≈ 20 min.

| split | arm | MAE | median | p90 | bias | sa1 | sa10 | ΔMAE | Δp90 | worst bucket |
|---|---|---|---|---|---|---|---|---|---|---|
| A | lv2 (prod) | 99.0 | 45.9 | 201.4 | −28.5 | 65.3 | 136.7 | — | — | — |
| A | legacy | 126.7 | 60.8 | 270.2 | −41.3 | 79.8 | 171.6 | +28.0% | +34.1% | +30.7% |
| A | lv2x_duty | 98.8 | 45.9 | 201.3 | −28.5 | 65.0 | 136.6 | −0.2% | −0.1% | −0.0% |
| A | lv2x_pa | 98.1 | 45.5 | 198.7 | −27.8 | 64.8 | 135.8 | −0.9% | −1.3% | −0.6% |
| A | lv2x_dutypa | 98.0 | 45.5 | 198.4 | −28.2 | 64.7 | 135.3 | −1.1% | −1.5% | −0.9% |
| B | lv2 (prod) | 96.6 | 44.5 | 199.9 | −33.0 | 62.3 | 136.3 | — | — | — |
| B | lv2x_dutypa | 95.5 | 44.2 | 197.5 | −32.4 | 61.7 | 135.6 | −1.1% | −1.2% | −0.5% |

Per day (A, lv2 → dutypa): 10-10 97.1 → 97.0, 10-11 91.6 → 90.5, 10-12 89.7 → 88.3, 10-13 109.7 → 108.6. Legacy per day 115.9 / 122.4 / 120.6 / 136.8.
Worst routes (A, lv2): 133 187, 137 166, 122 152, 106 140, 2302 140, 88 137. On B, 133 is 255 (lv2) / 262 (dutypa) — driver breaks, unchanged.

**Conclusion.** live_v2 reproduces its −22% vs legacy on fresh days (README: −22.2%). The two optional add-ons together give −1.1% on both splits,
every bucket and p90 better, consistent with the harness (−1.0%), but far below the 3% bar: not a winner, no notification.
`_PA` carries most of it; `_DUTY` alone is noise on the production trainer. Worth shipping only as a cheap optional add (< 0.5 day: persist
duty_t0 / duty_odo0 in LiveState.dump/load; inference needs no other change since it calls LiveState.features).

### Next steps
1. **Recommend pausing the schedule, or running it weekly** (owner's call; asked in this session). Feed-feature ideas are exhausted;
   the tail (133 / 137 / 122) needs operator break / layover data.
2. If continued: re-use this recipe (pipeline 27 days ≈ 3.7 h, feature build ≈ 25 min, six fits ≈ 4 h); the harness/addon chain is superseded.
3. Check the 10-12 / 10-13 row-count jump (≈ 1.6–2x) before the next weekly retrain.
4. Housekeeping due run 45: archive runs 38–44; prune ds.sh / run.sh / harness add-ons superseded by the production-trainer path.

## 2026-10-19 — run 45 (housekeeping; residual feedback on fresh days; no winner)

Lock taken 06:07 UTC (background refresh every 50 min). Main had moved (legacy model removed, New Relic); merged cleanly.
R2 credentials present. **Housekeeping** (own commit): main's removal of `src/train.py` broke harness.py, so the whole harness-era chain moved to
`archive/harness-v1/`; `resultslib.py` now carries `load_results` / `judge` for `report.py` (LEADERBOARD.md regenerates byte-identical);
runs 38–43 → `archive/JOURNAL-runs-38-43.md`; tl.sh lost the removed `--compare-baseline`.

**Data.** `python scripts/run_pipeline.py --all --parallel 3 --keep-pct 10 --since 2026-09-21 --until 2026-10-18` — 28 days in ≈ 6 h:
the pool broke (OOM, the doubled 10-12..10-14 days need more RAM) and retried the last 8 days serially. Use `--parallel 2` when those days are in range.

**Arm `rfb`** (research/model-search/rfb.py): at serving time the daemon knows what it predicted earlier for every stop a vehicle has since reached.
Features = mean signed / relative error of matured predictions (target stop reached ≥ 30 s ago) per vehicle over 600 / 1800 s, the latest one and
its age, and per route over 600 s (10 cols, sentinel −9999). Stage 1 = production `tl.fit`; train-day errors out-of-fold (3 folds by day,
OOF MAE 96.2), test-day errors from the baseline fit on all train days; features computed on all rows before the 5M cap (density matches test).
Command: `python research/model-search/rfb.py --train 2026-10-05..2026-10-14 --test 2026-10-15..2026-10-18 --seed 42 --split C`
(stage-1 predictions cached in data/rfb/C_s42; first attempt OOM-killed in the candidate fit, fixed by dropping the full frame first).

**Split C** = train 10-05..10-14 (5.0M of ≈ 7.0M rows, capped), test 10-15 Thu / 10-16 Fri / 10-17 Sat / 10-18 Sun (1.97M rows), seed 42.

| arm | MAE | median | p90 | bias | sa1 | sa5 | sa10 | ΔMAE | Δp90 | worst bucket |
|---|---|---|---|---|---|---|---|---|---|---|
| lv2 (prod) | 102.2 | 45.4 | 198.2 | −33.5 | 70.3 | 103.4 | 138.9 | — | — | — |
| rfb | 102.3 | 45.4 | 198.6 | −33.5 | 70.4 | 103.6 | 139.0 | +0.08% | +0.19% | +0.23% (sa1..sa10 all within ±0.2%) |

Per day (lv2 → rfb; rows): 10-15 85.3 → 85.6 (611k), 10-16 129.9 → 130.3 (569k), 10-17 79.6 → 79.5 (412k), 10-18 112.4 → 111.8 (381k).
Worst routes (lv2): 122 267, 133 248, 137 172, 106 163, 143 152, 88 133, 881 133, 94 133, 2460 132, 111 131 (rfb: same order, ±3).
Coverage on test: vehicle 600 s window 84%, 1800 s 95%, route 99%.

**Conclusion.** No signal: once the live state (link store, own laps, odometer, dwell table) is in the features, the model's own recent errors
add nothing. Ruled out; no notification. 10-16 (Fri) at 130 s MAE stands out vs 85 s on Thu — worth a look in the quality reports.
Also found the duplicated-snapshot issue above (data-quality flag).

### Next steps
1. **Recommend pausing the schedule** (owner's call): runs 41–45 tested every remaining feed-feature family on the production trainer; best is
   `_PA` + `_DUTY` at −1.1% (optional, below the bar). The tail (122 / 133 / 137) needs operator break / layover data.
2. Owner: de-duplicate raw snapshots (see the data-quality flag) before the next weekly retrain covers 10-12..10-14.
3. Next housekeeping due run 50.
