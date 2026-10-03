# Model-search journal

Research-only search for a better ETA model. Branch `claude/model-search`.
Harness: `harness.py` (prep/run), arms in `arms.py`, live features in
`features_live.py` (+ add-on scripts `addon_v*.py` that append cols to an existing
feature dir). Results: one compact JSON line per fit in `results.jsonl` (keyed by `name` = arm_tag_sSEED; written by `harness.save_result`); `report.py` regenerates
LEADERBOARD.md (current tags) and archive/LEADERBOARD-old.md (superseded tags).
Full run entries are archived in `archive/JOURNAL-runs-1-12.md`, `-13-16.md`, `-17-18.md`, `-19-20.md`, `-21-23.md`, `-24-27.md`.

## State of the search (updated run 33)

### Protocol set-up (fixed)
- Results live in `results.jsonl` (owner asked for this on 2026-09-28, run 20). Where the routine prompt says "keep results/*.json", read it
  as "keep results.jsonl". Never recreate per-fit JSON files; `harness.py run` appends through `save_result`.
- Fresh box: `pip install -e . tzdata` (without tzdata every pipeline day fails).
- Current day set ds4 (data 09-08..09-29): stage 1 = `ARMS_A=" " ARMS_B=" " sh run28.sh` (pipeline_lite `--parallel 3`, ≈ 2–2.5 h; feed-clock prep
  into ms_features_v10 ≈ 2.5 min/day; addons v26 + v28 → ms_features_v28), then `addon_v29.py` → ms_features_v29, then fits via run.sh.
  The box memcg limit is 14 GB: pipeline_lite OOMs at `--parallel 4`. Never run a prep or smoke test next to it by hand.
- Splits: ds4 = train 09-15..09-26, test 09-27 Sun / 09-28 Mon / 09-29 Tue, seed 42 (train 2.21M / test 1.61M rows);
  ds4shift = train 09-15..09-25, test 09-26 Sat / 09-27 / 09-28, seed 11 (2.06M / 1.42M). Train 1% of snapshots/day, test 3%.
  09-08..09-14 only feed the 14-day history tables. Reproduction controls: ds4 baseline 113.946, sc_big_mf3_age_fo 86.278, sc_big_fo_long 85.65.
- Fits: ~5 min baseline, ~12–15 min for 255-leaf arms. Background loop refreshes LOCK every 50 min (runs last > 3 h).
- `sh` reads scripts incrementally: never edit a runner while it runs. run.sh exits on PREP FAIL but callers still print "ALL DONE".

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
- Odometer-based crossing times in the link store: reasoned out (run 31). Snapshots are ~10 s apart and both odometer and dist_along interpolate linearly in between, so crossing times barely move.

### Open next steps
1. Hand-off of `sc_big_fo_long` (+ optional V30 `sc_big_fo_pa`, + optional V32 duty `sc_big_fo_duty`) is the owner's call. **Strongly recommend pausing the schedule**: run 33 found nothing; runs 16–32 moved the leader by ≈ 4.5% in total,
   and runs 31–32 each added ≤ 0.5%. Live-feature ideas from the feed (positions, odometer, speed, other vehicles) look exhausted.
2. What is left needs new data, not features: operator break / layover schedules (routes 133 / 137 / 122 dominate the tail), or more training days (data lever, run 16).

## 2026-10-01/02 — run 28 (feed-reported speed + odometer; feed-clock age ablation)

Lock taken at 20:15 UTC. Main was already merged. No housekeeping (journal 327 lines; the next 5th-run housekeeping falls on run 30). Deleted run26.sh.
**New signal.** I sampled 2026-09-29 raw snapshots. Each vehicle entity carries `position.speed` and `position.odometer` on 100% of rows. Neither is used by training,
src/features or the model search.
- The odometer is in km. Its deltas match the GPS chord distance (ratio p50 1.007), it never decreases, and it doesn't move while stopped (dgps < 5 m → Δodo 0).
- Reported speed is quantised to 1 km/h and correlates only 0.61 with the 60 s position speed, so it is an instantaneous reading.
- `pipeline_lite.py` now also keeps speed + odometer in `<day>.age.parquet`. The training rows are unchanged: baseline 113.946 and mf3_age 87.217 reproduce ds4v26 bit-for-bit.
- `addon_v28.py` (v26 → ms_features_v28):
  - fs_now and fs_mean60 (reported speed);
  - odo_spd_60 and odo_spd_180 (odometer m / fix-clock s);
  - odo_m_300.
- `addon_v28b.py` (v28 → v28b): fs_lag1, fs_acc30, fs_max120, fs_zero_sec, odo_spd_30.
- Arms: sc_big_mf3_age_spd / _odo / _fo (both groups) / _fo2 (+ V28b).
- `run.sh` ADDON now takes a space-separated list.

Commands: `sh research/model-search/run28.sh` (pipeline 09-08..09-29 `--parallel 3`, prep v10, addons v26 + v28, fits), then `sh research/model-search/run28b.sh`.
The container restarted twice, once during the pipeline and once during prep. Re-launching run28.sh resumed both (pipeline and prep skip finished days).
Note: editing run.sh while it runs is unsafe (sh reads scripts incrementally), so fits are not resumable yet. Prune ARMS_A/ARMS_B by hand after a restart.

### Results (MAE / median / p90 / bias; ds4 s42 train 2.21M / test 1.61M; ds4shift s11 train 2.06M / test 1.42M)
| arm | ds4 | Δ vs mf3_age | sa1 / sa5 / sa10 | Sun / Mon / Tue | ds4shift | Δ vs mf3_age | sa1 / sa5 / sa10 | Sat / Sun / Mon |
|---|---|---|---|---|---|---|---|---|
| baseline | 113.95 / 60.4 / 241.6 / −19.3 | | 73.4 / 116.2 / 157.8 | 104.5 / 115.5 / 118.5 | 109.95 / 55.9 / 228.7 / −23.5 (v25) | | 73.3 / 111.8 / 152.7 | 103.6 / 106.8 / 116.5 |
| sc_big_mf3_age | 87.22 / 44.3 / 180.3 / −12.8 | | 60.4 / 88.2 / 121.5 | 81.8 / 88.0 / 89.9 | 86.47 / 42.5 / 174.7 / −15.0 (v26) | | 61.1 / 87.0 / 120.0 | 84.4 / 84.3 / 89.4 |
| sc_big_mf3_age1 | 87.37 / 44.5 / 180.3 / −12.9 | +0.17% | 60.7 / 88.3 / 121.9 | 82.1 / 88.2 / 90.0 | 86.40 / 42.7 / 174.5 / −15.1 | −0.08% | 61.1 / 87.0 / 119.7 | 84.4 / 84.2 / 89.2 |
| sc_big_mf3_age_spd | 86.82 / 43.8 / 179.3 / −13.2 | −0.46% | 59.8 / 87.7 / 121.5 | 81.3 / 87.7 / 89.5 | — | | | |
| sc_big_mf3_age_odo | 86.81 / 44.2 / 179.1 / −13.0 | −0.47% | 59.9 / 87.7 / 121.3 | 81.0 / 87.8 / 89.5 | — | | | |
| **sc_big_mf3_age_fo** | **86.28** / 43.8 / 178.3 / −13.0 | **−1.08%** | 59.2 / 87.2 / 120.9 | 80.6 / 87.3 / 89.0 | **85.05** / 41.9 / 172.3 / −15.1 | **−1.64%** | 59.6 / 85.7 / 118.9 | 83.0 / 82.5 / 88.2 |
| sc_big_mf3_age_fo2 (v28b) | 86.12 / 43.6 / 177.9 / −13.2 | −1.26% | 59.3 / 86.9 / 120.6 | 80.1 / 87.3 / 88.9 | 84.99 / 41.7 / 171.9 / −15.0 | −1.71% | 59.7 / 85.6 / 118.5 | 82.7 / 82.5 / 88.3 |

Bucket Δ (sa1..10), fo vs mf3_age:
- ds4: −2.00 −1.83 −1.50 −1.00 −1.13 −0.93 −0.94 −0.67 −0.66 −0.54%.
- ds4shift: −2.58 −2.47 −2.30 −1.72 −1.57 −1.54 −1.48 −1.20 −1.12 −0.91%.

fo vs mf3 (the previous hand-off without age): −1.70% / −1.92%. fo vs baseline: −24.3% / −22.7%, every bucket at least −18.8% better, p90 241.6 → 178.3 / 228.7 → 172.3.
fo2 vs fo: −0.19% / −0.07%, sa1 +0.20% / +0.28% (worse), so V28b is not worth the extra cols.
spd and odo each give about half of fo, and they stack almost additively.
Worst routes, fo:
- ds4: 137 233, 118 189, 133 187, 122 163, 106 141, 2302 132, 2460 131, 131 129, 111 128, 143 126.
- ds4shift: 133 251, 137 216, 122 171, 106 156, 143 151, 2460 131, 1630 125, 2302 124, 111 121, 125 114.

**Conclusion: no protocol win vs the leader (< 3%), so no notification. This is the largest single gain since run 12 (vehicle ratio) and it is cheap to serve.**
- Both fields are read straight from the feed entity. odo_* needs a per-vehicle ring of (feed ts, vehicle ts, odometer) covering 300 s; fs_mean60 needs the last ~6 speeds.
- The new hand-off recipe is mf3 + V26 age cols + V28. Of the age cols, pos_age_s alone is enough (feed-clock ablation neutral: +0.17% / −0.08%).

### Next steps
1. Hand-off: sc_big_dwell + max_features 0.3 + pos_age_s (+ optional age ring) + V28 cols. The owner decides.
2. The odometer is a projection-free distance. Untested: odometer-based stationary_sec (no GPS jitter), and odometer progress between snapshots to sharpen crossing times in
   the live link store. Both need prep changes (features_live.py), so +1 h prep on the next run.
3. Run 30 = housekeeping (journal > 400 lines after this entry or 5th run, whichever comes first).

## 2026-10-02 — run 29 (odometer: longer windows, odometer vs shape progress)

Lock taken at 04:12 UTC. Main was already merged. No housekeeping (journal 390 lines; run 30 does it, and the journal is now > 400).
**Idea (run 28, next step 2).** The odometer was used only over ≤ 300 s. This run tests two other uses (`addon_v29.py`, v28 → ms_features_v29; arms in arms.py):
- Longer windows (`_ODOL`): odo_spd_600, odo_spd_1200 (odometer m / fix-clock s), fs_zero_frac600 (share of the vehicle's snapshots in 600 s with reported speed 0).
- Odometer vs shape (`_ODOG`): odo_shape_gap300 (odometer m − dist_along_m progress over 300 s, same trip), odo_trip_spd (odometer m / fix-clock s since the vehicle's first
  snapshot on this trip, ≥ 300 s). These would flag detours, loops and projection jumps.
- Coverage on 09-20: odo_spd_600 / 1200 97% / 96%, the gap cols ~88% (rows on trips with a 300 s history). Gap mean −15 m, p95 +37 m.
- Arms: sc_big_fo_long (fo + _ODOL), sc_big_fo_gap (fo + _ODOG), sc_big_fo_v29 (fo + both).
- `odometer stopped for N s` was not tested: run 28's fs_zero_sec (V28b) already covers it and added nothing.

Commands: pipeline_lite 09-08..09-29 `--parallel 3` (04:13–06:19, ≈ 2 h, the fastest yet), then run.sh prep v10 + addons v26, v28 (done 07:12),
then `sh research/model-search/run29.sh` (addon_v29 + fits), then sc_big_fo_long on ds4shift via run.sh. Fits finished ≈ 09:00.
Fix: run29.sh used `${ARMS_A:-…}`, so an empty list meant the defaults. It now uses `${ARMS_A-…}`, so empty means none.
Reproduction: the ds4v29 baseline (113.946) and sc_big_mf3_age_fo (86.278) equal ds4v28 bit-for-bit. ds4shiftv29 is judged against the ds4shiftv28 fo row / ds4shiftv25 baseline.

### Results (MAE / median / p90 / bias; ds4 s42 train 2.21M / test 1.61M; ds4shift s11 train 2.06M / test 1.42M)
| arm | ds4 | Δ vs fo | sa1 / sa5 / sa10 | Sun / Mon / Tue | ds4shift | Δ vs fo | sa1 / sa5 / sa10 | Sat / Sun / Mon |
|---|---|---|---|---|---|---|---|---|
| baseline | 113.95 / 60.4 / 241.6 / −19.3 | | 73.4 / 116.2 / 157.8 | 104.5 / 115.5 / 118.5 | 109.95 / 55.9 / 228.7 / −23.5 (v25) | | 73.3 / 111.8 / 152.7 | 103.6 / 106.8 / 116.5 |
| sc_big_mf3_age_fo | 86.28 / 43.8 / 178.3 / −13.0 | | 59.2 / 87.2 / 120.9 | 80.6 / 87.3 / 89.0 | 85.05 / 41.9 / 172.3 / −15.1 (v28) | | 59.6 / 85.7 / 118.9 | 83.0 / 82.5 / 88.2 |
| sc_big_fo_gap | 86.03 / 43.7 / 178.0 / −13.0 | −0.29% | 59.0 / 86.9 / 120.6 | 80.2 / 87.0 / 88.9 | — | | | |
| **sc_big_fo_long** | **85.65** / 43.6 / 177.1 / −13.5 | **−0.73%** | 58.8 / 86.7 / 120.1 | 79.3 / 86.9 / 88.5 | **84.20** / 41.8 / 170.6 / −14.7 | **−1.00%** | 59.1 / 84.9 / 117.7 | 81.8 / 81.3 / 87.8 |
| sc_big_fo_v29 | 85.71 / 43.6 / 177.0 / −13.2 | −0.66% | 58.8 / 86.7 / 120.2 | 79.1 / 87.0 / 88.7 | 84.35 / 41.8 / 170.6 / −14.6 | −0.83% | 59.0 / 85.0 / 118.0 | 82.0 / 81.5 / 87.9 |

Bucket Δ (sa1..10), fo_long vs fo:
- ds4: −0.76 −0.97 −0.82 −0.83 −0.54 −0.91 −0.72 −0.62 −0.60 −0.60%.
- ds4shift: −0.78 −1.07 −0.99 −0.95 −0.86 −1.00 −1.16 −1.16 −0.99 −0.99%.

fo_long vs baseline: −24.8% / −23.4%, p90 241.6 → 177.1 / 228.7 → 170.6.
Worst routes, fo_long:
- ds4: 137 223, 133 183, 118 178, 122 159, 106 135, 2460 132, 111 128, 2302 127, 131 125, 143 124.
- ds4shift: 133 245, 137 211, 122 166, 143 161, 106 145, 2460 131, 1630 126, 111 120, 2302 119, 125 115.

**Conclusion: no protocol win vs the leader (< 3%), so no notification.** The longer odometer windows are a small, consistent gain on both splits: every day, every bucket and p90 improve.
- The odometer-vs-shape gap helps a little alone (−0.29%) but doesn't stack with the long windows (v29 ≤ long on both splits). Drop it; it would also need dist_along in the ring.
- Serving: the per-vehicle ring from run 28 grows from 300 s to 1200 s (~120 entries per vehicle × ~600 vehicles; trivial memory). No new feed fields.
- **Hand-off recipe (run 29): sc_big_dwell + max_features 0.3 + pos_age_s (+ optional age ring) + V28 cols + _ODOL (arm `sc_big_fo_long`).**

### Next steps
1. Run 30 = housekeeping (journal > 400 lines). Candidates to prune (lost twice or superseded): addon_v28b / sc_big_mf3_age_fo2 (lost once only; keep for now),
   sc_big_fo_gap / _ODOG (lost on ds4, not stacked on either split), run27.sh, run28.sh, run28b.sh (keep the last 2 runners: run28*, run29).
2. Odometer ideas are close to exhausted (instant, 60–1200 s windows, trajectory, shape gap). Untested: an even longer window (2400–3600 s), but the 600 → 1200 step was already small.
3. Still recommend pausing the schedule. Runs 16–29 moved the leader by ≈ 3.5% in total (sc_big_dwell 88.1 → fo_long 85.65 on ds4), and the hand-off is the owner's call.

## 2026-10-02 — run 30 (housekeeping; vehicles on the path right now; serving ablations of fo_long)

Lock taken at 12:19 UTC. Main had not moved. **Housekeeping** (journal 439 lines) as its own commit: runs 24–27 → `archive/JOURNAL-runs-24-27.md`, state section rewritten,
leaderboard tags ds4v26 / v27 / v28b moved to the archive (report.py CURRENT_TAGS), run27.sh deleted. Arms sc_big_fo_gap / fo_v29 kept (run29.sh still names them).
**Idea.** The live link store only sees completed traversals (30 s detect lag, ≤ 30 min old). Other vehicles that are on the links between this vehicle and its target
right now are fresher evidence. `addon_v30.py` (v29 → ms_features_v30) places every vehicle in `.pos` on its current stop-pair link at each feed snapshot. Feed
timestamps are shared by all vehicles in a snapshot (~500 per snapshot), so the join on t is exact. It then takes the other vehicles on this row's path links, ahead of it
(any route, same link keys as the store). Cols: pa_n, pa_spd_mean / pa_spd_min (120 s odometer speed), pa_stop_frac (reported speed 0), pa_gap_m / pa_gap_spd (the nearest
one), pa_cov (occupied share of the path length). Coverage: 76% of weekday rows have ≥ 1 vehicle on the path (68% weekend), mean 3.8, nearest gap p50 ~400 m.
**Serving ablations** (run 29 next step): does the odometer make the per-vehicle crossing log (`_LAP`, `_VEH`) redundant? Arms sc_big_fo_nolap / _noveh / _novl.

Commands: `ARMS_A=" " ARMS_B=" " sh run28.sh` (pipeline 12:20–≈15:15 `--parallel 3`, prep v10, addons v26 + v28; done ≈ 16:30), smoke test
`addon_v29.py --days 2026-09-20 && addon_v30.py --days 2026-09-20`, then `sh run30.sh` (addons v29 + v30 ≈ 85 s/day, fits until ≈ 19:30).
Reproduction: the ds4v30 baseline (113.946) and sc_big_fo_long (85.650) equal ds4v29 bit-for-bit.

### Results (MAE / median / p90 / bias; ds4 s42 train 2.21M / test 1.61M; ds4shift s11 train 2.06M / test 1.42M)
| arm | ds4 | Δ vs fo_long | sa1 / sa5 / sa10 | Sun / Mon / Tue | ds4shift | Δ vs fo_long | sa1 / sa5 / sa10 | Sat / Sun / Mon |
|---|---|---|---|---|---|---|---|---|
| baseline | 113.95 / 60.4 / 241.6 / −19.3 | | 73.4 / 116.2 / 157.8 | 104.5 / 115.5 / 118.5 | 109.95 / 55.9 / 228.7 / −23.5 (v25) | | 73.3 / 111.8 / 152.7 | 103.6 / 106.8 / 116.5 |
| sc_big_fo_long | 85.65 / 43.6 / 177.1 / −13.5 | | 58.8 / 86.7 / 120.1 | 79.3 / 86.9 / 88.5 | 84.20 / 41.8 / 170.6 / −14.7 (v29) | | 59.1 / 84.9 / 117.7 | 81.8 / 81.3 / 87.8 |
| **sc_big_fo_pa** | **85.20** / 43.4 / 175.4 / −13.7 | **−0.53%** | 58.5 / 86.2 / 119.6 | 79.0 / 86.5 / 88.0 | **83.95** / 41.6 / 169.1 / −15.7 | **−0.30%** | 58.8 / 84.6 / 117.5 | 81.9 / 81.1 / 87.2 |
| sc_big_fo_nolap | 86.14 / 43.9 / 177.6 / −13.5 | +0.57% | 59.0 / 87.2 / 120.8 | 79.9 / 87.4 / 89.0 | — | | | |
| sc_big_fo_noveh | 86.56 / 43.7 / 178.2 / −14.0 | +1.06% | 59.8 / 87.4 / 121.2 | 80.1 / 87.9 / 89.4 | — | | | |
| sc_big_fo_novl | 87.10 / 44.0 / 179.1 / −13.6 | +1.69% | 59.7 / 88.2 / 122.1 | 80.9 / 88.3 / 89.9 | 85.85 / 42.3 / 173.3 / −16.2 | +1.96% | 59.8 / 86.4 / 120.1 | 83.5 / 83.1 / 89.3 |

Bucket Δ (sa1..10), fo_pa vs fo_long:
- ds4: −0.36 −0.62 −0.62 −0.66 −0.61 −0.37 −0.48 −0.60 −0.55 −0.44%.
- ds4shift: −0.44 −0.52 −0.44 −0.35 −0.40 −0.28 −0.18 −0.14 −0.19 −0.18%.

fo_pa vs baseline: −25.2% / −23.6%, p90 241.6 → 175.4 / 228.7 → 169.1.
Worst routes, fo_pa:
- ds4: 137 221, 133 185, 118 168, 122 161, 106 139, 2460 132, 2302 129, 131 128, 111 126, 143 125.
- ds4shift: 133 258, 137 210, 122 167, 143 152, 106 150, 2460 129, 1630 125, 2302 122, 111 118, 125 113.

**Conclusion: no protocol win vs the leader (< 3%), so no notification.**
- Vehicles on the path now: a small, consistent gain (−0.53% / −0.30%). Every bucket and p90 improve on both splits; 5 of 6 test days improve (ds4shift Sat 81.8 → 81.9).
- It is cheap to serve. The daemon already places every vehicle on its trip, so this is one link → [(vehicle, fraction, odometer speed)] dict per 10 s cycle (< 0.5 day).
  It is optional in the hand-off.
- Ablations: the per-vehicle crossing log still pays. Without `_LAP` and `_VEH` the cost is +1.7% / +2.0%, every bucket worse; `_LAP` alone +0.57%, `_VEH` alone +1.06%.
  The odometer does not replace them, so the serving work for the crossing log stays in the hand-off.
- **Hand-off recipe (run 30): sc_big_fo_long, optionally + V30 (`sc_big_fo_pa`).**

### Next steps
1. Hand-off is the owner's call. Recommend pausing the schedule: runs 16–30 moved the leader by ≈ 4% in total (sc_big_dwell 88.1 → fo_pa 85.2 on ds4).
2. Possible extensions of V30, each likely ≤ 0.3%: weight the occupancy by link (span / speed per occupied link, summed into a "now" path time), or use the
   vehicles that just left the path (last 120 s). Untested.
3. Untested, needs prep changes: odometer-based crossing times in the live link store.

## 2026-10-02/03 — run 31 (V31: "now" path time from vehicles on the path; 40/60 min odometer)

Lock taken at 20:13 UTC. Main was already merged. No housekeeping (journal 227 lines; run 30 did it).
**Ideas** (run 30 next step 2, run 29 next step 2). `addon_v31.py` (v30 → ms_features_v31):
- `_PAN`: pa_now_t = Σ over the occupied path links of (metres still ahead on that link) / max(median 120 s odometer speed of the vehicles on it, 0.5 m/s).
  Also pa_now_len (metres those links cover) and pa_slow_gap (path distance to the nearest vehicle ahead moving < 1 m/s over 120 s).
  Coverage: now_t > 0 on 75% of weekday rows and 68% of weekend rows (p50 ≈ 380 s / 260 s); slow_gap on 15% / 6%.
- `_ODOX`: odo_spd_2400 and odo_spd_3600 (coverage 93%).
- Odometer-based crossing times (run 30 next step 3) were reasoned out, not run: snapshots are ~10 s apart and both odometer and dist_along interpolate linearly between them.
- Arms: sc_big_fo_pan (fo_pa + _PAN), sc_big_fo_pax (fo_pa + _ODOX), sc_big_fo_v31 (both).

Commands:
- pipeline_lite 09-08..09-29 `--parallel 3` (20:13–≈22:40).
- `sh run31.sh`: prep v10 + addons v26, v28, v29, v30, v31 (≈ 110 s/day for v31), then fits.
- The container restarted twice (≈ 22:50 during the v31 add-on, then ≈ 23:20 with a proxy outage that broke the static-GTFS fetch). Relaunching run31.sh resumed both times, because prep and addons skip finished days.
- Fits finished ≈ 02:00 UTC.

Reproduction: the ds4v31 baseline (113.946) and sc_big_fo_pa (85.197) equal ds4v30 bit-for-bit. ds4shift is judged against the ds4shiftv30 fo_pa row (83.95).

### Results (MAE / median / p90 / bias; ds4 s42 train 2.21M / test 1.61M; ds4shift s11 train 2.06M / test 1.42M)
| arm | ds4 | Δ vs fo_pa | Sun / Mon / Tue | ds4shift | Δ vs fo_pa | Sat / Sun / Mon |
|---|---|---|---|---|---|---|
| baseline | 113.95 / 60.4 / 241.6 / −19.3 | | 104.5 / 115.5 / 118.5 | 109.95 / 55.9 / 228.7 / −23.5 (v25) | | 103.6 / 106.8 / 116.5 |
| sc_big_fo_pa | 85.20 / 43.4 / 175.4 / −13.7 | | 79.0 / 86.5 / 88.0 | 83.95 / 41.6 / 169.1 / −15.7 (v30) | | 81.9 / 81.1 / 87.2 |
| sc_big_fo_pan | 85.09 / 43.3 / 175.1 / −13.6 | −0.13% | 78.9 / 86.3 / 87.9 | 83.89 / 41.5 / 169.2 / −15.8 | −0.07% | 81.7 / 81.1 / 87.3 |
| sc_big_fo_pax | 85.03 / 43.3 / 174.8 / −14.1 | −0.19% | 79.0 / 86.2 / 87.8 | 83.54 / 41.5 / 168.9 / −15.4 | −0.49% | 81.2 / 80.7 / 87.0 |
| sc_big_fo_v31 | 85.07 / 43.3 / 174.6 / −13.8 | −0.15% | 79.1 / 86.3 / 87.7 | 83.48 / 41.5 / 168.4 / −15.8 | −0.56% | 81.2 / 80.6 / 87.0 |

Bucket Δ (sa1..10) vs fo_pa:
- pan ds4: +0.21 −0.17 −0.16 −0.13 −0.35 −0.21 −0.10 −0.11 −0.08 −0.17%. ds4shift: −0.03 −0.02 +0.08 −0.06 −0.03 −0.24 −0.13 +0.01 −0.18 −0.03%.
- pax ds4: +0.27 −0.04 −0.06 −0.31 −0.30 −0.30 −0.15 −0.22 −0.33 −0.34%. ds4shift: −0.73 −0.60 −0.60 −0.92 −0.56 −0.31 −0.27 −0.35 −0.35 −0.33%.
- v31 ds4: −0.02 −0.04 −0.21 −0.27 −0.34 −0.36 −0.07 −0.07 −0.09 −0.04%. ds4shift: −0.43 −0.58 −0.59 −0.65 −0.69 −0.54 −0.45 −0.56 −0.59 −0.51%.

v31 vs baseline: −25.3% / −24.1%, p90 241.6 → 174.6 / 228.7 → 168.4.
Worst routes, v31:
- ds4: 137 225, 133 191, 118 174, 122 159, 106 140, 2460 131, 2302 129, 111 126, 131 125, 143 122.
- ds4shift: 133 257, 137 208, 122 172, 143 146, 106 146, 2460 127, 1630 122, 111 120, 2302 116, 125 115.

**Conclusion: no protocol win (all < 3%), so no notification.**
- The "now" path time adds almost nothing on top of V30's raw cols (−0.13 / −0.07%): the trees already combine pa_spd / pa_cov with the link store.
- 40/60 min odometer is the only consistent part (−0.19 / −0.49%), but sa1 is worse on ds4 (+0.27%). v31 (both) is the most even (every bucket ≤ 0 on both splits).
  At ≤ 0.6% it is within the split-to-split noise seen in earlier runs, so I would not add it to the hand-off.
- Hand-off recipe unchanged: sc_big_fo_long, optionally + V30 (`sc_big_fo_pa`).

### Next steps
1. Pause the schedule (owner's call). Every feed-derived live signal has now been tried, and runs 28–31 each added ≤ 1.6%, the last two ≤ 0.6%.
2. If it continues: run 35 = housekeeping (or earlier if the journal passes 400 lines). Candidates to prune: addon_v28b / sc_big_mf3_age_fo2, sc_big_fo_gap / fo_v29 / _ODOG, and sc_big_fo_pan / _PAN (each lost once).

## 2026-10-03 — run 32 (V32: own break history and shift duty)

Lock taken at 04:14 UTC. Main was already merged. No housekeeping (journal 277 lines; due run 35).
**Idea.** The tail (133 / 137 / 122) is driver breaks. Trip-keyed hold tables gave only ±0.5% (run 20–23). This run tests the vehicle's own break state instead,
from the per-vehicle odometer ring that the hand-off already needs. A driver who just took a long stop is unlikely to take another soon, and one who has driven for hours is due one.
`addon_v32.py` (v31 → ms_features_v32):
- Stopped episodes come from distinct fixes: odometer step ≤ 2 m, fix gap ≤ 120 s, and length ≥ 180 s. Only episodes that already ended count.
  The day averages 7.6k episodes, length p50 367 s.
- `_BRK`: brk_since_s, brk_last_len, brk_odo_since (last ≥ 180 s episode, ≤ 4 h); brk10_since_s, brk10_odo_since (≥ 600 s, ≤ 6 h); brk_n3h.
  Coverage: brk 86% (p50 ≈ 2200 s weekday / 1800 s weekend), brk10 51–64%.
- `_DUTY`: duty_s and duty_odo, measured from the vehicle's first snapshot after its last ≥ 30 min feed gap (p50 ≈ 7 h).
- Arms: sc_big_fo_brk (fo_pa + _BRK), sc_big_fo_duty (fo_pa + _DUTY), sc_big_fo_v32 (both).

Commands:
- pipeline_lite 09-08..09-29 `--parallel 3`, 04:14–≈06:50.
- `sh research/model-search/run32.sh`: prep v10 plus addons v26, v28–v32, done 07:18 (v32 takes ≈ 10 s/day); then fits until 08:30.
- Then sc_big_fo_duty on ds4shift via `harness.py run`.

Reproduction: the ds4v32 baseline (113.946) and sc_big_fo_pa (85.197) equal ds4v31 bit-for-bit. ds4shift is judged against the ds4shiftv30 fo_pa row (83.95).

### Results (MAE / median / p90 / bias; ds4 s42 train 2.21M / test 1.61M; ds4shift s11 train 2.06M / test 1.42M)
| arm | ds4 | Δ vs fo_pa | Sun / Mon / Tue | ds4shift | Δ vs fo_pa | Sat / Sun / Mon |
|---|---|---|---|---|---|---|
| baseline | 113.95 / 60.4 / 241.6 / −19.3 | | 104.5 / 115.5 / 118.5 | 109.95 / 55.9 / 228.7 / −23.5 (v25) | | 103.6 / 106.8 / 116.5 |
| sc_big_fo_pa | 85.20 / 43.4 / 175.4 / −13.7 | | 79.0 / 86.5 / 88.0 | 83.95 / 41.6 / 169.1 / −15.7 (v30) | | 81.9 / 81.1 / 87.2 |
| sc_big_fo_brk | 85.43 / 43.3 / 175.4 / −14.1 | +0.27% | 79.3 / 86.6 / 88.3 | 83.90 / 41.6 / 169.5 / −15.7 | −0.06% | 81.6 / 81.0 / 87.4 |
| **sc_big_fo_duty** | **84.77** / 43.2 / 174.6 / −13.7 | **−0.50%** | 78.9 / 85.7 / 87.7 | **83.62** / 41.5 / 168.7 / −16.0 | **−0.39%** | 81.8 / 80.9 / 86.7 |
| sc_big_fo_v32 | 84.75 / 43.3 / 174.9 / −13.4 | −0.53% | 78.5 / 85.9 / 87.6 | 83.66 / 41.6 / 169.7 / −15.6 | −0.34% | 81.6 / 81.0 / 86.8 |

Bucket Δ (sa1..10) vs fo_pa:
- brk ds4: +0.50 +0.54 +0.31 +0.30 +0.17 +0.05 +0.25 +0.31 +0.23 +0.18%. ds4shift: −0.22 +0.15 −0.07 −0.09 −0.29 +0.04 −0.07 −0.04 −0.03 +0.05%.
- duty ds4: −0.25 −0.30 −0.52 −0.45 −0.50 −0.77 −0.57 −0.53 −0.55 −0.46%. ds4shift: −0.34 −0.26 −0.39 −0.41 −0.38 −0.36 −0.35 −0.48 −0.39 −0.47%.
- v32 ds4: −0.69 −0.44 −0.64 −0.50 −0.63 −0.47 −0.34 −0.56 −0.55 −0.52%. ds4shift: −0.52 −0.11 −0.10 −0.27 −0.36 −0.43 −0.42 −0.38 −0.37 −0.39%.

duty vs baseline: −25.6% / −24.0%, p90 241.6 → 174.6 / 228.7 → 168.7.
Worst routes, duty:
- ds4: 137 223, 133 188, 118 167, 122 160, 106 141, 2302 132, 2460 130, 111 125, 143 124, 131 124.
- ds4shift: 133 253, 137 209, 122 165, 106 152, 143 150, 2460 129, 1630 124, 111 120, 2302 118, 125 112.

**Conclusion: no protocol win (all < 3%), so no notification.**
- The vehicle's own break history doesn't help: +0.27% on ds4, −0.06% on ds4shift, and routes 133 / 137 are unchanged.
  The stop-state cols the leader already has (stationary_sec, the dwell table, _LAP, fs_zero_frac600) cover what is learnable; the breaks themselves stay unpredictable from the feed.
- Shift duty is a small, consistent gain: −0.50% / −0.39%, every bucket, p90 and 5 of 6 test days better. It is not driven by the break routes.
  Likely it encodes the vehicle's place in its block (morning vs evening shift, running late through the day).
  It is the cheapest add-on so far: per vehicle, one (t, odometer) at the first snapshot after a ≥ 30 min gap, persisted with the tracker state.
  Optional in the hand-off, like V30.
- Hand-off recipe unchanged: sc_big_fo_long, optionally + V30 (`sc_big_fo_pa`) + V32 duty (`sc_big_fo_duty`).

### Next steps
1. Pause the schedule (owner's call). Runs 30–32 each added ≤ 0.5%.
2. If it continues: a later day set once ≥ 10-11 is available (two new weekends) to re-check the leader + V30 + duty on fresh days and re-test trip-keyed holds (run 24–27 note).
   Run 35 = housekeeping; prune candidates as listed in run 31, plus `_BRK` / sc_big_fo_brk / sc_big_fo_v32 (lost once).

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
