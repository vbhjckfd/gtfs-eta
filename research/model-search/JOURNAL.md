# Model-search journal

Research-only search for a better ETA model. Branch `claude/model-search`.
Harness: `harness.py` (prep/run), arms in `arms.py`, live features in
`features_live.py` (+ add-on scripts `addon_v*.py` that append cols to an existing
feature dir). Results: one compact JSON line per fit in `results.jsonl` (keyed by `name` = arm_tag_sSEED; written by `harness.save_result`); `report.py` regenerates
LEADERBOARD.md (current tags) and archive/LEADERBOARD-old.md (superseded tags).
Full run entries are archived in `archive/JOURNAL-runs-1-12.md`, `-13-16.md`, `-17-18.md`, `-19-20.md`, `-21-23.md`, `-24-27.md`.

## State of the search (updated run 30)

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

### What won (in order; each Δ is vs the previous leader)
- live link path times: −7 to −9% vs baseline (run 1). + own speed, − headway: −8.3 / −9.0% (run 2).
- median of last 3/5 traversals, drop calendar cols (`stack`): −12.4 / −14.5% (run 3).
- historical link × hour table: ≈ −2% (run 4). weekday/weekend table + route_id categorical: ≈ −2.3% (run 6).
- target-stop categorical + 255 leaves: −1.6% (run 7). current-link cols: −0.6 to −1.1% (run 8).
- own previous lap: −1.2 to −1.5% (run 11). vehicle own/hist ratio: −1.0 to −1.2% (run 12). location dwell table: −0.8% (run 14).
- max_features 0.3: −0.4 to −0.65% (runs 23–25). fix-age cols: −0.3 to −0.6% (run 26).
- feed speed + odometer: −1.1 / −1.6% (run 28). odometer 600/1200 s windows: −0.7 / −1.0% (run 29).

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

### Open next steps
1. Hand-off of `sc_big_fo_long` is the owner's call. Recommend pausing the schedule: runs 16–29 moved the leader by ≈ 3.5% in total.
2. Serving simplification: ablate the expensive per-vehicle state (`_LAP`, `_VEH`) now that the odometer carries vehicle speed (run 30).
3. Untested, needs prep changes: odometer-based crossing times in the live link store.

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
