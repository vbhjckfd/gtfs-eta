# Model-search journal — runs 33–37 (archived in run 40)

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

## 2026-10-04 — run 36 (early exit, no fresh day set)

Lock taken at 04:12 UTC. Main had not moved. R2 credentials present. Newest complete day is 10-03; a fresh split needs ≥ 10-12 (state step 3), so no experiment,
no result rows, no notification. No housekeeping due (journal 167 lines, last done run 35).

### Next steps
1. Unchanged: pause the schedule (owner's call), or keep exiting early until ≥ 10-12 is in R2; then ds5 via ds.sh.

## 2026-10-04 — run 37 (early exit, no fresh day set)

Lock taken at 08:13 UTC. Main had not moved. R2 credentials present. Newest complete day is still 10-03; a fresh split needs ≥ 10-12 (state step 3),
so no experiment, no result rows, no notification. No housekeeping due (last done run 35).

### Next steps
1. Unchanged: pause the schedule (owner's call), or keep exiting early until ≥ 10-12 is in R2 (earliest useful run: 2026-10-12 UTC); then ds5 via ds.sh.

