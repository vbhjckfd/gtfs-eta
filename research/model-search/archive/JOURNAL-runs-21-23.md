# Model-search journal — runs 21–23 (archived in run 25)

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

## 2026-09-30 — run 23 (housekeeping; per-split feature subsampling; 2-model bag; route 137)

Lock taken at 04:13 UTC. Main was already merged. Commit "model-search: housekeeping" did the following:
- Run 19 went to `archive/JOURNAL-runs-19-20.md`, and run 20 joined it at the end of this run.
- The ds1v10f / ds2w / ds2 / ds1v19 / ds1v19t / ds1v13 tags moved to the old leaderboard.
- Arm `sc_big_hold` + `addon_v13.py` were removed (lost twice), as were `sc_big_it2k` / `sc_big_511`, which `sc_big_511_it2k` supersedes.
- run19/19b/20/20b/21.sh were deleted.

Rebuild: `pipeline_lite.py --parallel 4 --days 2026-09-07..2026-09-28` (04:14–05:57, ≈ 21 min/day/worker). The prep of 09-24 running next to it was
OOM-killed again (run.sh's 6 GB free-memory gate isn't enough while 4 pipeline workers peak). A re-run of `run23.sh` after the pipeline
finished skipped the finished days. Fits: `sh research/model-search/run23.sh` (tag ds3v13, features ms_features_v10; the arms don't use V13
cols). Baseline (109.3996) and leader (86.865) reproduce run 22 bit-for-bit. New arms (arms.py, no new cols):
- `sc_big_mf7` / `sc_big_mf5`: the leader with HGB `max_features` 0.7 / 0.5.
- `sc_big_bag2`: the mean of two `sc_big_mf7` fits with seeds s and s+1000.

### Results (MAE / median / p90 / bias; Δ vs the leader; ds3 s42 train 2.25M / test 1.42M; ds3shift s7 train 2.04M / test 1.44M)
| arm | ds3 | ΔL | sa1 / sa5 / sa10 | Sat / Sun / Mon | ds3shift | ΔL | sa1 / sa5 / sa10 | Fri / Sat / Sun | fit s |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 109.4 / 55.1 / 227.4 / −28.0 | | 73.0 / 111.1 / 152.0 | 103.4 / 106.6 / 115.4 | 111.4 / 57.0 / 233.3 / −27.5 | | 71.5 / 113.8 / 154.6 | 119.4 / 103.9 / 107.0 | 480 / 386 |
| sc_big_dwell | 86.87 / 43.0 / 175.7 / −15.9 | | 61.6 / 87.5 / 120.6 | 85.1 / 85.1 / 89.2 | 88.42 / 43.7 / 180.9 / −18.8 | | 59.9 / 89.2 / 123.6 | 92.4 / 85.3 / 85.5 | 747 / 684 |
| sc_big_mf7 | 86.59 / 42.9 / 175.2 / −15.4 | −0.31% | 61.3 / 87.3 / 120.3 | 84.8 / 84.8 / 89.0 | 88.07 / 43.7 / 180.6 / −18.6 | −0.40% | 59.6 / 88.8 / 123.4 | 92.1 / 85.1 / 84.9 | 772 / 726 |
| sc_big_mf5 | 86.43 / 42.8 / 174.8 / −15.8 | −0.50% | 61.1 / 87.0 / 120.0 | 84.5 / 84.6 / 88.9 | 88.03 / 43.6 / 180.3 / −18.4 | −0.45% | 59.6 / 88.5 / 123.4 | 92.1 / 85.0 / 84.8 | 787 / 751 |
| sc_big_bag2 | 86.21 / 42.7 / 174.6 / −15.3 | −0.76% | 60.9 / 86.9 / 119.9 | 84.4 / 84.4 / 88.7 | 87.56 / 43.4 / 179.5 / −18.4 | −0.98% | 59.3 / 88.3 / 122.7 | 91.6 / 84.6 / 84.4 | 1559 / 1403 |

Every stops_ahead bucket improves for all three arms on both splits (worst bucket vs the leader: −0.2% to −0.8%). Worst routes:
- mf5 ds3: 133 247, 137 221, 122 176, 143 157, 106 150. mf5 ds3shift: 133 345, 137 207, 143 197, 106 165, 2460 150.
- bag2 ds3: 133 249, 137 220, 122 178, 143 165, 106 151, 2460 132, 1630 129, 2302 125, 111 124, 125 115.
- Leader ds3: 133 258, 137 218, 122 179, 143 163, 106 158, 1630 131, 2460 130, 2302 128, 111 125, 125 116.

Route 137 diagnostic (`diag_route.py data/pred_ds3_leader.parquet 137`): MAE 218, bias −97 s, 4 vehicles (5838 has 47% of the rows).
28.1% of the rows have actual > hist path + 300 s; they carry 65.9% of the route's AE, with bias −473 s. Stationary > 900 s MAE is 340, 300–900 s is 379,
and 30–120 s is 329 (bias −211: the vehicle has just stopped for a long hold). The error grows with horizon (sa1 151 → sa10 291). It is the same
driver-hold class as 133 / 122, and no feature family has touched these.

**Conclusion: no protocol win vs the leader and no notification.**
- `max_features` 0.5 is a free, consistent −0.5% (every bucket and p90 better, same trees at serving). Fold it into the hand-off recipe.
- The 2-model bag adds ≈ −0.3 to −0.5% more at 2x serving cost. It is optional, the same trade as the 511/2400 capacity variant.

### Next steps
1. Cheap: confirm mf5 on ds1 (needs the 08-31..09-21 rebuild), or try mf 0.3 / 0.4 on ds3.
2. Hand-off of sc_big_dwell + max_features 0.5 (owner's call). The search on this feature family is saturated. The remaining large errors
   (133 / 137 / 122) are driver holds, which need operator break data.
3. Tip for the next run: start run.sh only after pipeline_lite has exited (the prep OOM repeated), run.sh's memory gate was raised from 6 to 9 GB at the end of run 23.

