# Model-search journal — runs 24–27 (archived in run 30)

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

## 2026-10-01 — run 26 (GPS-fix age features on ds4)

Lock taken at 04:13 UTC. Main (aeb652a) was already merged. No housekeeping this run (journal 229 lines; run 25 was the 5th-run housekeeping). Deleted run24.sh.
**New signal.** I sampled raw snapshots from 2026-09-29: feed ts − vehicle.timestamp is p50 14 s, p75 29 s, p90 62 s. 24% of rows are > 30 s and 3% are > 300 s (the
> 180 s ones are dropped at serving, STALE_VEHICLE_MAX_AGE_SEC). Training and src/features ignore the per-vehicle timestamp, and every row is anchored at the feed time.
- `pipeline_lite.py` now parses with a copy of `_fetch_and_parse` that also keeps vehicle_ts. It writes `data/ms_lite/<day>.age.parquet` (vehicle, feed ts, vehicle_ts) and drops the
  col before infer_trips, so the training rows are unchanged.
- `addon_v26.py` joins onto the v10 features (ms_features_v26):
  - `pos_age_s`;
  - `pos_age_med600` (the vehicle's rolling 600 s median age, i.e. its reporting cadence);
  - `pos_age_dist` = age × own_speed_180.
- Arm `sc_big_mf3_age` = mf3 + these cols.

Rebuild: `sh research/model-search/run26.sh` (pipeline 09-08..09-29, prep → ms_features_v10, addon → ms_features_v26, fits tagged ds4v26 / ds4shiftv26). First attempt at
`--parallel 4` was OOM-killed after 6 min (memcg limit 14.0 GB now; each worker ~3.7 GB while downloading). `--parallel 3` took 04:25–07:20, prep finished at 08:10, and the fits finished at 09:05.
In the 15 train/test days, 16–17% of rows have age > 30 s (p50 12 s, p90 40–44 s); coverage is 100%.
Reproduction: the ds4v26 baseline (113.95) and mf3 (87.7685) equal run 25 bit-for-bit. So I fit only the age arm on ds4shift and judged it against the ds4shiftv25 rows (report.py BASE_ALIAS).

### Results (MAE / median / p90 / bias)
| arm | ds4 (s42; train 2.21M, test 1.61M) | Δ vs mf3 | sa1 / sa5 / sa10 | Sun / Mon / Tue | ds4shift (s11; train 2.06M, test 1.42M) | Δ vs mf3 | sa1 / sa5 / sa10 | Sat / Sun / Mon | fit s |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 113.95 / 60.4 / 241.6 / −19.3 | | 73.4 / 116.2 / 157.8 | 104.5 / 115.5 / 118.5 | 109.95 / 55.9 / 228.7 / −23.5 (v25) | | 73.3 / 111.8 / 152.7 | 103.6 / 106.8 / 116.5 | 579 |
| sc_big_mf3 | 87.77 / 44.7 / 180.8 / −12.7 | | 61.0 / 88.6 / 122.3 | 82.4 / 88.4 / 90.6 | 86.71 / 42.9 / 175.5 / −14.9 (v25) | | 61.4 / 87.2 / 120.4 | 84.7 / 84.6 / 89.5 | 894 |
| **sc_big_mf3_age** | 87.22 / 44.3 / 180.3 / −12.8 | **−0.63%** | 60.4 / 88.2 / 121.5 | 81.8 / 88.0 / 89.9 | 86.47 / 42.5 / 174.7 / −15.0 | **−0.28%** | 61.1 / 87.0 / 120.0 | 84.4 / 84.3 / 89.4 | 888 / 857 |

Bucket Δ vs mf3 (sa1..10):
- ds4: −0.94 −0.71 −0.59 −0.75 −0.55 −0.46 −0.41 −0.60 −0.70 −0.67%.
- ds4shift: −0.50 −0.34 −0.23 −0.40 −0.21 −0.23 −0.08 −0.17 −0.33 −0.34%.

vs baseline: −23.5% (ds4) / −21.4% (ds4shift), p90 241.6 → 180.3 / 228.7 → 174.7.
Worst routes, age arm:
- ds4: 137 239, 133 183 (mf3 200), 118 171, 122 169, 106 145, 2302 134, 2460 130, 111 129, 131 129, 143 128.
- ds4shift: 133 253, 137 224, 122 177, 143 156, 106 156, 2460 134, 2302 132, 1630 126, 111 123, 125 114.

**Conclusion: a small, consistent gain, but no protocol win vs the leader (< 3%), so no notification.** Every test day, every stops_ahead bucket and p90 improve on both splits.
The gain is largest at sa1 (−0.9%), where a 40 s-old fix matters most. Serving cost is low: src/inference.py already reads vehicle_ts per vehicle, and the 600 s
median needs a small per-vehicle ring. Add it to the hand-off recipe as "mf3 + age" (≈ −0.3 to −0.6%).

### Next steps
1. Hand-off recipe is now sc_big_dwell + max_features 0.3 + V26 age cols. The owner decides.
2. Not done: an ablation of the 3 age cols, and age-correcting the live link-store crossing times (crossings are timed with the feed ts, so they lag by the fix age). The
   latter changes labels as well as features, which needs care. It is the one untested idea left from this signal.
3. Still recommend pausing the schedule; runs 16–26 moved the leader by < 1.3% in total.

## 2026-10-01 — run 27 (GPS fix clock for live link times and own speed)

Lock taken at 12:20 UTC. Main (aeb652a) was already merged. No housekeeping this run (journal 278 lines; the next 5th-run housekeeping falls on run 30).
**Idea (run 26, next step 2).** Crossing times in the live link store and the own-speed windows are measured on the feed clock, so every link traversal carries the
difference of two fix ages as noise (fix age p50 12 s, p90 ~40 s). Labels are unchanged.
- `MS_AGE_CORR=1` (features_live.py + harness.py prep):
  - Each crossing's fix-clock time is `t − age` (nearest snapshot of the vehicle within 60 s; age clipped to [0, 180] s). Link `lt` is the difference of fix-clock times.
  - Availability (`t`, the 30 s detect lag, windows) stays on the feed clock, so nothing peeks ahead.
  - Own speed divides by fix-clock elapsed time. When the fix didn't change between snapshots, el ≤ 0, so the value is NaN: own_speed_60 coverage drops from 96.7% to 95.5%.
  - The historical link tables (links_*.parquet) are corrected too, because they are built from the same links.
- On 09-08, link-time features moved by a median of 5–10 s (p90 19–35 s), and own_speed_60 by a median of 0.23 m/s.
- `addon_v26.py` SRC/DST can now be set with MS_ADDON_SRC / MS_ADDON_DST. New arm `sc_big_mf3_age1` = mf3 + pos_age_s only.

Commands: `sh research/model-search/run27.sh`.
- pipeline 09-08..09-29 at `--parallel 3` took 12:25–14:58, much faster than run 26 (≈ 23 min/day/worker).
- prep (MS_AGE_CORR=1) → ms_features_v27c, then addon_v26 → ms_features_v27, tags ds4v27 / ds4shiftv27.
- The first prep attempt failed on two bugs: a pandas-3 str-vs-object merge key, and an own-speed edit that hadn't applied. Both were fixed and re-run at 15:05.
  run.sh exits on PREP FAIL, but run27.sh still printed "RUN27 ALL DONE"; check for PREP FAIL in the log.
Reproduction: the ds4v27 baseline is 113.9459, bit-for-bit equal to ds4v25/v26, so the v27 rows are comparable with the run 26 rows.

### Results (MAE / median / p90 / bias; ds4 s42 train 2.21M / test 1.61M; ds4shift s11 train 2.06M / test 1.42M)
| arm | features | ds4 | sa1 / sa5 / sa10 | Sun / Mon / Tue | ds4shift | Sat / Sun / Mon |
|---|---|---|---|---|---|---|
| baseline | — | 113.95 / 60.4 / 241.6 / −19.3 | 73.4 / 116.2 / 157.8 | 104.5 / 115.5 / 118.5 | 109.95 (v25) | |
| sc_big_mf3 | feed clock (v26) | 87.77 / 44.7 / 180.8 / −12.7 | 61.0 / 88.6 / 122.3 | 82.4 / 88.4 / 90.6 | 86.71 (v25) | 84.7 / 84.6 / 89.5 |
| sc_big_mf3 | **fix clock (v27)** | 87.43 / 44.5 / 179.9 / −12.7 (−0.38%) | 60.7 / 88.2 / 122.0 | 82.2 / 88.1 / 90.2 | — | |
| sc_big_mf3_age | feed clock (v26) | **87.22** / 44.3 / 180.3 / −12.8 | 60.4 / 88.2 / 121.5 | 81.8 / 88.0 / 89.9 | 86.47 / 42.5 / 174.7 / −15.0 | 84.4 / 84.3 / 89.4 |
| sc_big_mf3_age | fix clock (v27) | 87.34 / 44.3 / 180.3 / −12.8 (+0.14% vs v26) | 60.6 / 88.1 / 122.0 | 82.0 / 88.2 / 90.0 | 86.43 / 42.5 / 174.5 / −15.4 (−0.05%) | 84.2 / 84.5 / 89.3 |
| sc_big_mf3_age1 | fix clock (v27) | 87.34 / 44.3 / 179.8 / −13.1 | 60.7 / 88.0 / 121.8 | 82.0 / 88.2 / 90.0 | — | |

Bucket Δ (sa1..10), mf3 v27 vs v26: −0.45 −0.48 −0.42 −0.48 −0.46 −0.33 −0.24 −0.26 −0.42 −0.32% (all better).
mf3_age v27 vs v26: ds4 +0.23 −0.04 +0.01 +0.15 −0.03 +0.09 −0.03 +0.29 +0.33 +0.38%; ds4shift +0.13 −0.10 −0.09 +0.08 −0.04 +0.03 −0.24 −0.01 −0.12 −0.09%.
Worst routes, mf3_age v27 ds4: 137 244, 133 188, 122 172, 118 166, 106 146, 2302 136, 2460 132, 131 129, 111 129, 143 125.
Worst routes, ds4shift: 133 253, 137 226, 122 174, 143 157, 106 157, 2460 133, 2302 132, 1630 129, 111 122, 149 114.

**Conclusion: no protocol win, no notification.**
- Timing on the fix clock recovers most of what the age cols give (−0.38% vs −0.63% on ds4), but the two don't stack. With the age cols present it is neutral to slightly worse (+0.14% / −0.05%).
- Fix-clock timing also needs vehicle_ts carried into the serving link store. The feed-clock recipe + V26 age cols stays the hand-off.
- Ablation: on v27, pos_age_s alone equals all 3 age cols. The 600 s median ring may be unnecessary, but that is only tested on fix-clock features; on the feed clock it is untested.

### Next steps
1. Hand-off recipe unchanged: sc_big_dwell + max_features 0.3 + V26 age cols (feed clock). Owner's call.
2. Optional, cheap: ablate pos_age_med600 / pos_age_dist on the feed-clock v26 features (needs a v10 prep, +1 h). If pos_age_s alone holds there, serving needs no age ring.
3. Still recommend pausing the schedule: runs 16–27 moved the leader by < 1.3% in total. The largest remaining errors (133 / 137 / 122 driver holds) need operator break data.
