# Leaderboard

Held-out raw-model metrics (seconds); Δ vs the baseline row with the same tag
(day set / split) and seed. Win = MAE ≤ −3%, p90 not worse, no stops_ahead bucket
worse by > 5%. Regenerate with `python research/model-search/report.py`.

Superseded tags (ds1 .. ds1v9, ds1shift .. ds1shiftv9, ds1v5k3, ds1v9k3, ds1v11, ds1v12,
ds1lag90w15, ds1h21 / ds1h7, ds1r32 / ds1r72, ds1v10f / ds2w / ds2, ds1v19 / ds1v19t, ds1v13, ds1v18, ds1v20, ds4v26 / ds4v27 / ds4v28b (run 30) and their shift twins) are in `archive/LEADERBOARD-old.md` with their definitions;
their baselines are identical to the ds1 / ds1shift ones below.

Tags: `ds4v28` / `ds4shiftv28` = run 28, ds4 rebuilt on feed-clock prep (v10) + addon_v26 age cols + addon_v28 feed-reported speed / odometer cols (ms_features_v28); ds4v28 baseline / mf3_age reproduce 113.95 / 87.22; ds4shiftv28 is judged against the ds4shiftv25 baseline.
`ds4v29` / `ds4shiftv29` = run 29, + addon_v29 odometer cols (600 / 1200 s odometer speed, 600 s zero-speed share, odometer-vs-shape gap, trip-average odometer speed; ms_features_v29); ds4v29 baseline / mf3_age_fo reproduce 113.95 / 86.28; ds4shiftv29 is judged against the ds4shiftv25 baseline.
`ds4v30` / `ds4shiftv30` = run 30, + addon_v30 cols (other vehicles on the path links right now: count, 120 s odometer speeds, stopped share, nearest gap, occupied share; ms_features_v30), and fo_long ablations without _LAP / _VEH; ds4shiftv30 is judged against the ds4shiftv25 baseline.
`ds4v25` / `ds4shiftv25` = run 25, the newest day set (data 09-08..09-29, v10 features): ds4 = train 09-15..09-26, test 09-27 (Sun) / 09-28 (Mon) / 09-29 (Tue), seed 42; ds4shift = train 09-15..09-25, test 09-26 (Sat) / 09-27 / 09-28, seed 11.
`ds3v13` / `ds3shiftv13` = runs 21–22, a later day set (data 09-07..09-28, v10 + V13 cols): ds3 = train 09-14..09-25, test 09-26 (Sat) / 09-27 (Sun) / 09-28 (Mon), seed 42; ds3shift = train 09-14..09-24, test 09-25 (Fri) / 09-26 / 09-27, seed 7. Its baselines differ from ds1 (different days).
`ds1v10` / `ds1shiftv10` = rebuilt in run 14 (+ V10 cols: conditional remaining dwell at the current location from prior days' stationary runs; MS_FEAT_DIR=ms_features_v10); baselines identical. `ds1v12k3` = run 15, the same v10 cols at 3x train rows (3% of snapshots).
`ds1` = train 2026-09-07..09-18, test 09-19 (Sat), 09-20 (Sun), 09-21 (Mon).
`ds1shift` = train 09-07..09-17, test 09-18 (Fri), 09-19, 09-20. Train rows 1% of
snapshots/day, test 3% (pipeline_lite 10% → prep 3% → run).

| tag | arm | seed | n_train | n_test | MAE | median | p90 | bias | sa1 | sa10 | ΔMAE | Δp90 | worst bucket Δ | win | serving |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ds1shiftv10 | baseline | 7 | 1,978,242 | 1,424,485 | 113.0 | 59.5 | 241.1 | -14.2 | 71.2 | 159.7 | — | — | — | — | prod |
| ds1shiftv10 | sc_big_dwell | 7 | 1,978,242 | 1,424,485 | 87.0 | 43.9 | 179.5 | -14.6 | 59.7 | 122.3 | -23.1% | -25.6% | -16.1% | **yes** | as sc_big_veh + static location -> sorted stop-duration table (prior 14 days, built at export like hist_dt); lookup by (last passed stop, 50 m bin) and stationary_sec; ~+0.3 day, ~3 days total |
| ds1shiftv10 | sc_big_dwell2 | 7 | 1,978,242 | 1,424,485 | 87.0 | 43.8 | 179.4 | -14.9 | 59.6 | 122.4 | -23.0% | -25.6% | -16.3% | **yes** | as sc_big_dwell (2 of its 5 cols) |
| ds1shiftv10 | sc_big_veh | 7 | 1,978,242 | 1,424,485 | 87.7 | 44.1 | 180.6 | -16.0 | 60.7 | 123.2 | -22.4% | -25.1% | -14.7% | **yes** | as sc_big_lap + per-vehicle 1 h deque of (own link time, hist link time) from the same crossing events; ~2.7 days total |
| ds1v10 | baseline | 42 | 2,178,139 | 1,414,363 | 110.6 | 56.0 | 230.9 | -28.2 | 72.6 | 153.5 | — | — | — | — | prod |
| ds1v10 | sc_big_dwell | 42 | 2,178,139 | 1,414,363 | 87.0 | 43.1 | 178.2 | -12.6 | 60.7 | 121.6 | -21.3% | -22.8% | -16.4% | **yes** | as sc_big_veh + static location -> sorted stop-duration table (prior 14 days, built at export like hist_dt); lookup by (last passed stop, 50 m bin) and stationary_sec; ~+0.3 day, ~3 days total |
| ds1v10 | sc_big_dwell2 | 42 | 2,178,139 | 1,414,363 | 87.2 | 43.1 | 177.7 | -13.8 | 60.8 | 122.0 | -21.1% | -23.0% | -16.2% | **yes** | as sc_big_dwell (2 of its 5 cols) |
| ds1v10 | sc_big_veh | 42 | 2,178,139 | 1,414,363 | 87.7 | 43.3 | 179.1 | -14.0 | 62.0 | 122.3 | -20.7% | -22.4% | -14.6% | **yes** | as sc_big_lap + per-vehicle 1 h deque of (own link time, hist link time) from the same crossing events; ~2.7 days total |
| ds1v12k3 | baseline | 42 | 6,510,353 | 1,414,363 | 108.1 | 54.4 | 226.9 | -27.5 | 70.4 | 149.9 | — | — | — | — | prod |
| ds1v12k3 | sc_big_dwell | 42 | 6,510,353 | 1,414,363 | 84.7 | 42.1 | 173.3 | -13.4 | 58.6 | 118.7 | -21.7% | -23.6% | -16.8% | **yes** | as sc_big_veh + static location -> sorted stop-duration table (prior 14 days, built at export like hist_dt); lookup by (last passed stop, 50 m bin) and stationary_sec; ~+0.3 day, ~3 days total |
| ds3shiftv13 | baseline | 7 | 2,040,811 | 1,443,531 | 111.4 | 57.0 | 233.3 | -27.5 | 71.5 | 154.6 | — | — | — | — | prod |
| ds3shiftv13 | sc_big_bag2 | 7 | 2,040,811 | 1,443,531 | 87.6 | 43.4 | 179.5 | -18.4 | 59.3 | 122.7 | -21.4% | -23.1% | -17.1% | **yes** | as sc_big_mf7 x 2 seeds, predictions averaged (2400 trees, ~2x serving time, ~3.8 s per 3000 rows) |
| ds3shiftv13 | sc_big_dwell | 7 | 2,040,811 | 1,443,531 | 88.4 | 43.7 | 180.9 | -18.8 | 59.9 | 123.6 | -20.7% | -22.4% | -16.1% | **yes** | as sc_big_veh + static location -> sorted stop-duration table (prior 14 days, built at export like hist_dt); lookup by (last passed stop, 50 m bin) and stationary_sec; ~+0.3 day, ~3 days total |
| ds3shiftv13 | sc_big_hold | 7 | 2,040,811 | 1,443,531 | 88.3 | 43.8 | 181.2 | -17.5 | 59.7 | 123.7 | -20.8% | -22.3% | -16.5% | **yes** | as sc_big_dwell + trip x location dwell table and trip -> per-day hold list (static, built at export); ~+0.5 day |
| ds3shiftv13 | sc_big_mf2 | 7 | 2,040,811 | 1,443,531 | 87.9 | 43.6 | 180.2 | -18.7 | 59.7 | 123.1 | -21.1% | -22.8% | -16.5% | **yes** | as sc_big_mf5 (max_features 0.2) |
| ds3shiftv13 | sc_big_mf3 | 7 | 2,040,811 | 1,443,531 | 87.9 | 43.5 | 180.4 | -18.5 | 59.5 | 122.9 | -21.2% | -22.7% | -16.7% | **yes** | as sc_big_mf5 (max_features 0.3) |
| ds3shiftv13 | sc_big_mf4 | 7 | 2,040,811 | 1,443,531 | 88.0 | 43.5 | 180.6 | -18.7 | 59.5 | 123.3 | -21.1% | -22.6% | -16.7% | **yes** | as sc_big_mf5 (max_features 0.4) |
| ds3shiftv13 | sc_big_mf5 | 7 | 2,040,811 | 1,443,531 | 88.0 | 43.6 | 180.3 | -18.4 | 59.6 | 123.4 | -21.0% | -22.7% | -16.6% | **yes** | as sc_big_dwell (training-only change: max_features 0.5; export unchanged) |
| ds3shiftv13 | sc_big_mf5_msl20 | 7 | 2,040,811 | 1,443,531 | 88.2 | 43.5 | 180.5 | -18.9 | 59.7 | 123.5 | -20.9% | -22.6% | -16.4% | **yes** | as sc_big_mf5 (min_samples_leaf 20) |
| ds3shiftv13 | sc_big_mf7 | 7 | 2,040,811 | 1,443,531 | 88.1 | 43.7 | 180.6 | -18.6 | 59.6 | 123.4 | -21.0% | -22.6% | -16.6% | **yes** | as sc_big_dwell (training-only change: max_features 0.7; export unchanged) |
| ds3v13 | baseline | 42 | 2,250,172 | 1,423,357 | 109.4 | 55.1 | 227.4 | -28.0 | 73.0 | 152.0 | — | — | — | — | prod |
| ds3v13 | sc_big_bag2 | 42 | 2,250,172 | 1,423,357 | 86.2 | 42.7 | 174.6 | -15.3 | 60.9 | 119.9 | -21.2% | -23.2% | -16.6% | **yes** | as sc_big_mf7 x 2 seeds, predictions averaged (2400 trees, ~2x serving time, ~3.8 s per 3000 rows) |
| ds3v13 | sc_big_dwell | 42 | 2,250,172 | 1,423,357 | 86.9 | 43.0 | 175.7 | -15.9 | 61.6 | 120.6 | -20.6% | -22.7% | -15.6% | **yes** | as sc_big_veh + static location -> sorted stop-duration table (prior 14 days, built at export like hist_dt); lookup by (last passed stop, 50 m bin) and stationary_sec; ~+0.3 day, ~3 days total |
| ds3v13 | sc_big_hold | 42 | 2,250,172 | 1,423,357 | 87.0 | 43.1 | 176.6 | -14.4 | 61.4 | 120.9 | -20.5% | -22.3% | -15.9% | **yes** | as sc_big_dwell + trip x location dwell table and trip -> per-day hold list (static, built at export); ~+0.5 day |
| ds3v13 | sc_big_mf2 | 42 | 2,250,172 | 1,423,357 | 86.5 | 42.7 | 174.7 | -15.9 | 61.2 | 119.7 | -20.9% | -23.2% | -16.2% | **yes** | as sc_big_mf5 (max_features 0.2) |
| ds3v13 | sc_big_mf3 | 42 | 2,250,172 | 1,423,357 | 86.3 | 42.7 | 174.7 | -15.3 | 61.0 | 119.7 | -21.1% | -23.2% | -16.4% | **yes** | as sc_big_mf5 (max_features 0.3) |
| ds3v13 | sc_big_mf4 | 42 | 2,250,172 | 1,423,357 | 86.5 | 42.7 | 175.0 | -15.6 | 61.2 | 120.0 | -21.0% | -23.1% | -16.2% | **yes** | as sc_big_mf5 (max_features 0.4) |
| ds3v13 | sc_big_mf5 | 42 | 2,250,172 | 1,423,357 | 86.4 | 42.8 | 174.8 | -15.8 | 61.1 | 120.0 | -21.0% | -23.1% | -16.3% | **yes** | as sc_big_dwell (training-only change: max_features 0.5; export unchanged) |
| ds3v13 | sc_big_mf5_msl20 | 42 | 2,250,172 | 1,423,357 | 86.5 | 42.8 | 175.4 | -15.1 | 61.1 | 120.1 | -21.0% | -22.9% | -16.3% | **yes** | as sc_big_mf5 (min_samples_leaf 20) |
| ds3v13 | sc_big_mf7 | 42 | 2,250,172 | 1,423,357 | 86.6 | 42.9 | 175.2 | -15.4 | 61.3 | 120.3 | -20.8% | -23.0% | -16.0% | **yes** | as sc_big_dwell (training-only change: max_features 0.7; export unchanged) |
| ds4shiftv25 | baseline | 11 | 2,064,786 | 1,423,357 | 110.0 | 55.9 | 228.7 | -23.5 | 73.3 | 152.7 | — | — | — | — | prod |
| ds4shiftv25 | sc_big_mf3 | 11 | 2,064,786 | 1,423,357 | 86.7 | 42.9 | 175.5 | -14.9 | 61.4 | 120.4 | -21.1% | -23.3% | -16.2% | **yes** | as sc_big_mf5 (max_features 0.3) |
| ds4shiftv25 | sc_big_mf3_fast | 11 | 2,064,786 | 1,423,357 | 87.3 | 43.2 | 177.0 | -14.9 | 61.7 | 121.2 | -20.6% | -22.6% | -15.9% | **yes** | as sc_big_mf3 with at most 600 trees (lr 0.1): ~half the serving time and export size |
| ds4shiftv28 | sc_big_mf3_age1 | 11 | 2,064,786 | 1,423,357 | 86.4 | 42.7 | 174.5 | -15.1 | 61.1 | 119.7 | -21.4% | -23.7% | -16.7% | **yes** | as sc_big_mf3 + pos_age_s only (no ring needed) |
| ds4shiftv28 | sc_big_mf3_age_fo | 11 | 2,064,786 | 1,423,357 | 85.1 | 41.9 | 172.3 | -15.1 | 59.6 | 118.9 | -22.6% | -24.7% | -18.8% | **yes** | as sc_big_mf3_age + feed speed + odometer (per-vehicle 300 s ring of ts / vehicle ts / speed / odometer; < 0.5 day) |
| ds4shiftv29 | sc_big_fo_long | 11 | 2,064,786 | 1,423,357 | 84.2 | 41.8 | 170.6 | -14.7 | 59.1 | 117.7 | -23.4% | -25.4% | -19.4% | **yes** | as sc_big_mf3_age_fo with the ring extended to 1200 s (~120 entries per vehicle) |
| ds4shiftv29 | sc_big_fo_v29 | 11 | 2,064,786 | 1,423,357 | 84.3 | 41.8 | 170.6 | -14.6 | 59.0 | 118.0 | -23.3% | -25.4% | -19.5% | **yes** | as sc_big_fo_long + sc_big_fo_gap |
| ds4shiftv30 | sc_big_fo_novl | 11 | 2,064,786 | 1,423,357 | 85.9 | 42.3 | 173.3 | -16.2 | 59.8 | 120.1 | -21.9% | -24.2% | -18.5% | **yes** | as sc_big_fo_long minus _LAP and _VEH (no per-vehicle crossing log): ~0.5 day less serving work |
| ds4shiftv30 | sc_big_fo_pa | 11 | 2,064,786 | 1,423,357 | 83.9 | 41.6 | 169.1 | -15.7 | 58.8 | 117.5 | -23.6% | -26.1% | -19.8% | **yes** | as sc_big_fo_long + per-cycle link -> [(vehicle, fraction, odometer speed)] dict from the daemon's own trip placement; < 0.5 day |
| ds4shiftv31 | sc_big_fo_pan | 11 | 2,064,786 | 1,423,357 | 83.9 | 41.5 | 169.2 | -15.8 | 58.8 | 117.5 | — | — | — | — | ? |
| ds4shiftv31 | sc_big_fo_pax | 11 | 2,064,786 | 1,423,357 | 83.5 | 41.5 | 168.9 | -15.4 | 58.4 | 117.1 | — | — | — | — | ? |
| ds4shiftv31 | sc_big_fo_v31 | 11 | 2,064,786 | 1,423,357 | 83.5 | 41.5 | 168.4 | -15.8 | 58.6 | 116.9 | — | — | — | — | ? |
| ds4shiftv32 | sc_big_fo_brk | 11 | 2,064,786 | 1,423,357 | 83.9 | 41.6 | 169.5 | -15.7 | 58.7 | 117.6 | — | — | — | — | as sc_big_fo_pa + per-vehicle stopped-episode state (last end / length / odometer, 3 h deque) |
| ds4shiftv32 | sc_big_fo_duty | 11 | 2,064,786 | 1,423,357 | 83.6 | 41.5 | 168.7 | -16.0 | 58.6 | 117.0 | — | — | — | — | as sc_big_fo_pa + per-vehicle shift start (feed t + odometer after the last >= 30 min gap); trivial |
| ds4shiftv32 | sc_big_fo_v32 | 11 | 2,064,786 | 1,423,357 | 83.7 | 41.6 | 169.7 | -15.6 | 58.5 | 117.1 | — | — | — | — | as sc_big_fo_brk + sc_big_fo_duty |
| ds4v25 | baseline | 42 | 2,205,756 | 1,607,392 | 113.9 | 60.4 | 241.6 | -19.3 | 73.4 | 157.8 | — | — | — | — | prod |
| ds4v25 | sc_big_dwell | 42 | 2,205,756 | 1,607,392 | 88.1 | 44.9 | 181.3 | -12.4 | 61.1 | 122.7 | -22.7% | -24.9% | -16.7% | **yes** | as sc_big_veh + static location -> sorted stop-duration table (prior 14 days, built at export like hist_dt); lookup by (last passed stop, 50 m bin) and stationary_sec; ~+0.3 day, ~3 days total |
| ds4v25 | sc_big_mf3 | 42 | 2,205,756 | 1,607,392 | 87.8 | 44.7 | 180.8 | -12.7 | 61.0 | 122.3 | -23.0% | -25.1% | -16.9% | **yes** | as sc_big_mf5 (max_features 0.3) |
| ds4v25 | sc_big_mf3_127 | 42 | 2,205,756 | 1,607,392 | 88.3 | 44.8 | 181.4 | -13.7 | 61.4 | 122.9 | -22.5% | -24.9% | -16.3% | **yes** | as sc_big_mf3 with 127-leaf trees (production per-tree size, ~half the export) |
| ds4v25 | sc_big_mf3_fast | 42 | 2,205,756 | 1,607,392 | 88.7 | 45.0 | 182.6 | -13.1 | 61.7 | 123.5 | -22.1% | -24.4% | -15.8% | **yes** | as sc_big_mf3 with at most 600 trees (lr 0.1): ~half the serving time and export size |
| ds4v28 | baseline | 42 | 2,205,756 | 1,607,392 | 113.9 | 60.4 | 241.6 | -19.3 | 73.4 | 157.8 | — | — | — | — | prod |
| ds4v28 | sc_big_mf3_age | 42 | 2,205,756 | 1,607,392 | 87.2 | 44.3 | 180.3 | -12.8 | 60.4 | 121.5 | -23.5% | -25.4% | -17.6% | **yes** | as sc_big_mf3 + GPS-fix age cols (vehicle_ts already in inference; + a per-vehicle 600 s age ring, < 0.5 day) |
| ds4v28 | sc_big_mf3_age1 | 42 | 2,205,756 | 1,607,392 | 87.4 | 44.5 | 180.3 | -12.9 | 60.7 | 121.9 | -23.3% | -25.4% | -17.2% | **yes** | as sc_big_mf3 + pos_age_s only (no ring needed) |
| ds4v28 | sc_big_mf3_age_fo | 42 | 2,205,756 | 1,607,392 | 86.3 | 43.8 | 178.3 | -13.0 | 59.2 | 120.9 | -24.3% | -26.2% | -19.3% | **yes** | as sc_big_mf3_age + feed speed + odometer (per-vehicle 300 s ring of ts / vehicle ts / speed / odometer; < 0.5 day) |
| ds4v28 | sc_big_mf3_age_odo | 42 | 2,205,756 | 1,607,392 | 86.8 | 44.2 | 179.1 | -13.0 | 59.9 | 121.3 | -23.8% | -25.9% | -18.4% | **yes** | as sc_big_mf3_age + odometer windows (entity field; 300 s ring) |
| ds4v28 | sc_big_mf3_age_spd | 42 | 2,205,756 | 1,607,392 | 86.8 | 43.8 | 179.3 | -13.2 | 59.8 | 121.5 | -23.8% | -25.8% | -18.5% | **yes** | as sc_big_mf3_age + feed-reported speed (entity field; 60 s ring) |
| ds4v29 | baseline | 42 | 2,205,756 | 1,607,392 | 113.9 | 60.4 | 241.6 | -19.3 | 73.4 | 157.8 | — | — | — | — | prod |
| ds4v29 | sc_big_fo_gap | 42 | 2,205,756 | 1,607,392 | 86.0 | 43.7 | 178.0 | -13.0 | 59.0 | 120.6 | -24.5% | -26.3% | -19.6% | **yes** | as sc_big_mf3_age_fo + dist_along in the ring + odometer at trip start |
| ds4v29 | sc_big_fo_long | 42 | 2,205,756 | 1,607,392 | 85.6 | 43.6 | 177.1 | -13.5 | 58.8 | 120.1 | -24.8% | -26.7% | -19.9% | **yes** | as sc_big_mf3_age_fo with the ring extended to 1200 s (~120 entries per vehicle) |
| ds4v29 | sc_big_fo_v29 | 42 | 2,205,756 | 1,607,392 | 85.7 | 43.6 | 177.0 | -13.2 | 58.8 | 120.2 | -24.8% | -26.7% | -19.8% | **yes** | as sc_big_fo_long + sc_big_fo_gap |
| ds4v29 | sc_big_mf3_age_fo | 42 | 2,205,756 | 1,607,392 | 86.3 | 43.8 | 178.3 | -13.0 | 59.2 | 120.9 | -24.3% | -26.2% | -19.3% | **yes** | as sc_big_mf3_age + feed speed + odometer (per-vehicle 300 s ring of ts / vehicle ts / speed / odometer; < 0.5 day) |
| ds4v30 | baseline | 42 | 2,205,756 | 1,607,392 | 113.9 | 60.4 | 241.6 | -19.3 | 73.4 | 157.8 | — | — | — | — | prod |
| ds4v30 | sc_big_fo_long | 42 | 2,205,756 | 1,607,392 | 85.6 | 43.6 | 177.1 | -13.5 | 58.8 | 120.1 | -24.8% | -26.7% | -19.9% | **yes** | as sc_big_mf3_age_fo with the ring extended to 1200 s (~120 entries per vehicle) |
| ds4v30 | sc_big_fo_nolap | 42 | 2,205,756 | 1,607,392 | 86.1 | 43.9 | 177.6 | -13.5 | 59.0 | 120.8 | -24.4% | -26.5% | -19.6% | **yes** | as sc_big_fo_long minus the own previous-lap cols |
| ds4v30 | sc_big_fo_noveh | 42 | 2,205,756 | 1,607,392 | 86.6 | 43.7 | 178.2 | -14.0 | 59.8 | 121.2 | -24.0% | -26.2% | -18.4% | **yes** | as sc_big_fo_long minus the vehicle own/hist ratio cols |
| ds4v30 | sc_big_fo_novl | 42 | 2,205,756 | 1,607,392 | 87.1 | 44.0 | 179.1 | -13.6 | 59.7 | 122.1 | -23.6% | -25.9% | -18.6% | **yes** | as sc_big_fo_long minus _LAP and _VEH (no per-vehicle crossing log): ~0.5 day less serving work |
| ds4v30 | sc_big_fo_pa | 42 | 2,205,756 | 1,607,392 | 85.2 | 43.4 | 175.4 | -13.7 | 58.5 | 119.6 | -25.2% | -27.4% | -20.2% | **yes** | as sc_big_fo_long + per-cycle link -> [(vehicle, fraction, odometer speed)] dict from the daemon's own trip placement; < 0.5 day |
| ds4v31 | baseline | 42 | 2,205,756 | 1,607,392 | 113.9 | 60.4 | 241.6 | -19.3 | 73.4 | 157.8 | — | — | — | — | prod |
| ds4v31 | sc_big_fo_pa | 42 | 2,205,756 | 1,607,392 | 85.2 | 43.4 | 175.4 | -13.7 | 58.5 | 119.6 | -25.2% | -27.4% | -20.2% | **yes** | as sc_big_fo_long + per-cycle link -> [(vehicle, fraction, odometer speed)] dict from the daemon's own trip placement; < 0.5 day |
| ds4v31 | sc_big_fo_pan | 42 | 2,205,756 | 1,607,392 | 85.1 | 43.3 | 175.1 | -13.6 | 58.7 | 119.4 | -25.3% | -27.5% | -20.0% | **yes** | ? |
| ds4v31 | sc_big_fo_pax | 42 | 2,205,756 | 1,607,392 | 85.0 | 43.3 | 174.8 | -14.1 | 58.7 | 119.2 | -25.4% | -27.6% | -20.0% | **yes** | ? |
| ds4v31 | sc_big_fo_v31 | 42 | 2,205,756 | 1,607,392 | 85.1 | 43.3 | 174.6 | -13.8 | 58.5 | 119.6 | -25.3% | -27.7% | -20.2% | **yes** | ? |
| ds4v32 | baseline | 42 | 2,205,756 | 1,607,392 | 113.9 | 60.4 | 241.6 | -19.3 | 73.4 | 157.8 | — | — | — | — | prod |
| ds4v32 | sc_big_fo_brk | 42 | 2,205,756 | 1,607,392 | 85.4 | 43.3 | 175.4 | -14.1 | 58.8 | 119.8 | -25.0% | -27.4% | -19.8% | **yes** | as sc_big_fo_pa + per-vehicle stopped-episode state (last end / length / odometer, 3 h deque) |
| ds4v32 | sc_big_fo_duty | 42 | 2,205,756 | 1,607,392 | 84.8 | 43.2 | 174.6 | -13.7 | 58.4 | 119.0 | -25.6% | -27.7% | -20.4% | **yes** | as sc_big_fo_pa + per-vehicle shift start (feed t + odometer after the last >= 30 min gap); trivial |
| ds4v32 | sc_big_fo_pa | 42 | 2,205,756 | 1,607,392 | 85.2 | 43.4 | 175.4 | -13.7 | 58.5 | 119.6 | -25.2% | -27.4% | -20.2% | **yes** | as sc_big_fo_long + per-cycle link -> [(vehicle, fraction, odometer speed)] dict from the daemon's own trip placement; < 0.5 day |
| ds4v32 | sc_big_fo_v32 | 42 | 2,205,756 | 1,607,392 | 84.7 | 43.3 | 174.9 | -13.4 | 58.1 | 119.0 | -25.6% | -27.6% | -20.7% | **yes** | as sc_big_fo_brk + sc_big_fo_duty |
