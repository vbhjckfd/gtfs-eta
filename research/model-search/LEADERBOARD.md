# Leaderboard

Held-out raw-model metrics (seconds); Δ vs the baseline row with the same tag
(day set / split) and seed. Win = MAE ≤ −3%, p90 not worse, no stops_ahead bucket
worse by > 5%. Regenerate with `python research/model-search/report.py`.

Superseded tags (ds1 .. ds1v9, ds1shift .. ds1shiftv9, ds1v5k3, ds1v9k3, ds1v11, ds1v12,
ds1lag90w15, ds1h21 / ds1h7, ds1r32 / ds1r72 and their shift twins) are in `archive/LEADERBOARD-old.md` with their definitions;
their baselines are identical to the ds1 / ds1shift ones below.

Tags: `ds1v20` / `ds1shiftv20` = run 20, v10 features rebuilt (baseline + leader reproduce ds1v18/v19). `ds1v13` / `ds1shiftv13` = run 20 add-on: v10 rows + V13 trip-keyed hold cols (addon_v13.py, MS_FEAT_DIR=ms_features_v13); baselines identical.
`ds1v19` = run 19, v10 features rebuilt (baseline + leader reproduce ds1v18). `ds1v19t` = run 19 timing refits (same data as ds1v19, models saved).
`ds1v18` / `ds1shiftv18` = run 18, v10 features rebuilt (V8/V9 losers pruned; the leader's cols are unchanged), hyper-parameter / monotone arms.
`ds1v10f` / `ds2w` / `ds2` = run 16, v10 features rebuilt from 08-17 (full 14-day history for every train day); ds1v10f = ds1 split; ds2w = train 08-31..09-18 at 0.63% (same rows as ds1); ds2 = same 19 days at 1%; test 09-19..09-21. Baselines differ between these tags (different training data).
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
| ds1shiftv13 | baseline | 7 | 1,978,242 | 1,424,485 | 113.0 | 59.5 | 241.1 | -14.2 | 71.2 | 159.7 | — | — | — | — | prod |
| ds1shiftv13 | sc_big_dwell | 7 | 1,978,242 | 1,424,485 | 87.0 | 43.9 | 179.5 | -14.6 | 59.7 | 122.3 | -23.1% | -25.6% | -16.1% | **yes** | as sc_big_veh + static location -> sorted stop-duration table (prior 14 days, built at export like hist_dt); lookup by (last passed stop, 50 m bin) and stationary_sec; ~+0.3 day, ~3 days total |
| ds1shiftv13 | sc_big_hold | 7 | 1,978,242 | 1,424,485 | 86.5 | 43.9 | 179.4 | -13.7 | 59.0 | 122.0 | -23.5% | -25.6% | -17.1% | **yes** | as sc_big_dwell + trip x location dwell table and trip -> per-day hold list (static, built at export); ~+0.5 day |
| ds1shiftv18 | baseline | 7 | 1,978,242 | 1,424,485 | 113.0 | 59.5 | 241.1 | -14.2 | 71.2 | 159.7 | — | — | — | — | prod |
| ds1shiftv18 | sc_big_511 | 7 | 1,978,242 | 1,424,485 | 86.5 | 43.8 | 178.7 | -14.4 | 59.1 | 122.0 | -23.5% | -25.9% | -17.0% | **yes** | as sc_big_dwell with 511-leaf trees (deeper walk, ~2x export size) |
| ds1shiftv18 | sc_big_511_it2k | 7 | 1,978,242 | 1,424,485 | 86.4 | 43.8 | 178.8 | -13.8 | 59.0 | 121.8 | -23.5% | -25.9% | -17.2% | **yes** | as sc_big_dwell with 511-leaf trees and up to ~2400 of them (~4x export size, ~2x serving time) |
| ds1shiftv18 | sc_big_dwell | 7 | 1,978,242 | 1,424,485 | 87.0 | 43.9 | 179.5 | -14.6 | 59.7 | 122.3 | -23.1% | -25.6% | -16.1% | **yes** | as sc_big_veh + static location -> sorted stop-duration table (prior 14 days, built at export like hist_dt); lookup by (last passed stop, 50 m bin) and stationary_sec; ~+0.3 day, ~3 days total |
| ds1shiftv18 | sc_big_it2k | 7 | 1,978,242 | 1,424,485 | 86.6 | 43.9 | 179.2 | -13.9 | 59.3 | 121.9 | -23.3% | -25.7% | -16.8% | **yes** | as sc_big_dwell with up to 2x the trees (~2400; serving time scales ~2x, ~3 s per 3000 rows) |
| ds1shiftv20 | baseline | 7 | 1,978,242 | 1,424,485 | 113.0 | 59.5 | 241.1 | -14.2 | 71.2 | 159.7 | — | — | — | — | prod |
| ds1shiftv20 | sc_big_dwell | 7 | 1,978,242 | 1,424,485 | 87.0 | 43.9 | 179.5 | -14.6 | 59.7 | 122.3 | -23.1% | -25.6% | -16.1% | **yes** | as sc_big_veh + static location -> sorted stop-duration table (prior 14 days, built at export like hist_dt); lookup by (last passed stop, 50 m bin) and stationary_sec; ~+0.3 day, ~3 days total |
| ds1shiftv20 | sc_big_nw | 7 | 1,978,242 | 1,424,485 | 86.9 | 43.8 | 179.6 | -14.7 | 59.7 | 122.2 | -23.1% | -25.5% | -16.2% | **yes** | as sc_big_dwell (training-only change) |
| ds1v10 | baseline | 42 | 2,178,139 | 1,414,363 | 110.6 | 56.0 | 230.9 | -28.2 | 72.6 | 153.5 | — | — | — | — | prod |
| ds1v10 | sc_big_dwell | 42 | 2,178,139 | 1,414,363 | 87.0 | 43.1 | 178.2 | -12.6 | 60.7 | 121.6 | -21.3% | -22.8% | -16.4% | **yes** | as sc_big_veh + static location -> sorted stop-duration table (prior 14 days, built at export like hist_dt); lookup by (last passed stop, 50 m bin) and stationary_sec; ~+0.3 day, ~3 days total |
| ds1v10 | sc_big_dwell2 | 42 | 2,178,139 | 1,414,363 | 87.2 | 43.1 | 177.7 | -13.8 | 60.8 | 122.0 | -21.1% | -23.0% | -16.2% | **yes** | as sc_big_dwell (2 of its 5 cols) |
| ds1v10 | sc_big_veh | 42 | 2,178,139 | 1,414,363 | 87.7 | 43.3 | 179.1 | -14.0 | 62.0 | 122.3 | -20.7% | -22.4% | -14.6% | **yes** | as sc_big_lap + per-vehicle 1 h deque of (own link time, hist link time) from the same crossing events; ~2.7 days total |
| ds1v10f | baseline | 42 | 2,178,139 | 1,414,363 | 110.6 | 56.0 | 230.9 | -28.2 | 72.6 | 153.5 | — | — | — | — | prod |
| ds1v10f | sc_big_dwell | 42 | 2,178,139 | 1,414,363 | 87.0 | 43.2 | 178.2 | -12.1 | 60.6 | 122.0 | -21.3% | -22.8% | -16.6% | **yes** | as sc_big_veh + static location -> sorted stop-duration table (prior 14 days, built at export like hist_dt); lookup by (last passed stop, 50 m bin) and stationary_sec; ~+0.3 day, ~3 days total |
| ds1v12k3 | baseline | 42 | 6,510,353 | 1,414,363 | 108.1 | 54.4 | 226.9 | -27.5 | 70.4 | 149.9 | — | — | — | — | prod |
| ds1v12k3 | sc_big_dwell | 42 | 6,510,353 | 1,414,363 | 84.7 | 42.1 | 173.3 | -13.4 | 58.6 | 118.7 | -21.7% | -23.6% | -16.8% | **yes** | as sc_big_veh + static location -> sorted stop-duration table (prior 14 days, built at export like hist_dt); lookup by (last passed stop, 50 m bin) and stationary_sec; ~+0.3 day, ~3 days total |
| ds1v13 | baseline | 42 | 2,178,139 | 1,414,363 | 110.6 | 56.0 | 230.9 | -28.2 | 72.6 | 153.5 | — | — | — | — | prod |
| ds1v13 | sc_big_dwell | 42 | 2,178,139 | 1,414,363 | 87.0 | 43.1 | 178.2 | -12.6 | 60.7 | 121.6 | -21.3% | -22.8% | -16.4% | **yes** | as sc_big_veh + static location -> sorted stop-duration table (prior 14 days, built at export like hist_dt); lookup by (last passed stop, 50 m bin) and stationary_sec; ~+0.3 day, ~3 days total |
| ds1v13 | sc_big_hold | 42 | 2,178,139 | 1,414,363 | 87.0 | 43.2 | 178.6 | -11.6 | 60.4 | 122.0 | -21.4% | -22.7% | -16.8% | **yes** | as sc_big_dwell + trip x location dwell table and trip -> per-day hold list (static, built at export); ~+0.5 day |
| ds1v18 | baseline | 42 | 2,178,139 | 1,414,363 | 110.6 | 56.0 | 230.9 | -28.2 | 72.6 | 153.5 | — | — | — | — | prod |
| ds1v18 | sc_big_511 | 42 | 2,178,139 | 1,414,363 | 86.7 | 43.1 | 177.4 | -12.7 | 60.3 | 121.4 | -21.6% | -23.2% | -17.0% | **yes** | as sc_big_dwell with 511-leaf trees (deeper walk, ~2x export size) |
| ds1v18 | sc_big_511_it2k | 42 | 2,178,139 | 1,414,363 | 86.5 | 43.1 | 177.3 | -11.6 | 59.9 | 121.2 | -21.7% | -23.2% | -17.4% | **yes** | as sc_big_dwell with 511-leaf trees and up to ~2400 of them (~4x export size, ~2x serving time) |
| ds1v18 | sc_big_dwell | 42 | 2,178,139 | 1,414,363 | 87.0 | 43.1 | 178.2 | -12.6 | 60.7 | 121.6 | -21.3% | -22.8% | -16.4% | **yes** | as sc_big_veh + static location -> sorted stop-duration table (prior 14 days, built at export like hist_dt); lookup by (last passed stop, 50 m bin) and stationary_sec; ~+0.3 day, ~3 days total |
| ds1v18 | sc_big_it2k | 42 | 2,178,139 | 1,414,363 | 86.7 | 43.0 | 177.9 | -11.9 | 60.3 | 121.3 | -21.6% | -22.9% | -17.0% | **yes** | as sc_big_dwell with up to 2x the trees (~2400; serving time scales ~2x, ~3 s per 3000 rows) |
| ds1v18 | sc_big_mono | 42 | 2,178,139 | 1,414,363 | 87.2 | 43.2 | 178.1 | -13.2 | 61.0 | 121.7 | -21.2% | -22.8% | -15.9% | **yes** | as sc_big_dwell (monotone constraints live inside the trees; export unchanged) |
| ds1v19 | baseline | 42 | 2,178,139 | 1,414,363 | 110.6 | 56.0 | 230.9 | -28.2 | 72.6 | 153.5 | — | — | — | — | prod |
| ds1v19 | sc_big_dwell | 42 | 2,178,139 | 1,414,363 | 87.0 | 43.1 | 178.2 | -12.6 | 60.7 | 121.6 | -21.3% | -22.8% | -16.4% | **yes** | as sc_big_veh + static location -> sorted stop-duration table (prior 14 days, built at export like hist_dt); lookup by (last passed stop, 50 m bin) and stationary_sec; ~+0.3 day, ~3 days total |
| ds1v19t | baseline | 42 | 2,178,139 | 1,414,363 | 110.6 | 56.0 | 230.9 | -28.2 | 72.6 | 153.5 | — | — | — | — | prod |
| ds1v19t | sc_big_511_it2k | 42 | 2,178,139 | 1,414,363 | 86.5 | 43.1 | 177.3 | -11.6 | 59.9 | 121.2 | -21.7% | -23.2% | -17.4% | **yes** | as sc_big_dwell with 511-leaf trees and up to ~2400 of them (~4x export size, ~2x serving time) |
| ds1v19t | sc_big_dwell | 42 | 2,178,139 | 1,414,363 | 87.0 | 43.1 | 178.2 | -12.6 | 60.7 | 121.6 | -21.3% | -22.8% | -16.4% | **yes** | as sc_big_veh + static location -> sorted stop-duration table (prior 14 days, built at export like hist_dt); lookup by (last passed stop, 50 m bin) and stationary_sec; ~+0.3 day, ~3 days total |
| ds1v20 | baseline | 42 | 2,178,139 | 1,414,363 | 110.6 | 56.0 | 230.9 | -28.2 | 72.6 | 153.5 | — | — | — | — | prod |
| ds1v20 | sc_big_dwell | 42 | 2,178,139 | 1,414,363 | 87.0 | 43.1 | 178.2 | -12.6 | 60.7 | 121.6 | -21.3% | -22.8% | -16.4% | **yes** | as sc_big_veh + static location -> sorted stop-duration table (prior 14 days, built at export like hist_dt); lookup by (last passed stop, 50 m bin) and stationary_sec; ~+0.3 day, ~3 days total |
| ds1v20 | sc_big_nw | 42 | 2,178,139 | 1,414,363 | 86.9 | 43.1 | 177.8 | -12.4 | 60.7 | 121.6 | -21.4% | -23.0% | -16.4% | **yes** | as sc_big_dwell (training-only change) |
| ds2 | baseline | 42 | 3,373,426 | 1,414,363 | 107.4 | 53.6 | 224.3 | -31.1 | 70.6 | 149.1 | — | — | — | — | prod |
| ds2 | sc_big_dwell | 42 | 3,373,426 | 1,414,363 | 85.3 | 42.5 | 175.1 | -13.2 | 59.3 | 119.4 | -20.5% | -21.9% | -16.1% | **yes** | as sc_big_veh + static location -> sorted stop-duration table (prior 14 days, built at export like hist_dt); lookup by (last passed stop, 50 m bin) and stationary_sec; ~+0.3 day, ~3 days total |
| ds2w | baseline | 42 | 2,126,692 | 1,414,363 | 108.9 | 54.3 | 227.7 | -31.0 | 71.5 | 151.1 | — | — | — | — | prod |
| ds2w | sc_big_dwell | 42 | 2,126,692 | 1,414,363 | 86.3 | 43.0 | 177.3 | -11.8 | 59.9 | 120.8 | -20.7% | -22.2% | -16.2% | **yes** | as sc_big_veh + static location -> sorted stop-duration table (prior 14 days, built at export like hist_dt); lookup by (last passed stop, 50 m bin) and stationary_sec; ~+0.3 day, ~3 days total |
