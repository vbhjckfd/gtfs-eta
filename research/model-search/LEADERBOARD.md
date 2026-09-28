# Leaderboard

Held-out raw-model metrics (seconds); Δ vs the baseline row with the same tag
(day set / split) and seed. Win = MAE ≤ −3%, p90 not worse, no stops_ahead bucket
worse by > 5%. Regenerate with `python research/model-search/report.py`.

Superseded tags (ds1 .. ds1v9, ds1shift .. ds1shiftv9, ds1v5k3, ds1v9k3, ds1v11, ds1v12,
ds1lag90w15 and their shift twins) are in `archive/LEADERBOARD-old.md` with their definitions;
their baselines are identical to the ds1 / ds1shift ones below.

Tags: `ds1v18` / `ds1shiftv18` = run 18, v10 features rebuilt (V8/V9 losers pruned; the leader's cols are unchanged), hyper-parameter / monotone arms.
`ds1h21` / `ds1h7` = run 17, ds1 split with the history link / dwell tables built from the prior 21 / 7 days (MS_HIST_DAYS; ds1v10f is the 14-day case); the baseline row is the ds1v10f baseline (it uses no history cols).
`ds1v10f` / `ds2w` / `ds2` = run 16, v10 features rebuilt from 08-17 (full 14-day history for every train day); ds1v10f = ds1 split; ds2w = train 08-31..09-18 at 0.63% (same rows as ds1); ds2 = same 19 days at 1%; test 09-19..09-21. Baselines differ between these tags (different training data).
`ds1v10` / `ds1shiftv10` = rebuilt in run 14 (+ V10 cols: conditional remaining dwell at the current location from prior days' stationary runs; MS_FEAT_DIR=ms_features_v10); baselines identical. `ds1v12k3` = run 15, the same v10 cols at 3x train rows (3% of snapshots).
`ds1` = train 2026-09-07..09-18, test 09-19 (Sat), 09-20 (Sun), 09-21 (Mon).
`ds1shift` = train 09-07..09-17, test 09-18 (Fri), 09-19, 09-20. Train rows 1% of
snapshots/day, test 3% (pipeline_lite 10% → prep 3% → run).

| tag | arm | seed | n_train | n_test | MAE | median | p90 | bias | sa1 | sa10 | ΔMAE | Δp90 | worst bucket Δ | win | serving |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ds1h21 | baseline | 42 | 2,178,139 | 1,414,363 | 110.6 | 56.0 | 230.9 | -28.2 | 72.6 | 153.5 | — | — | — | — | prod |
| ds1h21 | sc_big_dwell | 42 | 2,178,139 | 1,414,363 | 87.2 | 43.1 | 178.3 | -11.8 | 60.7 | 122.2 | -21.2% | -22.8% | -16.4% | **yes** | as sc_big_veh + static location -> sorted stop-duration table (prior 14 days, built at export like hist_dt); lookup by (last passed stop, 50 m bin) and stationary_sec; ~+0.3 day, ~3 days total |
| ds1h7 | baseline | 42 | 2,178,139 | 1,414,363 | 110.6 | 56.0 | 230.9 | -28.2 | 72.6 | 153.5 | — | — | — | — | prod |
| ds1h7 | sc_big_dwell | 42 | 2,178,139 | 1,414,363 | 87.8 | 43.5 | 178.8 | -14.2 | 61.6 | 123.0 | -20.6% | -22.6% | -15.2% | **yes** | as sc_big_veh + static location -> sorted stop-duration table (prior 14 days, built at export like hist_dt); lookup by (last passed stop, 50 m bin) and stationary_sec; ~+0.3 day, ~3 days total |
| ds1shiftv10 | baseline | 7 | 1,978,242 | 1,424,485 | 113.0 | 59.5 | 241.1 | -14.2 | 71.2 | 159.7 | — | — | — | — | prod |
| ds1shiftv10 | sc_big_dwell | 7 | 1,978,242 | 1,424,485 | 87.0 | 43.9 | 179.5 | -14.6 | 59.7 | 122.3 | -23.1% | -25.6% | -16.1% | **yes** | as sc_big_veh + static location -> sorted stop-duration table (prior 14 days, built at export like hist_dt); lookup by (last passed stop, 50 m bin) and stationary_sec; ~+0.3 day, ~3 days total |
| ds1shiftv10 | sc_big_dwell2 | 7 | 1,978,242 | 1,424,485 | 87.0 | 43.8 | 179.4 | -14.9 | 59.6 | 122.4 | -23.0% | -25.6% | -16.3% | **yes** | as sc_big_dwell (2 of its 5 cols) |
| ds1shiftv10 | sc_big_veh | 7 | 1,978,242 | 1,424,485 | 87.7 | 44.1 | 180.6 | -16.0 | 60.7 | 123.2 | -22.4% | -25.1% | -14.7% | **yes** | as sc_big_lap + per-vehicle 1 h deque of (own link time, hist link time) from the same crossing events; ~2.7 days total |
| ds1v10 | baseline | 42 | 2,178,139 | 1,414,363 | 110.6 | 56.0 | 230.9 | -28.2 | 72.6 | 153.5 | — | — | — | — | prod |
| ds1v10 | sc_big_dwell | 42 | 2,178,139 | 1,414,363 | 87.0 | 43.1 | 178.2 | -12.6 | 60.7 | 121.6 | -21.3% | -22.8% | -16.4% | **yes** | as sc_big_veh + static location -> sorted stop-duration table (prior 14 days, built at export like hist_dt); lookup by (last passed stop, 50 m bin) and stationary_sec; ~+0.3 day, ~3 days total |
| ds1v10 | sc_big_dwell2 | 42 | 2,178,139 | 1,414,363 | 87.2 | 43.1 | 177.7 | -13.8 | 60.8 | 122.0 | -21.1% | -23.0% | -16.2% | **yes** | as sc_big_dwell (2 of its 5 cols) |
| ds1v10 | sc_big_veh | 42 | 2,178,139 | 1,414,363 | 87.7 | 43.3 | 179.1 | -14.0 | 62.0 | 122.3 | -20.7% | -22.4% | -14.6% | **yes** | as sc_big_lap + per-vehicle 1 h deque of (own link time, hist link time) from the same crossing events; ~2.7 days total |
| ds1v10f | baseline | 42 | 2,178,139 | 1,414,363 | 110.6 | 56.0 | 230.9 | -28.2 | 72.6 | 153.5 | — | — | — | — | prod |
| ds1v10f | sc_big_dwell | 42 | 2,178,139 | 1,414,363 | 87.0 | 43.2 | 178.2 | -12.1 | 60.6 | 122.0 | -21.3% | -22.8% | -16.6% | **yes** | as sc_big_veh + static location -> sorted stop-duration table (prior 14 days, built at export like hist_dt); lookup by (last passed stop, 50 m bin) and stationary_sec; ~+0.3 day, ~3 days total |
| ds1v12k3 | baseline | 42 | 6,510,353 | 1,414,363 | 108.1 | 54.4 | 226.9 | -27.5 | 70.4 | 149.9 | — | — | — | — | prod |
| ds1v12k3 | sc_big_dwell | 42 | 6,510,353 | 1,414,363 | 84.7 | 42.1 | 173.3 | -13.4 | 58.6 | 118.7 | -21.7% | -23.6% | -16.8% | **yes** | as sc_big_veh + static location -> sorted stop-duration table (prior 14 days, built at export like hist_dt); lookup by (last passed stop, 50 m bin) and stationary_sec; ~+0.3 day, ~3 days total |
| ds2 | baseline | 42 | 3,373,426 | 1,414,363 | 107.4 | 53.6 | 224.3 | -31.1 | 70.6 | 149.1 | — | — | — | — | prod |
| ds2 | sc_big_dwell | 42 | 3,373,426 | 1,414,363 | 85.3 | 42.5 | 175.1 | -13.2 | 59.3 | 119.4 | -20.5% | -21.9% | -16.1% | **yes** | as sc_big_veh + static location -> sorted stop-duration table (prior 14 days, built at export like hist_dt); lookup by (last passed stop, 50 m bin) and stationary_sec; ~+0.3 day, ~3 days total |
| ds2w | baseline | 42 | 2,126,692 | 1,414,363 | 108.9 | 54.3 | 227.7 | -31.0 | 71.5 | 151.1 | — | — | — | — | prod |
| ds2w | sc_big_dwell | 42 | 2,126,692 | 1,414,363 | 86.3 | 43.0 | 177.3 | -11.8 | 59.9 | 120.8 | -20.7% | -22.2% | -16.2% | **yes** | as sc_big_veh + static location -> sorted stop-duration table (prior 14 days, built at export like hist_dt); lookup by (last passed stop, 50 m bin) and stationary_sec; ~+0.3 day, ~3 days total |
