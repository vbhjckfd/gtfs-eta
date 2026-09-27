# Leaderboard

Held-out raw-model metrics (seconds); Δ vs the baseline row with the same tag
(day set / split) and seed. Win = MAE ≤ −3%, p90 not worse, no stops_ahead bucket
worse by > 5%. Regenerate with `python research/model-search/report.py`.

Superseded tags (ds1 .. ds1v9, ds1shift .. ds1shiftv9, ds1v5k3, ds1lag90w15) are in
`archive/LEADERBOARD-old.md`; their baselines are identical to the ones below.

Tags: `ds1v12` / `ds1shiftv12` = rebuilt in run 15 (v10 files + V12 add-on cols from `addon_v12.py`: same-route headway / bunching, coarse dwell fallback; MS_FEAT_DIR=ms_features_v12); baselines identical. `ds1v12k3` = 3x train rows.
`ds1v10` / `ds1shiftv10` = rebuilt in run 14 (+ V10 cols: conditional remaining dwell at the current location from prior days' stationary runs; MS_FEAT_DIR=ms_features_v10); `ds1v11` / `ds1shiftv11` = same rows + V11 add-on cols (hour-banded dwell, live same-location dwell; MS_FEAT_DIR=ms_features_v11); baselines identical.
`ds1v9` / `ds1shiftv9` = rebuilt in run 13 (+ V9 cols: network ratio net_ratio15 / net_n15, veh_hist_ratio20, clipped veh_hist_ratio60c; MS_FEAT_DIR=ms_features_v9); baselines identical. `ds1v9k3` = ds1v9 with train rows at 3% of snapshots (3x).
`ds1v8` / `ds1shiftv8` = rebuilt in run 12 (+ V8 cols: 6 h previous lap lap6_*, vehicle-level own/hist ratio veh_hist_ratio60 / veh_hist_n60; MS_FEAT_DIR=ms_features_v8); baselines identical.
`ds1v7` / `ds1shiftv7` = rebuilt in run 11 (+ V7 own previous-lap cols: lap_path_sec / cov / age, lap_fill_path_sec; MS_FEAT_DIR=ms_features_v7); baselines identical.
`ds1v6` / `ds1shiftv6` = v4/v5 features rebuilt in run 8 (+ V6 current-link cols: next stop id, current-link live m5 / hist / fraction left); baselines identical.
`ds1v5` / `ds1shiftv5` = v4 features rebuilt in run 7 (+ own-vs-hist cols); baselines identical. `ds1v5k3` = ds1v5 with train rows at 3% of snapshots (3x) instead of 1%.
`ds1v4` / `ds1shiftv4` = same splits, features v4 (MS_FEAT_DIR=ms_features_v4: pipeline from 08-31 so the historical link table has 7-14 prior days for every train row, + day-type table, live/hist ratio, p75); baselines identical.
`ds1v3` / `ds1shiftv3` = same splits, features v3 (MS_FEAT_DIR=ms_features_v3: + m7, EWMA, historical link table from prior days); baselines identical.
`ds1v2` / `ds1shiftv2` = same splits, features prepped with the run-3 median-of-5 column (MS_FEAT_DIR=ms_features_v2); baselines identical.
`ds1lag90w15` = ds1 with live features rebuilt at 90 s detection lag, 15-min link window.
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
| ds1v11 | baseline | 42 | 2,178,139 | 1,414,363 | 110.6 | 56.0 | 230.9 | -28.2 | 72.6 | 153.5 | — | — | — | — | prod |
| ds1v11 | sc_big_dwell | 42 | 2,178,139 | 1,414,363 | 87.0 | 43.1 | 178.2 | -12.6 | 60.7 | 121.6 | -21.3% | -22.8% | -16.4% | **yes** | as sc_big_veh + static location -> sorted stop-duration table (prior 14 days, built at export like hist_dt); lookup by (last passed stop, 50 m bin) and stationary_sec; ~+0.3 day, ~3 days total |
| ds1v11 | sc_big_dwell_h | 42 | 2,178,139 | 1,414,363 | 87.1 | 43.0 | 177.9 | -13.5 | 60.4 | 122.1 | -21.2% | -22.9% | -16.8% | **yes** | as sc_big_dwell with the table also keyed by 3-hour band |
| ds1v11 | sc_big_dwell_v11 | 42 | 2,178,139 | 1,414,363 | 87.1 | 43.1 | 177.9 | -13.3 | 60.4 | 122.2 | -21.2% | -22.9% | -16.7% | **yes** | as sc_big_dwell_h + per-location ring of the last 3 live stop durations |
| ds1v12 | baseline | 42 | 2,178,139 | 1,414,363 | 110.6 | 56.0 | 230.9 | -28.2 | 72.6 | 153.5 | — | — | — | — | prod |
| ds1v12 | sc_big_dwc | 42 | 2,178,139 | 1,414,363 | 87.1 | 43.1 | 178.2 | -12.5 | 60.6 | 121.9 | -21.3% | -22.8% | -16.5% | **yes** | as sc_big_dwell + stop-keyed dwell table (same export step) |
| ds1v12 | sc_big_dwell | 42 | 2,178,139 | 1,414,363 | 87.0 | 43.1 | 178.2 | -12.6 | 60.7 | 121.6 | -21.3% | -22.8% | -16.4% | **yes** | as sc_big_veh + static location -> sorted stop-duration table (prior 14 days, built at export like hist_dt); lookup by (last passed stop, 50 m bin) and stationary_sec; ~+0.3 day, ~3 days total |
| ds1v12 | sc_big_hw | 42 | 2,178,139 | 1,414,363 | 87.2 | 43.1 | 178.1 | -13.5 | 60.9 | 121.9 | -21.1% | -22.8% | -16.1% | **yes** | as sc_big_dwell + (route, dir, stop) -> last 2 arrivals dict fed by the crossing events; ~+0.2 day |
| ds1v12 | sc_big_v12 | 42 | 2,178,139 | 1,414,363 | 87.3 | 43.1 | 178.2 | -12.9 | 60.8 | 122.2 | -21.1% | -22.8% | -16.2% | **yes** | as sc_big_hw + stop-keyed dwell table |
| ds1v12k3 | baseline | 42 | 6,510,353 | 1,414,363 | 108.1 | 54.4 | 226.9 | -27.5 | 70.4 | 149.9 | — | — | — | — | prod |
| ds1v12k3 | sc_big_dwell | 42 | 6,510,353 | 1,414,363 | 84.7 | 42.1 | 173.3 | -13.4 | 58.6 | 118.7 | -21.7% | -23.6% | -16.8% | **yes** | as sc_big_veh + static location -> sorted stop-duration table (prior 14 days, built at export like hist_dt); lookup by (last passed stop, 50 m bin) and stationary_sec; ~+0.3 day, ~3 days total |
| ds1v9k3 | baseline | 42 | 6,510,353 | 1,414,363 | 108.1 | 54.4 | 226.9 | -27.5 | 70.4 | 149.9 | — | — | — | — | prod |
| ds1v9k3 | sc_big_veh | 42 | 6,510,353 | 1,414,363 | 85.4 | 42.4 | 174.3 | -14.7 | 60.2 | 119.2 | -21.0% | -23.2% | -14.5% | **yes** | as sc_big_lap + per-vehicle 1 h deque of (own link time, hist link time) from the same crossing events; ~2.7 days total |
