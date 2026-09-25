# Leaderboard

Held-out raw-model metrics (seconds); Δ vs the baseline row with the same tag
(day set / split) and seed. Win = MAE ≤ −3%, p90 not worse, no stops_ahead bucket
worse by > 5%. Regenerate with `python research/model-search/report.py`.

Tags: `ds1` = train 2026-09-07..09-18, test 09-19 (Sat), 09-20 (Sun), 09-21 (Mon).
`ds1shift` = train 09-07..09-17, test 09-18 (Fri), 09-19, 09-20. Train rows 1% of
snapshots/day, test 3% (pipeline_lite 10% → prep 3% → run).

| tag | arm | seed | n_train | n_test | MAE | median | p90 | bias | sa1 | sa10 | ΔMAE | Δp90 | worst bucket Δ | win | serving |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

| ds1 | baseline | 42 | 2,178,139 | 1,414,363 | 110.6 | 56.0 | 230.9 | -28.2 | 72.6 | 153.5 | — | — | — | — | prod |
| ds1 | headway | 42 | 2,178,139 | 1,414,363 | 110.0 | 55.5 | 229.9 | -29.2 | 72.5 | 153.4 | -0.5% | -0.4% | -0.1% | no | needs serving work (NaN routing + stop-arrival store) |
| ds1 | live_all | 42 | 2,178,139 | 1,414,363 | 103.7 | 53.5 | 214.5 | -17.8 | 73.4 | 142.7 | -6.2% | -7.1% | +1.2% | **yes** | needs serving work (NaN routing + live link store) |
| ds1 | live_all_s | 42 | 2,178,139 | 1,414,363 | 102.5 | 52.9 | 210.7 | -18.8 | 71.6 | 142.3 | -7.3% | -8.7% | -1.3% | **yes** | needs serving work (live link store, ~1-2 days) |
| ds1 | live_all_s_cold | 42 | 2,178,139 | 1,414,363 | 163.5 | 104.0 | 344.8 | +18.4 | 102.0 | 220.5 | +47.8% | +49.4% | +59.7% | no | ? |
| ds1 | live_all_s_lr10 | 42 | 2,178,139 | 1,414,363 | 102.7 | 53.0 | 213.4 | -16.5 | 73.0 | 141.0 | -7.1% | -7.6% | +0.6% | **yes** | ? |
| ds1 | live_path | 42 | 2,178,139 | 1,414,363 | 104.5 | 52.9 | 216.6 | -20.1 | 73.6 | 143.6 | -5.5% | -6.2% | +1.4% | **yes** | needs serving work (NaN routing + live link store) |
| ds1 | own_speed | 42 | 2,178,139 | 1,414,363 | 110.0 | 56.6 | 229.5 | -24.6 | 72.9 | 153.7 | -0.6% | -0.6% | +0.4% | no | needs serving work (NaN routing + 5-min position buffer) |
| ds1 | path_min_s | 42 | 2,178,139 | 1,414,363 | 103.2 | 52.3 | 211.7 | -22.1 | 70.6 | 142.9 | -6.7% | -8.3% | -2.7% | **yes** | ? |
| ds1 | path_own_s | 42 | 2,178,139 | 1,414,363 | 101.4 | 51.8 | 208.3 | -20.7 | 70.0 | 141.1 | -8.3% | -9.8% | -3.5% | **yes** | ? |
| ds1 | path_s | 42 | 2,178,139 | 1,414,363 | 104.8 | 53.1 | 218.8 | -19.1 | 73.9 | 143.8 | -5.2% | -5.2% | +1.8% | **yes** | ? |
| ds1shift | baseline | 7 | 1,978,242 | 1,424,485 | 113.0 | 59.5 | 241.1 | -14.2 | 71.2 | 159.7 | — | — | — | — | prod |
| ds1shift | live_all | 7 | 1,978,242 | 1,424,485 | 103.0 | 54.1 | 212.4 | -14.7 | 69.9 | 144.0 | -8.9% | -11.9% | -1.9% | **yes** | needs serving work (NaN routing + live link store) |
| ds1shift | live_all_s | 7 | 1,978,242 | 1,424,485 | 102.4 | 53.2 | 211.8 | -16.3 | 68.8 | 143.4 | -9.4% | -12.2% | -3.4% | **yes** | needs serving work (live link store, ~1-2 days) |
