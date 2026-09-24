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
| ds1 | live_path | 42 | 2,178,139 | 1,414,363 | 104.5 | 52.9 | 216.6 | -20.1 | 73.6 | 143.6 | -5.5% | -6.2% | +1.4% | **yes** | needs serving work (NaN routing + live link store) |
| ds1 | own_speed | 42 | 2,178,139 | 1,414,363 | 110.0 | 56.6 | 229.5 | -24.6 | 72.9 | 153.7 | -0.6% | -0.6% | +0.4% | no | needs serving work (NaN routing + 5-min position buffer) |
