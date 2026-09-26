# Leaderboard

Held-out raw-model metrics (seconds); Δ vs the baseline row with the same tag
(day set / split) and seed. Win = MAE ≤ −3%, p90 not worse, no stops_ahead bucket
worse by > 5%. Regenerate with `python research/model-search/report.py`.

Tags: `ds1v5` / `ds1shiftv5` = v4 features rebuilt in run 7 (+ own-vs-hist cols); baselines identical. `ds1v5k3` = ds1v5 with train rows at 3% of snapshots (3x) instead of 1%.
`ds1v4` / `ds1shiftv4` = same splits, features v4 (MS_FEAT_DIR=ms_features_v4: pipeline from 08-31 so the historical link table has 7-14 prior days for every train row, + day-type table, live/hist ratio, p75); baselines identical.
`ds1v3` / `ds1shiftv3` = same splits, features v3 (MS_FEAT_DIR=ms_features_v3: + m7, EWMA, historical link table from prior days); baselines identical.
`ds1v2` / `ds1shiftv2` = same splits, features prepped with the run-3 median-of-5 column (MS_FEAT_DIR=ms_features_v2); baselines identical.
`ds1lag90w15` = ds1 with live features rebuilt at 90 s detection lag, 15-min link window.
`ds1` = train 2026-09-07..09-18, test 09-19 (Sat), 09-20 (Sun), 09-21 (Mon).
`ds1shift` = train 09-07..09-17, test 09-18 (Fri), 09-19, 09-20. Train rows 1% of
snapshots/day, test 3% (pipeline_lite 10% → prep 3% → run).

| tag | arm | seed | n_train | n_test | MAE | median | p90 | bias | sa1 | sa10 | ΔMAE | Δp90 | worst bucket Δ | win | serving |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

| ds1 | baseline | 42 | 2,178,139 | 1,414,363 | 110.6 | 56.0 | 230.9 | -28.2 | 72.6 | 153.5 | — | — | — | — | prod |
| ds1 | headway | 42 | 2,178,139 | 1,414,363 | 110.0 | 55.5 | 229.9 | -29.2 | 72.5 | 153.4 | -0.5% | -0.4% | -0.1% | no | needs serving work (NaN routing + stop-arrival store) |
| ds1 | live_all | 42 | 2,178,139 | 1,414,363 | 103.7 | 53.5 | 214.5 | -17.8 | 73.4 | 142.7 | -6.2% | -7.1% | +1.2% | **yes** | needs serving work (NaN routing + live link store) |
| ds1 | live_all_s | 42 | 2,178,139 | 1,414,363 | 102.5 | 52.9 | 210.7 | -18.8 | 71.6 | 142.3 | -7.3% | -8.7% | -1.3% | **yes** | needs serving work (live link store, ~1-2 days) |
| ds1 | live_all_s_cold | 42 | 2,178,139 | 1,414,363 | 163.5 | 104.0 | 344.8 | +18.4 | 102.0 | 220.5 | +47.8% | +49.4% | +59.7% | no | diagnostic (cold start) |
| ds1 | live_all_s_lr10 | 42 | 2,178,139 | 1,414,363 | 102.7 | 53.0 | 213.4 | -16.5 | 73.0 | 141.0 | -7.1% | -7.6% | +0.6% | **yes** | as live_all_s |
| ds1 | live_path | 42 | 2,178,139 | 1,414,363 | 104.5 | 52.9 | 216.6 | -20.1 | 73.6 | 143.6 | -5.5% | -6.2% | +1.4% | **yes** | needs serving work (NaN routing + live link store) |
| ds1 | own_speed | 42 | 2,178,139 | 1,414,363 | 110.0 | 56.6 | 229.5 | -24.6 | 72.9 | 153.7 | -0.6% | -0.6% | +0.4% | no | needs serving work (NaN routing + 5-min position buffer) |
| ds1 | path_min_s | 42 | 2,178,139 | 1,414,363 | 103.2 | 52.3 | 211.7 | -22.1 | 70.6 | 142.9 | -6.7% | -8.3% | -2.7% | **yes** | needs serving work (link store, 2 cols) |
| ds1 | path_own_m3 | 42 | 2,178,139 | 1,414,363 | 98.4 | 49.8 | 199.2 | -19.9 | 69.9 | 136.1 | -11.1% | -13.7% | -3.7% | **yes** | as path_own_s; store keeps last 3 traversals per link |
| ds1 | path_own_m3only | 42 | 2,178,139 | 1,414,363 | 98.3 | 49.3 | 199.8 | -20.7 | 69.0 | 136.1 | -11.1% | -13.5% | -4.9% | **yes** | as path_own_s; store keeps last 3 traversals per link |
| ds1 | path_own_s | 42 | 2,178,139 | 1,414,363 | 101.4 | 51.8 | 208.3 | -20.7 | 70.0 | 141.1 | -8.3% | -9.8% | -3.5% | **yes** | needs serving work (link store + 5-min position ring, ~1 day), state persisted across the 5-min push-feed restarts |
| ds1 | path_own_s_big | 42 | 2,178,139 | 1,414,363 | 100.5 | 51.4 | 206.3 | -18.7 | 69.3 | 140.2 | -9.1% | -10.7% | -4.5% | **yes** | as path_own_s; 255-leaf trees ~2x export size |
| ds1 | path_own_s_cold | 42 | 2,178,139 | 1,414,363 | 182.0 | 121.6 | 391.8 | +66.0 | 99.6 | 259.7 | +64.5% | +69.7% | +75.3% | no | diagnostic (cold start) |
| ds1 | path_own_s_drop | 42 | 2,178,139 | 1,414,363 | 103.6 | 52.8 | 214.9 | -18.7 | 73.3 | 142.8 | -6.3% | -6.9% | +1.1% | **yes** | as path_own_s; degrades less when cold |
| ds1 | path_own_s_drop_cold | 42 | 2,178,139 | 1,414,363 | 124.4 | 66.3 | 265.5 | -24.3 | 80.4 | 174.9 | +12.5% | +15.0% | +14.0% | no | diagnostic (cold start) |
| ds1 | path_own_s_gate | 42 | 2,178,139 | 1,414,363 | 101.7 | 51.7 | 208.7 | -21.2 | 69.8 | 141.5 | -8.0% | -9.6% | -3.8% | **yes** | as path_own_s; baseline trees also shipped for cov=0 rows |
| ds1 | path_own_s_linkcold | 42 | 2,178,139 | 1,414,363 | 150.5 | 84.2 | 338.5 | -90.8 | 77.7 | 242.2 | +36.1% | +46.6% | +57.8% | no | diagnostic (link store cold) |
| ds1 | path_own_s_nocal | 42 | 2,178,139 | 1,414,363 | 100.6 | 50.9 | 206.3 | -19.2 | 68.9 | 140.4 | -9.1% | -10.6% | -5.0% | **yes** | as path_own_s |
| ds1 | path_own_s_nw | 42 | 2,178,139 | 1,414,363 | 101.4 | 51.9 | 208.4 | -19.7 | 70.6 | 140.5 | -8.3% | -9.7% | -2.8% | **yes** | as path_own_s |
| ds1 | path_s | 42 | 2,178,139 | 1,414,363 | 104.8 | 53.1 | 218.8 | -19.1 | 73.9 | 143.8 | -5.2% | -5.2% | +1.8% | **yes** | needs serving work (link store) |
| ds1lag90w15 | baseline | 42 | 2,178,139 | 1,414,363 | 110.6 | 56.0 | 230.9 | -28.2 | 72.6 | 153.5 | — | — | — | — | prod |
| ds1lag90w15 | path_own_s | 42 | 2,178,139 | 1,414,363 | 101.9 | 52.2 | 209.6 | -21.1 | 70.4 | 141.5 | -7.8% | -9.2% | -3.0% | **yes** | needs serving work (link store + 5-min position ring, ~1 day), state persisted across the 5-min push-feed restarts |
| ds1shift | baseline | 7 | 1,978,242 | 1,424,485 | 113.0 | 59.5 | 241.1 | -14.2 | 71.2 | 159.7 | — | — | — | — | prod |
| ds1shift | live_all | 7 | 1,978,242 | 1,424,485 | 103.0 | 54.1 | 212.4 | -14.7 | 69.9 | 144.0 | -8.9% | -11.9% | -1.9% | **yes** | needs serving work (NaN routing + live link store) |
| ds1shift | live_all_s | 7 | 1,978,242 | 1,424,485 | 102.4 | 53.2 | 211.8 | -16.3 | 68.8 | 143.4 | -9.4% | -12.2% | -3.4% | **yes** | needs serving work (live link store, ~1-2 days) |
| ds1shift | m3_nocal | 7 | 1,978,242 | 1,424,485 | 97.9 | 49.7 | 199.9 | -21.3 | 66.9 | 136.7 | -13.4% | -17.1% | -6.0% | **yes** | link store (last 3 traversals/link) + 5-min position ring, persisted in tracker_state.json; ~1 day |
| ds1shift | path_own_m3 | 7 | 1,978,242 | 1,424,485 | 99.2 | 51.0 | 203.8 | -16.5 | 68.7 | 138.2 | -12.2% | -15.5% | -3.5% | **yes** | as path_own_s; store keeps last 3 traversals per link |
| ds1shift | path_own_s | 7 | 1,978,242 | 1,424,485 | 102.8 | 53.6 | 214.2 | -14.8 | 69.6 | 144.0 | -9.0% | -11.2% | -2.2% | **yes** | needs serving work (link store + 5-min position ring, ~1 day), state persisted across the 5-min push-feed restarts |
| ds1shift | path_own_s_big | 7 | 1,978,242 | 1,424,485 | 100.7 | 52.4 | 208.8 | -14.5 | 67.7 | 141.7 | -10.9% | -13.4% | -4.9% | **yes** | as path_own_s; 255-leaf trees ~2x export size |
| ds1shift | path_own_s_drop | 7 | 1,978,242 | 1,424,485 | 104.2 | 54.5 | 216.4 | -13.7 | 71.4 | 144.9 | -7.8% | -10.3% | +0.2% | **yes** | as path_own_s; degrades less when cold |
| ds1shiftv2 | baseline | 7 | 1,978,242 | 1,424,485 | 113.0 | 59.5 | 241.1 | -14.2 | 71.2 | 159.7 | — | — | — | — | prod |
| ds1shiftv2 | stack | 7 | 1,978,242 | 1,424,485 | 96.7 | 48.7 | 196.9 | -21.3 | 66.6 | 134.7 | -14.5% | -18.3% | -6.5% | **yes** | link store (last 5 traversals/link) + 5-min position ring, persisted in tracker_state.json; ~1 day |
| ds1shiftv2 | stack_big | 7 | 1,978,242 | 1,424,485 | 95.9 | 48.5 | 196.1 | -20.1 | 66.0 | 133.9 | -15.1% | -18.7% | -7.2% | **yes** | as stack; 255-leaf trees ~2x export size |
| ds1shiftv3 | baseline | 7 | 1,978,242 | 1,424,485 | 113.0 | 59.5 | 241.1 | -14.2 | 71.2 | 159.7 | — | — | — | — | prod |
| ds1shiftv3 | stack | 7 | 1,978,242 | 1,424,485 | 96.7 | 48.7 | 196.9 | -21.3 | 66.6 | 134.7 | -14.5% | -18.3% | -6.5% | **yes** | link store (last 5 traversals/link) + 5-min position ring, persisted in tracker_state.json; ~1 day |
| ds1shiftv3 | stack_hist | 7 | 1,978,242 | 1,424,485 | 94.5 | 46.8 | 192.1 | -22.7 | 66.0 | 131.6 | -16.4% | -20.3% | -7.4% | **yes** | as stack + static link x hour median table built at export (like priors) |
| ds1shiftv3 | stack_v3 | 7 | 1,978,242 | 1,424,485 | 94.3 | 46.5 | 191.6 | -22.6 | 66.1 | 131.5 | -16.5% | -20.5% | -7.1% | **yes** | as stack_hist + last 7 traversals + EWMA per link |
| ds1shiftv4 | baseline | 7 | 1,978,242 | 1,424,485 | 113.0 | 59.5 | 241.1 | -14.2 | 71.2 | 159.7 | — | — | — | — | prod |
| ds1shiftv4 | stack_hist | 7 | 1,978,242 | 1,424,485 | 94.4 | 46.6 | 192.2 | -22.9 | 66.0 | 131.6 | -16.5% | -20.3% | -7.3% | **yes** | as stack + static link x hour median table built at export (like priors) |
| ds1shiftv4 | stack_hist_catroute | 7 | 1,978,242 | 1,424,485 | 93.0 | 46.3 | 189.7 | -21.0 | 64.9 | 129.7 | -17.7% | -21.3% | -8.8% | **yes** | as stack_hist + categorical-split (bitset) support in export/inference, ~0.5-1 day |
| ds1shiftv4 | stack_hist_dt | 7 | 1,978,242 | 1,424,485 | 93.4 | 45.9 | 190.4 | -21.1 | 65.6 | 129.9 | -17.4% | -21.0% | -7.9% | **yes** | as stack_hist + weekday/weekend link x hour table |
| ds1shiftv4 | stack_hist_dt_cat | 7 | 1,978,242 | 1,424,485 | 92.1 | 45.6 | 187.6 | -19.5 | 64.5 | 128.3 | -18.5% | -22.2% | -9.4% | **yes** | as stack_hist_dt + categorical-split (bitset) support in export/inference; ~2 days total |
| ds1shiftv5 | baseline | 7 | 1,978,242 | 1,424,485 | 113.0 | 59.5 | 241.1 | -14.2 | 71.2 | 159.7 | — | — | — | — | prod |
| ds1shiftv5 | dtcat_big | 7 | 1,978,242 | 1,424,485 | 91.3 | 45.5 | 187.3 | -17.4 | 63.2 | 127.3 | -19.2% | -22.3% | -11.2% | **yes** | as stack_hist_dt_cat; 255-leaf trees ~2x export size |
| ds1shiftv5 | dtcat_stopcat | 7 | 1,978,242 | 1,424,485 | 91.2 | 45.1 | 186.1 | -19.2 | 63.6 | 127.2 | -19.3% | -22.8% | -10.7% | **yes** | as stack_hist_dt_cat + stop->code map (254 stops) shipped with the model |
| ds1shiftv5 | dtcat_stopcat_big | 7 | 1,978,242 | 1,424,485 | 90.5 | 44.9 | 185.4 | -18.3 | 62.6 | 126.6 | -19.9% | -23.1% | -12.1% | **yes** | as dtcat_stopcat; 255-leaf trees ~2x export size; ~2-2.5 days total |
| ds1shiftv5 | stack_hist_dt_cat | 7 | 1,978,242 | 1,424,485 | 92.1 | 45.6 | 187.6 | -19.5 | 64.5 | 128.3 | -18.5% | -22.2% | -9.4% | **yes** | as stack_hist_dt + categorical-split (bitset) support in export/inference; ~2 days total |
| ds1v2 | baseline | 42 | 2,178,139 | 1,414,363 | 110.6 | 56.0 | 230.9 | -28.2 | 72.6 | 153.5 | — | — | — | — | prod |
| ds1v2 | m3_nocal | 42 | 2,178,139 | 1,414,363 | 98.0 | 48.8 | 198.4 | -20.7 | 68.9 | 135.8 | -11.3% | -14.1% | -5.1% | **yes** | link store (last 3 traversals/link) + 5-min position ring, persisted in tracker_state.json; ~1 day |
| ds1v2 | path_own_m3 | 42 | 2,178,139 | 1,414,363 | 98.4 | 49.8 | 199.2 | -19.9 | 69.9 | 136.1 | -11.1% | -13.7% | -3.7% | **yes** | as path_own_s; store keeps last 3 traversals per link |
| ds1v2 | path_own_m35 | 42 | 2,178,139 | 1,414,363 | 97.6 | 49.1 | 197.7 | -18.9 | 69.5 | 135.0 | -11.7% | -14.4% | -4.2% | **yes** | as path_own_s; store keeps last 5 traversals per link |
| ds1v2 | path_own_m3_big | 42 | 2,178,139 | 1,414,363 | 99.7 | 50.6 | 205.6 | -15.6 | 71.8 | 136.2 | -9.9% | -10.9% | -1.0% | **yes** | as path_own_m3; 255-leaf trees ~2x export size |
| ds1v2 | stack | 42 | 2,178,139 | 1,414,363 | 96.9 | 48.0 | 196.1 | -20.5 | 68.3 | 134.2 | -12.4% | -15.0% | -5.8% | **yes** | link store (last 5 traversals/link) + 5-min position ring, persisted in tracker_state.json; ~1 day |
| ds1v2 | stack_big | 42 | 2,178,139 | 1,414,363 | 96.0 | 47.7 | 194.3 | -19.4 | 67.3 | 133.1 | -13.2% | -15.8% | -7.2% | **yes** | as stack; 255-leaf trees ~2x export size |
| ds1v3 | baseline | 42 | 2,178,139 | 1,414,363 | 110.6 | 56.0 | 230.9 | -28.2 | 72.6 | 153.5 | — | — | — | — | prod |
| ds1v3 | stack | 42 | 2,178,139 | 1,414,363 | 96.9 | 48.0 | 196.1 | -20.5 | 68.3 | 134.2 | -12.4% | -15.0% | -5.8% | **yes** | link store (last 5 traversals/link) + 5-min position ring, persisted in tracker_state.json; ~1 day |
| ds1v3 | stack_cold | 42 | 2,178,139 | 1,414,363 | 177.2 | 116.1 | 385.2 | +58.0 | 92.8 | 255.7 | +60.2% | +66.9% | +70.3% | no | diagnostic (cold start) |
| ds1v3 | stack_ewm | 42 | 2,178,139 | 1,414,363 | 96.2 | 47.7 | 194.6 | -20.2 | 67.8 | 133.5 | -13.0% | -15.7% | -6.6% | **yes** | as stack; + per-link EWMA in the store |
| ds1v3 | stack_fill | 42 | 2,178,139 | 1,414,363 | 94.6 | 46.0 | 190.5 | -21.3 | 67.4 | 131.2 | -14.5% | -17.5% | -7.1% | **yes** | as stack_hist |
| ds1v3 | stack_hist | 42 | 2,178,139 | 1,414,363 | 94.8 | 46.1 | 190.9 | -21.6 | 68.0 | 131.5 | -14.2% | -17.3% | -6.3% | **yes** | as stack + static link x hour median table built at export (like priors) |
| ds1v3 | stack_hist_cold | 42 | 2,178,139 | 1,414,363 | 154.8 | 95.9 | 324.4 | +18.1 | 94.8 | 217.9 | +40.0% | +40.5% | +50.4% | no | diagnostic (cold start, hist table kept) |
| ds1v3 | stack_m7 | 42 | 2,178,139 | 1,414,363 | 96.4 | 47.5 | 194.7 | -21.2 | 68.5 | 133.3 | -12.8% | -15.7% | -5.6% | **yes** | as stack; store keeps last 7 traversals per link |
| ds1v3 | stack_v3 | 42 | 2,178,139 | 1,414,363 | 94.5 | 45.9 | 190.1 | -21.5 | 67.5 | 131.1 | -14.6% | -17.7% | -7.0% | **yes** | as stack_hist + last 7 traversals + EWMA per link |
| ds1v4 | baseline | 42 | 2,178,139 | 1,414,363 | 110.6 | 56.0 | 230.9 | -28.2 | 72.6 | 153.5 | — | — | — | — | prod |
| ds1v4 | stack_hist | 42 | 2,178,139 | 1,414,363 | 94.3 | 46.1 | 190.1 | -19.6 | 67.1 | 130.7 | -14.7% | -17.7% | -7.5% | **yes** | as stack + static link x hour median table built at export (like priors) |
| ds1v4 | stack_hist_catroute | 42 | 2,178,139 | 1,414,363 | 93.0 | 45.7 | 188.3 | -17.8 | 66.2 | 128.6 | -15.9% | -18.5% | -8.8% | **yes** | as stack_hist + categorical-split (bitset) support in export/inference, ~0.5-1 day |
| ds1v4 | stack_hist_dt | 42 | 2,178,139 | 1,414,363 | 93.3 | 45.2 | 188.1 | -18.0 | 66.8 | 129.1 | -15.6% | -18.5% | -8.0% | **yes** | as stack_hist + weekday/weekend link x hour table |
| ds1v4 | stack_hist_dt_cat | 42 | 2,178,139 | 1,414,363 | 92.1 | 44.9 | 186.0 | -16.5 | 65.9 | 127.2 | -16.7% | -19.4% | -9.2% | **yes** | as stack_hist_dt + categorical-split (bitset) support in export/inference; ~2 days total |
| ds1v4 | stack_hist_l2 | 42 | 2,178,139 | 1,414,363 | 94.3 | 46.0 | 190.1 | -19.8 | 66.9 | 130.9 | -14.7% | -17.6% | -7.8% | **yes** | as stack_hist |
| ds1v4 | stack_hist_leaf100 | 42 | 2,178,139 | 1,414,363 | 94.2 | 46.0 | 189.8 | -19.6 | 67.0 | 130.5 | -14.8% | -17.8% | -7.7% | **yes** | as stack_hist |
| ds1v4 | stack_hist_r | 42 | 2,178,139 | 1,414,363 | 94.3 | 45.9 | 190.6 | -19.9 | 67.1 | 131.0 | -14.7% | -17.5% | -7.5% | **yes** | as stack_hist |
| ds1v4 | stack_v4 | 42 | 2,178,139 | 1,414,363 | 93.5 | 45.3 | 188.0 | -18.2 | 66.8 | 129.4 | -15.5% | -18.6% | -7.9% | **yes** | as stack_hist + day-type and p75 tables |
| ds1v5 | baseline | 42 | 2,178,139 | 1,414,363 | 110.6 | 56.0 | 230.9 | -28.2 | 72.6 | 153.5 | — | — | — | — | prod |
| ds1v5 | dtcat_big | 42 | 2,178,139 | 1,414,363 | 91.6 | 44.8 | 185.4 | -15.5 | 64.9 | 126.7 | -17.2% | -19.7% | -10.6% | **yes** | as stack_hist_dt_cat; 255-leaf trees ~2x export size |
| ds1v5 | dtcat_logratio | 42 | 2,178,139 | 1,414,363 | 92.6 | 44.4 | 184.8 | -18.6 | 67.1 | 127.5 | -16.3% | -19.9% | -7.6% | **yes** | as dtcat_resid (exp transform) |
| ds1v5 | dtcat_own | 42 | 2,178,139 | 1,414,363 | 92.0 | 44.6 | 185.8 | -17.6 | 66.1 | 126.8 | -16.8% | -19.5% | -9.0% | **yes** | as stack_hist_dt_cat + own last-k link sums vs hist table (per-vehicle crossing log) |
| ds1v5 | dtcat_own_x | 42 | 2,178,139 | 1,414,363 | 92.0 | 44.6 | 185.9 | -17.4 | 66.3 | 126.8 | -16.8% | -19.5% | -8.7% | **yes** | as dtcat_own |
| ds1v5 | dtcat_prune | 42 | 2,178,139 | 1,414,363 | 93.3 | 45.4 | 187.9 | -17.6 | 67.3 | 128.6 | -15.6% | -18.6% | -7.3% | **yes** | as stack_hist_dt_cat |
| ds1v5 | dtcat_recency | 42 | 2,178,139 | 1,414,363 | 92.5 | 45.0 | 186.6 | -17.1 | 66.4 | 127.8 | -16.3% | -19.2% | -8.6% | **yes** | as stack_hist_dt_cat |
| ds1v5 | dtcat_resid | 42 | 2,178,139 | 1,414,363 | 91.9 | 44.7 | 185.3 | -16.2 | 66.0 | 126.9 | -16.9% | -19.7% | -9.1% | **yes** | as stack_hist_dt_cat + add base ETA to tree output |
| ds1v5 | dtcat_stopcat | 42 | 2,178,139 | 1,414,363 | 91.6 | 44.4 | 185.0 | -17.0 | 65.1 | 126.6 | -17.2% | -19.9% | -10.4% | **yes** | as stack_hist_dt_cat + stop->code map (254 stops) shipped with the model |
| ds1v5 | dtcat_stopcat_big | 42 | 2,178,139 | 1,414,363 | 90.7 | 44.3 | 183.8 | -15.5 | 63.7 | 126.0 | -18.0% | -20.4% | -12.2% | **yes** | as dtcat_stopcat; 255-leaf trees ~2x export size; ~2-2.5 days total |
| ds1v5 | stack_hist_dt_cat | 42 | 2,178,139 | 1,414,363 | 92.1 | 44.9 | 186.0 | -16.5 | 65.9 | 127.2 | -16.7% | -19.4% | -9.2% | **yes** | as stack_hist_dt + categorical-split (bitset) support in export/inference; ~2 days total |
| ds1v5k3 | baseline | 42 | 6,510,353 | 1,414,363 | 108.1 | 54.4 | 226.9 | -27.5 | 70.4 | 149.9 | — | — | — | — | prod |
| ds1v5k3 | dtcat_stopcat_big | 42 | 6,510,353 | 1,414,363 | 88.8 | 43.5 | 180.2 | -15.5 | 61.8 | 123.4 | -17.9% | -20.6% | -12.3% | **yes** | as dtcat_stopcat; 255-leaf trees ~2x export size; ~2-2.5 days total |
| ds1v5k3 | stack_hist_dt_cat | 42 | 6,510,353 | 1,414,363 | 90.5 | 44.2 | 182.8 | -17.1 | 64.0 | 124.8 | -16.3% | -19.4% | -9.1% | **yes** | as stack_hist_dt + categorical-split (bitset) support in export/inference; ~2 days total |
