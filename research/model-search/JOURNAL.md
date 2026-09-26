# Model-search journal

Research-only search for a better ETA model. Branch `claude/model-search`.
Harness: `harness.py` (prep/run), arms in `arms.py`, live features in
`features_live.py`. Results JSON per arm in `results/`.

## Fixed day set (DS1)
- Pipeline days: 2026-09-07 .. 2026-09-21 (`scripts/run_pipeline.py --date … --parallel 2`).
  Cloud box = 4 cores / 15.7 GB; `--parallel 3` OOM-killed (each worker ~4.7 GB
  RSS and still growing at ~15 min) and a BrokenProcessPool loses every day, so
  use 2. Disk is wiped between runs → every run regenerates; keep the set small.
- Train: 2026-09-07 .. 2026-09-18 (12 days). Test: 2026-09-19 (Sat), 09-20 (Sun),
  09-21 (Mon).
- Features built per day against `get_gtfs_for_date(day)` (static archive
  snapshots 2026-08-24, 2026-09-04 cover this range), sampled per day by
  hashing (vehicle_id, snapshot_ts) → whole snapshots kept, 10% default.
- Baseline arm = `src/train.py` recipe (features, sanity filters, bad routes,
  train-only route×hour priors, hour sample weights, HGBT hyper-params).

## Known from repo history (don't redo)
- Segment-additive link-time model (scripts/compare_models.py, 2026-08-14) loses
  to direct HGBT at every horizon (170 vs 139s); blends worse than A alone.

## 2026-09-24 — run 1 (setup + live link-time features)

### Infra
- `pipeline_lite.py`: same download → `infer_trips` → `labeling.training_rows_for_trajectory`
  as production, but samples 10% of snapshots per trajectory on the fly
  (hash of vehicle_id+snapshot_ts) and writes full-day side tables
  (`.cross` = unique stop arrivals, `.pos` = per-snapshot positions). Peak 3.0 GB
  vs 7.6 GB for `run_pipeline.py` (which OOM-killed the pool at `--parallel 3` and 2).
  ~19 min/day single, ~19-30 min/day at `--parallel 4`; 15 days ≈ 75 min.
  `python research/model-search/pipeline_lite.py --days 2026-09-07..2026-09-21 --parallel 4`
- `harness.py prep --days 2026-09-07..2026-09-21` (3% of snapshots, era-pinned
  static, + live features) ≈ 1.5 min/day. `harness.py run --arm A --train … --test … --tag T [--seed S]`
  subsamples train to 1%/day (≈2.2M rows), test 3% (≈1.4M rows). A fit takes ~5 min
  on 4 free cores; never fit while the pipeline runs (OpenMP oversubscription made a
  600k-row fit take >55 min).
- NB: prod `n_iter_` hits the 1200 cap on 2.2M rows here too (no early stop).

### Experiment: live features (`features_live.py`)
All causal: only crossings completed ≥ 30 s (DETECT_LAG_SEC) before the snapshot.
- `live_path_sec`: for each stop→stop link on the path vehicle→target (keyed by
  stop-id pair, shared across routes), the latest traversal time by any vehicle
  completed in the last 30 min; current link weighted by the fraction left; missing
  links extrapolated from covered weight. Plus `live_path_cov`, `live_path_age`,
  `live_speed_mps`. Alone as a predictor: MAE 134.7 s on 09-21 (≈ whole prod model
  a month ago); coverage 96%.
- `own_speed_{60,180,300}`: vehicle's own advance over the last W s.
- `since_last_at_target`: s since any vehicle last arrived at the target stop.

### Results (tag ds1: train 09-07..09-18, test 09-19 Sat / 09-20 Sun / 09-21 Mon; seed 42)
| arm | MAE | p90 | ΔMAE | Δp90 | worst bucket |
|---|---|---|---|---|---|
| baseline | 110.6 | 230.9 | | | |
| own_speed | 110.0 | 229.5 | −0.6% | −0.6% | +0.4% |
| headway | 110.0 | 229.9 | −0.5% | −0.4% | −0.1% |
| live_path | 104.5 | 216.6 | −5.5% | −6.2% | +1.4% (sa1) |
| live_all (NaN) | 103.7 | 214.5 | −6.2% | −7.1% | +1.2% (sa1) |
| **live_all_s** (−1 sentinel) | **102.5** | **210.7** | **−7.3%** | **−8.7%** | −1.3% |

Confirmation (tag ds1shift: train 09-07..09-17, test 09-18 Fri / 09-19 / 09-20; seed 7):
baseline 113.0 / p90 241.1 → live_all_s 102.4 / 211.8 (−9.4%, −12.2%), every
stops_ahead bucket better (sa1 71.2→68.8, sa10 159.7→143.4).
Per day (live_all_s vs baseline): Fri 126.8→107.3, Mon 120.5→106.1, Sat 100.2→94.8 / 99.6→95.3,
Sun 107.1→105.4 / 106.6→102.4 — the gain is live traffic, so mostly weekdays.

**Conclusion: `live_all_s` is a WIN under the protocol (both splits, two seeds).**

### Serving notes (needs serving work, est. 1-2 days)
- Exported trees (`src/inference.py` `_traverse_tree`/`predict_rows`) have no
  missing-value direction (NaN always goes right), hence the sentinel variant
  `live_all_s` (NaN → −1). It is also the better one.
- Daemon needs: (1) a shared store of the latest traversal per stop-pair link from
  all tracked vehicles (detect stop crossings in `update_tracker`, keep
  `{(from_stop,to_stop): (t_done, link_sec)}`, drop > 30 min); (2) a ~5 min position
  ring per vehicle (own_speed); (3) last-arrival time per stop. All O(vehicles) per
  tick. State is lost on restart → features read −1/cov 0 for up to 30 min; the
  model has seen cov=0 rows in training (4% of rows) but restart behaviour should be
  checked. Append the 8 columns after `stationary_sec` so feature indices stay put.
- Offline crossing times are interpolated from full trajectories; the live daemon
  sees them one snapshot later — the 30 s lag emulates that, but a live shadow run
  should confirm the gain before switching.

### Next steps
1. Ablate `live_all_s` (drop own_speed/headway/age) to minimise serving work;
   try window 15/60 min and lag 60 s for robustness.
2. Worst routes unchanged (122 ~350 s, 133 ~230 s): inspect route 122 (new in top list).
3. Other ideas still open: hyper-params (lr/leaves, n_iter cap is binding),
   drop calendar features, log target, sample weights off.

## 2026-09-25 — run 2 (ablation, cold start, robustness)

Rebuilt DS1 from scratch (`pip install -e . tzdata`; `pipeline_lite.py --days 2026-09-07..2026-09-21 --parallel 4`
≈ 75 min; `harness.py prep` ≈ 20 min). Baseline and live_all_s reproduced
bit-for-bit (110.6 / 102.5), so run-1 numbers stay comparable.
New: `queue.sh TAG TRAIN TEST SEED arm…`; env overrides `MS_DETECT_LAG`,
`MS_LINK_WINDOW`, `MS_FEAT_DIR` for features_live / prep.

### Ablation (ds1, seed 42; baseline 110.6 / p90 230.9)
| arm | cols added | MAE | p90 | ΔMAE |
|---|---|---|---|---|
| path_min_s | live_path_sec, live_path_cov | 103.2 | 211.7 | −6.7% |
| path_s | 4 path cols | 104.8 | 218.8 | −5.2% |
| live_all_s | all 8 | 102.5 | 210.7 | −7.3% |
| **path_own_s** | 4 path + 3 own_speed (no headway) | **101.4** | **208.3** | **−8.3%** |
| path_own_s_big | same, 255 leaves / min_leaf 50 | 100.5 | 206.3 | −9.1% |
| live_all_s_lr10 | lr 0.1 | 102.7 | 213.4 | −7.1% (no gain) |

`path_own_s` confirmed on ds1shift (seed 7): 113.0 → 102.8 (−9.0%), p90 241.1 → 214.2,
all buckets better. Robust to a 90 s detection lag + 15-min link window
(ds1lag90w15: −7.8%, p90 −9.2%). → drop `since_last_at_target` (no stop-arrival
store needed). path_s < path_min_s is odd (likely noise from age/speed cols).

### Cold start — IMPORTANT for serving
Model trained with live features, tested with all live features at their
cold values (daemon just restarted):
- live_all_s cold: 163.5 (+48%); path_own_s cold: 182.0 (+65%).
- path_own_s link store cold, own_speed warm (5–30 min after restart): 150.5 (+36%).
- Dropout (15% of training snapshots forced cold, `path_own_s_drop`): cold 124.4
  (+12.5%) but warm only −6.3% / −7.8% (shift) instead of −8.3% / −9.0%.
Cold rows are rare in training (they coincide with the start of service), so the
trees extrapolate badly. Recommendation: serve the live model only when the link
store is warm (e.g. ≥ 30 min uptime or path coverage > 0) and fall back to the
current production trees otherwise — a gate, not dropout, keeps the full gain.
Alternatively persist the link store (+ position ring) across restarts.

### Next steps
1. Confirm path_own_s_big on ds1shift; weigh ~2x tree size vs −0.8 pt.
2. Gate evaluation: per-row fallback to baseline where `live_path_cov == 0`
   (simulate on the cold test sets) — measure the served-mix MAE.
3. Worst routes still 122 (~340 s), 133 (~230 s), 106, 125 — not helped by live
   features; inspect route 122 labels/shape.
4. Remaining ideas: drop calendar cols, sample-weight off, quantile/huber loss.

## 2026-09-25 — run 3 (smoothed link times, gate, ablations, stack)

Rebuilt DS1 again (pipeline_lite 75 min, prep 20 min); baseline 110.6 / 113.0 and
path_own_s 101.4 / 102.8 reproduced exactly.

### Serving reality check (read scripts/push_feed.py + push-feed.yml)
The "daemon" is not long-lived: push-feed runs `push_feed.py --loop 10 --count 33`
every 5 min (~5.5 min per process) and already carries per-vehicle anchors across
restarts in R2 `feed/tracker_state.json`. So a cold start happens every 5 min
unless the live state is persisted — the link store (≤ 5 traversals × a few
thousand stop-pair links, 30-min TTL) and the 5-min position ring must go into
that same JSON (a few hundred KB). With that, cold start only happens after an R2
failure / >30 min outage, which is what cov=0 already models.

### Gate (path_own_s_gate): not needed
Serve path_own_s where live_path_cov>0, else baseline: 101.7 vs 101.4 ungated.
On naturally cold rows (cov=0, 3.7% of test, start of service) path_own_s is
already *better* than baseline (211 vs 221 s). By coverage band (live / base MAE):
cov 0: 211/221, (0,0.5): 153/152, [0.5,0.9): 222/231, ≥0.9 (92%): 91/101.
→ ship without a gate; only the artificial "everything cold mid-day" case is bad.

### New feature: median of the last k traversals per link
`features_live.py` now also emits `live_path_sec_m3` / `live_path_sec_m5`: same
path sum, but each link's time is the median of its last 3 / 5 traversals (the
latest must still be within the 30-min window; older ones are taken as-is — the
serving store keeps the last k per link), and `live_path_n` (#links observed).

| arm (seed 42 ds1 / seed 7 shift) | ds1 MAE | Δ | shift MAE | Δ | p90 ds1 / shift |
|---|---|---|---|---|---|
| baseline | 110.6 | | 113.0 | | 230.9 / 241.1 |
| path_own_s (run-2 leader) | 101.4 | −8.3% | 102.8 | −9.0% | 208.3 / 214.2 |
| path_own_s_nw (no hour weights) | 101.4 | −8.3% | | | 208.4 |
| path_own_s_nocal (− month/dow/stop_seq) | 100.6 | −9.1% | | | 206.3 |
| path_own_s_big (255 leaves) | 100.5 | −9.1% | 100.7 | −10.9% | 206.3 / 208.8 |
| path_own_m3 (+m3, +n) | 98.4 | −11.1% | 99.2 | −12.2% | 199.2 / 203.8 |
| path_own_m3only (latest→m3) | 98.3 | −11.1% | | | 199.8 |
| path_own_m35 (+m3,+n,+m5) | 97.6 | −11.7% | | | 197.7 |
| path_own_m3_big | 99.7 | −9.9% | | | 205.6 |
| m3_nocal | 98.0 | −11.3% | 97.9 | −13.4% | 198.4 / 199.9 |
| **stack** (m3+m5 replace latest, own speed, no calendar) | **96.9** | **−12.4%** | **96.7** | **−14.5%** | **196.1 / 196.9** |
| stack_big (255 leaves) | 96.0 | −13.2% | 95.9 | −15.1% | 194.3 / 196.1 |

`stack` cols = FEATURE_COLS − {month, day_of_week, stop_sequence} +
{live_path_sec_m3, live_path_cov, live_path_age, live_speed_mps, own_speed_60/180/300,
live_path_sec_m5}. Every stops_ahead bucket improves on both splits
(ds1 sa1 72.6→68.3, sa5 112.0→97.1, sa10 153.5→134.2; shift sa1 71.2→66.6,
sa10 159.7→134.7). Per day ds1: Sat 100.2→91.8, Sun 107.1→95.3, Mon 120.5→101.6;
shift: Fri 126.8→101.0. Rows: train 2.18M / 1.98M, test 1.41M / 1.42M.
Worst routes (stack, ds1): 122 330.7 (base 352.7), 133 224.6 (231.2), 106 177.4,
125 170.7, 131 148.6, 113 148.4, 94 147.0, 129 146.5, 878 143.9, 881 143.2.

**Conclusion: `stack` is the new leader, WIN on both splits/seeds (−12.4% / −14.5%,
p90 −15% / −18%, all buckets better).** Big trees add ~0.8 pt more (not on m3
alone — noisy); keep 127 leaves unless the export budget allows 2x.
Cmds: `MS_FEAT_DIR=ms_features_v2 python research/model-search/harness.py prep --days 2026-09-07..2026-09-21`;
`MS_FEAT_DIR=ms_features_v2 sh research/model-search/queue.sh ds1v2 2026-09-07..2026-09-18 2026-09-19..2026-09-21 42 baseline stack stack_big`
(and `ds1shiftv2 2026-09-07..2026-09-17 2026-09-18..2026-09-20 7`).

### Serving estimate for `stack` (~1-1.5 days)
In push_feed/inference: detect stop crossings between consecutive pushes per
vehicle; per link keep a deque of the last 5 (t_done, link_sec); per vehicle a
5-min ring of (t, dist_along); persist both in tracker_state.json; compute the 8
columns per row (path walk over the trip profile = O(stops_ahead)). Drop 3
calendar cols → feature indices change, so export + inference must switch
together (bump the model blob format).

### Next steps
1. Larger k / time-decayed mean (m5 > m3 → try m7, trimmed mean, EWMA).
2. Route 122 still ~330 s: inspect labels/shape (not helped by anything so far).
3. Live shadow check of crossing detection at 10 s cadence vs offline interpolated
   crossings (lag90 robustness already −7.8% for path_own_s).
4. Second seed on ds1 for stack (both splits were run once each with distinct seeds).

## 2026-09-25 — run 4 (historical link table, longer/decayed smoothing)

Rebuilt DS1 (pipeline_lite `--parallel 4` ≈ 105 min this time, ~28 min/day/worker;
prep ≈ 2.5 min/day, sequential because of the history lookup). Baseline 110.6 / 113.0
and stack 96.9 / 96.7 reproduced exactly on the v3 features.

### New features (`features_live.py` V3_COLS, `MS_FEAT_DIR=ms_features_v3`)
- `live_path_sec_m7` — median of last 7 traversals per link.
- `live_path_sec_ewm` — per-link EWMA (alpha 0.4 per traversal).
- `hist_path_sec`, `hist_path_cov` — static historical table: median link time per
  (stop-pair link, local hour) over the PRIOR 14 days (≥ 3 obs, fallback: link
  all-hours median). Prep saves each day's link traversals to
  `FEAT_DIR/links_<day>.parquet` and day D only reads days < D, so it is causal;
  early train days have little/no history (09-07: none) — test days have 12+.
  Serving: a table built at export time from the training window, like the
  route/hour priors — no live state. Single-predictor MAE on 09-08 (1 prior day):
  hist 122.1 vs live latest 131.1 / m5 118.8.
- `fill_path_sec` — live m5 where observed, historical otherwise.

### Results
| arm | ds1v3 (s42) MAE | Δ vs base | p90 | ds1shiftv3 (s7) MAE | Δ | p90 |
|---|---|---|---|---|---|---|
| baseline | 110.6 | | 230.9 | 113.0 | | 241.1 |
| stack (run-3 leader) | 96.9 | −12.4% | 196.1 | 96.7 | −14.5% | 196.9 |
| stack_m7 | 96.4 | −12.8% | 194.7 | | | |
| stack_ewm | 96.2 | −13.0% | 194.6 | | | |
| **stack_hist** | **94.8** | **−14.2%** | **190.9** | **94.5** | **−16.4%** | **192.1** |
| stack_fill | 94.6 | −14.5% | 190.5 | | | |
| stack_v3 (all) | 94.5 | −14.6% | 190.1 | 94.3 | −16.5% | 191.6 |
| stack_cold (diag) | 177.2 | +60.2% | 385.2 | | | |
| stack_hist_cold (diag) | 154.8 | +40.0% | 324.4 | | | |

stack_hist vs stack: −2.2% (ds1) / −2.3% (shift), p90 −2.7% / −2.4%, every
stops_ahead bucket better on both (ds1 sa1 68.3→68.0, sa5 97.1→94.7, sa10 134.2→131.5;
shift sa1 66.6→66.0, sa10 134.7→131.6). Per day ds1: Sat 91.8→90.3, Sun 95.3→94.0,
Mon 101.6→98.7; shift Fri 101.0→97.8. Rows: train 2.18M / 1.98M, test 1.41M / 1.42M.
Worst routes (stack_hist, ds1): 122 321, 133 223, 106 174, 125 168, 113 148, 94 147,
131 146, 881 145, 878 143, 129 142.
m7 / EWMA add only ~0.5 pt each and stack_v3 ≈ stack_hist + 0.3 pt → not worth the
extra store state. The hist table also cuts the cold-start penalty (177 → 155) but
cold is still far worse than baseline → still persist the live store.

**Conclusion: `stack_hist` = new leader, WIN on both splits/seeds
(ds1 110.6→94.8, −14.2%, p90 230.9→190.9; shift 113.0→94.5, −16.4%, p90 241.1→192.1).**
Cols: `_NOCAL + _STACK + [hist_path_sec, hist_path_cov]` (sentinel −1).
Cmds: `MS_FEAT_DIR=ms_features_v3 python research/model-search/harness.py prep --days 2026-09-07..2026-09-21`
(must run days in ascending order);
`MS_FEAT_DIR=ms_features_v3 sh research/model-search/queue.sh ds1v3 2026-09-07..2026-09-18 2026-09-19..2026-09-21 42 baseline stack stack_hist …`,
`… ds1shiftv3 2026-09-07..2026-09-17 2026-09-18..2026-09-20 7 …`.

### Serving estimate for `stack_hist`
As `stack` (~1 day: link store of last 5 traversals + 5-min position ring persisted in
tracker_state.json) plus: at export build the link×hour median table from the training
days' crossings (a few k links × 24 h ≈ 100k floats, ship alongside the trees) and
sum it along the path per row (same path walk as the live sum). +0.5 day.

### Next steps
1. Historical table with more history (train rows early in DS1 have little; production
   would have ≥ 14 days for all rows) — extend pipeline to 08-31..09-06 for link tables
   only, expect a bit more gain.
2. Day-type split (weekday/weekend) in the hist table; ratio live/hist as explicit col.
3. Route 122 still ~320 s — inspect.
4. Hyper-params on stack_hist (lr 0.1 did nothing before; try l2_regularization,
   min_samples_leaf 100).

## 2026-09-25 — run 6 (retry of the stalled run 5: v4 features, catroute)

Run 5 took the lock at 14:18 and committed the v4 feature code, then stalled while
rebuilding data (no journal entry and no results). On the user's "try again" this
run took the lock over at 17:35 (it was 2h43m old).

Rebuilt with `pipeline_lite.py --days 2026-08-31..2026-09-21 --parallel 4`
(weekdays ~25-47 min per day per worker, weekends ~16 min, about 2.5 h in total). Prep ran
in a chain as days landed (`MS_FEAT_DIR=ms_features_v4 harness.py prep --days D`,
ascending). History now starts 08-31, so 09-07 train rows have 7 prior days and
09-14+ have 14.

Cmds: `MS_FEAT_DIR=ms_features_v4 sh research/model-search/queue.sh ds1v4 2026-09-07..2026-09-18 2026-09-19..2026-09-21 42 <arms>`
and `… ds1shiftv4 2026-09-07..2026-09-17 2026-09-18..2026-09-20 7 <arms>`.

### Results (MAE / p90, Δ vs baseline on the same tag)
| arm | ds1v4 (s42) | Δ | ds1shiftv4 (s7) | Δ |
|---|---|---|---|---|
| baseline | 110.6 / 230.9 | | 113.0 / 241.1 | |
| stack_hist (run-4 leader, longer history) | 94.3 / 190.1 | −14.7% | 94.4 / 192.2 | −16.5% |
| stack_hist_r (+ live/hist ratio) | 94.3 / 190.6 | −14.7% | | |
| stack_hist_l2 (l2 = 1.0) | 94.3 / 190.1 | −14.7% | | |
| stack_hist_leaf100 | 94.2 / 189.8 | −14.8% | | |
| stack_v4 (+ dt, ratio, p75) | 93.5 / 188.0 | −15.5% | | |
| stack_hist_dt (+ weekday/weekend hist table) | 93.3 / 188.1 | −15.6% | 93.4 / 190.4 | −17.4% |
| stack_hist_catroute (route_id native categorical) | 93.0 / 188.3 | −15.9% | 93.0 / 189.7 | −17.7% |
| **stack_hist_dt_cat** | **92.1 / 186.0** | **−16.7%** | **92.1 / 187.6** | **−18.5%** |

- Longer history on its own: stack_hist went 94.8 → 94.3 on ds1 (v3 → v4 features) and
  94.5 → 94.4 on shift. The gain is small, so 14 days of history is enough.
- The ratio, p75, l2 and leaf100 arms all come out as noise (±0.1).
- The day-type table and categorical route are each worth about 1% and add up.
  Against stack_hist, stack_hist_dt_cat is −2.3% (ds1) and −2.4% (shift), and every
  stops_ahead bucket improves by 1.8–2.7% on both splits. By day on ds1: Sat
  89.7→87.0, Sun 93.7→90.7, Mon 98.1→96.8. On shift: Fri 97.8→96.4, Sat 90.1→87.1,
  Sun 93.9→90.9. By horizon (ds1): sa1 65.9, sa5 92.1, sa10 127.2 (baseline 72.6 / — / 153.5).
  Rows: train 2.18M / 1.98M, test 1.41M / 1.42M.
- Worst routes (dt_cat, ds1): 122 309, 133 215, 106 171, 125 155, 113 143, 94 140,
  878 138, 137 136, 129 134, 131 133. On shift, 122 is 228 and 133 is 160.

**Conclusion: new leader `stack_hist_dt_cat`. It wins per protocol on both splits
and seeds (ds1 110.6→92.1, −16.7%, p90 230.9→186.0; shift 113.0→92.1, −18.5%,
p90 241.1→187.6).**
Cols: `_NOCAL + _STACK + [hist_path_sec, hist_path_cov, hist_path_sec_dt]`, with
`categorical_features=[0]` (route_id).

### Serving estimate
- Base: same as stack_hist (link store plus position ring persisted, historical
  link×hour table at export), about 1.5 days.
- Day-type table: build two tables (weekday and weekend) instead of one. This
  costs nothing extra.
- Categorical route: the exported trees need categorical splits. For each
  categorical node, export sklearn's `raw_left_cat_bitsets` (8×uint32 per node)
  and use a bitset membership test in inference, instead of the ordinal
  threshold now used for route_id (export_worker_data.py route_to_int). Also
  mirror the change in the worker if it walks the trees. About 0.5–1 day.
- Total is about 2–2.5 days.

### Next steps
1. Route 122 (~230–310 s) is still the worst route. Inspect its labels and shape.
2. Other categoricals: stops_ahead / hour as native categoricals? Probably not
   useful. A better candidate is a stop-level categorical (next stop id, too
   many levels for 255 bins, so it would need a hashed or grouped version).
3. Day-type split of the live store? (Probably none; the live store is already same-day.)
4. Weight recent training days more heavily, or train on a longer window
   (more days, now that history is cheap).

## 2026-09-26 — run 7 (own-vs-hist, recency, prune, residual targets, stop categorical)

Rebuilt DS1 as in run 6 (`pipeline_lite.py --days 2026-08-31..2026-09-21 --parallel 4`,
23:17→01:34 UTC with prep chained per day; `MS_FEAT_DIR=ms_features_v4`). Baseline
110.6 / 113.0 and stack_hist_dt_cat 92.1 / 92.1 reproduced exactly, so v5 = v4 numbers.
New V5 columns in `features_live.py` (appended; the existing columns are unchanged):
`own_hist_ratio3/8`, `own_excess8` (this vehicle's own last 3 or 8 completed links:
actual vs the historical link×hour median), `hist_x_own`.
New `cmp.py TAG [REF]` prints a tag's arms vs the baseline and vs a reference arm.

Cmds: `MS_FEAT_DIR=ms_features_v4 sh research/model-search/queue.sh ds1v5 2026-09-07..2026-09-18 2026-09-19..2026-09-21 42 <arms>`,
`… ds1shiftv5 2026-09-07..2026-09-17 2026-09-18..2026-09-20 7 <arms>`.

### Results (MAE / p90; Δ vs baseline, then Δ vs stack_hist_dt_cat)
| arm | ds1v5 (s42) | Δbase | Δleader | ds1shiftv5 (s7) | Δbase | Δleader |
|---|---|---|---|---|---|---|
| baseline | 110.6 / 230.9 | | | 113.0 / 241.1 | | |
| stack_hist_dt_cat (run-6 leader) | 92.1 / 186.0 | −16.7% | | 92.1 / 187.6 | −18.5% | |
| dtcat_own (+ own vs hist) | 92.0 / 185.8 | −16.8% | −0.1% | | | |
| dtcat_own_x (+ hist×own ratio) | 92.0 / 185.9 | −16.8% | −0.1% | | | |
| dtcat_recency (half-life 7 d) | 92.5 / 186.6 | −16.3% | +0.5% | | | |
| dtcat_prune (− is_holiday, trip_progress_frac, stops_remaining) | 93.3 / 187.9 | −15.6% | +1.3% | | | |
| dtcat_resid (target y − hist path) | 91.9 / 185.3 | −16.9% | −0.2% | | | |
| dtcat_logratio (log ratio target) | 92.6 / 184.8 | −16.3% | +0.5% | | | |
| dtcat_big (255 leaves, min 50) | 91.6 / 185.4 | −17.2% | −0.6% | 91.3 / 187.3 | −19.2% | −0.9% |
| dtcat_stopcat (+ target stop categorical) | 91.6 / 185.0 | −17.2% | −0.6% | 91.2 / 186.1 | −19.3% | −1.0% |
| **dtcat_stopcat_big** | **90.7 / 183.8** | **−18.0%** | **−1.6%** | **90.5 / 185.4** | **−19.9%** | **−1.7%** |

- Own-vs-historical ratio: noise. Own speed plus live links already carry it.
- Recency weighting and the residual/log-ratio targets: no gain. The trees already
  track the historical path closely.
- Pruning hurts by 1.3%. trip_progress_frac and stops_remaining carry signal even
  though the schedule is synthetic, so keep them.
- The target-stop categorical (`stop_cat`: the 254 most frequent train stops get
  their own code, the rest share code 254; native categorical next to route_id) and
  255-leaf trees add up (−0.6 and −0.6 on ds1, −1.6 together; −1.0 and −0.9 on
  shift, −1.7 together).
- dtcat_stopcat_big vs baseline: every stops_ahead bucket improves on both splits.
  ds1: sa1 72.6→63.7, sa5 112.0→90.9, sa10 153.5→126.0. Shift: sa1 71.2→62.6,
  sa10 159.7→126.6. By day, ds1 Sat 100.2→85.5, Sun 107.1→89.2, Mon 120.5→95.4;
  shift Fri 126.8→94.7. Rows: train 2.18M / 1.98M, test 1.41M / 1.42M. n_iter still
  hits the 1200 cap.
- Worst routes (dtcat_stopcat_big, ds1): 122 317, 133 216, 106 168, 125 156, 113 138,
  137 137, 94 136, 129 133, 131 132, 111 130. On shift, 122 is 227 and 133 is 154.
  Route 122 gets no better from anything we have tried.

**Conclusion: new leader `dtcat_stopcat_big`. It wins per protocol on both splits
and seeds (ds1 110.6→90.7, −18.0%, p90 230.9→183.8; shift 113.0→90.5, −19.9%, p90
241.1→185.4).** It is only 1.6–1.7% better than the run-6 leader, so if 2x trees
are too much for the push-feed runner, `dtcat_stopcat` at 127 leaves keeps
−0.6 to −1.0%.

### Serving estimate
Same as stack_hist_dt_cat (about 2–2.5 days, including categorical bitsets), plus:
- a stop_id→code map with 254 entries, shipped with the model;
- a second categorical column in the bitset path;
- 255-leaf trees, so about 2x the node arrays in the export and about 1 more
  level per tree walk.
Per-row cost grows only by about log2(2) = 1 comparison per tree.

### Scale check: 3x training rows (tag `ds1v5k3`, `--keep-pct 3`, 6.51M train rows, seed 42)
| arm | MAE / p90 | Δbase | sa1 / sa5 / sa10 | fit |
|---|---|---|---|---|
| baseline | 108.1 / 226.9 | | 70.4 / 109.7 / 149.9 | 18 min |
| stack_hist_dt_cat | 90.5 / 182.8 | −16.3% | 64.0 / 90.8 / 124.8 | 20 min |
| dtcat_stopcat_big | 88.8 / 180.2 | −17.9% | 61.8 / 89.2 / 123.4 | 22 min |

More data helps every arm by about 2 s. The relative gain holds, so it is not an
artifact of the 1% subsample. dtcat_stopcat_big vs the old leader is −1.8% here,
slightly more than at 1%. n_iter still hits the 1200 cap.

### Next steps
1. Route 122 (230–320 s) is untouched by every feature so far. Inspect its labels,
   shape and trip inference directly (a data/labeling issue is likely).
2. Stop coverage: 254 codes leave the tail pooled. Try a second categorical for the
   vehicle's current/previous stop, or hash the rest into the spare codes by region.
3. The n_iter cap binds everywhere. Try lr 0.08 with 255 leaves on stopcat
   (lr 0.1 on live_all_s did nothing, but the tree size is different now).
4. Serving: settle on 127 vs 255 leaves with a timing test of `src/inference.py`
   predict_rows at 2x nodes.
