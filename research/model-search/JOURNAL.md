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

## 2026-09-26 — run 8 (current-link features, next-stop categorical, lr 0.08, route 122)

Lock taken at 04:17 UTC (run 7's lock was 5 h old; run 7's session released it later
and merged, keeping this lock). Rebuilt DS1 with `pipeline_lite.py --days 2026-08-31..2026-09-21 --parallel 4`
(04:18→06:26 UTC). NB for later runs: prep must run for every day from 08-31, not
only 09-07+, because each prep writes `links_<day>` that later days read as history.
The first chain only prepped 09-07+ and was also OOM-killed next to 4 pipeline
workers. The fixed chain (in the journal cmds below) sets `oom_score_adj=1000` on
itself and waits for MemAvailable ≥ 6 GB.
```
# prep chain: for day in 08-31..09-21 ascending: wait for data/ms_lite/<day>.parquet,
#   MS_FEAT_DIR=ms_features_v4 python research/model-search/harness.py prep --days <day>
sh research/model-search/run8.sh    # 11 fits, ~2 h
```
New V6 columns in `features_live.py` (appended to the existing ones):
`next_stop_id` (the vehicle's next stop; for sa=1 it matches the target stop 97% of the time),
`cur_link_live` (live m5 of the link the vehicle is on), `cur_link_hist` (day-type
historical median of that link), `cur_link_frac` (fraction of the link still ahead).

### Results (MAE / p90; Δ vs baseline, then Δ vs dtcat_stopcat_big)
| arm | ds1v6 (s42) | Δbase | Δleader | ds1shiftv6 (s7) | Δbase | Δleader |
|---|---|---|---|---|---|---|
| baseline | 110.6 / 230.9 | | | 113.0 / 241.1 | | |
| dtcat_stopcat_big (run-7 leader) | 90.7 / 183.8 | −18.0% | | 90.5 / 185.4 | −19.9% | |
| sc_big_nxt (+ next-stop categorical) | 90.5 / 184.7 | −18.1% | −0.1% | 90.3 / 186.2 | −20.1% | −0.2% |
| **sc_big_cur (+ current-link live/hist/frac)** | **90.1 / 182.9** | **−18.5%** | **−0.6%** | **89.5 / 183.9** | **−20.8%** | **−1.1%** |
| sc_big_v6 (both) | 90.3 / 183.6 | −18.4% | −0.4% | 89.8 / 185.1 | −20.6% | −0.8% |
| sc_big_lr08 (lr 0.08) | 90.9 / 184.3 | −17.8% | +0.2% | | | |

- Baseline and the run-7 leader reproduced exactly (110.6 / 113.0, 90.7 / 90.5).
- The current-link columns give a small gain that holds on both splits: every
  stops_ahead bucket is 0.3–1.4% better than the run-7 leader and p90 improves.
  ds1 sa1/sa5/sa10: 63.4 / 90.1 / 125.3 (baseline 72.6 / 112.0 / 153.5).
  Shift: 61.7 / 89.8 / 125.5 (baseline 71.2 / 114.3 / 159.7).
  By day, ds1 Sat 100.2→84.9, Sun 107.1→88.5, Mon 120.5→94.9. Shift Fri 126.8→93.9,
  Sat 99.6→84.6, Sun 106.6→88.2. Median AE 56.0→44.0 (ds1) and 59.5→44.6 (shift).
  Bias −15.5 / −17.5. Rows: train 2.18M / 1.98M, test 1.41M / 1.42M.
- Next-stop categorical: noise, and it slightly worsens p90. The target-stop categorical
  already covers it.
- lr 0.08: no gain, even though the 1200-iteration cap binds. Leave lr at 0.05.
- Worst routes (sc_big_cur, ds1): 122 322, 133 219, 106 165, 125 160, 137 138, 94 136,
  113 135, 111 130, 131 130, 129 129. On shift: 122 224, 106 165, 133 146, 125 145.
- Route 122 diagnostic (09-07..09-09, 3% prep): 4.5k rows, 17 trips, only 2 vehicles.
  Live-link coverage is 43% vs 96% on other routes, since its links are not shared
  and the headway is long. Links are ~130 s per stop vs ~60 s elsewhere, and there is
  a service gap around hours 8–10. The historical path alone has median AE 89 s vs
  55 s for other routes. So 122 is data-sparse, not mislabelled. It is expected to
  stay the worst route unless its own-vehicle history is used more (e.g. the same
  vehicle's traversal of the same link on its previous round trip).

**Conclusion: new leader `sc_big_cur`, an incremental gain on the run-7 leader.
It wins per protocol on both splits and seeds (ds1 110.6→90.1, −18.5%, p90
230.9→182.9; shift 113.0→89.5, −20.8%, p90 241.1→183.9).** Against dtcat_stopcat_big
it is −0.6% / −1.1%, so it comes almost for free.

### Serving estimate
Same as dtcat_stopcat_big (about 2–2.5 days in total). The three new columns come
from state that serving already needs: the current link's m5 from the live link
store, its day-type median from the historical table, and the fraction left from
the shape projection. That adds about 0.1 day.

### Next steps
1. Route 122 / sparse routes: an own-vehicle previous-lap link time
   (same vehicle, same link, last traversal within ~2 h) as a fallback where live coverage is low.
2. A timing test of `src/inference.py` at 255 leaves with 2 categorical bitsets, to
   decide between 127 and 255 leaves before any serving work.
3. Try a stronger regulariser on 255 leaves (min_samples_leaf 100, l2 1) with 3x
   data. Run 7's 3x scale check showed that data helps by about 2 s.

## 2026-09-26 — runs 9 and 10 (lost) and run 11

Runs 9 (08:21 UTC) and 10 (~15:20 UTC) committed code and lock refreshes but no
results or journal; their sessions ended mid-rebuild. Run 11 (lock 19:17 UTC) re-runs
run 9's plan with `research/model-search/run11.sh`, which commits and pushes results
after every fit. Setup notes for a fresh box: `pip install -e . tzdata` (without
tzdata every pipeline day fails with ZoneInfoNotFoundError 'Europe/Kiev').
```
python research/model-search/pipeline_lite.py --parallel 4 --days 2026-08-31..2026-09-21 &
sh research/model-search/run11.sh     # prep chain -> ms_features_v7, then fits
```
Plan: ds1v7 / ds1shiftv7 with baseline, sc_big_cur (leader), sc_big_lap (+ own
previous-lap link times, V7 cols), sc_big_cur_reg (min_samples_leaf 100, l2 1); then
the serving timing test (`timing.py` on the saved baseline and sc_big_cur fits).

### Run 11 results (rebuild 19:25→21:40 UTC, fits until ~23:00)
The rebuild was slower than in run 8 (first days took about 35 min per worker; 22
days in 2 h 15 min at `--parallel 4`, with peak use around 11 GB). Baseline and sc_big_cur
reproduced exactly (110.6 / 113.0 and 90.1 / 89.5), so v7 = v6 plus the new columns.

| arm | ds1v7 (s42) MAE / p90 | Δbase | Δleader | ds1shiftv7 (s7) | Δbase | Δleader |
|---|---|---|---|---|---|---|
| baseline | 110.6 / 230.9 | | | 113.0 / 241.1 | | |
| sc_big_cur (run-8 leader) | 90.1 / 182.9 | −18.5% | | 89.5 / 183.9 | −20.8% | |
| **sc_big_lap (+ own previous lap)** | **88.8 / 180.6** | **−19.7%** | **−1.45%** | **88.5 / 182.1** | **−21.7%** | **−1.19%** |
| sc_lap_127 (sc_big_lap, 127 leaves) | 89.7 / 181.5 | −18.9% | −0.5% | 89.4 / 183.1 | −20.9% | −0.1% |
| sc_big_cur_reg (msl 100, l2 1) | 90.1 / 182.9 | −18.5% | +0.0% | | | |

- sc_big_lap improves p90 and every stops_ahead bucket against sc_big_cur on both splits.
  Against the leader, the worst bucket is only −0.3% / −0.3%, so no bucket gets worse.
  Against the baseline, ds1 sa1/sa5/sa10 is 72.6→63.2, 112.0→88.7, 153.5→123.6; shift
  is 71.2→61.5, 114.3→88.6, 159.7→124.1. Median AE 56.0→43.6 / 59.5→44.2. Bias −15.0 / −17.1.
  By day, ds1 Sat 100.2→83.6, Sun 107.1→87.0, Mon 120.5→93.7; shift Fri 126.8→93.2,
  Sat 99.6→83.2, Sun 106.6→86.9. Rows: train 2.18M / 1.98M, test 1.41M / 1.42M
  (Sat 422k, Sun 404k, Mon 589k, Fri 599k).
- Worst routes (sc_big_lap, ds1): 122 319, 133 207, 106 158, 125 156, 113 134, 137 134,
  131 131, 111 131, 129 127, 881 123. Shift: 122 228, 106 159, 125 144, 111 136, 137 135.
  Route 133 improved most (219→207 on ds1). Route 122 barely moved (322→319); its
  previous lap is often more than 3 h back or on the other direction.
- Regularisation (min_samples_leaf 100, l2 1): exactly neutral. Stop tuning regularisation.
- At 127 leaves the previous-lap gain mostly disappears (sc_lap_127 ≈ sc_big_cur at 255 leaves).
- Loss: production already trains with `absolute_error`, so the loss idea is moot.

**Serving timing test** (`timing.py`, single thread, this 4-vCPU box, 3000-row batch,
walking the exported trees with a categorical bitset test added):
baseline 1200 trees / 304k nodes: 0.75 s (export ~15 MB). sc_big_cur 1200 trees / 611k
nodes, 116k categorical nodes: 1.55 s (~29 MB). Parity with sklearn is exact (max
|d| 1e-11). 255 leaves plus bitsets cost 2x per push but fit easily in the 10 s cycle, so
there is no serving reason to drop to 127 leaves. Serving needs: export `is_categorical`,
`bitset_idx`, `missing_go_to_left` and `raw_left_cat_bitsets` per tree, and the
categorical column permutation / re-encoding from `model._preprocessor`.

**Conclusion: new leader `sc_big_lap`, which wins per protocol on both splits and seeds (ds1
110.6→88.8, −19.7%, p90 230.9→180.6; shift 113.0→88.5, −21.7%, p90 241.1→182.1). It is
−1.2 to −1.5% against sc_big_cur.** Serving cost on top of sc_big_cur: the link store also
keeps the last traversal per (vehicle, link) for 3 h, a dict updated from the same crossing
events. That is about +0.3 day, so ~2.5 days in total.

### Next steps
1. Lap window: re-prep with `MS_LAP_WINDOW=21600` (6 h) and 5400 (1.5 h) into new feature
   dirs (~50 min of prep each, sequential because of links history). Route 122 needs the
   longer window.
2. Direction-agnostic previous lap: the same vehicle's traversal of the *same stop pair* in
   either direction, or the time of its previous full round trip.
3. A 3x-data check of sc_big_lap (`--keep-pct 3`, ~25 min per fit).
4. Session budget: the rebuild alone is about 2.25 h. Keep each run to at most one re-prep plus
   about 6 fits.

## 2026-09-26/27 — run 12 (6 h previous-lap window, vehicle-level own/historical ratio)

Lock taken at 23:16 UTC. Rebuilt DS1 with `pipeline_lite.py --parallel 4 --days 2026-08-31..2026-09-21`
(about 1.5 h this time). Ran `sh research/model-search/run12.sh`, which runs the prep chain into
`ms_features_v8` and then the fits. The first prep crashed with a pandas 3 merge-key dtype
mismatch (StringDtype vs object) in the new vehicle merge; fixed by casting both sides with
`.astype(str)`. Everything finished around 03:00 UTC.

New V8 columns in `features_live.py` (the lap code is refactored into `_lap(window)`; the V7
outputs are unchanged, and sc_big_lap reproduced exactly):
- `lap6_path_sec / cov / age / fill`: the same own previous-lap link times with a 6 h window
  instead of 3 h. Coverage on 08-31 rises from 64% to 80% of rows, and on route 122 from
  41% to 68%.
- `veh_hist_ratio60`, `veh_hist_n60`: the vehicle's own link times / day-hour historical
  medians over its links completed in the last hour, **across trips** (cumulative sums per
  vehicle and merge_asof at tq and tq−3600). The run-7 own_hist_ratio* cols are keyed by
  vehicle|trip, so they are empty right after a terminus turnaround. Coverage on 09-01 is
  546k rows vs 463k for own_hist_ratio8.

Arms (arms.py): `sc_big_lap6` (lap cols → 6 h versions), `sc_big_veh` (sc_big_lap + veh
cols), `sc_big_v8` (lap6 + veh).

### Results (MAE / p90; Δ vs baseline, Δ vs the run-11 leader sc_big_lap)
| arm | ds1v8 (s42) | Δbase | Δleader | ds1shiftv8 (s7) | Δbase | Δleader |
|---|---|---|---|---|---|---|
| baseline | 110.6 / 230.9 | | | 113.0 / 241.1 | | |
| sc_big_lap (run-11 leader) | 88.8 / 180.6 | −19.7% | | 88.5 / 182.1 | −21.7% | |
| sc_big_lap6 | 88.9 / 180.7 | −19.6% | +0.16% | 88.4 / 182.1 | −21.7% | −0.05% |
| **sc_big_veh** | **87.7 / 179.1** | **−20.7%** | **−1.19%** | **87.7 / 180.6** | **−22.4%** | **−0.95%** |
| sc_big_v8 (lap6 + veh) | 87.7 / 179.1 | −20.7% | −1.22% | 87.5 / 180.1 | −22.6% | −1.10% |

- sc_big_veh vs baseline: ds1 sa1/sa5/sa10 72.6→62.0, 112.0→87.6, 153.5→122.3; shift
  71.2→60.7, 114.3→87.8, 159.7→123.2. Every stops_ahead bucket is better than the leader
  (worst bucket −0.9% / −0.6%). Median AE 56.0→43.3 / 59.5→44.1. Bias −14.0 / −16.0
  (baseline −28.2 / −14.2).
  By day, ds1 Sat 100.2→82.5, Sun 107.1→85.7, Mon 120.5→92.9; shift Fri 126.8→92.5,
  Sat 99.6→82.5, Sun 106.6→85.9. Rows: test 1.41M / 1.42M (Sat 422k, Sun 404k, Mon 589k,
  Fri 599k); train 2.18M / 1.98M.
- Worst routes (sc_big_veh, ds1): 122 325, 133 210, 106 161, 125 157, 137 131, 131 131,
  129 130, 113 129, 111 128, 881 124. Shift: 122 231, 106 157, 133 146 (leader 134, so it
  got worse on this split), 125 142, 98 141, 111 134, 878 132, 137 132, 2302 132, 94 123.
- The 6 h lap window does nothing, even though it adds route-122 coverage (41%→68%). Route
  122 stays at 320–325 (ds1) / 230 (shift). A 3–6 h old lap does not predict this route's
  current link times, so its error is not about missing features on the path. Stop pursuing
  lap windows.
- sc_big_v8 vs sc_big_veh is −0.03% / −0.15%, which is noise, so prefer the simpler sc_big_veh.

**Conclusion: new leader `sc_big_veh`. It wins per protocol on both splits and seeds (ds1
110.6→87.7, −20.7%, p90 230.9→179.1; shift 113.0→87.7, −22.4%, p90 241.1→180.6), and it
beats sc_big_lap on both by −1.0 to −1.2% with p90 and every bucket better.**

### Serving estimate
On top of sc_big_lap (~2.5 days): a per-vehicle deque of (t, own link time, historical
link time) for the last hour, fed by the same crossing events that the link store already
consumes. Two running sums per vehicle, so about +0.2 day. No new artifacts.

### Next steps
1. Route 133 got worse on shift with veh (134→146). Check whether veh_hist_ratio60 is
   polluted by deadhead / layover links at the terminus, e.g. with a cap on link time or
   by excluding the first link after a trip change.
2. Vehicle-level ratio windows: 30 min and 2 h, and a count-based (last 15 links) version.
3. Route 122: features on the path are exhausted. Look at the target side instead
   (dwell/terminus layover at its stops, schedule headway gaps 8–10h) or accept it.
4. A 3x-data check of sc_big_veh (`--keep-pct 3`).

## 2026-09-27 — run 13 (network-wide ratio, short / clipped vehicle ratios)

Lock taken at 03:17 UTC. Rebuilt DS1 with `pipeline_lite.py --parallel 4 --days 2026-08-31..2026-09-21`,
at about 41 min per batch of 4 days, then ran `sh research/model-search/run13.sh`, which runs the prep chain
into `ms_features_v9` and then the fits. A background loop refreshes LOCK every 50 min, because the
run lasts longer than the 3 h lock age.

New V9 columns in `features_live.py` (V8 outputs unchanged):
- `net_ratio15`, `net_n15`: network-wide sum(link time, clipped to [hist/3, 3·hist]) / sum(historical
  day-hour median) over *all* vehicles' links completed in the last 15 min (citywide traffic state; a
  global cumsum plus searchsorted, O(1) per row when serving). On 09-01 it ranges 0.77–1.24 (median 1.04),
  with n ≈ 3k links per 15 min.
- `veh_hist_ratio20`: the run-12 vehicle-level ratio over a 20 min window.
- `veh_hist_ratio60c`: the 60 min vehicle ratio with each link clipped to [hist/3, 3·hist], so that
  layover and deadhead links at the terminus can't dominate it (run 12 next step 1, the route 133 regression).

Arms: `sc_big_net` (sc_big_veh + net), `sc_big_v9` (net + ratio20 + ratio60c).

The rebuild ran from 03:20 to 05:36 UTC (first batch about 41 min per day, later days about 20 min).
Baseline and sc_big_veh reproduced exactly (110.6 / 113.0 and 87.7 / 87.7).

### Results (MAE / p90; Δ vs baseline, Δ vs the leader sc_big_veh)
| arm | ds1v9 (s42) | Δbase | Δleader | ds1shiftv9 (s7) | Δbase | Δleader |
|---|---|---|---|---|---|---|
| baseline | 110.6 / 230.9 | | | 113.0 / 241.1 | | |
| sc_big_veh (leader) | 87.7 / 179.1 | −20.7% | | 87.7 / 180.6 | −22.4% | |
| sc_big_net | 87.8 / 179.0 | −20.6% | +0.07% | 87.5 / 179.8 | −22.5% | −0.13% |
| sc_big_v9 | 87.6 / 178.1 | −20.8% | −0.13% | 87.5 / 179.4 | −22.6% | −0.16% |

- Both are noise-level. sc_big_v9 trims p90 by 0.5–0.65% and the long horizons (ds1 sa10
  122.3→121.6), but sa1 gets 0.5–0.9% worse. The network ratio alone does nothing: the
  shared live link store already carries current traffic on each link, so a citywide scalar adds
  no information.
- The clipped vehicle ratio does not fix route 133 on shift (146.0 → 146.9), so the run-12
  regression there is not caused by layover links. Route 122 gets worse with the net
  cols (ds1 325→334–337, shift 230→246–247). 122 is sparse and long-headway, so a global
  scalar is a spurious split for it.
- Rows: train 2.18M / 1.98M, test 1.41M / 1.42M (unchanged).

**Conclusion: no new leader. sc_big_veh stays. The vehicle/network "state ratio" family is
exhausted: 3 h / 6 h lap, 20 / 60 min vehicle ratio, clipped, and network-wide all land within ±0.2%.**

### Scale check: 3x training rows (tag `ds1v9k3`, `--keep-pct 3`, 6.51M train rows, seed 42)
The first attempt was OOM-killed for both arms, because the v9 feature files are too wide for 3x in 15.7 GB.
`harness.py` now reads `MS_DROP_COLS` (a comma list of feature cols to skip at load; numerics
unchanged). run13b.sh drops the 21 live cols that neither arm uses.
| arm | MAE / median / p90 | Δbase | sa1 / sa5 / sa10 | Sat / Sun / Mon | fit |
|---|---|---|---|---|---|
| baseline | 108.1 / 54.4 / 226.9 | | 70.4 / 109.7 / 149.9 | 96.8 / 106.3 / 117.4 | 20 min |
| sc_big_veh | 85.4 / 42.4 / 174.3 | −21.0% | 60.2 / 85.5 / 119.2 | 79.7 / 83.8 / 90.7 | 28 min |
- The baseline reproduces run 7's 3x baseline exactly (108.1 / 226.9). 3x data helps the leader by 2.3 s
  (87.7→85.4), a little more than the baseline's 2.5 s relative to its size. The relative gain holds and grows
  (−20.7% → −21.0%), with p90 −23.2% and the worst bucket −14.5%. Bias is −14.7 (baseline −27.5).
- Worst routes at 3x: 122 296, 133 207, 125 150, 106 143, 137 132, 129 131, 111 126, 113 126,
  131 125, 881 124. Route 122 gains the most from more data (325→296), so part of its error is data
  sparsity in training, not only in the live features.

### Next steps
1. Stop adding live-state ratio features (exhausted, see above). The remaining levers are data volume
   (3x = −2.3 s; production trains on 100% of snapshots over more days, so the served gain is likely
   larger than measured here) and the target side.
2. Target-side / dwell: model arrival at the target as arrival at the previous stop plus the target's
   own link, i.e. the historical dwell at the target stop × hour (a static table like hist_dt, keyed by
   stop) and the observed dwell of the last vehicle at the target stop (live, from `.pos`
   stationary runs near the stop).
3. Route 133 (early by ~200 s at long horizons): check whether its trips have a long layover *inside*
   the trip (a timed point), which the model can't see. If so, add a "timed-point stops between
   vehicle and target" count from historical dwell > 60 s.
4. Serving: sc_big_veh is ready to hand off (≈2.7 days, see the run-12 serving estimate; timing test
   in run 11). A productionisation PR is the owner's call, not this routine's.

## 2026-09-27 — run 14 (dwell at the vehicle's current location)

Lock taken at 08:16 UTC. The fresh box had no deps (`pip install -e . tzdata`, as noted in run 11).
Rebuilt DS1 with `pipeline_lite.py --parallel 4 --days 2026-08-31..2026-09-21` (08:20–10:15, ~35 min
per 4-day batch), and ran `sh research/model-search/run14.sh` (prep chain into `ms_features_v10`, then fits).
A background loop refreshes LOCK every 50 min.

Idea (run 13 next step 2, turned around): the tail is stopped vehicles, and `stationary_sec` tells the
model how long a vehicle has been stopped, but not how long stops *at this place* usually last (a red light
vs a terminus layover vs a timed point). New V10 cols in `features_live.py`:
- `stationary_runs(pos)`: every stationary run of the full day, same 25 m anchor rule as
  `labeling._stationary_seconds` (plus a break on >120 s gaps), keyed by location = (last passed stop id,
  50 m bin of the offset past it). ~480k runs/day, median 18 s, 1.4% longer than 120 s. Saved per day as
  `runs_<day>.parquet` next to `links_<day>.parquet`.
- `dwell_features`: over the prior 14 days' runs at the row's location, conditional on the run lasting
  longer than the row's current `stationary_sec` s: `dwell_rem_med` (median of D − s), `dwell_rem_p75`,
  `dwell_p_more120` (P(D > s + 120 | D > s)), `dwell_n`, `dwell_long_share` (share of runs at the key > 120 s).
  NaN when fewer than 5 runs survive. Coverage on 09-10: 89% of all rows; for rows stopped > 60 s, 72%
  (47% with only one prior day); corr with the target on those rows 0.44.
  Serving: a static table key -> sorted durations built at export (like hist_dt), one searchsorted per row.

Arms: `sc_big_dwell` (sc_big_veh + all 5 cols), `sc_big_dwell2` (+ dwell_rem_med, dwell_p_more120 only).

### Results (MAE / p90; Δ vs baseline, Δ vs the leader sc_big_veh)
| arm | ds1v10 (s42) | Δbase | Δleader | ds1shiftv10 (s7) | Δbase | Δleader |
|---|---|---|---|---|---|---|
| baseline | 110.6 / 230.9 | | | 113.0 / 241.1 | | |
| sc_big_veh (leader) | 87.7 / 179.1 | −20.7% | | 87.7 / 180.6 | −22.4% | |
| **sc_big_dwell** | **87.0 / 178.2** | **−21.3%** | **−0.86%** | **87.0 / 179.5** | **−23.1%** | **−0.79%** |
| sc_big_dwell2 | 87.2 / 177.7 | −21.1% | −0.61% | 87.0 / 179.4 | −23.0% | −0.75% |

- Baselines and the leader reproduced exactly (110.6 / 113.0, 87.7 / 87.7). Rows unchanged: train
  2.18M / 1.98M, test 1.41M / 1.42M (Sat 422k, Sun 404k, Mon 589k, Fri 599k).
- sc_big_dwell vs the leader improves every stops_ahead bucket on both splits, most at short horizons
  where a remaining dwell is a big part of the ETA: ds1 sa1 −2.1%, sa2 −1.4%, sa5 −0.7%, sa10 −0.5%; shift
  sa1 −1.7%, sa10 −0.8%. Every test day improves (ds1 Sat 82.5→81.5, Sun 85.7→84.9, Mon 92.9→92.3; shift
  Fri 92.5→91.9, Sat 82.5→81.8, Sun 85.9→85.1). Median 43.3→43.1 / 44.1→43.9, bias −14.0→−12.6 / −16.0→−14.6.
- Route 133 on shift (the run-12 regression) 146→138; ds1 210→208. Route 122 unchanged-to-worse
  (ds1 325→332, shift 230→234).
- Worst routes (sc_big_dwell): ds1 122 332, 133 208, 106 161, 125 155, 129 138, 881 129, 111 129, 137 127,
  113 126, 131 124; shift 122 234, 106 167, 125 144, 98 140, 133 138, 137 134, 111 133, 2302 130, 129 129, 88 121.
- It is the first gain above ±0.2% since run 12; small in MAE but consistent on both splits/seeds and in
  every bucket and day. Both fits hit the 1200-iteration cap, like the leader.

**Conclusion: new leader `sc_big_dwell` (ds1 110.6→87.0, −21.3%, p90 230.9→178.2; shift 113.0→87.0,
−23.1%, p90 241.1→179.5), −0.8% vs sc_big_veh on both splits with every bucket better.**

### Add-on: V11 dwell cols (`addon_v11.py`, `run14b.sh`, tag `ds1v11`, same rows)
`addon_v11.py` appends cols to the v10 day files (into `ms_features_v11`) without a full re-prep (~50 s/day):
- `dwell_h_rem_med / dwell_h_p_more120 / dwell_h_n`: the V10 conditional remaining dwell keyed by
  (location, 3-hour local band). For stopped > 60 s rows its corr with the target is 0.47 (V10: 0.44), but
  coverage is 58% (V10: 72%).
- `dwell_live_last / dwell_live_age / dwell_live_m3`: today's most recent completed stop (run ≥ 30 s) by any
  vehicle at the same location, ended ≤ t − 30 s within the last hour, plus the mean of the last 3.
  Coverage for stopped > 60 s rows is 81%, corr 0.23 / 0.28.

| arm (ds1v11, s42) | MAE / p90 | Δ vs sc_big_dwell | sa1 / sa5 / sa10 Δ |
|---|---|---|---|
| baseline | 110.6 / 230.9 | | |
| sc_big_dwell | 87.0 / 178.2 | (reproduced exactly) | |
| sc_big_dwell_h | 87.1 / 177.9 | +0.19% | −0.4 / +0.2 / +0.4% |
| sc_big_dwell_v11 | 87.1 / 177.9 | +0.13% | −0.4 / −0.0 / +0.5% |

Both are noise with a slight loss, so I stopped the shift-split fits for them early (only one split
was run, so they are not claims either way). The prior-day location table already carries the signal; hour
banding thins it, and the last live stop at a location is mostly a different vehicle in a different phase
(33% of live rows lack it and it correlates weakly).

### Serving estimate (sc_big_dwell)
On top of sc_big_veh (~2.7 days): at export, build location → sorted stop-duration arrays from the last
14 days of stationary runs (from the same positions the pipeline already projects; ~10k keys/day, a few MB).
At serving, derive the location key from the tracker's dist_along + the trip profile, then do one
searchsorted per row against the stationary_sec the daemon already computes (`inference.stationary_seconds`).
No new live state. About +0.3 day, so ~3 days total.

### Next steps
1. Hand-off candidate is now `sc_big_dwell`. Productionizing it is the owner's decision.
2. Dwell family: try the location key without the 50 m bin (stop only) for more coverage, or a
   fallback chain bin→stop. Also try conditioning on route (terminus layover lengths are route-specific).
3. Route 122 stays the worst (230–330 s) through every feature family. Treat it as a data / target issue.
4. If time allows, do a 3x-data check of sc_big_dwell (`--keep-pct 3` with `MS_DROP_COLS`, as in run 13).
