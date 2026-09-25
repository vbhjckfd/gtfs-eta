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
