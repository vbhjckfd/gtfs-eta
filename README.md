# gtfs-eta

ML-based GTFS-RT TripUpdates feed for Lviv public transport. Predicts per-stop arrival times using a gradient-boosted model trained on historical vehicle position snapshots, and serves the result as a drop-in replacement for the operator's own `trip_updates` feed.

## How it works

```
track.ua-gis.com/vehicle_position  (GTFS-RT)
        │  ~11 s (median gap between archived snapshots)
        ▼
  Cloudflare R2 (Bronze)           raw/*.pb  — immutable protobuf snapshots
        │  run_pipeline.py
        ▼
  data/training/YYYY-MM-DD.parquet (Gold)    — snapshot-anchored rows with actual arrivals
        │  make train
        ▼
  models/eta_pipeline.joblib       HistGradientBoostingRegressor
        │  make export
        ▼
  R2: worker/*.pkl                 compact GTFS + model for inference
        │  push_feed.py (GitHub Actions, every 5 min)
        ▼
  R2: feed/trip_updates.pb         pre-computed GTFS-RT TripUpdates
        │  GET /
        ▼
  Cloudflare Worker  →  consumers (apps, journey planners, …)
```

**Cron reliability trick**: GitHub Actions scheduled triggers are unreliable on low-activity repos. Instead, a Cloudflare Worker cron fires every 5 minutes and dispatches the `push-feed.yml` GitHub Actions workflow, which pushes a fresh feed snapshot every 10 seconds for ~4 minutes.

## Repository layout

```
src/                 Python library (features, labeling, training, inference)
scripts/             Pipeline and operational scripts
worker/              Cloudflare Worker — /health + cron + legacy feed redirects (wrangler)
tests/               Smoke tests against the live worker
data/
  gtfs_static/       Local copy of GTFS static (stops, trips, shapes)
  training/          Gold parquets — one per day, training input
models/              Trained sklearn pipeline (joblib)
docs/
  collector_rules.md Data contract for R2 storage (medallion architecture)
```

## Requirements

- Python ≥ 3.11
- Node ≥ 22 (for `wrangler deploy`)
- Cloudflare account with R2 enabled
- `.env` populated from `.env.example`

## Setup

```bash
pip install -e ".[dev]"
cp .env.example .env   # fill in R2 credentials
```

## Common commands

```bash
make pipeline            # process all days from R2 → training parquets (incremental, PARALLEL=4)
make pipeline-date DATE=2026-06-01   # single date
make train               # build features + train model from data/training/
make learn               # pipeline + train in one step
make measure-dwell       # measure real per-route-type stop dwell → models/dwell.joblib
make export              # serialise GTFS + model, upload to R2
make deploy              # deploy Cloudflare Worker
make release             # export + deploy in one step

make push-feed           # push one TripUpdates snapshot to R2 now
make serve-feed          # push every 10 s (local daemon)
make smoke               # smoke-test the live worker against the network
make sanity              # check R2 collection health (snapshot counts, staleness)
make check-gtfs          # verify GTFS static loading
make check-snapshots     # verify R2 Bronze snapshot reading
```

## Environment variables

| Variable | Description |
|---|---|
| `R2_ACCOUNT_ID` | Cloudflare account ID |
| `R2_ACCESS_KEY_ID` | R2 S3-compat access key |
| `R2_SECRET_ACCESS_KEY` | R2 S3-compat secret |
| `R2_BUCKET` | R2 bucket name (default: `gtfs-lviv`) |
| `GTFS_STATIC_URL` | URL for the static GTFS zip |
| `GTFS_RT_URL` | URL for the vehicle positions GTFS-RT feed |
| `PROJECTED_CRS` | UTM CRS for geometry (default: `EPSG:32635`) |

The worker also requires a Cloudflare secret:

```bash
wrangler secret put GITHUB_TOKEN            # PAT with workflow scope
wrangler secret put SENTRY_DSN              # optional — error reporting to Sentry
wrangler secret put NEW_RELIC_LICENSE_KEY   # optional — telemetry to New Relic
```

### New Relic

The New Relic Node APM agent cannot run in a Worker — it is CommonJS, spawns
background harvest timers and reads Node internals that a V8 isolate does not
have. So [worker/newrelic.js](worker/newrelic.js) posts custom events straight
to the Event API over `fetch`, the same no-SDK approach as `sendSentryEvent`.
Without the key, reporting is a no-op.

Two event types, both tagged `service = gtfs-eta-worker` and stamped with
`GIT_COMMIT`:

- `GtfsEtaWorkerHealth` — one per `/health` hit, mirroring the verdict:
  `status`, `ageSec`, `entities`, `arrivals`, `workingHours`, `vehiclesIn`,
  `vehiclesStale`, `feedSkewSec`, `feedCommit`. Every field is already public in
  the `/health` body. The legacy redirect paths are deliberately not
  instrumented — they are edge-cacheable and carry no signal.
- `GtfsEtaWorkerCron` — one per cron fire: which `cron` matched, the `workflow`
  dispatched, whether the dispatch and the feed archive succeeded, `durationMs`.
  This is the only record that the 5-minute pipeline actually fired.

```sql
SELECT max(ageSec) FROM GtfsEtaWorkerHealth TIMESERIES SINCE 1 day ago
SELECT count(*) FROM GtfsEtaWorkerCron WHERE dispatched IS false SINCE 1 day ago
```

The account is in New Relic's **EU** region: ingest goes to
`insights-collector.eu01.nr-data.net` and the key is the 40-character licence
key starting `eu01xx`, not an `NRAK-...` user API key.

## Model

**Target**: `seconds_to_arrival` — time from the snapshot to the actual arrival at each upcoming stop (direct multi-horizon, up to 10 stops ahead).

Training examples are **snapshot-anchored**: every vehicle position snapshot yields one row per upcoming stop, so the model sees vehicles mid-segment and dwelling at stops — not only at stop crossings. This is what lets it predict ≈0 s when a bus is already at the stop.

**Algorithm**: `sklearn.ensemble.HistGradientBoostingRegressor` wrapped in a `Pipeline` with `OrdinalEncoder` for `route_id`. The full pipeline is saved to `models/eta_pipeline.joblib` — a single file contains everything needed for inference.

**Features** (16 total — 13 base + 3 prior-derived):

| Feature | Description |
|---|---|
| `route_id` | Route identifier (categorical) |
| `stop_sequence` | Target stop index in the trip |
| `stops_ahead` | Prediction horizon in stops (1 = next stop) |
| `hour`, `day_of_week`, `month` | Temporal context |
| `is_weekend`, `is_holiday` | Calendar flags (Ukrainian holidays) |
| `remaining_dist_m` | Shape distance from the vehicle's projected position to the target stop |
| `sched_remaining_sec` | Schedule's expectation for that remaining distance (interpolated at the vehicle's position) |
| `progress_speed_mps` | Observed speed over the last snapshot interval (−1 when unknown) |
| `stops_remaining` | Stops left after the target |
| `trip_progress_frac` | Position along route [0, 1] |
| `dist_per_stop_m` | `remaining_dist_m / stops_ahead` |
| `speed_eta_warm` | `remaining_dist_m / effective_speed`, warm-started from the prior when speed is unknown |
| `hist_speed_mps` | Route×hour historical median speed |
| `hist_travel_time_est` | `stops_ahead ×` historical seconds-per-stop (dwell-aware) |

The last three are computed by `apply_priors()` from a route×hour speed/dwell
table built on the training split only, so they carry no test-set leakage. The
feature vector is indexed **positionally** by the exported tree traversal, so
`FEATURE_COLS` in [src/features.py](src/features.py), `build_features` in
[src/inference.py](src/inference.py) and `_extract_trees` in
[scripts/export_worker_data.py](scripts/export_worker_data.py) must stay in sync.

**Baseline**: scheduled remaining time (`sched_remaining_sec`). The model is evaluated against this baseline and must beat it on the held-out test set (last 20% of days by date).

### live_v2 (current production model)

The model above sees one snapshot of one vehicle. `live_v2` ([src/train_live.py](src/train_live.py), features in [src/live_features.py](src/live_features.py)) keeps 13 of its base features (drops `month`, `day_of_week`, `stop_sequence`), makes `route_id` and the target stop (top 254) native categoricals, and adds 36 live columns:

| Group | Columns | Source |
|---|---|---|
| Live link store | `live_path_sec_m3/_m5`, `live_path_cov/_age`, `live_speed_mps`, `cur_link_*` | last traversals of every stop-to-stop link on the path by **any** vehicle (≤ 30 min), summed to the target stop |
| Historical links | `hist_path_sec`, `hist_path_sec_dt`, `hist_path_cov` | median link time × local hour (and × weekday/weekend) over the prior 14 days |
| Own history | `lap_path_*`, `veh_hist_ratio60/_n60`, `own_speed_60/180/300` | this vehicle's previous traversal of the path links (≤ 3 h), its own-vs-historical link times over the last hour, its recent speed |
| Dwell | `dwell_rem_med/_p75`, `dwell_p_more120`, `dwell_n`, `dwell_long_share` | remaining stop time at its current location (stop + 50 m bin) given `stationary_sec`, from prior days' stops |
| Feed fields | `pos_age_*`, `fs_now`, `fs_mean60`, `fs_zero_frac600`, `odo_spd_60/180/600/1200`, `odo_m_300` | GPS-fix age, reported speed and odometer windows |

Missing live values are `-1`. Trees: 255 leaves, ≤ 1200 iterations, `max_features=0.3`, absolute-error loss.

**One engine for training and serving.** `LiveState` is fed the same two streams in both: every vehicle entity's raw feed fields, and every on-route vehicle's projected distance along its trip. `run_pipeline.py` writes both streams per day (`data/training/side/<day>.feed|pos.parquet`), the trainer replays them to build features for each training snapshot, and `run_inference` feeds the live ones. Information from other vehicles only becomes visible on the next snapshot, so the order the daemon processes vehicles in doesn't matter. `push_feed.py` carries the state across restarts in `feed/live_state.pkl.gz`. The static tables (historical link medians, dwell) are built by the trainer from the 14 days up to its last day and shipped inside the model export.

**Held-out result** (protocol of branch `claude/model-search`, same rows for both models; train 09-19..09-30, test 10-01 Thu / 10-02 Fri / 10-03 Sat, 1.58 M rows):

| | legacy | live_v2 |
|---|---|---|
| MAE | 119.3 s | 92.8 s (−22.2%) |
| median | 60.7 s | 45.0 s |
| p90 | 256.1 s | 192.3 s (−24.9%) |
| MAE at 1 / 5 / 10 stops ahead | 73.8 / 121.7 / 167.9 s | 60.8 / 93.6 / 131.7 s |

```bash
make pipeline     # label the last 36 days (rows + side tables, 10% of snapshots)
make train        # python -m src.train_live: latest 21 days, last 20% held out
make export       # exports live_v2 when models/eta_live.joblib exists (EXPORT_MODEL=legacy|live to force)
python -m src.train_live --train 2026-09-19..2026-09-30 --test 2026-10-01..2026-10-03 --compare-baseline --no-save
```

**Rollback**: `make train-legacy && EXPORT_MODEL=legacy make export`. The daemon serves whichever format is in `worker/eta_pipeline.pkl`.

## Inference & serving

Inference runs in `scripts/push_feed.py` (the GitHub Actions push pipeline), using a compact, pure-Python re-implementation of GBT tree traversal (`src/inference.py`) — no sklearn or pandas at runtime.

Per cycle (every ~10 s):
1. Fetch the current vehicle positions GTFS-RT feed from upstream.
2. For each vehicle: project lat/lon to UTM, match to the best trip shape (off-route filter with hysteresis), project the vehicle's exact position along the shape, and measure progress speed vs the previous cycle.
3. Build feature rows for the next ≤10 stops, anchored at the vehicle's projected position.
4. Run the serialised GBT model.
5. Encode a GTFS-RT TripUpdates protobuf and a cleaned VehiclePositions protobuf, and upload both to R2.

A Lviv route averages 216 trips over just 2.5 distinct shapes, so trip matching
memoises the polyline walk per shape, caches the schedule-progress lookup per
push (every vehicle in a push shares one `now_sec`), and skips any shape whose
bounding box already scores worse than the best full candidate. Together these
cut a full inference pass from **4.0 s to 0.76 s** on the live fleet, with
byte-identical output. The caches live on the loaded data dict, so they expire
with the export they describe.

Each `StopTimeUpdate` carries a `StopTimeEvent.uncertainty` (seconds) so a consumer can widen the arrival window for far-horizon stops instead of treating a 1-stop and a 10-stop ETA as equally certain. The bands are **calibrated from live serving error**: `make export` pools the per-stops-ahead MAE from the last 7 days of the quality archive (`quality/*.json`) — which runs ~2× the training-test split — and bakes the result into the model blob. The training-test MAE (`models/uncertainty.joblib`) is only a cold-start fallback for before any day has been scored.

### What else each TripUpdate carries

Every field below is optional in GTFS-RT, so a consumer that ignores them sees
exactly the feed it saw before. Each is omitted rather than guessed when the
underlying data is missing (an older export, a trip with no parseable schedule).

| Field | Why |
|---|---|
| `TripUpdate.timestamp` | Age of the vehicle fix the prediction was computed from. The feed header is the *publish* time and is always fresh, so this is the only thing distinguishing a prediction made from a 2 s-old position from one made from a 170 s-old one (the staleness filter tolerates up to 180 s). |
| `TripDescriptor.start_date` / `start_time` | Binds an update to a specific *run* of the trip. `trip_id` alone cannot: a loop route repeats through the day, and an after-midnight trip belongs to the previous service day. |
| `TripDescriptor.direction_id` | Parsed from `trips.txt` all along, previously never exported — consumers had to infer direction from the shape. |
| `TripUpdate.delay` | Deviation from the timetable, **tram and trolleybus only**. Lviv's bus timetable is unreliable enough that no schedule feature survived into the model at all, and measured live, bus delay spans roughly ±30 min at the 10th/90th percentile. Since consumers use `delay` to extrapolate stops that aren't listed, publishing that would mislead; the two electric modes are the same measured exception the terminus path already makes. |
| Trailing `NO_DATA` `StopTimeUpdate` | Fences off the stops past the 10-stop horizon. Per the spec a consumer applies the last listed stop's deviation to every later stop of the trip — so without the fence, our 10th stop's error was being extrapolated across the whole remainder of the route. It carries a stop but deliberately no times. |
| `StopTimeEvent.departure` | `arrival + dwell`, where dwell is **measured**, not assumed: 32 s for trams, 27 s for buses and trolleybuses, from 1.76 M stationary events over 14 days of labelled history (`make measure-dwell`). It was a flat 15 s guess before. |

A prediction whose arrival has already elapsed is published as *arriving now*
rather than dropped. Dropping it hid the one arrival a waiting rider most wants,
and — because the horizon it served vanished from the emitted list — silently
renumbered every later stop. The quality scorer reads the horizon from exactly
that position, so serving and calibration had been keying their per-horizon
uncertainty and bias tables off horizons that differed by one.

The **cleaned VehiclePositions feed** (`feed/vehicle_positions.pb`) re-emits the upstream positions enriched with this project's corrected trip match (often better than the operator's reported `trip_id`), the next stop + `current_status` (STOPPED_AT / IN_TRANSIT_TO), and a `congestion_level` derived from observed-vs-historical speed. It is a by-product of the same inference pass, so it costs no extra geometry work.

The feeds are served from the **R2 public custom domain** `eta.lad.lviv.ua`, which Cloudflare edge-caches natively (`Cache-Control: max-age=10` stamped on the objects), so they cost no worker invocation or R2 read per poll:
- `https://eta.lad.lviv.ua/feed/trip_updates.pb`
- `https://eta.lad.lviv.ua/feed/vehicle_positions.pb`

The `worker/` directory is a plain JS Cloudflare Worker that no longer proxies the feeds — it only:
- `GET /` and `GET /vehicle_positions` — **301-redirect** to the R2 domain above (backward compatibility for unmigrated consumers; no R2 read).
- `GET /health` — parses the feed (hand-rolled protobuf wire walk, no deps) and returns 200/503 based on header freshness and, during working hours, predicted arrivals at stop 60.
- cron (every 5 min) — dispatches `push-feed.yml` (see the cron reliability trick above).

It was originally a Python Worker; Pyodide isolates intermittently entered a poisoned state where every request failed in ~2 ms before handler code ran (`scriptThrewException` storms), so it was rewritten in JS.

## R2 storage layout (medallion)

```
gtfs-lviv/
  raw/        YYYY-MM-DD/<feedTsISO8601Z>.pb   # Bronze — immutable raw protobuf
  positions/  YYYY-MM-DD.parquet               # Silver — consolidated rows
  static/     <feedVersion>/static.zip         # Versioned GTFS static
  static/     index.json                       # day → feedVersion mapping
  _meta/      collector_health.json            # collection health counters
  feed/       trip_updates.pb                  # pre-computed TripUpdates feed (served via eta.lad.lviv.ua)
  feed/       vehicle_positions.pb             # cleaned VehiclePositions feed (served via eta.lad.lviv.ua)
  predictions/ YYYY-MM-DD/<feedTsISO>.pb       # sampled archive of the served feed (quality scoring)
  quality/    YYYY-MM-DD.json + latest.json    # scored live-prediction quality
  worker/     gtfs_worker_data.pkl             # compact GTFS for push_feed.py inference
  worker/     eta_pipeline.pkl                 # serialised GBT for push_feed.py inference
```

The worker's 5-min cron archives the currently served feed into `predictions/`
before dispatching the next refresh, so live-prediction quality can be scored
offline against the actual arrivals derived from `raw/`. The archive is capped
by a 14-day R2 object-lifecycle rule (managed in the Cloudflare R2 dashboard).

See [docs/collector_rules.md](docs/collector_rules.md) for the full data contract.

## Smoke tests

```bash
make smoke
# or
pytest tests/test_smoke.py -v
```

Tests hit the live worker over the network and verify:
- HTTP 200, `application/x-protobuf` content type
- Feed parses as a valid `FeedMessage`, timestamp is fresh
- All entities are `TripUpdate` with well-formed stop sequences
- Arrival times are monotonically increasing and in the future
- Every TripUpdate carries a prediction `timestamp`
- The `NO_DATA` horizon fence, where present, is last, names a stop and carries no times
- Trip ID format matches Lviv's `DIGITS_DIGIT_DIGIT` scheme
- Stop codes are numeric (matching physical signage)
- ≥50% vehicle coverage vs the upstream vehicle positions feed
- ≥10% trip overlap with the operator's own `trip_updates` feed

Override the target URL:
```bash
SMOKE_URL=https://your-preview-url.workers.dev make smoke
```
