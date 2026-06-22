# gtfs-eta

ML-based GTFS-RT TripUpdates feed for Lviv public transport. Predicts per-stop arrival times using a gradient-boosted model trained on historical vehicle position snapshots, and serves the result as a drop-in replacement for the operator's own `trip_updates` feed.

## How it works

```
track.ua-gis.com/vehicle_position  (GTFS-RT)
        │  every 60 s
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
worker/              Cloudflare Worker — JS feed passthrough + /health (wrangler)
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
wrangler secret put GITHUB_TOKEN   # PAT with workflow scope
wrangler secret put SENTRY_DSN    # optional — enables error reporting to Sentry
```

## Model

**Target**: `seconds_to_arrival` — time from the snapshot to the actual arrival at each upcoming stop (direct multi-horizon, up to 10 stops ahead).

Training examples are **snapshot-anchored**: every vehicle position snapshot yields one row per upcoming stop, so the model sees vehicles mid-segment and dwelling at stops — not only at stop crossings. This is what lets it predict ≈0 s when a bus is already at the stop.

**Algorithm**: `sklearn.ensemble.HistGradientBoostingRegressor` wrapped in a `Pipeline` with `OrdinalEncoder` for `route_id`. The full pipeline is saved to `models/eta_pipeline.joblib` — a single file contains everything needed for inference.

**Features** (13 total):

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

**Baseline**: scheduled remaining time (`sched_remaining_sec`). The model is evaluated against this baseline and must beat it on the held-out test set (last 20% of days by date).

## Inference & serving

Inference runs in `scripts/push_feed.py` (the GitHub Actions push pipeline), using a compact, pure-Python re-implementation of GBT tree traversal (`src/inference.py`) — no sklearn or pandas at runtime.

Per cycle (every ~10 s):
1. Fetch the current vehicle positions GTFS-RT feed from upstream.
2. For each vehicle: project lat/lon to UTM, match to the best trip shape (off-route filter with hysteresis), project the vehicle's exact position along the shape, and measure progress speed vs the previous cycle.
3. Build feature rows for the next ≤10 stops, anchored at the vehicle's projected position.
4. Run the serialised GBT model.
5. Encode a GTFS-RT TripUpdates protobuf and a cleaned VehiclePositions protobuf, and upload both to R2.

Each `StopTimeUpdate` carries a `StopTimeEvent.uncertainty` (seconds) — the model's per-horizon held-out MAE, baked into the export at training time — so a consumer can widen the arrival window for far-horizon stops instead of treating a 1-stop and a 10-stop ETA as equally certain.

The **cleaned VehiclePositions feed** (`feed/vehicle_positions.pb`) re-emits the upstream positions enriched with this project's corrected trip match (often better than the operator's reported `trip_id`), the next stop + `current_status` (STOPPED_AT / IN_TRANSIT_TO), and a `congestion_level` derived from observed-vs-historical speed. It is a by-product of the same inference pass, so it costs no extra geometry work.

The `worker/` directory is a plain JS Cloudflare Worker that serves the pre-computed feeds — no inference CPU at request time:
- `GET /` — streams `feed/trip_updates.pb` from R2.
- `GET /vehicle_positions` — streams the cleaned `feed/vehicle_positions.pb`.
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
  feed/       trip_updates.pb                  # pre-computed TripUpdates feed served at /
  feed/       vehicle_positions.pb             # cleaned VehiclePositions feed served at /vehicle_positions
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
- Trip ID format matches Lviv's `DIGITS_DIGIT_DIGIT` scheme
- Stop codes are numeric (matching physical signage)
- ≥50% vehicle coverage vs the upstream vehicle positions feed
- ≥10% trip overlap with the operator's own `trip_updates` feed

Override the target URL:
```bash
SMOKE_URL=https://your-preview-url.workers.dev make smoke
```
