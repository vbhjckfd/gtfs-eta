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
  data/labeled/YYYY-MM-DD.parquet  (Gold)    — stop-crossing events with actual arrivals
        │  make train
        ▼
  models/eta_pipeline.joblib       HistGradientBoostingRegressor
        │  make export
        ▼
  R2: gtfs-data.msgpack + model    compact format for Pyodide
        │  push_feed.py (GitHub Actions, every 5 min)
        ▼
  R2: feed/trip_updates.pb         pre-computed GTFS-RT TripUpdates
        │  on_fetch
        ▼
  Cloudflare Worker  →  consumers (apps, journey planners, …)
```

**Cron reliability trick**: GitHub Actions scheduled triggers are unreliable on low-activity repos. Instead, a Cloudflare Worker cron fires every 5 minutes and dispatches the `push-feed.yml` GitHub Actions workflow, which pushes a fresh feed snapshot every 30 seconds for ~4 minutes.

## Repository layout

```
src/                 Python library (features, labeling, training, inference)
scripts/             Pipeline and operational scripts
worker/              Cloudflare Python Worker (wrangler)
tests/               Smoke tests against the live worker
data/
  gtfs_static/       Local copy of GTFS static (stops, trips, shapes)
  labeled/           Gold parquets — one per day, training input
models/              Trained sklearn pipeline (joblib)
docs/
  collector_rules.md Data contract for R2 storage (medallion architecture)
```

## Requirements

- Python ≥ 3.11
- Node ≥ 18 (for `wrangler deploy`)
- Cloudflare account with R2 enabled
- `.env` populated from `.env.example`

## Setup

```bash
pip install -e ".[dev]"
cp .env.example .env   # fill in R2 credentials
```

## Common commands

```bash
make pipeline            # process all days from R2 → labeled parquets (incremental)
make pipeline-date DATE=2026-06-01   # single date
make train               # build features + train model from data/labeled/
make export              # serialise GTFS + model, upload to R2
make deploy              # deploy Cloudflare Worker
make release             # export + deploy in one step

make push-feed           # push one TripUpdates snapshot to R2 now
make serve-feed          # push every 30 s (local daemon)
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
```

## Model

**Target**: `seconds_to_target` — inter-stop travel time from the vehicle's current position to each upcoming stop.

**Algorithm**: `sklearn.ensemble.HistGradientBoostingRegressor` wrapped in a `Pipeline` with `OrdinalEncoder` for `route_id`. The full pipeline is saved to `models/eta_pipeline.joblib` — a single file contains everything needed for inference.

**Features** (13 total):

| Feature | Description |
|---|---|
| `route_id` | Route identifier (categorical) |
| `stop_sequence` | Target stop index in the trip |
| `prediction_stop_sequence` | Current stop (context) |
| `hour`, `day_of_week`, `month` | Temporal context |
| `is_weekend`, `is_holiday` | Calendar flags (Ukrainian holidays) |
| `current_delay_sec` | Observed delay at the prediction point |
| `segment_distance_m` | Shape-projected distance to target stop |
| `scheduled_segment_sec` | Schedule's own prediction (strong prior) |
| `stops_remaining` | Stops left after the target |
| `trip_progress_frac` | Position along route [0, 1] |

**Baseline**: scheduled inter-stop time (`scheduled_segment_sec`). The model is evaluated against this baseline and must beat it on the held-out test set (last 20% of days by date).

## Worker inference

The `worker/` directory is a Cloudflare Python Worker that runs inside Pyodide. It uses a compact, pure-Python re-implementation of GBT tree traversal (`src/inference.py`) — no sklearn or pandas at runtime.

Per request:
1. Fetch the current vehicle positions GTFS-RT feed from R2.
2. For each vehicle: project lat/lon to UTM, match to the best trip shape (off-route filter with hysteresis), find the current stop sequence.
3. Build feature rows for the next ≤10 stops.
4. Run the serialised GBT model.
5. Encode and return a GTFS-RT TripUpdates protobuf.

## R2 storage layout (medallion)

```
gtfs-lviv/
  raw/        YYYY-MM-DD/<feedTsISO8601Z>.pb   # Bronze — immutable raw protobuf
  positions/  YYYY-MM-DD.parquet               # Silver — consolidated rows
  static/     <feedVersion>/static.zip         # Versioned GTFS static
  static/     index.json                       # day → feedVersion mapping
  _meta/      collector_health.json            # collection health counters
  feed/       trip_updates.pb                  # served by the worker on_fetch
  gtfs-data.msgpack                            # compact GTFS for the worker
  model.msgpack                                # serialised GBT for the worker
```

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
