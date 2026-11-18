# gtfs-eta — open realtime feeds for Lviv public transport

🇬🇧 **English** · 🇺🇦 [Українська](README.uk.md)

Free, open **GTFS Realtime** feeds for every tram, trolleybus and bus in Lviv, Ukraine. Arrival times are **predicted by a machine-learning model** trained on months of real vehicle movements. They are not copied from the timetable. You need no API key and no sign-up.

| | |
|---|---|
| **TripUpdates** (predicted arrivals) | `https://eta.lad.lviv.ua/feed/trip_updates.pb` |
| **VehiclePositions** (cleaned, trip-matched) | `https://eta.lad.lviv.ua/feed/vehicle_positions.pb` |
| **GTFS static** (operator's schedule) | `https://track.ua-gis.com/gtfs/lviv/static.zip` |
| Docs page for consumers (EN/UA) | [eta.lad.lviv.ua](https://eta.lad.lviv.ua/) |
| Health check | [gtfs-eta.vbhjckfd.workers.dev/health](https://gtfs-eta.vbhjckfd.workers.dev/health) |
| Rider-facing app built on it | [lad.lviv.ua](https://lad.lviv.ua) |

**Accuracy** (held out, all horizons): mean absolute error **92.8 s**, median **45 s**. At 1 / 5 / 10 stops ahead the MAE is 61 / 94 / 132 s. That is 22% better than the single-snapshot model it replaced. [Details](#evaluation).

**Contents**

- [For feed consumers](#for-feed-consumers): [quick start](#quick-start) · [GTFS static](#gtfs-static) · [TripUpdates](#tripupdates) · [VehiclePositions](#vehiclepositions) · [freshness and status](#freshness-and-status) · [terms](#terms)
- [Methodology](#methodology): [data](#data) · [labelling](#labelling) · [model](#model) · [evaluation](#evaluation) · [serving](#inference-and-serving) · [uncertainty](#uncertainty) · [quality monitoring](#live-quality-monitoring)
- [Operations](#operations): [architecture](#architecture) · [schedules](#schedules) · [safeguards](#safeguards) · [worker](#the-worker) · [R2 layout](#r2-storage-layout-medallion) · [telemetry](#telemetry)
- [Development](#development): [setup](#setup) · [commands](#common-commands) · [env vars](#environment-variables) · [smoke tests](#smoke-tests)

---

## For feed consumers

### Quick start

```bash
pip install gtfs-realtime-bindings requests
```

```python
import requests
from google.transit import gtfs_realtime_pb2

feed = gtfs_realtime_pb2.FeedMessage()
feed.ParseFromString(requests.get(
    "https://eta.lad.lviv.ua/feed/trip_updates.pb", timeout=10).content)

for e in feed.entity[:3]:
    tu = e.trip_update
    for stu in tu.stop_time_update:
        if stu.HasField("arrival"):
            print(tu.trip.trip_id, stu.stop_id, stu.arrival.time, "±", stu.arrival.uncertainty, "s")
```

Poll every **5–10 s**. The feed is regenerated about every 10 s. Cloudflare's edge serves it with `Cache-Control: public, max-age=3, stale-while-revalidate=2, stale-if-error=60`, so polling faster than that returns the same bytes.

Plug the two URLs into anything that speaks GTFS-RT, such as OpenTripPlanner, MOTIS or Transitous, together with the static zip.

### GTFS static

The schedule is the operator's own feed (publisher **Львівавтодор**). We do not modify it. The realtime feeds use its identifiers unchanged, so joins work directly. We check it every 5 minutes and switch to a new version within about 30 minutes of its publication.

| File | Notes |
|---|---|
| `agency.txt` | Timezone is Kyiv, written as `Europe/Kiev`. |
| `routes.txt` | About 70 routes. `route_type` **0 = tram**, **3 = bus *and* trolleybus**. Use the short name to tell them apart: `Т…` is a tram, `Тр…` a trolleybus, `А…` a bus. |
| `stops.txt` | About 1,060 stops. `stop_id` is the internal key that the realtime feeds use. `stop_code` is the number printed on the stop sign, the one riders know. |
| `trips.txt` | About 15,700 trips, with `direction_id`, `shape_id` and `block_id`. `trip_id` looks like `18550_0_1`. |
| `stop_times.txt` | Planned times for every stop of every trip (about 430k rows). |
| `shapes.txt` | Route geometry. A route averages 216 trips over only 2.5 distinct shapes. |
| `calendar.txt`, `calendar_dates.txt` | Service days. |

Known quirks:

- Trolleybuses share `route_type` 3 with buses. If you need the mode, use the short-name prefix.
- **The bus timetable is unreliable.** Measured live, bus delay spans roughly ±30 min between the 10th and 90th percentile. That is why the model uses no schedule feature, and why the feed publishes `delay` only for trams and trolleybuses (see below).
- The operator republishes the static zip without notice, sometimes during the day, and trip IDs can change. A trip_id in the realtime feed always refers to the static version that was current when it was published.

### TripUpdates

One `TripUpdate` per vehicle that is matched to a trip and on its route. It carries predictions for the **next 10 stops at most**.

| Field | Meaning |
|---|---|
| `trip.trip_id`, `route_id` | From GTFS static, unchanged. |
| `trip.start_date` / `start_time` | Binds the update to one *run* of the trip. `trip_id` alone cannot: a loop route repeats through the day, and an after-midnight trip belongs to the previous service day. |
| `trip.direction_id` | From `trips.txt`. |
| `vehicle.id` | The operator's vehicle ID, the same one VehiclePositions carries. |
| `timestamp` | Time of the **vehicle fix** the prediction was computed from. The header timestamp is the *publish* time and is always fresh, so this is the only thing that separates a prediction made from a 2 s-old position from one made from a 170 s-old one. Positions older than 180 s are dropped. |
| `stop_time_update[].stop_id`, `stop_sequence` | GTFS `stop_id`, not the sign number. |
| `arrival.time` | Predicted arrival, POSIX seconds. |
| `arrival.uncertainty` | Typical error for this horizon, in seconds, calibrated on live serving error. Widen your window for far stops. |
| `departure.time` | `arrival + dwell`. Dwell is **measured**: 32 s for trams and 27 s for buses and trolleybuses, from 1.76 M stationary events. |
| `delay` | Deviation from the timetable, **tram and trolleybus only**. It is omitted for buses on purpose: consumers use `delay` to extrapolate unlisted stops, and the bus timetable is too unreliable for that. |
| Trailing `NO_DATA` stop | Fences off the stops past the 10-stop horizon. It names a stop but deliberately carries no times. Without it, the spec tells consumers to apply the 10th stop's deviation to the rest of the route. |

Every optional field is **omitted rather than guessed** when its source data is missing. A prediction whose arrival has already passed is published as *arriving now* rather than dropped. Dropping it would hide the one arrival a waiting rider most wants, and would renumber every later horizon.

### VehiclePositions

This feed re-emits the upstream positions, enriched by the same inference pass:

- A **corrected trip match**, which is often better than the operator's reported `trip_id`. Each trip instance goes to exactly one vehicle, with `start_date` and `start_time`. It is the same match TripUpdates names, so the two feeds agree.
- The next stop, and `current_status` (`STOPPED_AT` / `IN_TRANSIT_TO`).
- `congestion_level`, derived from the observed speed against the historical speed for that route and hour.

### Freshness and status

- `GET https://gtfs-eta.vbhjckfd.workers.dev/health` returns JSON with HTTP **200** when the feed is fresh. It returns **503** if the feed is older than 180 s, or if no arrivals are predicted during service hours at a central stop (sign 60, `stop_id` 4577).

  ```json
  {"status":"ok","age_sec":4,"entities":544,"stop_code":60,"arrivals":34,
   "working_hours":true,"vehicles_in":610,"vehicles_stale":20,"feed_skew_sec":3,
   "worker_commit":"a083227","feed_commit":"a083227"}
  ```

- `https://eta.lad.lviv.ua/feed/status.json` is a tiny sidecar rewritten at most once a minute, and at once when a push is blocked. It is a cheap freshness check that needs no protobuf parsing: `feed_timestamp`, `entities`, `vehicles_in`, `vehicles_stale`, `feed_skew_sec`, `published`, `commit`.

### Terms

- **Feed data** (TripUpdates, VehiclePositions, `status.json`) is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Use it in apps, journey planners and research, commercial use included. Credit **lad.lviv.ua**.
- **Code** in this repository is under the [MIT License](LICENSE).
- **GTFS static** belongs to its publisher, Львівавтодор, and is not covered by either licence.

Please keep polling to a reasonable rate. Predictions are estimates and come with no warranty; the service is best-effort with no SLA.

Older URLs keep working: `https://gtfs-eta.vbhjckfd.workers.dev/` and `/vehicle_positions` return a 301 to the R2 URLs above.

---

## Methodology

This section is for data analysts and anyone checking whether the numbers can be trusted.

### Data

- **Input**: the operator's GTFS-RT VehiclePositions (`track.ua-gis.com`), archived since June 2026 with a median gap of about **11 s** between snapshots. The archive is kept as immutable raw protobuf (Bronze) in Cloudflare R2.
- **Schedule**: GTFS static, versioned per feed version. Every day is mapped to the static version that was in force that day (`static/index.json`), so old days are labelled against the right shapes.
- **Calendar**: weekend and Ukrainian public-holiday flags.

### Labelling

`run_pipeline.py` turns each day of snapshots into training rows (Gold parquet):

1. Match every vehicle to a trip (see [trip matching](#inference-and-serving)) and project it onto the trip's shape.
2. Find the **actual arrival time** at every stop by interpolating the projected distance along the shape between consecutive snapshots.
3. Build rows anchored on snapshots. Every snapshot of a vehicle produces one row per upcoming stop, up to 10 ahead, with target `seconds_to_arrival`. Because rows come from every snapshot and not only from stop crossings, the model sees vehicles mid-segment and dwelling at stops. This is what lets it predict ≈0 s when a bus is already at the stop.
4. Write side tables per day (`data/training/side/<day>.feed|pos.parquet`): every vehicle's raw feed fields and every on-route vehicle's projected distance. The trainer replays these to rebuild the live state exactly as serving saw it.

10% of snapshots are sampled for training rows. The side tables are complete.

### Model

**Target**: `seconds_to_arrival`, the time from the snapshot to the actual arrival at each upcoming stop. It is predicted directly for every horizon from 1 to 10 stops ahead.

**Algorithm** (`live_v2`, [src/train_live.py](src/train_live.py), live features in [src/live_features.py](src/live_features.py)): `sklearn.ensemble.HistGradientBoostingRegressor` with absolute-error loss. `route_id` and the target stop (top 254) are native categoricals, alongside 13 base features and 36 live columns. Trees: 255 leaves, ≤ 1200 iterations, `max_features=0.3`.

**Base features**, from one snapshot of one vehicle:

| Feature | Description |
|---|---|
| `stops_ahead` | Prediction horizon in stops (1 = next stop) |
| `hour` | Temporal context |
| `is_weekend`, `is_holiday` | Calendar flags (Ukrainian holidays) |
| `remaining_dist_m` | Shape distance from the vehicle's projected position to the target stop |
| `progress_speed_mps` | Observed speed over the last snapshot interval (−1 when unknown) |
| `stops_remaining` | Stops left after the target |
| `trip_progress_frac` | Position along route [0, 1] |
| `dist_per_stop_m` | `remaining_dist_m / stops_ahead` |
| `speed_eta_warm` | `remaining_dist_m / effective_speed`, warm-started from the prior when speed is unknown |
| `hist_speed_mps` | Route×hour historical median speed |
| `hist_travel_time_est` | `stops_ahead ×` historical seconds-per-stop (dwell-aware) |
| `stationary_sec` | Seconds since the vehicle last advanced > 25 m |

`apply_priors()` computes `speed_eta_warm`, `hist_speed_mps` and `hist_travel_time_est` from a route×hour speed and dwell table built **on the training split only**, so they carry no test-set leakage. No timetable feature survived feature selection.

**Live features**, which describe what is happening on the network right now:

| Group | Columns | Source |
|---|---|---|
| Live link store | `live_path_sec_m3/_m5`, `live_path_cov/_age`, `live_speed_mps`, `cur_link_*` | The last traversals of every stop-to-stop link on the path by **any** vehicle (≤ 30 min), summed to the target stop |
| Historical links | `hist_path_sec`, `hist_path_sec_dt`, `hist_path_cov` | Median link time × local hour (and × weekday/weekend) over the prior 14 days |
| Own history | `lap_path_*`, `veh_hist_ratio60/_n60`, `own_speed_60/180/300` | This vehicle's previous traversal of the path links (≤ 3 h), its own-vs-historical link times over the last hour, and its recent speed |
| Dwell | `dwell_rem_med/_p75`, `dwell_p_more120`, `dwell_n`, `dwell_long_share` | Remaining stop time at its current location (stop + 50 m bin) given `stationary_sec`, from prior days' stops |
| Feed fields | `pos_age_*`, `fs_now`, `fs_mean60`, `fs_zero_frac600`, `odo_spd_60/180/600/1200`, `odo_m_300` | GPS-fix age, reported speed and odometer windows |

Missing live values are `-1`.

**One engine for training and serving.** `LiveState` is fed the same two streams in both: every vehicle's raw feed fields, and every on-route vehicle's projected distance along its trip. The trainer replays the side tables, and serving feeds the live stream. Information from other vehicles becomes visible only on the next snapshot, so the order in which vehicles are processed does not matter. This removes the usual train/serve skew of live features. The static tables (historical link medians, dwell) are built by the trainer from the 14 days up to its last day and shipped inside the model export.

The serving row is assembled **positionally**. `build_features` in [src/inference.py](src/inference.py) emits rows in `FEATURE_COLS` order ([src/features.py](src/features.py)). `build_features_live` picks the base columns out of them by index (`_LIVE_BASE_IDX`), and that index must match `MODEL_COLS` in [src/train_live.py](src/train_live.py).

### Evaluation

The held-out comparison uses a **time-based split**, so no future data leaks into training. Both models were scored on the same rows: train 2026-09-19..09-30, test 10-01 Thu / 10-02 Fri / 10-03 Sat, 1.58 M rows.

| | single-snapshot (retired) | live_v2 |
|---|---|---|
| MAE | 119.3 s | **92.8 s** (−22.2%) |
| median | 60.7 s | **45.0 s** |
| p90 | 256.1 s | **192.3 s** (−24.9%) |
| MAE at 1 / 5 / 10 stops ahead | 73.8 / 121.7 / 167.9 s | **60.8 / 93.6 / 131.7 s** |

To reproduce it:

```bash
python -m src.train_live --train 2026-09-19..2026-09-30 --test 2026-10-01..2026-10-03 --no-save
```

Live serving error runs about **2× the offline test error**. Live predictions use positions up to 180 s old, upstream feed gaps and trip matches that are less certain. That is why the uncertainty bands come from live scoring and not from the test split.

**Retraining** happens weekly (`retrain.yml`, Monday 03:00 UTC). It labels the last ~5 weeks, trains on the latest 21 days with the last 20% held out, and exports to R2. The push daemon picks up the new model on its next restart, with no deploy.

### Inference and serving

Inference runs in [scripts/push_feed.py](scripts/push_feed.py). It uses a compact, pure-Python re-implementation of GBT tree traversal ([src/inference.py](src/inference.py)), with no sklearn or pandas at runtime.

Each cycle (about every 10 s):

1. Fetch the upstream VehiclePositions feed. If it is more than 90 s stale, fall back to the last cached copy.
2. **Match every vehicle to a trip**, scoring shape distance, heading and how far off schedule it would be on that run. No run more than 1 h off is accepted. Trips are assigned for the whole snapshot at once, so each run goes to one vehicle, and a bunched second vehicle takes the run before or after.
3. For each vehicle, apply an off-route filter with hysteresis, project the exact position along the shape, and measure progress speed against the previous cycle.
4. Build feature rows for the next ≤ 10 stops and update `LiveState`.
5. Run the serialised GBT model.
6. Encode TripUpdates and VehiclePositions protobufs, then upload both to R2 in parallel, with `status.json` at most once a minute.

Trip matching memoises the polyline walk per shape. It caches the schedule-progress lookup per push, since every vehicle in a push shares one `now_sec`. It also skips any shape whose bounding box already scores worse than the best full candidate. Together these cut a full inference pass from **4.0 s to 0.76 s** on the live fleet, with byte-identical output. The caches live on the loaded data dict, so they expire with the export they describe.

State survives restarts: `feed/live_state.pkl.gz` (max age 3 h) and `feed/tracker_state.json` (max age 6 h).

### Uncertainty

Each `StopTimeEvent.uncertainty` is the **live** MAE for its horizon. `make export` pools the per-stops-ahead MAE from the last 7 days of the quality archive (`quality/*.json`) and bakes it into the model blob. The training-test MAE (`models/uncertainty.joblib`) is only a cold-start fallback for before any day has been scored.

### Live quality monitoring

- Every 5 minutes the worker archives the served feed into `predictions/` (kept 14 days).
- `score-quality.yml` (daily, 02:15 UTC) scores yesterday's archived predictions against the actual arrivals derived from `raw/`. It writes `quality/<day>.json`, `latest.json` and `latest.md`: MAE, bias and coverage per horizon, mode and route.
- `route-mae.yml` (22:00 UTC) opens an issue with the per-route MAE table and its day-over-day change, so a route that degrades shows up within a day.

---

## Operations

### Architecture

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
  models/eta_live.joblib           live_v2 HistGradientBoostingRegressor
        │  make export
        ▼
  R2: worker/*.pkl                 compact GTFS + model for inference
        │  push_feed.py (GitHub Actions, every ~10 s)
        ▼
  R2: feed/trip_updates.pb         pre-computed GTFS-RT TripUpdates
      feed/vehicle_positions.pb    cleaned VehiclePositions
        │  R2 public custom domain, edge-cached
        ▼
  eta.lad.lviv.ua/feed/*  →  consumers (apps, journey planners, …)
```

There are no servers. The whole system runs on GitHub Actions, Cloudflare R2 and one Cloudflare Worker, all on free tiers.

### Schedules

| Workflow | When | What |
|---|---|---|
| `push-feed.yml` | every 5 min, dispatched by the worker cron | Runs `push_feed.py --loop 10` for about 5.5 min. It polls upstream every 2 s and pushes as soon as positions change, at least once every 20 s. |
| `refresh-gtfs.yml` | daily 02:00 UTC, plus on demand | Rebuilds `worker/gtfs_worker_data.pkl` from the operator's `static.zip`. |
| `score-quality.yml` | 02:15 UTC, dispatched by the worker cron | Scores yesterday's predictions (see above). |
| `retrain.yml` | Monday 03:00 UTC | Labels, trains and exports the model. |
| `route-mae.yml` | 22:00 UTC | Opens the per-route MAE digest issue. |
| `cleanup-runs.yml` | daily 03:17 UTC | Deletes completed workflow runs older than 2 days. |

**Cron reliability trick.** GitHub's scheduled triggers are unreliable on low-activity repos. A Cloudflare Worker cron fires every 5 minutes instead and dispatches `push-feed.yml`, so the runs overlap seamlessly.

**Static refresh.** The same 5-minute cron HEADs the operator's `static.zip` and dispatches `refresh-gtfs.yml` when it is newer than our `worker/gtfs_worker_data.pkl`, with a 30 min cooldown (`REFRESH_COOLDOWN_SEC`). The daily cron alone used to leave the feed predicting against a stale static for hours whenever the operator republished mid-day, and every renamed trip went out with a trip_id the new static no longer had.

### Safeguards

- **Feed watchdog.** Between crons, a Durable Object alarm (`FeedWatchdog` in [worker/worker.js](worker/worker.js)) HEADs the feed blob every 30 s. If no push has landed for 90 s because a run died mid-cycle, it dispatches `push-feed.yml` immediately instead of waiting up to 5 minutes. A 180 s cooldown follows each dispatch. The cron re-arms the alarm on every fire, so the chain restarts by itself after a deploy. Tune or disable it with the `WATCHDOG_*` vars in `wrangler.toml` (`WATCHDOG_DISABLED = "true"` stops it). Everything it uses is on the Workers Free plan.
- **Collapse guard.** If a pass produces less than 30% of the previous feed's entities, and the previous feed had at least 20, the old feed is kept and a `FeedCollapse` warning is reported. This stops a bad upstream snapshot from blanking the feed for every consumer.
- **Stale upstream.** VehiclePositions more than 90 s old are replaced by the cached copy. Individual vehicle fixes older than 180 s are dropped.
- **Model rollback.** Every export that publishes new trees first copies the old ones to `worker/eta_pipeline.pkl.prev`. Copying that object back over `worker/eta_pipeline.pkl` restores the previous model within one daemon restart (about 5 min).

### The worker

[worker/](worker/) is a plain JS Cloudflare Worker. It does **not** proxy the feeds, which R2 serves directly from its edge-cached custom domain, so a poll costs no worker invocation or R2 read. It only handles:

- `GET /` from a browser (`Accept: text/html`): the bilingual EN/UA docs page, [worker/index.html](worker/index.html). It is also routed at `https://eta.lad.lviv.ua/`, root only; `/feed/*` stays on R2.
- `GET /` from any other client, and `GET /vehicle_positions`: a **301 redirect** to the R2 URLs, for backward compatibility with consumers that have not migrated. The root responses are `no-store` / `private` because the edge cache keys by path and ignores `Vary`.
- `GET /health`: parses the feed with a hand-rolled protobuf wire walk (no dependencies) and returns 200 or 503.
- The cron: dispatches workflows, archives the served feed and runs the static freshness check.

It was originally a Python Worker. Pyodide isolates intermittently entered a poisoned state where every request failed in about 2 ms before handler code ran (`scriptThrewException` storms), so it was rewritten in JS.

### R2 storage layout (medallion)

```
gtfs-lviv/
  raw/        YYYY-MM-DD/<feedTsISO8601Z>.pb   # Bronze — immutable raw protobuf
  positions/  YYYY-MM-DD.parquet               # Silver — consolidated rows
  static/     <feedVersion>/static.zip         # Versioned GTFS static
  static/     index.json                       # day → feedVersion mapping
  _meta/      collector_health.json            # collection health counters
  feed/       trip_updates.pb                  # served TripUpdates (eta.lad.lviv.ua)
  feed/       vehicle_positions.pb             # served VehiclePositions (eta.lad.lviv.ua)
  feed/       status.json                      # freshness sidecar
  feed/       live_state.pkl.gz, tracker_state.json   # daemon state across restarts
  predictions/ YYYY-MM-DD/<feedTsISO>.pb       # sampled archive of the served feed (14-day lifecycle)
  quality/    YYYY-MM-DD.json + latest.json/.md # scored live-prediction quality
  worker/     gtfs_worker_data.pkl             # compact GTFS for inference
  worker/     eta_pipeline.pkl (+ .prev)       # serialised GBT for inference
```

See [docs/collector_rules.md](docs/collector_rules.md) for the full data contract.

### Telemetry

#### New Relic

The New Relic Node APM agent cannot run in a Worker: it is CommonJS, spawns background harvest timers and reads Node internals that a V8 isolate does not have. So [worker/newrelic.js](worker/newrelic.js) posts custom events straight to the Event API over `fetch`. Without the key, reporting is a no-op. Errors go there too; there is no Sentry anywhere in the project.

The worker sends three event types, all tagged `service = gtfs-eta-worker` and stamped with `GIT_COMMIT`:

- `GtfsEtaWorkerHealth`: one per uncached `/health` check, mirroring the verdict (`status`, `ageSec`, `entities`, `arrivals`, `workingHours`, `vehiclesIn`, `vehiclesStale`, `feedSkewSec`, `feedCommit`). The response is cached for 5 s by Workers Caching, and a cache hit never runs the worker. Every field is already public in the `/health` body. The legacy redirect paths are deliberately not instrumented: they are edge-cacheable and carry no signal.
- `GtfsEtaWorkerCron`: one per cron fire. It records which `cron` matched, the `workflow` dispatched, whether the dispatch and the feed archive succeeded, what the static freshness check did (`gtfsRefresh`), and `durationMs`. This is the only record that the 5-minute pipeline actually fired.
- `GtfsEtaWorkerError`: a caught exception or a failed workflow dispatch, with `level`, `where` (e.g. `scheduled.dispatch:push-feed.yml`, `watchdog.alarm`, `fetch`), `errorClass`, `message` and `stack`. A CPU-limit kill never reaches it, because the isolate dies before any handler runs; use `wrangler tail` for those.

The push daemon reports the same way ([src/nr_events.py](src/nr_events.py), gated on the `NEW_RELIC_LICENSE_KEY` GitHub secret). Its events are tagged `service = gtfs-eta-daemon` and stamped with the feed commit:

- `GtfsEtaDaemonError`: `level = 'error'` for a failed push iteration (`errorClass`, `message`, the last 4000 chars of `traceback`), and `level = 'warning'` with `errorClass = 'FeedCollapse'` when the collapse guard holds back a feed.
- `GtfsEtaDaemonRun`: one per push-feed job, with `pushes`, `failedPushes` and `durationSec`. It is the only event a healthy run sends, so it doubles as the daemon's heartbeat.

```sql
SELECT max(ageSec) FROM GtfsEtaWorkerHealth TIMESERIES SINCE 1 day ago
SELECT count(*) FROM GtfsEtaWorkerCron WHERE dispatched IS false SINCE 1 day ago
SELECT count(*) FROM GtfsEtaDaemonError FACET level, errorClass SINCE 1 day ago
SELECT sum(failedPushes) FROM GtfsEtaDaemonRun TIMESERIES SINCE 1 day ago
```

The account is in New Relic's **EU** region. Ingest goes to `insights-collector.eu01.nr-data.net`, and the key is the 40-character licence key starting `eu01xx`, not an `NRAK-...` user API key.

#### Analytics Engine

The same signals also go to Workers Analytics Engine through [worker/analytics.js](worker/analytics.js). It is free and needs no secret or outbound fetch. The data lands in dataset `gtfs_eta_worker`. Columns are positional: `index1` is the point kind and `blob1` is the worker commit.

| `index1` | blobs | doubles |
|---|---|---|
| `health` | `blob2` status, `blob3` feedCommit | `double1` httpStatus, `double2` ageSec, `double3` entities, `double4` arrivals, `double5` workingHours, `double6` vehiclesIn, `double7` vehiclesStale, `double8` feedSkewSec |
| `cron` | `blob2` cron, `blob3` workflow, `blob4` gtfsRefresh | `double1` dispatched, `double2` archived, `double3` durationMs |
| `watchdog` | `blob2` action (`ok`/`cooldown`/`dispatched`/`dispatch_failed`/`error`) | `double1` feed ageSec, `double2` dispatched |

Unknown numbers are written as `-1`. Query it over the SQL API with an API token that has *Account Analytics: Read*:

```sql
-- watchdog interventions per hour
SELECT toStartOfHour(timestamp) AS h, count() AS n
FROM gtfs_eta_worker WHERE index1 = 'watchdog' AND blob2 = 'dispatched'
  AND timestamp > NOW() - INTERVAL '1' DAY GROUP BY h ORDER BY h
-- worst feed age seen by the watchdog, per 5 min
SELECT toStartOfFiveMinutes(timestamp) AS t, max(double1) AS max_age
FROM gtfs_eta_worker WHERE index1 = 'watchdog'
  AND timestamp > NOW() - INTERVAL '1' DAY GROUP BY t ORDER BY t
```

```bash
curl "https://api.cloudflare.com/client/v4/accounts/$CF_ACCOUNT_ID/analytics_engine/sql" \
  -H "Authorization: Bearer $CF_API_TOKEN" --data "SELECT ..."
```

---

## Development

### Repository layout

```
src/                 Python library (features, labelling, training, inference)
scripts/             Pipeline and operational scripts (run_pipeline.py, push_feed.py, …)
worker/              Cloudflare Worker — docs page, /health, cron, watchdog, legacy redirects
tests/               Smoke tests against the live feed
data/
  gtfs_static/       Local copy of GTFS static
  training/          Gold parquets, one per day (+ side/ tables)
models/              Trained sklearn pipeline (joblib), dwell and uncertainty tables
docs/
  collector_rules.md Data contract for R2 storage (medallion architecture)
.github/workflows/   push-feed, refresh-gtfs, score-quality, retrain, route-mae, cleanup-runs
```

### Requirements

- Python ≥ 3.11
- Node ≥ 22 (for `wrangler deploy`)
- A Cloudflare account with R2 enabled
- `.env` populated from `.env.example`

### Setup

```bash
pip install -e ".[dev]"
cp .env.example .env   # fill in R2 credentials
```

### Common commands

```bash
make pipeline            # label all days from R2 → training parquets (incremental, PARALLEL=4)
make pipeline-date DATE=2026-06-01   # single date
make train               # python -m src.train_live: latest 21 days, last 20% held out
make learn               # pipeline + train in one step
make measure-dwell       # measure real per-route-type stop dwell → models/dwell.joblib
make export              # serialise GTFS + model, upload to R2
make deploy              # deploy the Cloudflare Worker
make release             # export + deploy in one step
make r2-lifecycle        # apply R2 lifecycle rules (predictions/ expire after 14 d)

make push-feed           # push one TripUpdates snapshot to R2 now
make serve-feed          # push every 10 s (local daemon)
make smoke               # smoke-test the live feed over the network
make sanity              # check R2 collection health (snapshot counts, staleness)
make check-gtfs          # verify GTFS static loading
make check-snapshots     # verify R2 Bronze snapshot reading
```

### Environment variables

| Variable | Description |
|---|---|
| `R2_ACCOUNT_ID` | Cloudflare account ID |
| `R2_ACCESS_KEY_ID` | R2 S3-compat access key |
| `R2_SECRET_ACCESS_KEY` | R2 S3-compat secret |
| `R2_BUCKET` | R2 bucket name (default: `gtfs-lviv`) |
| `GTFS_STATIC_URL` | URL of the static GTFS zip |
| `GTFS_RT_URL` | URL of the upstream VehiclePositions GTFS-RT feed |
| `PROJECTED_CRS` | UTM CRS for geometry (default: `EPSG:32635`) |

The worker also needs Cloudflare secrets:

```bash
wrangler secret put GITHUB_TOKEN            # PAT with workflow scope
wrangler secret put NEW_RELIC_LICENSE_KEY   # optional — telemetry + errors to New Relic
```

### Smoke tests

```bash
make smoke
# or
pytest tests/test_smoke.py -v
```

The tests hit the live feed over the network and verify:

- HTTP 200 and an `application/x-protobuf` content type
- The feed parses as a valid `FeedMessage` and its timestamp is fresh
- All entities are `TripUpdate` with well-formed stop sequences
- Arrival times increase monotonically and are in the future
- Every TripUpdate carries a prediction `timestamp`
- The `NO_DATA` horizon fence, where present, is last, names a stop and carries no times
- Trip IDs match Lviv's `DIGITS_DIGIT_DIGIT` scheme
- Stop IDs are numeric
- Vehicle coverage is ≥ 50% of the upstream VehiclePositions feed
- Trip overlap with the operator's own `trip_updates` feed is ≥ 10%

Override the target URL:

```bash
SMOKE_URL=https://your-preview-url.workers.dev make smoke
```
