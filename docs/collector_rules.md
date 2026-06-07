# GTFS Collector Rules

Operational contract for the Cloudflare Worker that polls Lviv GTFS-RT into R2,
and for the compaction job that builds the training-ready position history.

These rules exist to (a) keep raw data an immutable, replayable source of truth,
and (b) make model training fast by reading consolidated columnar history instead
of millions of tiny protobuf objects.

Endpoints (from `.env`):
- Realtime: `https://track.ua-gis.com/gtfs/lviv/vehicle_position`
- Static:   `https://track.ua-gis.com/gtfs/lviv/static.zip`
- Storage:  Cloudflare R2 bucket `gtfs-lviv`, S3-compatible.
- Feed timezone: `Europe/Kiev`. All stored timestamps are **UTC**.

---

## 1. Storage layout (medallion)

```
gtfs-lviv/
  raw/        YYYY-MM-DD/<feedTsISO8601Z>.pb     # Bronze — immutable protobuf
  positions/  YYYY-MM-DD.parquet                 # Silver — consolidated rows (training input)
  static/     <feedVersion>/static.zip           # Versioned static GTFS
  static/     index.json                         # day -> feedVersion mapping
  _meta/      collector_health.json              # last run, counts, last feed_ts
```

`data/labeled/YYYY-MM-DD.parquet` (Gold) is produced locally by the pipeline and
is out of scope for the collector.

---

## 2. Bronze rules — raw protobuf (the collector Worker)

1. **Poll every 60 s.** Cron `* * * * *`. Do not poll faster than the upstream
   feed refreshes; do not skip minutes silently (a gap is a data loss event, see §6).
2. **Store bytes verbatim.** The Worker performs *no* parsing, filtering, or
   transformation beyond what §2.4–2.6 require. Bronze must be byte-replayable.
3. **Key by feed timestamp, not wall-clock.** Parse only `feed.header.timestamp`,
   format it as `YYYY-MM-DDTHH:MM:SSZ` (UTC), and use it for both the date
   partition and the object name: `raw/2026-06-05/2026-06-05T13:45:00Z.pb`.
   This makes keys deterministic and idempotent.
4. **Immutability — never overwrite.** Before `PutObject`, check existence
   (`If-None-Match: *` / conditional put). If the key exists, skip. Re-running the
   Worker must never mutate or duplicate a snapshot.
5. **Skip stale duplicates.** If `feed.header.timestamp` equals the last written
   feed_ts (tracked in `_meta/collector_health.json`), the upstream feed has not
   advanced — **do not write**. This is the first line of dedup and saves storage.
6. **Never write empty or corrupt objects.** Reject and log (do not store) when:
   - HTTP status is not 200, or body is empty;
   - the body fails `FeedMessage.ParseFromString`;
   - `header.timestamp` is missing, before `2020-01-01`, or more than 5 min in the
     future (clock-skew / garbage feed).
   A skipped write is recorded in health metadata, not silently dropped.
7. **Record fetch wall-clock separately.** Set object metadata
   `x-fetch-wallclock` to the Worker's `Date.now()` UTC. Staleness =
   `wallclock - feed_ts`; alert if median staleness > 120 s (feed is lagging).
8. **Optional gzip.** Protobuf is already binary; gzip yields modest gains on the
   repeated string IDs. If enabled, set `Content-Encoding: gzip` and keep the
   `.pb` name. Bronze size is small — clarity over compression here.

### Schema captured per vehicle entity
Match the parser in `src/snapshots.py` (`_COLUMNS`): `timestamp` (= feed_ts),
`vehicle_id`, `trip_id`, `route_id`, `lat`, `lon`, `bearing`, `speed`, `stop_id`,
`current_status`. The collector stores the full feed; column projection happens in
Silver.

---

## 3. Silver rules — consolidated positions (compaction job)

Runs once per day (after the day closes in `Europe/Kiev`) or on demand to backfill.

1. **One Parquet per UTC day**, `positions/YYYY-MM-DD.parquet`, built by parsing
   every `raw/YYYY-MM-DD/*.pb` for that day exactly once.
2. **Lossless dedup** on `(vehicle_id, feed_ts)` — drop exact repeats that arise
   when the feed header timestamp recurs across polls. Do **not** drop stationary
   pings: dwell time at stops is signal for stop-crossing labeling.
3. **Drop rows with null `lat`/`lon`** (no position fix), matching
   `run_pipeline.py`.
4. **Sort by `(vehicle_id, timestamp)`** so trajectory reads are sequential.
5. **Compression & encoding:** zstd, with dictionary encoding on `vehicle_id`,
   `route_id`, `trip_id`, `current_status`, `stop_id`. Target row group size
   ~128 MB.
6. **Stable dtypes:** `timestamp` = UTC `datetime64[ns]`; `lat`/`lon`/`bearing`/
   `speed` = `float32`; `current_status` = small int; IDs = string/dictionary.
7. **Idempotent & atomic:** write to a temp key, then move into place; rebuilding a
   day must produce an equivalent file. Skip days whose `positions/` already exists
   unless `--force`.
8. **Verify before publishing:** row count ≥ 0.9 × (n_snapshots × median vehicles);
   fewer means a parse regression or a raw gap — fail loudly, don't publish.

The training pipeline reads Silver. Re-deriving Gold (improved inference/labeling)
must **never** require re-reading Bronze.

---

## 4. Static GTFS versioning rules

1. **Snapshot static on change.** Fetch `static.zip` daily; read `feed_info.txt`
   (`feed_version` / `feed_start_date`). If the version differs from the latest
   stored, save `static/<feedVersion>/static.zip`.
2. **Maintain `static/index.json`** mapping each collection day → the
   `feedVersion` in effect that day.
3. **Label against the era's static.** Historical labeling/feature-building for day
   D must load the static version active on D, not the newest. A single mutable
   `data/gtfs_static/static.zip` is a correctness bug once the schedule changes.
4. Static snapshots are immutable once written.

---

## 5. Retention & cost

1. **Bronze: keep indefinitely** — it is the replay source. R2 storage is cheap;
   re-deriving Silver/Gold from Bronze is the recovery path.
2. **Silver/Gold are regenerable** — safe to delete and rebuild; treat as cache.
3. **Read amplification is the cost to optimize**, not storage. One Silver GET per
   day replaces ~1440 Bronze GETs per training pass — that is the whole point.
4. Optional: after Silver is verified for a day, Bronze for that day may be
   tar-archived (`raw_archive/YYYY-MM-DD.tar`) to cut object count, keeping bytes
   intact for replay.

---

## 6. Health & monitoring

1. **Per-day snapshot count.** Expect ~1440. `< 1000` for a closed day = gap →
   alert (mirrors `scripts/sanity_check.py`).
2. **Update `_meta/collector_health.json` every run:** last feed_ts, last
   wall-clock, today's write count, today's skip/error counts.
3. **Bearing presence check** on a sample (bearing all-zero indicates a degraded
   feed; bearing is a trip-inference feature).
4. **Feed staleness alert** when `wallclock - feed_ts` median > 120 s.
5. Surface gaps to the pipeline: a day with a known Bronze gap is flagged so the
   model isn't trained on silently-incomplete trajectories.

---

## 7. Invariants (the short list)

- Bronze is **immutable, verbatim, keyed by feed_ts, never empty/corrupt**.
- Time is **UTC from `feed.header.timestamp`**; wall-clock is metadata only.
- Static GTFS is **versioned**; days are pinned to their era.
- Silver is **derived, deduped, columnar, regenerable**; training reads Silver.
- Re-deriving Gold **never re-reads Bronze**.
