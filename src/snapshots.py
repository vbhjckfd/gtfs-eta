"""Download and parse GTFS-RT protobuf snapshots stored in Cloudflare R2."""

from __future__ import annotations

import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Iterable, Iterator

import boto3
import pandas as pd
from dotenv import load_dotenv
from google.transit import gtfs_realtime_pb2

load_dotenv()

R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET = os.environ.get("R2_BUCKET", "gtfs-lviv")
R2_PREFIX = "raw/"

# R2 S3-compatible endpoint
_R2_ENDPOINT = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

_COLUMNS = [
    "timestamp",
    "vehicle_id",
    "trip_id",
    "route_id",
    "lat",
    "lon",
    "bearing",
    "speed",
    "stop_id",
    "current_status",
]


def _make_client() -> "boto3.client":
    return boto3.client(
        "s3",
        endpoint_url=_R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
    )


def list_snapshot_keys(
    start: datetime | None = None,
    end: datetime | None = None,
    date_str: str | None = None,
) -> list[str]:
    """
    List R2 object keys under raw/.

    Pass date_str (e.g. "2024-11-15") to list a single day, or pass
    start/end datetimes to filter across multiple days.
    """
    client = _make_client()
    prefix = R2_PREFIX if date_str is None else f"{R2_PREFIX}{date_str}/"

    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=R2_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if start is not None or end is not None:
                ts = _key_to_timestamp(key)
                if ts is None:
                    continue
                if start is not None and ts < start:
                    continue
                if end is not None and ts > end:
                    continue
            keys.append(key)

    return sorted(keys)


def list_snapshot_days() -> list[str]:
    """Days (YYYY-MM-DD) that have a raw/ snapshot prefix, sorted."""
    client = _make_client()
    days: set[str] = set()
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=R2_BUCKET, Prefix=R2_PREFIX, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            day = cp["Prefix"][len(R2_PREFIX):].rstrip("/")
            if day:
                days.add(day)
    return sorted(days)


def _key_to_timestamp(key: str) -> datetime | None:
    """Extract UTC datetime from key like raw/2024-11-15/2024-11-15T13:45:00Z.pb"""
    try:
        fname = key.rsplit("/", 1)[-1].replace(".pb", "")
        return datetime.fromisoformat(fname.replace("Z", "+00:00"))
    except ValueError:
        return None


def _fetch_and_parse(client, key: str) -> list[dict]:
    """Download one protobuf key and return a list of vehicle rows. Returns [] on parse errors."""
    try:
        obj = client.get_object(Bucket=R2_BUCKET, Key=key)
        data = obj["Body"].read()
    except Exception:
        return []

    if not data:
        return []

    feed = gtfs_realtime_pb2.FeedMessage()
    try:
        feed.ParseFromString(data)
    except Exception:
        return []

    feed_ts = datetime.fromtimestamp(feed.header.timestamp, tz=timezone.utc)
    rows = []
    for entity in feed.entity:
        if not entity.HasField("vehicle"):
            continue
        v = entity.vehicle
        pos = v.position if v.HasField("position") else None
        trip = v.trip if v.HasField("trip") else None

        row = {
            "timestamp": feed_ts,
            "vehicle_id": v.vehicle.id if v.HasField("vehicle") else entity.id,
            "trip_id": trip.trip_id if trip else None,
            "route_id": trip.route_id if trip else None,
            "lat": pos.latitude if pos else None,
            "lon": pos.longitude if pos else None,
            "bearing": pos.bearing if pos else None,
            "speed": pos.speed if pos else None,
            "stop_id": str(v.stop_id) if v.stop_id else None,
            "current_status": v.current_status,
            # The vehicle's own fix time and odometer (km) — the live ETA
            # features (src/live_features.py) use both; not in _COLUMNS, so
            # load_snapshots_df leaves its frame unchanged.
            "vehicle_ts": int(v.timestamp) if v.HasField("timestamp") else None,
            "odometer": pos.odometer if pos is not None and pos.HasField("odometer") else None,
        }
        rows.append(row)

    return rows


def drop_repeated_snapshots(snapshots: Iterable[list[dict]]) -> Iterator[list[dict]]:
    """Skip parsed snapshots whose feed header timestamp was already seen.

    The collector keys raw/ objects by fetch wall-clock, not by feed timestamp,
    so one upstream feed can land under several keys: the first poll of each
    cron fire repeats the previous one when the feed has not advanced, and
    2026-10-12 09:30 .. 10-14 19:30 UTC Cloudflare fired the collector cron
    twice a minute, doubling every snapshot. A repeat adds no positions; kept,
    it turns into zero-dt "speed unknown" training rows and double weight.
    """
    seen: set = set()
    for rows in snapshots:
        if rows:
            ts = rows[0]["timestamp"]
            if ts in seen:
                continue
            seen.add(ts)
        yield rows


# The collector stores ~360 snapshots per hour (one 10 s poller); normal hours
# measure 355-365. Hours far above that come from overlapping pollers —
# 2026-10-12..14 ran ~716/h — and are thinned back to the single-poller cadence.
DENSE_KEYS_PER_HOUR = 400
THIN_MIN_GAP_SEC = 7.0


def _key_hour(key: str) -> str:
    """'raw/2026-10-13/2026-10-13T10:00:03.070Z.pb' -> '2026-10-13T10'."""
    return key.rsplit("/", 1)[-1][:13]


def thin_dense_hours(
    keys: list[str],
    snapshots: Iterable[list[dict]],
    max_per_hour: int = DENSE_KEYS_PER_HOUR,
    min_gap_sec: float = THIN_MIN_GAP_SEC,
) -> Iterator[list[dict]]:
    """Thin over-sampled hours back to the collector's normal ~10 s cadence.

    ``snapshots`` are the parsed row lists of ``keys``, in the same order. In an
    hour with more than ``max_per_hour`` keys, a snapshot is kept only when its
    feed timestamp is at least ``min_gap_sec`` after the last kept one. The
    upstream feed advances every ~5-6 s, so two phase-shifted pollers catch
    ~1.5x the distinct feeds of one: without thinning such a day carries 1.5x
    the training weight, and progress_speed_mps is measured over ~5 s instead
    of the ~10 s the push daemon sees. Normal hours pass through untouched, so
    their training rows do not change.
    """
    per_hour = Counter(_key_hour(k) for k in keys)
    dense = {h for h, n in per_hour.items() if n > max_per_hour}
    last_kept = None
    for key, rows in zip(keys, snapshots):
        if rows:
            ts = rows[0]["timestamp"]
            if (_key_hour(key) in dense and last_kept is not None
                    and (ts - last_kept).total_seconds() < min_gap_sec):
                continue
            last_kept = ts
        yield rows


def iter_snapshots(keys: list[str], max_workers: int = 8) -> Iterator[list[dict]]:
    """Yield parsed row lists for each key, in order, fetched concurrently."""
    client = _make_client()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch_and_parse, client, k): k for k in keys}
        # preserve order
        key_to_fut = {k: f for f, k in futures.items()}
        for key in keys:
            fut = key_to_fut[key]
            yield fut.result()


def load_snapshots_df(
    start: datetime | None = None,
    end: datetime | None = None,
    date_str: str | None = None,
    max_workers: int = 8,
) -> pd.DataFrame:
    """
    Download all snapshots in the given time window and return a single DataFrame.

    Columns: timestamp, vehicle_id, trip_id, route_id, lat, lon, bearing,
             speed, stop_id, current_status
    """
    keys = list_snapshot_keys(start=start, end=end, date_str=date_str)
    if not keys:
        return pd.DataFrame(columns=_COLUMNS)

    all_rows: list[dict] = []
    client = _make_client()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_fetch_and_parse, client, k) for k in keys]
        for rows in drop_repeated_snapshots(fut.result() for fut in futures):
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows, columns=_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.sort_values(["timestamp", "vehicle_id"], inplace=True, ignore_index=True)
    return df


def load_local_protobuf(path: str) -> pd.DataFrame:
    """Parse a single local .pb file — useful for development without R2 access."""
    with open(path, "rb") as f:
        data = f.read()

    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(data)

    feed_ts = datetime.fromtimestamp(feed.header.timestamp, tz=timezone.utc)
    rows = []
    for entity in feed.entity:
        if not entity.HasField("vehicle"):
            continue
        v = entity.vehicle
        pos = v.position if v.HasField("position") else None
        trip = v.trip if v.HasField("trip") else None

        rows.append({
            "timestamp": feed_ts,
            "vehicle_id": v.vehicle.id if v.HasField("vehicle") else entity.id,
            "trip_id": trip.trip_id if trip else None,
            "route_id": trip.route_id if trip else None,
            "lat": pos.latitude if pos else None,
            "lon": pos.longitude if pos else None,
            "bearing": pos.bearing if pos else None,
            "speed": pos.speed if pos else None,
            "stop_id": str(v.stop_id) if v.stop_id else None,
            "current_status": v.current_status,
        })

    df = pd.DataFrame(rows, columns=_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df
