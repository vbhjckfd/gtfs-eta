"""Download and parse GTFS-RT protobuf snapshots stored in Cloudflare R2."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Iterator

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
        }
        rows.append(row)

    return rows


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
        for fut in futures:
            all_rows.extend(fut.result())

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
