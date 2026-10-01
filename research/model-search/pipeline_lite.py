"""
Memory-bounded variant of scripts/run_pipeline.py for the research box.

Same download -> infer_trips -> labeling.training_rows_for_trajectory path as
production, but each trajectory's rows are sampled (whole snapshots, hashed on
vehicle_id + snapshot_ts, the same key harness.prep_day uses) as soon as they
are produced, so a worker never holds the ~3M-row day in Python dicts (a full
production worker peaked at 7.6 GB and OOM-killed the pool on a 15.7 GB box).

Also writes two compact full-day side tables for features_live.py:
  data/ms_lite/<day>.cross.parquet  unique stop arrivals (vehicle, trip, stop, t)
  data/ms_lite/<day>.pos.parquet    per-snapshot vehicle positions along the trip

Run 26: also writes data/ms_lite/<day>.age.parquet (vehicle_id, timestamp = feed
time, vehicle_ts = the vehicle's own GPS fix time, epoch s) for addon_v26.py; the
rows fed to infer_trips are unchanged.

Usage: python research/model-search/pipeline_lite.py --parallel 4 --days 2026-09-07..2026-09-21
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

OUT = REPO / "data" / "ms_lite"
KEEP_PCT = 10.0


def sample_mask(df: pd.DataFrame, keep_pct: float) -> np.ndarray:
    key = pd.util.hash_pandas_object(
        df[["vehicle_id", "snapshot_ts"]].astype(str), index=False).to_numpy()
    return (key % 10000) < keep_pct * 100


def _parse_with_vts(client, key: str) -> list[dict]:
    """Copy of src.snapshots._fetch_and_parse that also keeps the per-vehicle
    GPS timestamp as vehicle_ts (run 26); every other field is identical."""
    from datetime import datetime, timezone
    from google.transit import gtfs_realtime_pb2
    from src.snapshots import R2_BUCKET
    try:
        data = client.get_object(Bucket=R2_BUCKET, Key=key)["Body"].read()
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
            "vehicle_ts": int(v.timestamp) if v.HasField("timestamp") else None,
        })
    return rows


def process_day(day: str) -> dict:
    from src.gtfs_static import get_gtfs_for_date
    from src.labeling import training_rows_for_trajectory
    from src.snapshots import _make_client, list_snapshot_keys
    from src.trip_inference import infer_trips

    out = OUT / f"{day}.parquet"
    if out.exists():
        return {"day": day, "status": "skipped"}
    t0 = time.time()
    client = _make_client()
    gtfs = get_gtfs_for_date(day, client=client)
    keys = list_snapshot_keys(date_str=day)
    if not keys:
        return {"day": day, "status": "no_keys"}
    rows = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        for r in ex.map(lambda k: _parse_with_vts(client, k), keys):
            rows.extend(r)
    df = pd.DataFrame(rows)
    del rows
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.dropna(subset=["lat", "lon"])
    OUT.mkdir(parents=True, exist_ok=True)
    df[["vehicle_id", "timestamp", "vehicle_ts"]].to_parquet(OUT / f"{day}.age.parquet", index=False)
    df = df.drop(columns=["vehicle_ts"])
    df = infer_trips(df, gtfs)
    if "off_route" in df.columns:
        df = df[~df["off_route"]]

    samp, cross, pos = [], [], []
    for (vid, tid), traj in df.groupby(["vehicle_id", "inferred_trip_id"], sort=False):
        if not tid or pd.isna(tid):
            continue
        r = training_rows_for_trajectory(str(vid), str(tid), traj.sort_values("timestamp"), gtfs)
        if r is None or r.empty:
            continue
        cross.append(r[["vehicle_id", "trip_id", "stop_id", "stop_sequence",
                        "stop_dist_along_m", "actual_arrival"]].drop_duplicates())
        pos.append(r[["vehicle_id", "trip_id", "snapshot_ts", "dist_along_m"]]
                   .drop_duplicates(["snapshot_ts"]))
        samp.append(r[sample_mask(r, KEEP_PCT)])
    OUT.mkdir(parents=True, exist_ok=True)
    s = pd.concat(samp, ignore_index=True)
    s.sort_values(["date", "trip_id", "snapshot_ts", "stop_sequence"], inplace=True,
                  ignore_index=True)
    pd.concat(cross, ignore_index=True).to_parquet(OUT / f"{day}.cross.parquet", index=False)
    pd.concat(pos, ignore_index=True).to_parquet(OUT / f"{day}.pos.parquet", index=False)
    s.to_parquet(out, index=False)
    return {"day": day, "status": "ok", "n": len(s), "secs": round(time.time() - t0)}


def main():
    from harness import expand_days
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", required=True)
    ap.add_argument("--parallel", type=int, default=3)
    a = ap.parse_args()
    days = expand_days(a.days)
    with ProcessPoolExecutor(max_workers=a.parallel, max_tasks_per_child=1) as ex:
        futs = {ex.submit(process_day, d): d for d in days}
        for f in as_completed(futs):
            try:
                print(f.result(), flush=True)
            except Exception as e:  # keep other days going
                print({"day": futs[f], "status": f"error {e!r}"}, flush=True)


if __name__ == "__main__":
    main()
