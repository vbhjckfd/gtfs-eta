"""
Spot-check the R2 snapshot collection:
  - Count files per day (flag days with < 1000 snapshots)
  - Sample-parse one file from the oldest, middle, and newest day
  - Confirm entity count and bearing presence
"""
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone

import boto3
from dotenv import load_dotenv
from google.transit import gtfs_realtime_pb2

load_dotenv()

R2_ACCOUNT_ID = os.environ["R2_ACCOUNT_ID"]
R2_ACCESS_KEY_ID = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET_ACCESS_KEY = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET = os.environ.get("R2_BUCKET", "gtfs-lviv")
R2_ENDPOINT = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

client = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    region_name="auto",
)


def list_all_keys(prefix="raw/") -> list[str]:
    keys = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=R2_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def parse_pb(key: str) -> dict:
    obj = client.get_object(Bucket=R2_BUCKET, Key=key)
    data = obj["Body"].read()
    size_kb = len(data) / 1024

    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(data)

    n_entities = len(feed.entity)
    n_with_pos = sum(1 for e in feed.entity if e.HasField("vehicle") and e.vehicle.HasField("position"))
    n_with_bearing = sum(
        1 for e in feed.entity
        if e.HasField("vehicle") and e.vehicle.HasField("position") and e.vehicle.position.bearing != 0
    )
    feed_ts = datetime.fromtimestamp(feed.header.timestamp, tz=timezone.utc)

    return {
        "key": key,
        "size_kb": size_kb,
        "feed_ts": feed_ts,
        "n_entities": n_entities,
        "n_with_pos": n_with_pos,
        "n_with_bearing": n_with_bearing,
    }


def main():
    print("Listing all keys in R2 (this may take a minute)…")
    keys = list_all_keys()
    print(f"Total keys found: {len(keys)}\n")

    # Group by day prefix (raw/YYYY-MM-DD/)
    by_day: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        parts = k.split("/")
        if len(parts) >= 3:
            by_day[parts[1]].append(k)

    days = sorted(by_day.keys())
    print(f"{'Day':<14} {'Files':>6}  {'Status'}")
    print("-" * 35)
    for day in days:
        n = len(by_day[day])
        flag = "⚠ LOW" if n < 1000 else "✓"
        print(f"{day:<14} {n:>6}  {flag}")

    print()
    if not days:
        print("No days found — check bucket/prefix.")
        sys.exit(1)

    # Pick oldest, middle, newest day
    sample_days = [days[0], days[len(days) // 2], days[-1]]
    print("Sampling one file from each of: " + ", ".join(sample_days))
    print()

    for day in sample_days:
        day_keys = by_day[day]
        key = random.choice(day_keys)
        try:
            r = parse_pb(key)
            bearing_ok = "✓" if r["n_with_bearing"] > 0 else "✗ NONE"
            print(
                f"Day {day}: {len(day_keys)} files, "
                f"sample parses OK, "
                f"{r['n_entities']} entities "
                f"({r['n_with_pos']} with pos, bearing {bearing_ok}), "
                f"{r['size_kb']:.1f} KB, "
                f"feed_ts={r['feed_ts'].strftime('%H:%M:%S UTC')}"
            )
        except Exception as e:
            print(f"Day {day}: ✗ PARSE ERROR — {e}")


if __name__ == "__main__":
    main()
