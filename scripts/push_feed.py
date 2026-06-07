"""
GTFS-RT ETA push daemon.

Loads the compact GTFS data and model from R2 once at startup, then on each
iteration:
  1. Fetches live vehicle positions from the upstream GTFS-RT feed.
  2. Runs ETA inference (geometry + GBT tree traversal in native Python).
  3. Pushes the resulting TripUpdates protobuf to R2 as FEED_KEY.

The Cloudflare Worker (worker/worker.py) simply reads that pre-computed blob
from R2 and returns it — no inference CPU needed, so it fits in Cloudflare's
free-plan 10 ms CPU budget.

Usage:
    python scripts/push_feed.py              # push once and exit
    python scripts/push_feed.py --loop 30   # push every 30 seconds

Environment (read from .env):
    R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
    R2_BUCKET       (default: gtfs-lviv)
    GTFS_KEY        (default: worker/gtfs_worker_data.pkl)
    MODEL_KEY       (default: worker/eta_pipeline.pkl)
    FEED_KEY        (default: feed/trip_updates.pb)
    GTFS_RT_URL     upstream vehicle-position feed
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, ".")

import boto3
import requests
from dotenv import load_dotenv

from src.inference import run_inference

load_dotenv()

R2_ACCOUNT_ID       = os.environ["R2_ACCOUNT_ID"]
R2_ACCESS_KEY_ID    = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET_ACCESS_KEY = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET           = os.environ.get("R2_BUCKET",   "gtfs-lviv")
GTFS_KEY            = os.environ.get("GTFS_KEY",    "worker/gtfs_worker_data.pkl")
MODEL_KEY           = os.environ.get("MODEL_KEY",   "worker/eta_pipeline.pkl")
FEED_KEY            = os.environ.get("FEED_KEY",    "feed/trip_updates.pb")
VP_URL              = os.environ.get(
    "GTFS_RT_URL", "https://track.ua-gis.com/gtfs/lviv/vehicle_position"
)
REQUEST_TIMEOUT = 20


def _make_client():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
    )


def _load_resources(client) -> tuple[dict, dict]:
    print(f"Loading GTFS data from R2:{GTFS_KEY}…", flush=True)
    gtfs_data = pickle.loads(
        client.get_object(Bucket=R2_BUCKET, Key=GTFS_KEY)["Body"].read()
    )
    print(f"Loading model from R2:{MODEL_KEY}…", flush=True)
    model_data = pickle.loads(
        client.get_object(Bucket=R2_BUCKET, Key=MODEL_KEY)["Body"].read()
    )
    n_trips  = len(gtfs_data.get("trip_index", {}))
    n_routes = len(gtfs_data.get("route_trips", {}))
    n_trees  = len(model_data.get("trees", []))
    print(f"Ready — {n_trips} trips, {n_routes} routes, {n_trees} trees.", flush=True)
    return gtfs_data, model_data


def _push_once(client, gtfs_data: dict, model_data: dict, trackers: dict) -> None:
    t0 = time.monotonic()
    try:
        vp_bytes = requests.get(VP_URL, timeout=REQUEST_TIMEOUT).content
    except Exception as exc:
        print(f"[warn] VP fetch failed: {exc}", flush=True)
        return

    result = run_inference(gtfs_data, model_data, trackers, vp_bytes)

    client.put_object(
        Bucket=R2_BUCKET,
        Key=FEED_KEY,
        Body=result,
        ContentType="application/x-protobuf",
    )
    elapsed = (time.monotonic() - t0) * 1000
    print(f"Pushed {len(result):,} B → R2:{FEED_KEY}  ({elapsed:.0f} ms)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Push pre-computed GTFS-RT TripUpdates to R2."
    )
    parser.add_argument(
        "--loop", metavar="SECONDS", type=float, default=None,
        help="Re-push every SECONDS seconds (omit to run once and exit)",
    )
    parser.add_argument(
        "--count", metavar="N", type=int, default=None,
        help="Stop after N pushes (use with --loop to cap a CI job)",
    )
    args = parser.parse_args()

    client = _make_client()
    gtfs_data, model_data = _load_resources(client)
    trackers: dict = {}

    if args.loop:
        n = 0
        print(f"Looping every {args.loop:.0f}s — Ctrl-C to stop.", flush=True)
        while True:
            _push_once(client, gtfs_data, model_data, trackers)
            n += 1
            if args.count and n >= args.count:
                break
            time.sleep(args.loop)
    else:
        _push_once(client, gtfs_data, model_data, trackers)


if __name__ == "__main__":
    main()
