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
import sentry_sdk
from dotenv import load_dotenv
from google.protobuf.message import DecodeError
from google.transit import gtfs_realtime_pb2

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

SENTRY_DSN = os.environ.get("SENTRY_DSN", "")


def _init_sentry() -> None:
    """Enable Sentry error reporting when SENTRY_DSN is configured.

    No-op without a DSN, so local runs and tests stay unchanged. Only
    exceptions are reported (no performance tracing) to keep quota usage low.
    """
    if not SENTRY_DSN:
        return
    # A missing DSN is the normal "disabled" path above; a *malformed* one must
    # also never take down the feed pipeline, so fall back to silent no-logging.
    try:
        sentry_sdk.init(
            dsn=SENTRY_DSN,
            environment=os.environ.get("SENTRY_ENVIRONMENT", "production"),
            release=os.environ.get("SENTRY_RELEASE"),
            traces_sample_rate=0.0,
        )
        # This DSN is shared with other services (e.g. timetable-api-node), so
        # tag every event to make gtfs-eta's daemon issues unmistakable.
        sentry_sdk.set_tag("service", "gtfs-eta-daemon")
        print("Sentry error reporting enabled (service=gtfs-eta-daemon).", flush=True)
    except Exception as exc:  # noqa: BLE001 — Sentry must never break startup
        print(f"[warn] Sentry init failed, continuing without it: {exc!r}", flush=True)


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
    vp_bytes = None
    for attempt in range(5):
        try:
            resp = requests.get(VP_URL, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            trial = gtfs_realtime_pb2.FeedMessage()
            trial.ParseFromString(resp.content)
            vp_bytes = resp.content
            break
        except (requests.exceptions.RequestException, DecodeError) as exc:
            if attempt == 4:
                print(f"[warn] VP fetch failed after 5 attempts: {exc}", flush=True)
                return
            time.sleep(0.2 * (2 ** attempt))
            print(f"[warn] VP fetch attempt {attempt + 1} failed: {exc}, retrying…", flush=True)

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

    _init_sentry()

    client = _make_client()
    gtfs_data, model_data = _load_resources(client)
    trackers: dict = {}

    if args.loop:
        n = 0
        print(f"Looping every {args.loop:.0f}s — Ctrl-C to stop.", flush=True)
        while True:
            # A single failed iteration (inference bug, transient R2 error) is
            # reported to Sentry but must not kill the daemon — the next cycle
            # 30 s later usually recovers, keeping the feed fresh.
            try:
                _push_once(client, gtfs_data, model_data, trackers)
            except Exception as exc:  # noqa: BLE001 — daemon must stay alive
                sentry_sdk.capture_exception(exc)
                print(f"[error] push iteration failed: {exc!r}", flush=True)
            n += 1
            if args.count and n >= args.count:
                break
            time.sleep(args.loop)
    else:
        # One-shot mode: report, then re-raise so the exit code reflects failure.
        try:
            _push_once(client, gtfs_data, model_data, trackers)
        except Exception as exc:  # noqa: BLE001
            sentry_sdk.capture_exception(exc)
            raise


if __name__ == "__main__":
    main()
