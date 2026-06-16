"""
Score the *live* prediction feed against what actually happened.

The worker archives the served TripUpdates feed every 5 min to
`predictions/YYYY-MM-DD/<feedTsISO>.pb` (see worker/worker.js, archiveFeed).
This module joins those archived predictions against the actual arrival times
derived from the raw vehicle positions for the same day, and reports how good
the live ETAs really were — the number offline training MAE can't tell us.

Pipeline:
    predictions/  ──parse──▶  one row per (feed_ts, vehicle, trip, stop, predicted_arrival)
    raw/          ──infer_trips──▶ build_labels ──▶ one actual_arrival per (vehicle, trip, stop)
    join on (vehicle_id, trip_id, stop_id, stop_sequence)
    ▶ error_sec = predicted_arrival − actual_arrival   (signed; +ve = predicted late)
    ▶ lead_sec  = predicted_arrival − feed_ts          (the horizon the rider saw)

Each (vehicle, trip, stop) has one actual but many predictions (one per 5-min
snapshot, at decreasing lead times) — exactly what horizon-stratified error
needs.  The join key is unambiguous because the encoder stamps vehicle.id on
every TripUpdate (src/inference.py, encode_trip_updates).
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from google.transit import gtfs_realtime_pb2

from src.labeling import build_labels
from src.snapshots import R2_BUCKET, _fetch_and_parse, _make_client, list_snapshot_keys
from src.trip_inference import infer_trips

PREDICTIONS_PREFIX = "predictions/"

# Lead-time (horizon) buckets in seconds — how far ahead the prediction was
# when the rider saw it.  Error almost always grows with the horizon, so this
# is the most informative stratification.
LEAD_BUCKETS_SEC = [0, 120, 300, 600, 1200, np.inf]
LEAD_LABELS = ["0-2m", "2-5m", "5-10m", "10-20m", "20m+"]

# A prediction is "arriving now" if it promised arrival within this window;
# calibration = did the vehicle actually show up within it?
ARRIVING_NOW_SEC = 60

# Ignore absurd residuals from trip-inference mismatches (e.g. a vehicle id
# reused across two trips a day): a |error| beyond this is almost certainly a
# bad join, not a bad prediction, and would swamp the means.
MAX_PLAUSIBLE_ERROR_SEC = 3600


# ---------------------------------------------------------------------------
# Load archived predictions
# ---------------------------------------------------------------------------

def _parse_prediction_feed(data: bytes) -> list[dict]:
    """One row per stop_time_update in an archived TripUpdates feed.

    stops_ahead is the 1-based position of the stop within the vehicle's
    update list at that snapshot — a post-hoc proxy for the prediction horizon
    in stops (the feed doesn't carry the vehicle's own stop_sequence).
    """
    feed = gtfs_realtime_pb2.FeedMessage()
    try:
        feed.ParseFromString(data)
    except Exception:
        return []

    feed_ts = int(feed.header.timestamp)
    rows: list[dict] = []
    for entity in feed.entity:
        if not entity.HasField("trip_update"):
            continue
        tu = entity.trip_update
        vehicle_id = tu.vehicle.id if tu.HasField("vehicle") else entity.id
        trip_id = tu.trip.trip_id
        route_id = tu.trip.route_id or None
        for ahead, stu in enumerate(tu.stop_time_update, start=1):
            if not stu.HasField("arrival"):
                continue
            rows.append({
                "feed_ts": feed_ts,
                "vehicle_id": str(vehicle_id),
                "trip_id": str(trip_id),
                "route_id": route_id,
                "stop_id": str(stu.stop_id),
                "stop_sequence": int(stu.stop_sequence),
                "stops_ahead": ahead,
                "predicted_arrival": int(stu.arrival.time),
            })
    return rows


def load_predictions(date_str: str, client=None, max_workers: int = 16) -> pd.DataFrame:
    """All archived predictions for a UTC day as a flat DataFrame."""
    client = client or _make_client()
    paginator = client.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=R2_BUCKET, Prefix=f"{PREDICTIONS_PREFIX}{date_str}/"):
        keys.extend(o["Key"] for o in page.get("Contents", []))

    rows: list[dict] = []

    def _get(key: str) -> list[dict]:
        try:
            data = client.get_object(Bucket=R2_BUCKET, Key=key)["Body"].read()
        except Exception:
            return []
        return _parse_prediction_feed(data)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for r in ex.map(_get, keys):
            rows.extend(r)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Derive actual arrivals from raw vehicle positions
# ---------------------------------------------------------------------------

def load_actuals(date_str: str, gtfs, client=None, max_workers: int = 12) -> pd.DataFrame:
    """Actual arrival time per (vehicle_id, trip_id, stop_id, stop_sequence).

    Reuses the exact trip-inference + stop-crossing logic the training pipeline
    uses, so the inferred trip_id here matches the one the live feed predicted
    against (both come from src.trip_inference).
    """
    client = client or _make_client()
    keys = list_snapshot_keys(date_str=date_str)
    if not keys:
        return pd.DataFrame()

    raw_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for r in ex.map(lambda k: _fetch_and_parse(client, k), keys):
            raw_rows.extend(r)

    df = pd.DataFrame(raw_rows)
    if df.empty:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.dropna(subset=["lat", "lon"])

    df = infer_trips(df, gtfs)
    labels = build_labels(df, gtfs, trip_col="inferred_trip_id")
    if labels.empty:
        return pd.DataFrame()

    labels = labels.copy()
    labels["actual_arrival_ts"] = _to_epoch_seconds(labels["actual_arrival"])
    return labels[
        ["vehicle_id", "trip_id", "route_id", "stop_id", "stop_sequence", "actual_arrival_ts"]
    ]


def _to_epoch_seconds(ts) -> pd.Series:
    """UTC datetime series → int64 epoch seconds, resolution-independent.

    pandas 3.0 builds datetime64 at microsecond (not nanosecond) resolution, so a
    hard-coded `.astype("int64") // 1e9` divides microseconds by a nanosecond
    factor and lands 1000x low — which silently zeroed every join (every residual
    exceeded the plausibility filter). Subtracting the epoch and floor-dividing by
    a 1-second Timedelta is correct under ns/us/ms/s alike.
    """
    ts = pd.to_datetime(ts, utc=True)
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    return ((ts - epoch) // pd.Timedelta(seconds=1)).astype("int64")


# ---------------------------------------------------------------------------
# Join + score
# ---------------------------------------------------------------------------

def join_predictions_actuals(
    predictions: pd.DataFrame, actuals: pd.DataFrame, feed_tz=None
) -> pd.DataFrame:
    """Inner-join predictions to their actual arrival and compute residuals."""
    if predictions.empty or actuals.empty:
        return pd.DataFrame()

    keys = ["vehicle_id", "trip_id", "stop_id", "stop_sequence"]
    actual_one = actuals.drop_duplicates(subset=keys)[keys + ["actual_arrival_ts"]]

    joined = predictions.merge(actual_one, on=keys, how="inner")
    if joined.empty:
        return joined

    joined["error_sec"] = joined["predicted_arrival"] - joined["actual_arrival_ts"]
    joined["abs_error_sec"] = joined["error_sec"].abs()
    joined["lead_sec"] = joined["predicted_arrival"] - joined["feed_ts"]

    # Keep only forward-looking predictions (the rider saw a future ETA) and
    # drop implausible residuals that signal a bad join, not a bad model.
    joined = joined[joined["lead_sec"] > 0]
    joined = joined[joined["abs_error_sec"] <= MAX_PLAUSIBLE_ERROR_SEC]

    joined["lead_bucket"] = pd.cut(
        joined["lead_sec"], bins=LEAD_BUCKETS_SEC, labels=LEAD_LABELS, right=False
    )
    tz = feed_tz or timezone.utc
    joined["hour"] = pd.to_datetime(joined["feed_ts"], unit="s", utc=True).dt.tz_convert(
        tz
    ).dt.hour
    return joined


def _metrics(g: pd.DataFrame) -> dict:
    err = g["error_sec"].to_numpy()
    abs_err = g["abs_error_sec"].to_numpy()
    return {
        "n": int(len(g)),
        "bias_sec": round(float(err.mean()), 1),
        "mae_sec": round(float(abs_err.mean()), 1),
        "median_ae_sec": round(float(np.median(abs_err)), 1),
        "p90_ae_sec": round(float(np.percentile(abs_err, 90)), 1),
    }


def _grouped(joined: pd.DataFrame, col: str, top: int | None = None) -> dict:
    out = {}
    for key, g in joined.groupby(col, observed=True):
        out[str(key)] = _metrics(g)
    if top is not None:
        out = dict(sorted(out.items(), key=lambda kv: kv[1]["n"], reverse=True)[:top])
    return out


def score_report(joined: pd.DataFrame, actuals: pd.DataFrame, date_str: str) -> dict:
    """Aggregate residuals into the structured quality report."""
    if joined.empty:
        return {"date": date_str, "status": "no_matches", "overall": None}

    # Coverage: of the actual arrivals observed, how many got *any* prediction.
    pred_keys = set(
        map(tuple, joined[["vehicle_id", "trip_id", "stop_id", "stop_sequence"]].values)
    )
    actual_keys = set(
        map(tuple, actuals[["vehicle_id", "trip_id", "stop_id", "stop_sequence"]].values)
    )
    coverage = len(pred_keys & actual_keys) / len(actual_keys) if actual_keys else 0.0

    # Arriving-now calibration: of predictions promising arrival within
    # ARRIVING_NOW_SEC, how often the vehicle truly arrived within it.
    now_preds = joined[joined["lead_sec"] <= ARRIVING_NOW_SEC]
    if len(now_preds):
        hit = (now_preds["abs_error_sec"] <= ARRIVING_NOW_SEC).mean()
        arriving_now = {"n": int(len(now_preds)), "within_window_frac": round(float(hit), 3)}
    else:
        arriving_now = {"n": 0, "within_window_frac": None}

    return {
        "date": date_str,
        "status": "ok",
        "commit_feed_age_days": None,
        "n_predictions_scored": int(len(joined)),
        "n_actual_arrivals": int(len(actual_keys)),
        "coverage_frac": round(coverage, 3),
        "overall": _metrics(joined),
        "by_lead_bucket": _grouped(joined, "lead_bucket"),
        "by_hour": _grouped(joined, "hour"),
        "by_route": _grouped(joined, "route_id", top=25),
        "by_stops_ahead": _grouped(joined, "stops_ahead"),
        "arriving_now": arriving_now,
        "worst_routes": _worst_routes(joined, top=8),
    }


def _worst_routes(joined: pd.DataFrame, top: int = 8, min_n: int = 50) -> list[dict]:
    """Routes with the highest MAE (min sample size) — exemplars for the AI step."""
    rows = []
    for route, g in joined.groupby("route_id", observed=True):
        if len(g) < min_n:
            continue
        m = _metrics(g)
        m["route_id"] = str(route)
        rows.append(m)
    rows.sort(key=lambda m: m["mae_sec"], reverse=True)
    return rows[:top]


def score_date(date_str: str, gtfs=None, client=None) -> dict:
    """End-to-end: load predictions + actuals for a day and return the report."""
    if gtfs is None:
        from src.gtfs_static import get_gtfs

        gtfs = get_gtfs()
    client = client or _make_client()

    predictions = load_predictions(date_str, client=client)
    if predictions.empty:
        return {"date": date_str, "status": "no_predictions", "overall": None}

    actuals = load_actuals(date_str, gtfs, client=client)
    if actuals.empty:
        return {"date": date_str, "status": "no_actuals", "overall": None}

    joined = join_predictions_actuals(predictions, actuals, feed_tz=gtfs.feed_tz)
    return score_report(joined, actuals, date_str)


def main() -> int:
    parser = argparse.ArgumentParser(description="Score live ETA quality for a day.")
    parser.add_argument(
        "--date",
        default=(datetime.now(timezone.utc).date()).isoformat(),
        help="UTC day YYYY-MM-DD (default: today)",
    )
    parser.add_argument("--out", help="write the JSON report to this path")
    args = parser.parse_args()

    report = score_date(args.date)
    text = json.dumps(report, indent=2, default=str)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
