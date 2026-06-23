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
QUALITY_PREFIX = "quality/"

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
    n_forward = len(joined)
    implausible = joined["abs_error_sec"] > MAX_PLAUSIBLE_ERROR_SEC
    n_dropped = int(implausible.sum())
    joined = joined[~implausible]
    # Stash how aggressive the plausibility filter was so the report can flag a
    # join that's silently discarding a large share of residuals (which would
    # make the published MAE look better than it is).
    joined.attrs["n_implausible_dropped"] = n_dropped
    joined.attrs["n_forward"] = n_forward

    joined["lead_bucket"] = pd.cut(
        joined["lead_sec"], bins=LEAD_BUCKETS_SEC, labels=LEAD_LABELS, right=False
    )
    tz = feed_tz or timezone.utc
    joined["hour"] = pd.to_datetime(joined["feed_ts"], unit="s", utc=True).dt.tz_convert(
        tz
    ).dt.hour
    return joined


def join_predictions_actuals_relaxed(
    predictions: pd.DataFrame, actuals: pd.DataFrame, feed_tz=None
) -> pd.DataFrame:
    """Rider-centric join: match each prediction to the nearest-in-time actual
    arrival at the same physical ``(vehicle_id, stop_id)``, within
    MAX_PLAUSIBLE_ERROR_SEC.

    The strict join (``join_predictions_actuals``) keys on ``trip_id`` *and*
    ``stop_sequence``. The issue #3 probe showed both are trip-relative: when the
    live per-snapshot trip inference labels a vehicle with a different trip than
    the batch ``infer_trips`` labeling (38% of uncovered arrivals, 100% on the
    correct route), the stop_sequence numbering differs too — measured at **85%
    of (vehicle, stop_id) pairs** — so even dropping trip_id alone recovers almost
    nothing. Matching on the physical stop the rider actually waits at, with time
    proximity separating a vehicle's repeat visits, is the rider's own question:
    "did this stop get a usable ETA near this time?". Empirically this lifts
    measured coverage only ~3.4pp (73.2% → 76.6% on 2026-06-22) — i.e. coverage is
    *mostly a real gap*, not a join artifact; the rest of trip_mismatch reflects
    physical stops the live feed never predicted at a plausible time.

    Caveat: a vehicle that visits the same stop_id twice within
    MAX_PLAUSIBLE_ERROR_SEC (loop / out-and-back) can mis-bind; that makes this an
    upper-bound diagnostic, not the authoritative metric. The matched actual's
    trip_id / stop_sequence are preserved (``actual_trip_id`` /
    ``actual_stop_sequence``) so coverage uses the same strict denominator.
    """
    if predictions.empty or actuals.empty:
        return pd.DataFrame()

    by = ["vehicle_id", "stop_id"]
    left = predictions.sort_values("predicted_arrival")
    right = (
        actuals[by + ["trip_id", "stop_sequence", "actual_arrival_ts"]]
        .rename(columns={"trip_id": "actual_trip_id",
                         "stop_sequence": "actual_stop_sequence"})
        .sort_values("actual_arrival_ts")
    )
    joined = pd.merge_asof(
        left, right,
        left_on="predicted_arrival", right_on="actual_arrival_ts",
        by=by, direction="nearest", tolerance=MAX_PLAUSIBLE_ERROR_SEC,
    )
    joined = joined[joined["actual_arrival_ts"].notna()].copy()
    if joined.empty:
        return joined

    joined["error_sec"] = joined["predicted_arrival"] - joined["actual_arrival_ts"]
    joined["abs_error_sec"] = joined["error_sec"].abs()
    joined["lead_sec"] = joined["predicted_arrival"] - joined["feed_ts"]
    joined = joined[joined["lead_sec"] > 0]
    if joined.empty:
        return joined

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


def score_report(
    joined: pd.DataFrame, actuals: pd.DataFrame, date_str: str, feed_tz=None,
    predictions: pd.DataFrame | None = None,
) -> dict:
    """Aggregate residuals into the structured quality report."""
    if joined.empty:
        out = {"date": date_str, "status": "no_matches", "overall": None}
        # The strict trip_id join can find nothing while the relaxed join still
        # matches everything (a day where the live trip labels diverge wholesale
        # from batch). Surface that rather than reporting a bare no_matches.
        if predictions is not None and not predictions.empty and not actuals.empty:
            actual_keys = set(
                map(tuple, actuals[["vehicle_id", "trip_id", "stop_id", "stop_sequence"]].values)
            )
            rb = _relaxed_block(predictions, actuals, actual_keys, feed_tz=feed_tz)
            if rb.get("status") != "no_matches":
                out["relaxed_join"] = rb
        return out

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

    rep = {
        "date": date_str,
        "status": "ok",
        "commit_feed_age_days": None,
        "n_predictions_scored": int(len(joined)),
        "n_actual_arrivals": int(len(actual_keys)),
        "coverage_frac": round(coverage, 3),
        "implausible_dropped": {
            "n": int(joined.attrs.get("n_implausible_dropped", 0)),
            "frac": round(
                joined.attrs.get("n_implausible_dropped", 0)
                / max(joined.attrs.get("n_forward", 0), 1),
                4,
            ),
            "threshold_sec": MAX_PLAUSIBLE_ERROR_SEC,
        },
        "overall": _metrics(joined),
        "by_lead_bucket": _grouped(joined, "lead_bucket"),
        "by_hour": _grouped(joined, "hour"),
        "by_route": _grouped(joined, "route_id", top=25),
        "by_stops_ahead": _grouped(joined, "stops_ahead"),
        "arriving_now": arriving_now,
        "worst_routes": _worst_routes(joined, top=8),
        "coverage_gap": _coverage_gap_breakdown(
            actuals, pred_keys, predictions=predictions, feed_tz=feed_tz
        ),
    }
    # Diagnostic-only: what coverage and error look like under a trip-id-agnostic
    # join (see join_predictions_actuals_relaxed). Kept separate so the strict
    # metrics above stay the authoritative, historically-continuous series the
    # uncertainty calibration pools — this block just makes the trip_mismatch
    # measurement loss visible.
    if predictions is not None and not predictions.empty:
        rep["relaxed_join"] = _relaxed_block(
            predictions, actuals, actual_keys, feed_tz=feed_tz
        )
    return rep


def _relaxed_block(
    predictions: pd.DataFrame, actuals: pd.DataFrame, actual_keys: set, feed_tz=None
) -> dict:
    """Coverage + error under the trip-id-agnostic join, vs the same denominator."""
    rj = join_predictions_actuals_relaxed(predictions, actuals, feed_tz=feed_tz)
    if rj.empty:
        return {"status": "no_matches"}
    matched_keys = set(
        map(tuple, rj[["vehicle_id", "actual_trip_id", "stop_id", "actual_stop_sequence"]].to_numpy())
    )
    coverage = len(matched_keys & actual_keys) / len(actual_keys) if actual_keys else 0.0
    return {
        "coverage_frac": round(coverage, 3),
        "n_predictions_scored": int(len(rj)),
        "overall": _metrics(rj),
        "by_lead_bucket": _grouped(rj, "lead_bucket"),
        "by_stops_ahead": _grouped(rj, "stops_ahead"),
        "worst_routes": _worst_routes(rj, top=8),
    }


def _coverage_gap_breakdown(
    actuals: pd.DataFrame, pred_keys: set, predictions: pd.DataFrame | None = None,
    feed_tz=None,
) -> dict:
    """Per-route and per-hour breakdown of actual arrivals that got no prediction.

    Helps distinguish a concentrated gap (bad feed for a few routes / off-peak
    hours) from a uniform one (systematic pipeline dropout).

    When the served ``predictions`` feed is supplied, each uncovered arrival is
    tagged with *why* it was missed — the actionable distinction the headline
    coverage number hides:

      vehicle_absent — the vehicle never appeared in the served feed that day
                       (off-route suppressed throughout, or absent upstream).
      trip_mismatch  — the vehicle was served, but never under this trip_id: the
                       live trip-inference matched it to a different trip than the
                       actuals labeling did (the matching disagreement behind the
                       worst-route MAE and the loop-route coverage holes).
      stop_missing   — the (vehicle, trip) was served, but not this stop: the stop
                       sat beyond the served horizon (max_stops_ahead) or had
                       already been passed when the vehicle entered the feed.
    """
    tz = feed_tz or timezone.utc
    key_cols = ["vehicle_id", "trip_id", "stop_id", "stop_sequence"]
    df = actuals[key_cols + ["route_id", "actual_arrival_ts"]].copy()
    df["covered"] = [tuple(row) in pred_keys for row in df[key_cols].values]
    df["hour"] = (
        pd.to_datetime(df["actual_arrival_ts"], unit="s", utc=True)
        .dt.tz_convert(tz)
        .dt.hour
    )

    have_causes = predictions is not None and not predictions.empty
    if have_causes:
        served_vehicles = set(predictions["vehicle_id"].astype(str))
        served_vtrips = set(
            map(tuple, predictions[["vehicle_id", "trip_id"]].astype(str).to_numpy())
        )
        vid = df["vehicle_id"].astype(str)
        in_vehicle = vid.isin(served_vehicles).to_numpy()
        vtrips = zip(vid, df["trip_id"].astype(str))
        in_vtrip = np.fromiter(
            (vt in served_vtrips for vt in vtrips), dtype=bool, count=len(df)
        )
        df["cause"] = np.where(
            df["covered"].to_numpy(), "covered",
            np.where(~in_vehicle, "vehicle_absent",
                     np.where(~in_vtrip, "trip_mismatch", "stop_missing")),
        )

    def _gap(g: pd.DataFrame) -> dict:
        n = len(g)
        nc = int(g["covered"].sum())
        out = {
            "n_actual": int(n),
            "n_uncovered": int(n - nc),
            "coverage_frac": round(nc / n, 3) if n else 0.0,
        }
        if have_causes:
            unc = g.loc[~g["covered"], "cause"].value_counts()
            out["uncovered_by_cause"] = {k: int(v) for k, v in unc.items()}
        return out

    by_route = {str(k): _gap(g) for k, g in df.groupby("route_id")}
    # Surface the routes with the most uncovered arrivals first; cap at 20.
    by_route = dict(
        sorted(by_route.items(), key=lambda kv: kv[1]["n_uncovered"], reverse=True)[:20]
    )
    by_hour = {str(int(k)): _gap(g) for k, g in df.groupby("hour")}
    result = {"by_route": by_route, "by_hour": by_hour}
    if have_causes:
        unc_all = df.loc[~df["covered"], "cause"].value_counts()
        result["by_cause"] = {k: int(v) for k, v in unc_all.items()}
    return result


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


# ---------------------------------------------------------------------------
# Live-calibrated prediction uncertainty
# ---------------------------------------------------------------------------

def _aggregate_stops_ahead_mae(reports: list[dict], min_n: int = 200) -> dict:
    """Pool per-horizon MAE across daily quality reports → {horizon:int -> sec:int}.

    Each report's ``by_stops_ahead[h] = {"n", "mae_sec", ...}``. Since mae_sec is
    itself a mean of |error|, the n-weighted mean of the daily per-horizon values
    is exactly the pooled multi-day MAE for that horizon. Horizons below *min_n*
    total support are dropped (too noisy to publish). The result is made
    non-decreasing in horizon: a longer-horizon confidence band must never be
    tighter than a shorter one, even if a sparse far horizon scores lower by luck.
    """
    acc: dict[int, list[float]] = {}  # horizon -> [sum_abs_err, n]
    for rep in reports:
        bsa = (rep or {}).get("by_stops_ahead") or {}
        for h_str, m in bsa.items():
            try:
                h, n, mae = int(h_str), int(m["n"]), float(m["mae_sec"])
            except (ValueError, KeyError, TypeError):
                continue
            if n <= 0:
                continue
            slot = acc.setdefault(h, [0.0, 0.0])
            slot[0] += mae * n
            slot[1] += n
    pooled = {h: s / n for h, (s, n) in acc.items() if n >= min_n}
    if not pooled:
        return {}
    out: dict[int, int] = {}
    running = 0.0
    for h in sorted(pooled):
        running = max(running, pooled[h])
        out[h] = int(round(running))
    return out


def live_uncertainty_by_horizon(
    days: int = 7, client=None, min_n: int = 200
) -> tuple[dict, list[str]]:
    """Per-horizon uncertainty (seconds) calibrated from the live quality archive.

    Reads the most recent *days* ``quality/<date>.json`` reports from R2 and pools
    their per-stops-ahead MAE (see ``_aggregate_stops_ahead_mae``). This reflects
    real serving error, which runs ~2x the training-test split the bands were
    originally derived from. Returns ``({horizon:int -> sec:int}, dates_used)``;
    an empty dict when no usable scored history exists (caller should fall back).
    """
    client = client or _make_client()
    paginator = client.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=R2_BUCKET, Prefix=QUALITY_PREFIX):
        for o in page.get("Contents", []):
            base = o["Key"][len(QUALITY_PREFIX):]
            # Only immutable per-day files: quality/YYYY-MM-DD.json (15 chars).
            # Excludes latest.json / latest.md.
            if len(base) == 15 and base.endswith(".json") and base[:10].count("-") == 2:
                keys.append(o["Key"])
    keys.sort(reverse=True)

    reports: list[dict] = []
    dates_used: list[str] = []
    for key in keys[:days]:
        try:
            rep = json.loads(client.get_object(Bucket=R2_BUCKET, Key=key)["Body"].read())
        except Exception:  # noqa: BLE001 — a single unreadable report must not abort
            continue
        if rep.get("status") == "ok" and rep.get("by_stops_ahead"):
            reports.append(rep)
            dates_used.append(rep.get("date") or key)
    return _aggregate_stops_ahead_mae(reports, min_n=min_n), sorted(dates_used)


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

    feed_tz = getattr(gtfs, "feed_tz", None)
    joined = join_predictions_actuals(predictions, actuals, feed_tz=feed_tz)
    return score_report(
        joined, actuals, date_str, feed_tz=feed_tz, predictions=predictions
    )


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
