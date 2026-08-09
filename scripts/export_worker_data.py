"""
Export GTFS static data + trained model to R2 for the Cloudflare Worker.

Produces two R2 objects:
  worker/gtfs_worker_data.pkl  — compact GTFS dict (shapes, trips, stops, distances)
  worker/eta_pipeline.pkl      — sklearn Pipeline (pickle, joblib-free)

Usage:
    python scripts/export_worker_data.py
"""

import sys
sys.path.insert(0, ".")

import io
import os
import pickle
from pathlib import Path

import boto3
import joblib
from dotenv import load_dotenv

from src.gtfs_static import get_gtfs
from src.train import MODEL_PATH, PRIORS_PATH, UNCERTAINTY_PATH

load_dotenv()

R2_ACCOUNT_ID = os.environ["R2_ACCOUNT_ID"]
R2_ACCESS_KEY_ID = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET_ACCESS_KEY = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET = os.environ.get("R2_BUCKET", "gtfs-lviv")

GTFS_KEY = "worker/gtfs_worker_data.pkl"
MODEL_KEY = "worker/eta_pipeline.pkl"

# Measured stop-dwell sidecar (scripts/measure_dwell.py). Optional: without it
# the served feed falls back to src/inference._DWELL_SECS.
DWELL_PATH = MODEL_PATH.parent / "dwell.joblib"


def _make_client():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
    )


# The Lviv feed types every trolleybus route as 3 (bus); only trams carry a
# correct route_type. Their short names are the only reliable marker, so the
# spec value (11, trolleybus) is restored here — src/inference.py keys its
# schedule-anchored terminus ETAs off route_type, and trolleybuses keep the
# schedule at a terminus just like trams do.
_TROLLEYBUS_NAME_PREFIX = "Тр"      # Cyrillic Т + р, e.g. "Тр22"
_ROUTE_TYPE_TROLLEYBUS = 11


def _build_route_types(gtfs) -> dict[str, int]:
    """route_id → GTFS route_type, with Lviv's mistyped trolleybuses restored."""
    out: dict[str, int] = {}
    routes = gtfs._routes
    if routes is None:
        return out
    for r in routes.itertuples():
        try:
            rtype = int(r.route_type)
        except (TypeError, ValueError):
            continue
        name = str(getattr(r, "route_short_name", "") or "")
        if name.startswith(_TROLLEYBUS_NAME_PREFIX):
            rtype = _ROUTE_TYPE_TROLLEYBUS
        out[str(r.route_id)] = rtype
    return out


def build_gtfs_worker_data(gtfs, existing_priors: dict | None = None) -> dict:
    """
    Serialise GTFSStatic to a plain-dict format loadable without pyproj/pyproject.

    Scheduled times are baked into stop_times as *cumulative* seconds since
    the trip's first stop (sched_cum_sec), so inference can interpolate the
    schedule at the vehicle's projected position — matching how
    sched_remaining_sec is computed at training time (src/features.py).

    ``existing_priors`` is the ``route_hour_priors`` dict already live on R2,
    used when this run has no local models/route_hour_priors.joblib (e.g.
    refresh-gtfs.yml, which only refreshes GTFS static and never checks out
    model artifacts) — without it, every such run silently wiped priors back
    to fallback constants until the next manual `make export`.
    """
    print("Building worker GTFS data…")

    from src.gtfs_static import _parse_gtfs_time
    from datetime import date, datetime, time as dtime
    base = date(2000, 1, 1)  # dummy date — only schedule deltas are used

    # Trips: route_id, shape_id, stop_times with sched_cum_sec
    trip_index = {}
    for trip_id, info in gtfs._trip_index.items():
        stop_times = []
        t0 = None
        cum = 0.0
        for st in info.stop_times:
            t = _parse_gtfs_time(st.arrival_time or st.departure_time, base)
            if t is not None:
                if t0 is None:
                    t0 = t
                cum = max(cum, (t - t0).total_seconds())
            # Compact tuple (stop_id, stop_sequence, sched_cum_sec) —
            # unparseable times carry the previous cumulative value.
            stop_times.append((st.stop_id, st.stop_sequence, cum))

        trip_index[trip_id] = {
            "route_id": info.route_id,
            "shape_id": info.shape_id,
            "service_id": info.service_id,
            # Parsed from trips.txt all along but never exported, so the feed
            # could not publish TripDescriptor.direction_id — leaving consumers
            # to infer a trip's direction from its shape.
            "direction_id": info.direction_id,
            "stop_times": stop_times,
            # Absolute scheduled start, as seconds since local midnight of the
            # service day (≥86400 for after-midnight trips).  The cumulative
            # offsets above are enough for mid-route interpolation, but the
            # terminus path in src/inference.py needs the wall-clock departure
            # itself.  None when the trip has no parseable time at all.
            "start_sec": None if t0 is None else (t0 - datetime.combine(base, dtime.min)).total_seconds(),
        }

    # Store shapes as packed float64 bytes — 16 bytes/point instead of
    # ~104 bytes/point as Python (float, float) tuples. This keeps the worker
    # well under the 128 MB Cloudflare memory limit.
    import struct as _struct
    shapes_coords = {
        sid: _struct.pack(f"{2 * len(pts)}d", *(v for xy in pts for v in xy))
        for sid, geom in gtfs._shapes.items()
        for pts in (list(geom.coords),)
    }

    # Axis-aligned bounds per shape, so trip matching can prove a shape cannot
    # win before walking its polyline (src/inference.infer_trip). Four floats
    # per shape against ~180 shapes — no meaningful size cost.
    shape_bboxes = {
        sid: tuple(float(v) for v in geom.bounds)   # (minx, miny, maxx, maxy)
        for sid, geom in gtfs._shapes.items()
    }

    # Route+hour speed/dwell priors — converted to string keys for fast dict lookup.
    # Format: {"ROUTE_ID:HOUR": (hist_speed_mps, hist_time_per_stop_sec), "_global": (...)}
    priors_raw: dict = {}
    if PRIORS_PATH.exists():
        raw = joblib.load(PRIORS_PATH)
        priors_raw = {f"{rh[0]}:{rh[1]}": v for rh, v in raw["lookup"].items()}
        priors_raw["_global"] = (raw["global_speed"], raw["global_tps"])
        print(f"  Loaded {len(priors_raw) - 1} route×hour priors")
    elif existing_priors:
        priors_raw = existing_priors
        print(f"  {PRIORS_PATH} not found — preserving {len(priors_raw) - 1} priors already live on R2")
    else:
        print(f"  WARNING: {PRIORS_PATH} not found and no existing R2 priors — using fallback constants")

    route_types = _build_route_types(gtfs)

    data = {
        "shapes": shapes_coords,                          # shape_id → bytes (packed float64 pairs)
        "shape_bboxes": shape_bboxes,                     # shape_id → (minx, miny, maxx, maxy)
        "shape_lengths": dict(gtfs._shape_lengths),       # shape_id → float metres
        # stop_id → stop_name. Nothing in the serving path needs it, but every
        # diagnostic that reports a stop currently prints a bare internal id and
        # has to reload the whole static feed to name it.
        "stop_names": {sid: s.stop_name for sid, s in gtfs._stops.items()},
        "stop_distances": dict(gtfs._stop_distances),     # (shape_id, stop_id) → float
        "trip_index": trip_index,
        "route_trips": dict(gtfs._route_trips),           # route_id → [trip_id, ...]
        "route_hour_priors": priors_raw,                  # route+hour speed/dwell priors
        "route_types": route_types,                       # route_id → GTFS route_type int
        "feed_timezone": str(gtfs.feed_tz),               # for local-time schedule lookups
        # calendar.txt / calendar_dates.txt, as plain string-keyed rows — lets
        # infer_trip() restrict same-shape weekday/weekend trip variants (e.g.
        # route 105) to the ones actually running today, instead of matching
        # on geometry alone across trips that were never scheduled today.
        "calendar": gtfs._calendar.to_dict("records") if not gtfs._calendar.empty else [],
        "calendar_dates": (
            gtfs._calendar_dates.to_dict("records") if not gtfs._calendar_dates.empty else []
        ),
        # Self-intersecting shapes (out-and-back routes, tram turnarounds) —
        # see GTFSStatic.is_ambiguous_shape. vehicle_dist_along() clamps raw
        # nearest-point projection on these to avoid snapping to a distant,
        # wrong occurrence of the same physical road (e.g. route 122).
        "ambiguous_shapes": set(gtfs._ambiguous_shapes),
    }

    n_shapes = len(data["shapes"])
    n_trips = len(data["trip_index"])
    n_routes = len(data["route_trips"])
    print(f"  {n_shapes} shapes, {n_trips} trips, {n_routes} routes")
    return data


def _extract_trees(pipeline) -> dict:
    """
    Convert sklearn Pipeline → plain-Python tree dict for Pyodide inference.

    Schema:
      route_to_int  : {route_id_str: int_index}   (OrdinalEncoder mapping)
      baseline      : float                        (_baseline_prediction)
      learning_rate : float
      trees         : list of list of 6-tuples
                      (feat_idx, threshold, left, right, is_leaf, value)

    Feature vector order after ColumnTransformer (matches FEATURE_COLS in
    src/features.py — keep in sync with build_features in src/inference.py):
      0  route_id (encoded)
      1  stop_sequence
      2  stops_ahead
      3  hour
      4  day_of_week
      5  month
      6  is_weekend
      7  is_holiday
      8  remaining_dist_m
      9  progress_speed_mps
      10 stops_remaining
      11 trip_progress_frac
      12 dist_per_stop_m
      13 speed_eta_warm       (remaining_dist / effective_speed, warm-started)
      14 hist_speed_mps       (route+hour historical median speed)
      15 hist_travel_time_est (stops_ahead * hist seconds-per-stop, dwell-aware)
    """
    prep = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]

    encoder = prep.transformers_[0][1]
    route_to_int = {str(r): i for i, r in enumerate(encoder.categories_[0])}

    baseline = float(model._baseline_prediction.flat[0])
    learning_rate = float(model.learning_rate)

    trees = []
    for estimators_at_iter in model._predictors:
        predictor = estimators_at_iter[0]
        nodes = predictor.nodes
        trees.append([
            (int(n["feature_idx"]), float(n["num_threshold"]),
             int(n["left"]), int(n["right"]),
             bool(n["is_leaf"]), float(n["value"]))
            for n in nodes
        ])

    n_trees = len(trees)
    n_nodes = sum(len(t) for t in trees)
    print(f"  {n_trees} trees, {n_nodes} nodes, {len(route_to_int)} routes")
    return {
        "route_to_int": route_to_int,
        "baseline": baseline,
        "learning_rate": learning_rate,
        "trees": trees,
    }


def main():
    client = _make_client()

    # Preserve whatever priors are already live on R2 when this run has no
    # local models/route_hour_priors.joblib — see build_gtfs_worker_data.
    existing_priors: dict | None = None
    if not PRIORS_PATH.exists():
        try:
            obj = client.get_object(Bucket=R2_BUCKET, Key=GTFS_KEY)
            existing_priors = pickle.loads(obj["Body"].read()).get("route_hour_priors") or None
        except Exception as exc:  # noqa: BLE001 — no existing object is fine, fall back
            print(f"  Could not fetch existing priors from R2: {exc!r}")

    # ── GTFS ──
    print("Loading GTFS static…")
    gtfs = get_gtfs(force_download=True, force_rebuild=True)
    worker_data = build_gtfs_worker_data(gtfs, existing_priors=existing_priors)

    print(f"Serialising GTFS data…")
    buf = io.BytesIO()
    pickle.dump(worker_data, buf, protocol=4)
    size_mb = buf.tell() / 1e6
    print(f"  {size_mb:.1f} MB")

    print(f"Uploading → R2:{GTFS_KEY}")
    buf.seek(0)
    client.put_object(Bucket=R2_BUCKET, Key=GTFS_KEY, Body=buf.read())
    print("  done")

    # ── Model ──
    model_path = Path(MODEL_PATH)
    if model_path.exists():
        import joblib
        pipeline = joblib.load(model_path)
        tree_data = _extract_trees(pipeline)
    else:
        # refresh-gtfs.yml (daily CI) checks out the repo only — it never has a
        # local models/eta_pipeline.joblib (gitignored, only produced by a real
        # `make train`). This used to `return` here, so every CI run silently
        # skipped the model upload entirely — the live-calibrated bands below
        # (uncertainty, bias) never refreshed in CI at all, only whenever
        # someone happened to run `make export` by hand from a machine with a
        # trained model. Reuse the trees already live on R2 (unchanged since the
        # last real retrain) and just refresh the bands, same fallback shape as
        # existing_priors above.
        try:
            existing_model = pickle.loads(
                client.get_object(Bucket=R2_BUCKET, Key=MODEL_KEY)["Body"].read()
            )
            tree_data = {
                k: existing_model[k]
                for k in ("route_to_int", "baseline", "learning_rate", "trees")
            }
            print(f"  Model not found at {model_path} — reusing trees already live "
                  "on R2, refreshing live-calibrated bands only")
        except Exception as exc:  # noqa: BLE001 — no existing object, nothing to refresh
            print(f"  Model not found at {model_path} and no existing model on R2 "
                  f"({exc!r}) — skipping model upload")
            return

    # Per-horizon uncertainty bands (seconds) → served as GTFS-RT
    # StopTimeEvent.uncertainty. Prefer LIVE calibration — pooled per-stops-ahead
    # MAE from the quality archive, which reflects real serving error (~2x the
    # training-test split). The train-split sidecar (models/uncertainty.joblib) is
    # only a cold-start fallback for before any day has been scored. The feed omits
    # the field entirely when neither source is available (backward-compatible).
    unc_table: dict | None = None
    try:
        from src.scoring import live_uncertainty_by_horizon
        unc_table, dates_used = live_uncertainty_by_horizon(days=7)
        if unc_table:
            span = f"{dates_used[0]}..{dates_used[-1]}" if dates_used else "?"
            print(f"  Uncertainty bands (live, {len(dates_used)} days {span}): {unc_table}")
    except Exception as exc:  # noqa: BLE001 — calibration must never block an export
        print(f"  WARNING: live uncertainty calibration failed: {exc!r}")

    if not unc_table and UNCERTAINTY_PATH.exists():
        unc_table = joblib.load(UNCERTAINTY_PATH)
        print(f"  Uncertainty bands (train-split fallback): {unc_table}")

    if unc_table:
        tree_data["uncertainty_by_horizon"] = unc_table
    else:
        print("  WARNING: no uncertainty bands available — feed will omit the field")

    # Per-horizon live bias correction (seconds): corrected = raw - bias, cancels
    # a systematic per-horizon over/under-estimate the model itself doesn't fix
    # (e.g. the flat ~-44s optimism found across every stops_ahead bucket on
    # 2026-07-25). No offline fallback — an unavailable live signal means no
    # correction, not a guessed/stale one baked in from a different model.
    bias_table: dict | None = None
    try:
        from src.scoring import live_bias_by_horizon
        bias_table, bias_dates = live_bias_by_horizon(days=7)
        if bias_table:
            span = f"{bias_dates[0]}..{bias_dates[-1]}" if bias_dates else "?"
            print(f"  Bias correction (live, {len(bias_dates)} days {span}): {bias_table}")
    except Exception as exc:  # noqa: BLE001 — calibration must never block an export
        print(f"  WARNING: live bias calibration failed: {exc!r}")

    if bias_table:
        tree_data["bias_by_horizon"] = bias_table
    else:
        print("  No live bias correction available — serving uncorrected predictions")

    # Weekday/weekend split (found 2026-07-25: weekday rush bias runs ~2-3x
    # weekend rush bias — a single blended number under-corrects one and
    # over-corrects the other). 14-day window (vs the flat table's 7) since
    # weekend days are ~2/7 of the week and need more history for the same
    # per-horizon support. Each bucket is merged over the flat table so a
    # horizon too thin on just-weekday or just-weekend data still gets the
    # blended correction rather than none.
    try:
        from src.scoring import live_bias_by_horizon_weekend
        bias_weekend, bias_weekend_dates = live_bias_by_horizon_weekend(days=14)
        if bias_weekend and (bias_weekend.get("weekday") or bias_weekend.get("weekend")):
            span = (
                f"{bias_weekend_dates[0]}..{bias_weekend_dates[-1]}"
                if bias_weekend_dates else "?"
            )
            print(f"  Bias correction (weekday/weekend, {len(bias_weekend_dates)} days {span}): "
                  f"{bias_weekend}")
            merged = {
                bucket: {**(bias_table or {}), **(bias_weekend.get(bucket) or {})}
                for bucket in ("weekday", "weekend")
            }
            tree_data["bias_by_horizon_weekend"] = merged
    except Exception as exc:  # noqa: BLE001 — calibration must never block an export
        print(f"  WARNING: live weekday/weekend bias calibration failed: {exc!r}")

    # Measured per-route-type stop dwell (scripts/measure_dwell.py), published
    # as the gap between StopTimeEvent arrival and departure. Static — it comes
    # off the labelled history, not the live archive — so unlike the bands above
    # there is nothing to recalibrate per export; it is simply carried when the
    # sidecar exists. Absent, src/inference.py keeps the old flat 15 s.
    if DWELL_PATH.exists():
        dwell_table = joblib.load(DWELL_PATH)
        tree_data["dwell_by_route_type"] = dwell_table
        print(f"  Dwell by route_type: {dwell_table}")
    else:
        try:
            existing = pickle.loads(
                client.get_object(Bucket=R2_BUCKET, Key=MODEL_KEY)["Body"].read()
            ).get("dwell_by_route_type")
        except Exception:  # noqa: BLE001 — no existing object is fine
            existing = None
        if existing:
            tree_data["dwell_by_route_type"] = existing
            print(f"  {DWELL_PATH} not found — preserving dwell table live on R2")
        else:
            print(f"  {DWELL_PATH} not found — feed keeps the flat fallback dwell")

    model_bytes = pickle.dumps(tree_data, protocol=4)
    size_mb = len(model_bytes) / 1e6
    print(f"Uploading model trees ({size_mb:.1f} MB) → R2:{MODEL_KEY}")
    client.put_object(Bucket=R2_BUCKET, Key=MODEL_KEY, Body=model_bytes)
    print("  done")

    print("\nAll artifacts uploaded. Deploy with: make deploy")


if __name__ == "__main__":
    main()
