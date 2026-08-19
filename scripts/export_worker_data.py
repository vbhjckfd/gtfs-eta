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

import hashlib
import io
import os
import pickle
from datetime import date, timedelta
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


def _day_after(day: str | None) -> str | None:
    """``YYYY-MM-DD`` + 1 day, or None. Unparseable input is treated as absent
    so a corrupt stamp reopens the window rather than freezing calibration."""
    if not day:
        return None
    try:
        return (date.fromisoformat(day) + timedelta(days=1)).isoformat()
    except ValueError:
        return None


# Fraction of a freshly measured residual folded in per calibration. A full
# step would be exact if the residual described the table now live, but it
# never does: a day is scored only after it has been served, so what gets
# folded in was measured against an older table. Under that one-cycle lag
# ``T <- T + R`` puts both poles on the unit circle (z^2 - z + 1 = 0) — the
# loop swings around the true bias forever without losing amplitude. Which is
# what live bias did over 2026-08-15..18, the first four days under full
# accumulation: -21.7s, +1.7s, -26.6s, -23.9s, sign flipping daily while
# arriving-now calibration fell 87% -> 65%. A half step moves the poles to
# |z| = 0.707, so the same loop decays instead of ringing, and one
# unrepresentative day drags the served correction half as far.
_BIAS_GAIN = 0.5


def _accumulate_bias(previous: dict | None, residual: dict,
                     gain: float = _BIAS_GAIN) -> dict:
    """Fold a freshly measured residual into the bias correction already live.

    The served prediction is ``raw - T``, so a report's bias is what is *left*
    after T was applied, not the model's own bias. Replacing T with that
    residual therefore only ever removes half the error: with ``T <- R`` and
    ``R = B - T`` the loop settles at ``T = B/2``. Measured 2026-08-13 —
    sa=1 served a +34s residual under a table of +18, i.e. a raw bias near +52
    that the correction had been chasing for weeks without ever closing.

    Accumulating (``T <- T + gain*R``) puts the fixed point at ``R = 0``
    instead. The residual being measured against whatever T was live is also
    what makes this survive a model change: the inherited T is wrong for the
    new trees, but it is still the baseline the next residual is relative to,
    so the loop re-converges rather than needing to be reset (zeroing T would
    throw away the only measurement of the raw bias).

    Keys are horizons; ``previous`` may be missing any of them (treated as 0),
    which is also what happens on the first calibration after a model change.
    """
    previous = previous or {}
    return {h: int(round(previous.get(h, 0) + gain * r)) for h, r in residual.items()}


def _merge_weekend_bias(flat: dict | None, live: dict | None, residual: dict | None) -> dict:
    """Accumulate a weekday/weekend residual over what is already live.

    Layered lowest-first: the flat table covers horizons a bucket has no split
    support for, the live bucket keeps everything it has already accumulated,
    and the residual moves only the horizons actually measured this round.
    That middle layer matters — a bucket whose residual is empty (weekend
    support is thin under a ``since`` floor, and thinner still right after a
    model change) would otherwise be demoted to the flat table, throwing away
    weeks of accumulation on the strength of one under-supported window.
    """
    live = live or {}
    return {**(flat or {}), **live, **_accumulate_bias(live, residual or {})}


def _model_fingerprint(tree_data: dict) -> str:
    """Stable digest of the serving trees, to tell a real retrain from a re-export."""
    payload = pickle.dumps(
        (tree_data.get("baseline"), tree_data.get("learning_rate"), tree_data.get("trees")),
        protocol=4,
    )
    return hashlib.sha256(payload).hexdigest()


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
      16 stationary_sec       (seconds since the vehicle last advanced >25 m)
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
    # Date the serving trees were first published. The live bands below describe
    # whatever model produced the residuals they are pooled from, so calibrating
    # across a model change subtracts the previous model's correction from the
    # new one — measured 2026-08-13, when live bias doubled to +28s the day after
    # a retrain. Stamped when new trees are uploaded and read back on every
    # band-only refresh, so the pool self-restricts to days this model served.
    model_since: str | None = None
    # What is already live, so the bias correction can build on it — see
    # _accumulate_bias.
    live_model: dict = {}
    try:
        live_model = pickle.loads(
            client.get_object(Bucket=R2_BUCKET, Key=MODEL_KEY)["Body"].read()
        )
    except Exception:  # noqa: BLE001 — first ever export has nothing to read
        live_model = {}

    model_path = Path(MODEL_PATH)
    if model_path.exists():
        import joblib
        pipeline = joblib.load(model_path)
        tree_data = _extract_trees(pipeline)
        fingerprint = _model_fingerprint(tree_data)
        if fingerprint and fingerprint == live_model.get("model_fingerprint"):
            # Same trees as those already live: a hand `make export`, or a band
            # refresh from a machine that happens to hold the pkl. Stamping
            # today here would restrict every live pool below to a model change
            # that never happened, starving the bands of history — and the bias
            # loop of the days it still has to absorb.
            model_since = live_model.get("model_since")
        else:
            model_since = date.today().isoformat()
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
            existing_model = live_model or {}
            tree_data = {
                k: existing_model[k]
                for k in ("route_to_int", "baseline", "learning_rate", "trees")
            }
            model_since = existing_model.get("model_since")
            fingerprint = existing_model.get("model_fingerprint") or _model_fingerprint(tree_data)
            print(f"  Model not found at {model_path} — reusing trees already live "
                  "on R2, refreshing live-calibrated bands only")
            if model_since:
                print(f"  Live bands will pool only days from {model_since} "
                      "(when these trees went live)")
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
        unc_table, dates_used = live_uncertainty_by_horizon(days=7, since=model_since)
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
    bias_table: dict | None = live_model.get("bias_by_horizon")
    bias_through: str | None = live_model.get("bias_calibrated_through")
    try:
        from src.scoring import live_bias_by_horizon

        # Only days this model served AND that the live table has not already
        # absorbed — folding a day in twice would correct for it twice.
        bias_floor = max([d for d in (model_since, _day_after(bias_through)) if d], default=None)
        residual, bias_dates = live_bias_by_horizon(days=7, since=bias_floor)
        if residual:
            span = f"{bias_dates[0]}..{bias_dates[-1]}" if bias_dates else "?"
            bias_table = _accumulate_bias(bias_table, residual)
            bias_through = bias_dates[-1] if bias_dates else bias_through
            print(f"  Bias residual (live, {len(bias_dates)} days {span}): {residual}")
            print(f"  Bias correction (accumulated): {bias_table}")
        else:
            print(f"  No unabsorbed bias residual since {bias_floor} — "
                  f"keeping the live correction: {bias_table}")
    except Exception as exc:  # noqa: BLE001 — calibration must never block an export
        print(f"  WARNING: live bias calibration failed: {exc!r}")

    if bias_table:
        tree_data["bias_by_horizon"] = bias_table
        if bias_through:
            tree_data["bias_calibrated_through"] = bias_through
    else:
        print("  No live bias correction available — serving uncorrected predictions")

    # Weekday/weekend split (found 2026-07-25: weekday rush bias runs ~2-3x
    # weekend rush bias — a single blended number under-corrects one and
    # over-corrects the other). 14-day window (vs the flat table's 7) since
    # weekend days are ~2/7 of the week and need more history for the same
    # per-horizon support. Each bucket is merged over the flat table so a
    # horizon too thin on just-weekday or just-weekend data still gets the
    # blended correction rather than none.
    bias_weekend_table: dict | None = live_model.get("bias_by_horizon_weekend")
    try:
        from src.scoring import live_bias_by_horizon_weekend
        bias_weekend, bias_weekend_dates = live_bias_by_horizon_weekend(
            days=14, since=bias_floor
        )
        if bias_weekend and (bias_weekend.get("weekday") or bias_weekend.get("weekend")):
            span = (
                f"{bias_weekend_dates[0]}..{bias_weekend_dates[-1]}"
                if bias_weekend_dates else "?"
            )
            print(f"  Bias residual (weekday/weekend, {len(bias_weekend_dates)} days {span}): "
                  f"{bias_weekend}")
            # These buckets take precedence over the flat table in
            # src/inference.run_inference, so they must accumulate too —
            # publishing a raw residual here would quietly override the
            # accumulated correction on whichever bucket it covers.
            live_weekend = live_model.get("bias_by_horizon_weekend") or {}
            merged = {
                bucket: _merge_weekend_bias(
                    bias_table, live_weekend.get(bucket), bias_weekend.get(bucket)
                )
                for bucket in ("weekday", "weekend")
            }
            print(f"  Bias correction (weekday/weekend, accumulated): {merged}")
            bias_weekend_table = merged
    except Exception as exc:  # noqa: BLE001 — calibration must never block an export
        print(f"  WARNING: live weekday/weekend bias calibration failed: {exc!r}")

    # Carry the split forward when this run folded in nothing new — otherwise a
    # refresh with no unabsorbed residual would drop the key and silently demote
    # every prediction to the flat table.
    if bias_weekend_table:
        tree_data["bias_by_horizon_weekend"] = bias_weekend_table

    # Carry the serving date forward so the next band-only refresh knows which
    # scored days this model actually produced (see model_since above), and the
    # digest that says whether the next export is looking at the same trees.
    if model_since:
        tree_data["model_since"] = model_since
    if fingerprint:
        tree_data["model_fingerprint"] = fingerprint

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
