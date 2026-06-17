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
from dotenv import load_dotenv

from src.gtfs_static import get_gtfs
from src.train import MODEL_PATH

load_dotenv()

R2_ACCOUNT_ID = os.environ["R2_ACCOUNT_ID"]
R2_ACCESS_KEY_ID = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET_ACCESS_KEY = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET = os.environ.get("R2_BUCKET", "gtfs-lviv")

GTFS_KEY = "worker/gtfs_worker_data.pkl"
MODEL_KEY = "worker/eta_pipeline.pkl"


def _make_client():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
    )


def build_gtfs_worker_data(gtfs) -> dict:
    """
    Serialise GTFSStatic to a plain-dict format loadable without pyproj/pyproject.

    Scheduled times are baked into stop_times as *cumulative* seconds since
    the trip's first stop (sched_cum_sec), so inference can interpolate the
    schedule at the vehicle's projected position — matching how
    sched_remaining_sec is computed at training time (src/features.py).
    """
    print("Building worker GTFS data…")

    from src.gtfs_static import _parse_gtfs_time
    from datetime import date
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
            "stop_times": stop_times,
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

    data = {
        "shapes": shapes_coords,                          # shape_id → bytes (packed float64 pairs)
        "shape_lengths": dict(gtfs._shape_lengths),       # shape_id → float metres
        "stop_distances": dict(gtfs._stop_distances),     # (shape_id, stop_id) → float
        "trip_index": trip_index,
        "route_trips": dict(gtfs._route_trips),           # route_id → [trip_id, ...]
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
      13 speed_eta_sec        (remaining_dist_m / speed; -1 when speed unknown)
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

    # ── GTFS ──
    print("Loading GTFS static…")
    gtfs = get_gtfs(force_download=True, force_rebuild=True)
    worker_data = build_gtfs_worker_data(gtfs)

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
    if not model_path.exists():
        print(f"Model not found at {model_path} — skipping model upload")
        return

    import joblib
    pipeline = joblib.load(model_path)
    tree_data = _extract_trees(pipeline)
    model_bytes = pickle.dumps(tree_data, protocol=4)
    size_mb = len(model_bytes) / 1e6
    print(f"Uploading model trees ({size_mb:.1f} MB) → R2:{MODEL_KEY}")
    client.put_object(Bucket=R2_BUCKET, Key=MODEL_KEY, Body=model_bytes)
    print("  done")

    print("\nAll artifacts uploaded. Deploy with: make deploy")


if __name__ == "__main__":
    main()
