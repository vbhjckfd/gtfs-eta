"""
Real-time ETA prediction.

Consumes a live GTFS-RT snapshot, infers trip IDs (including off-route detection),
and returns predicted arrival ISO timestamps for the next N stops per vehicle.

Output format per vehicle:
    {
      "vehicle_id": "...",
      "route_id":   "A46",
      "lat": ..., "lon": ..., "bearing": ...,
      "status":     "on_route" | "off_route",
      "predictions": [{"stop_id": ..., "stop_sequence": ..., "predicted_arrival_iso": ...}]
      // predictions is [] when status == "off_route"
    }
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from google.protobuf.message import DecodeError
from google.transit import gtfs_realtime_pb2

from src.baseline import compute_eta
from src.features import FEATURE_COLS, compute_features_for_inference
from src.gtfs_static import GTFSStatic, _project_xy, get_gtfs
from src.train import MODEL_PATH, load_model, predict as pipeline_predict
from src.trip_inference import (
    RouteStatus,
    VehicleRouteTracker,
    _bearing_diff_for_shape,
    infer_trip_for_vehicle,
    score_trip,
    MAX_DIST_M,
)

load_dotenv()

GTFS_RT_URL = os.environ.get("GTFS_RT_URL", "https://track.ua-gis.com/gtfs/lviv/vehicle_position")
MAX_STOPS_AHEAD = 10

# Module-level persistent state — survives across repeated calls to predict_all()
# so the off-route state machine sees a real history of snapshots.
_live_trackers: dict[str, VehicleRouteTracker] = {}
_live_prev_trips: dict[str, str | None] = {}


def fetch_live_feed() -> gtfs_realtime_pb2.FeedMessage:
    for attempt in range(5):
        try:
            r = requests.get(GTFS_RT_URL, timeout=10)
            r.raise_for_status()
            feed = gtfs_realtime_pb2.FeedMessage()
            feed.ParseFromString(r.content)
            return feed
        except (requests.exceptions.RequestException, DecodeError) as exc:
            if attempt == 4:
                raise
            time.sleep(0.2 * (2 ** attempt))
            print(f"[warn] VP fetch attempt {attempt + 1} failed: {exc}, retrying…", flush=True)
    raise RuntimeError("unreachable")


def parse_feed(feed: gtfs_realtime_pb2.FeedMessage) -> list[dict]:
    """Parse a FeedMessage into a list of vehicle state dicts."""
    ts = datetime.fromtimestamp(feed.header.timestamp, tz=timezone.utc)
    vehicles = []
    for entity in feed.entity:
        if not entity.HasField("vehicle"):
            continue
        v = entity.vehicle
        pos = v.position if v.HasField("position") else None
        trip = v.trip if v.HasField("trip") else None
        vehicles.append({
            "timestamp": ts,
            "vehicle_id": v.vehicle.id if v.HasField("vehicle") else entity.id,
            "trip_id": trip.trip_id if trip else None,
            "route_id": trip.route_id if trip else None,
            "lat": pos.latitude if pos else None,
            "lon": pos.longitude if pos else None,
            "bearing": pos.bearing if pos else None,
            "speed": pos.speed if pos else None,
        })
    return vehicles


def _current_stop_sequence(
    trip_id: str,
    lat: float,
    lon: float,
    gtfs: GTFSStatic,
) -> tuple[int, float]:
    """Return (last_passed_stop_sequence, current_delay_sec)."""
    from shapely.geometry import Point
    trip = gtfs.get_trip(trip_id)
    if trip is None:
        return 0, 0.0
    shape = gtfs.get_shape_linestring(trip.shape_id)
    if shape is None:
        return 0, 0.0

    vx, vy = _project_xy(lon, lat)
    vehicle_dist = shape.project(Point(vx, vy))

    last_seq = 0
    for st in trip.stop_times:
        d = gtfs.get_stop_distance_along_shape(trip.shape_id, st.stop_id) or 0.0
        if d <= vehicle_dist:
            last_seq = st.stop_sequence
    return last_seq, 0.0


def _check_off_route(
    vid: str,
    route_id: str,
    vx: float,
    vy: float,
    bearing: float | None,
    now: datetime,
    gtfs: GTFSStatic,
    best_trip_id: str | None,
    best_dist: float,
) -> RouteStatus:
    """Update persistent off-route tracker for one vehicle and return status."""
    tracker = _live_trackers.setdefault(vid, VehicleRouteTracker())

    # Compute bearing diff against best shape for the tracker
    b_diff: float | None = None
    if best_trip_id and bearing is not None:
        trip = gtfs.get_trip(best_trip_id)
        if trip:
            shape = gtfs.get_shape_linestring(trip.shape_id)
            if shape:
                b_diff = _bearing_diff_for_shape(shape, vx, vy, bearing)

    # min_route_dist: use best_dist if it's real, else query active candidates
    min_route_dist = best_dist
    if min_route_dist > MAX_DIST_M:
        # Score wasn't computed (fast-path miss or no candidates) — find nearest shape
        from math import inf
        from shapely.geometry import Point
        for tid in gtfs.active_trips_for_route(route_id, now):
            trip = gtfs.get_trip(tid)
            if trip is None:
                continue
            shape = gtfs.get_shape_linestring(trip.shape_id)
            if shape is not None:
                d = shape.distance(Point(vx, vy))
                min_route_dist = min(min_route_dist, d)

    return tracker.update(min_route_dist, b_diff)


def predict_all(
    vehicles: list[dict] | None = None,
    gtfs: GTFSStatic | None = None,
    model_path: str | Path = MODEL_PATH,
    use_ml: bool = True,
) -> list[dict]:
    """
    Predict ETAs for all vehicles in the live feed.

    Returns a list of per-vehicle dicts:
        {vehicle_id, route_id, lat, lon, bearing, status, predictions}
    """
    if vehicles is None:
        feed = fetch_live_feed()
        vehicles = parse_feed(feed)

    if gtfs is None:
        gtfs = get_gtfs()

    model = load_model(model_path) if use_ml and Path(model_path).exists() else None
    known_trips = set(gtfs._trip_index.keys())

    results: list[dict] = []

    for v in vehicles:
        vid = str(v["vehicle_id"])
        route_id = str(v["route_id"]) if v.get("route_id") else None
        lat = v.get("lat")
        lon = v.get("lon")

        if not route_id or lat is None or lon is None:
            continue

        ts = v["timestamp"]
        now = ts if isinstance(ts, datetime) else ts.to_pydatetime()
        vx, vy = _project_xy(float(lon), float(lat))
        bearing = float(v["bearing"]) if v.get("bearing") is not None else None

        # --- Trip inference ---
        infer = infer_trip_for_vehicle(
            vehicle_id=vid,
            route_id=route_id,
            reported_trip_id=v.get("trip_id"),
            lat=float(lat),
            lon=float(lon),
            bearing=bearing,
            now=now,
            gtfs=gtfs,
            previous_trip_id=_live_prev_trips.get(vid),
        )
        trip_id = infer.inferred_trip_id
        best_dist = (1.0 - infer.score / 1.0) if infer.high_confidence else 200.0
        # Re-derive best_dist from the inference score (score = spatial_dist/MAX_DIST)
        # Use raw dist from score: score only meaningful on slow path, so query directly
        best_dist_raw = infer.score * MAX_DIST_M if infer.high_confidence else 999.0

        # --- Off-route detection (persistent across calls) ---
        route_status = _check_off_route(
            vid=vid,
            route_id=route_id,
            vx=vx,
            vy=vy,
            bearing=bearing,
            now=now,
            gtfs=gtfs,
            best_trip_id=trip_id,
            best_dist=best_dist_raw,
        )

        is_off_route = route_status == RouteStatus.OFF_ROUTE
        _live_prev_trips[vid] = trip_id if not is_off_route else None

        base = {
            "vehicle_id": vid,
            "route_id": route_id,
            "lat": float(lat),
            "lon": float(lon),
            "bearing": bearing,
            "status": route_status.value,
            "predictions": [],
        }

        if is_off_route or not trip_id:
            results.append(base)
            continue

        # --- ETA computation ---
        current_seq, current_delay = _current_stop_sequence(trip_id, float(lat), float(lon), gtfs)

        if model is not None:
            feat_df = compute_features_for_inference(
                vehicle_id=vid,
                trip_id=trip_id,
                current_stop_sequence=current_seq,
                current_delay_sec=current_delay,
                snapshot_time=ts,
                recent_speed_mps=float(v["speed"]) if v.get("speed") is not None else None,
                gtfs=gtfs,
            )
            if not feat_df.empty:
                feat_df = feat_df.head(MAX_STOPS_AHEAD)
                preds_sec = pipeline_predict(model, feat_df[FEATURE_COLS])
                snap_ts = pd.Timestamp(ts)
                base["predictions"] = [
                    {
                        "stop_id": row["stop_id"],
                        "stop_sequence": int(row["stop_sequence"]),
                        "predicted_arrival_iso": (
                            snap_ts + pd.Timedelta(seconds=float(preds_sec[i]))
                        ).isoformat(),
                    }
                    for i, (_, row) in enumerate(feat_df.iterrows())
                ]
        else:
            # Baseline fallback
            etas = compute_eta(trip_id, float(lat), float(lon), now, gtfs)
            base["predictions"] = [
                {
                    "stop_id": e.stop_id,
                    "stop_sequence": e.stop_sequence,
                    "predicted_arrival_iso": (
                        e.estimated_arrival.isoformat() if e.estimated_arrival else None
                    ),
                }
                for e in etas[:MAX_STOPS_AHEAD]
            ]

        results.append(base)

    return results


def reset_live_state() -> None:
    """Clear persistent tracker state — useful for testing."""
    _live_trackers.clear()
    _live_prev_trips.clear()


if __name__ == "__main__":
    import json
    predictions = predict_all()
    on_route = sum(1 for p in predictions if p["status"] == "on_route")
    off_route = sum(1 for p in predictions if p["status"] == "off_route")
    print(f"Vehicles: {len(predictions)} total, {on_route} on-route, {off_route} off-route")
    print(json.dumps(predictions[:3], indent=2, default=str))
