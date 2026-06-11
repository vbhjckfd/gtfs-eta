"""
Pure-Python GTFS-RT ETA inference for the worker compact data format.

Compatible with the data produced by scripts/export_worker_data.py:
  shapes        shape_id → bytes  (struct-packed float64 UTM pairs)
  trip_index    trip_id  → {route_id, shape_id,
                            stop_times: [(stop_id, seq, sched_cum_sec), ...]}
                where sched_cum_sec is scheduled seconds since the trip's
                first stop (cumulative)
  stop_distances (shape_id, stop_id) → float metres along shape
  shape_lengths  shape_id → float
  route_trips    route_id → [trip_id, ...]
  model          {route_to_int, baseline, learning_rate, trees}

The model predicts seconds_to_arrival directly per upcoming stop (multi-
horizon), anchored at the vehicle's *projected position along the shape* —
not at the last passed stop — so a bus sitting at a stop gets
remaining_dist ≈ 0 and a near-zero ETA instead of a full segment time.

No R2 / JS / Pyodide APIs — runs on standard CPython or inside Pyodide.
"""
from __future__ import annotations

import math
import struct
import time
from datetime import datetime, timezone

from google.transit import gtfs_realtime_pb2

from src.utm import project_xy

MAX_STOPS_AHEAD = 10

_UA_HOLIDAYS: set[tuple[int, int]] = {
    (1, 1), (1, 7), (3, 8), (5, 1), (5, 9),
    (6, 28), (8, 24), (10, 14), (12, 25),
}

_OFF_ROUTE_DIST = 150.0
_ON_ROUTE_DIST  = 100.0
_OFF_CONSEC     = 3
_ON_CONSEC      = 3


# ---------------------------------------------------------------------------
# Geometry — shapes stored as packed float64 bytes
# ---------------------------------------------------------------------------

def _seg_nearest(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    seg2 = dx * dx + dy * dy
    if seg2 == 0.0:
        return math.hypot(px - ax, py - ay), 0.0
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy)), t


def poly_distance(shape_bytes: bytes, px: float, py: float) -> float:
    n = len(shape_bytes) // 16
    min_d = math.inf
    for i in range(n - 1):
        ax, ay = struct.unpack_from("dd", shape_bytes, i * 16)
        bx, by = struct.unpack_from("dd", shape_bytes, (i + 1) * 16)
        d, _ = _seg_nearest(px, py, ax, ay, bx, by)
        if d < min_d:
            min_d = d
    return min_d


def poly_project(shape_bytes: bytes, px: float, py: float) -> float:
    n = len(shape_bytes) // 16
    min_d = math.inf
    best = cum = 0.0
    for i in range(n - 1):
        ax, ay = struct.unpack_from("dd", shape_bytes, i * 16)
        bx, by = struct.unpack_from("dd", shape_bytes, (i + 1) * 16)
        seg_len = math.hypot(bx - ax, by - ay)
        d, t = _seg_nearest(px, py, ax, ay, bx, by)
        if d < min_d:
            min_d = d
            best = cum + t * seg_len
        cum += seg_len
    return best


# ---------------------------------------------------------------------------
# Off-route tracker
# ---------------------------------------------------------------------------

def update_tracker(trackers: dict, vid: str, min_dist: float) -> bool:
    """Update hysteresis state for vehicle *vid*. Returns True if off-route."""
    state = trackers.setdefault(vid, {"status": "on_route", "off": 0, "on": 0})
    if min_dist > _OFF_ROUTE_DIST:
        state["off"] += 1
        state["on"] = 0
        if state["off"] >= _OFF_CONSEC:
            state["status"] = "off_route"
    else:
        state["on"] += 1
        state["off"] = 0
        if state["on"] >= _ON_CONSEC and min_dist < _ON_ROUTE_DIST:
            state["status"] = "on_route"
    return state["status"] == "off_route"


# ---------------------------------------------------------------------------
# Trip inference
# ---------------------------------------------------------------------------

def infer_trip(route_id, reported_trip_id, vx, vy, data):
    candidates = data["route_trips"].get(str(route_id), [])
    if not candidates:
        return None, 9999.0
    if reported_trip_id and reported_trip_id in data["trip_index"]:
        shape_id = data["trip_index"][reported_trip_id]["shape_id"]
        coords = data["shapes"].get(shape_id)
        if coords is not None:
            d = poly_distance(coords, vx, vy)
            if d < 75.0:
                return reported_trip_id, d
    best_id, best_dist = None, math.inf
    for tid in candidates:
        trip = data["trip_index"].get(tid)
        if trip is None:
            continue
        coords = data["shapes"].get(trip["shape_id"])
        if coords is None:
            continue
        d = poly_distance(coords, vx, vy)
        if d < best_dist:
            best_dist = d
            best_id = tid
    return best_id, best_dist


# ---------------------------------------------------------------------------
# ML inference (pure-Python GBT tree traversal)
# ---------------------------------------------------------------------------

def _traverse_tree(nodes, feat):
    idx = 0
    while True:
        f_idx, threshold, left, right, is_leaf, value = nodes[idx]
        if is_leaf:
            return value
        idx = left if feat[f_idx] <= threshold else right


def predict_rows(model_data: dict, rows: list) -> list:
    route_to_int = model_data["route_to_int"]
    baseline     = model_data["baseline"]
    trees        = model_data["trees"]
    preds = []
    for row in rows:
        route_int = float(route_to_int.get(str(row[0]), -1))
        feat = [route_int] + [float(v) for v in row[1:]]
        total = baseline
        for tree_nodes in trees:
            # HistGradientBoosting leaf values already include the learning
            # rate (shrinkage is applied at fit time), so they are summed raw.
            total += _traverse_tree(tree_nodes, feat)
        preds.append(total)
    return preds


# ---------------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------------

# Progress-speed measurement between consecutive pushes (mirrors the training
# computation in src/labeling.py — keep the constants in sync).
SPEED_UNKNOWN = -1.0
_SPEED_MIN_GAP_SEC = 3.0
_SPEED_MAX_GAP_SEC = 120.0
_SPEED_MAX_BACKWARD_M = 30.0


def vehicle_dist_along(trip_id: str, vx: float, vy: float, data: dict) -> float:
    """Vehicle's projected distance (m) along the trip's shape."""
    trip = data["trip_index"].get(trip_id)
    if trip is None:
        return 0.0
    coords = data["shapes"].get(trip["shape_id"])
    if coords is None:
        return 0.0
    return poly_project(coords, vx, vy)


def progress_speed(trackers: dict, vid: str, trip_id: str, v_dist: float,
                   ts_sec: float) -> float:
    """Speed (m/s) from the previous push's projection; SPEED_UNKNOWN when
    there is no usable history (first sighting, trip change, stale gap)."""
    state = trackers.setdefault(vid, {"status": "on_route", "off": 0, "on": 0})
    prev = state.get("pos")

    # Upstream VP can update slower than our push cadence: the same snapshot
    # gets re-served with dt ≈ 0. Keep the anchor and the last measurement
    # instead of flipping to UNKNOWN every other push.
    if (prev is not None and prev[2] == trip_id
            and ts_sec - prev[0] < _SPEED_MIN_GAP_SEC):
        return state.get("speed", SPEED_UNKNOWN)

    state["pos"] = (ts_sec, v_dist, trip_id)
    speed = SPEED_UNKNOWN
    if prev is not None:
        prev_ts, prev_dist, prev_trip = prev
        dt = ts_sec - prev_ts
        dd = v_dist - prev_dist
        if (prev_trip == trip_id
                and _SPEED_MIN_GAP_SEC <= dt <= _SPEED_MAX_GAP_SEC
                and dd >= -_SPEED_MAX_BACKWARD_M):
            speed = max(0.0, dd) / dt
    state["speed"] = speed
    return speed


def _sched_sec_at_dist(stop_dists: list, sched_cums: list, d: float) -> float:
    """Interpolated cumulative scheduled seconds at shape distance *d*.
    stop_dists/sched_cums are parallel lists sorted by distance."""
    if not stop_dists:
        return 0.0
    if d <= stop_dists[0]:
        return sched_cums[0]
    for i in range(1, len(stop_dists)):
        if d <= stop_dists[i]:
            span = stop_dists[i] - stop_dists[i - 1]
            if span <= 0:
                return sched_cums[i]
            frac = (d - stop_dists[i - 1]) / span
            return sched_cums[i - 1] + frac * (sched_cums[i] - sched_cums[i - 1])
    return sched_cums[-1]


def build_features(trip_id: str, v_dist: float, speed: float,
                   snap_ts: datetime, data: dict) -> list:
    """Returns [(feat_row, stop_id, stop_sequence), ...] for upcoming stops.

    Feature order must match FEATURE_COLS in src/features.py (the exported
    trees index features positionally).
    """
    trip = data["trip_index"].get(trip_id)
    if trip is None:
        return []
    shape_len = max(data["shape_lengths"].get(trip["shape_id"], 1.0), 1.0)
    d = snap_ts.date()
    is_holiday = int((d.month, d.day) in _UA_HOLIDAYS)

    # Per-stop (dist_along, sched_cum_sec), sorted by distance for interpolation
    sts = trip["stop_times"]  # [(stop_id, seq, sched_cum_sec), ...]
    n_stops_total = len(sts)
    entries = sorted(
        (data["stop_distances"].get((trip["shape_id"], st[0]), 0.0), st[2], st[0], st[1], i)
        for i, st in enumerate(sts)
    )
    stop_dists = [e[0] for e in entries]
    sched_cums = [e[1] for e in entries]
    sched_at_pos = _sched_sec_at_dist(stop_dists, sched_cums, v_dist)

    result = []
    stops_ahead = 0
    for d_target, sched_cum, stop_id, stop_seq, orig_idx in entries:
        if d_target <= v_dist:
            continue
        stops_ahead += 1
        if stops_ahead > MAX_STOPS_AHEAD:
            break
        feat_row = [
            trip["route_id"], stop_seq, stops_ahead,
            snap_ts.hour, snap_ts.weekday(), snap_ts.month,
            int(snap_ts.weekday() >= 5), is_holiday,
            d_target - v_dist,
            max(0.0, sched_cum - sched_at_pos),
            speed,
            n_stops_total - 1 - orig_idx,
            d_target / shape_len,
        ]
        result.append((feat_row, stop_id, stop_seq))
    return result


# ---------------------------------------------------------------------------
# Protobuf encoder
# ---------------------------------------------------------------------------

def encode_trip_updates(updates: list[dict], feed_ts: int) -> bytes:
    # Use wall-clock time for the header and staleness filter.
    # feed_ts is the VP feed's capture timestamp and can be 30–90 s behind
    # real time by the time our feed is read from R2 cache, so filtering
    # against feed_ts only removes negative predictions, not stale ones.
    now_ts = int(time.time())
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.header.gtfs_realtime_version = "2.0"
    feed.header.incrementality = gtfs_realtime_pb2.FeedHeader.FULL_DATASET
    feed.header.timestamp = now_ts
    for u in updates:
        entity = feed.entity.add()
        entity.id = u["vehicle_id"]
        tu = entity.trip_update
        tu.trip.trip_id = u["trip_id"]
        if u.get("route_id"):
            tu.trip.route_id = u["route_id"]
        tu.vehicle.id = u["vehicle_id"]
        t0 = u["snap_ts"]
        stop_count = 0
        last_arr_ts = 0
        for pred in u["predictions"]:
            # Direct multi-horizon seconds from the snapshot; clamp to keep
            # arrival times non-decreasing along the trip.
            arr_ts = max(int(t0.timestamp() + pred["seconds"]), last_arr_ts)
            last_arr_ts = arr_ts
            if arr_ts <= now_ts:
                continue
            stu = tu.stop_time_update.add()
            stu.stop_id = pred["stop_id"]
            stu.stop_sequence = pred["stop_sequence"]
            stu.arrival.time = arr_ts
            stu.departure.time = arr_ts
            stop_count += 1
        if stop_count == 0:
            del feed.entity[-1]
    return feed.SerializeToString()


# ---------------------------------------------------------------------------
# Main inference pass
# ---------------------------------------------------------------------------

def run_inference(gtfs_data: dict, model_data: dict, trackers: dict,
                  vp_bytes: bytes) -> bytes:
    """Vehicle-positions protobuf bytes → TripUpdates protobuf bytes."""
    vp_feed = gtfs_realtime_pb2.FeedMessage()
    vp_feed.ParseFromString(vp_bytes)
    feed_ts = int(vp_feed.header.timestamp) or int(time.time())
    snap_ts = datetime.fromtimestamp(feed_ts, tz=timezone.utc)

    updates = []
    for entity in vp_feed.entity:
        if not entity.HasField("vehicle"):
            continue
        v   = entity.vehicle
        pos = v.position if v.HasField("position") else None
        trp = v.trip     if v.HasField("trip")     else None
        if pos is None:
            continue

        lat, lon      = pos.latitude, pos.longitude
        route_id      = str(trp.route_id) if trp else None
        reported_tid  = str(trp.trip_id)  if trp else None
        vid           = v.vehicle.id if v.HasField("vehicle") else entity.id

        if not route_id or not lat or not lon:
            continue

        vx, vy = project_xy(lon, lat)
        trip_id, min_dist = infer_trip(route_id, reported_tid, vx, vy, gtfs_data)
        if update_tracker(trackers, vid, min_dist) or trip_id is None:
            continue

        v_dist = vehicle_dist_along(trip_id, vx, vy, gtfs_data)
        speed  = progress_speed(trackers, vid, trip_id, v_dist, float(feed_ts))
        feature_rows = build_features(trip_id, v_dist, speed, snap_ts, gtfs_data)
        if not feature_rows:
            continue

        preds_sec = predict_rows(model_data, [r[0] for r in feature_rows])
        updates.append({
            "vehicle_id": vid,
            "trip_id":    trip_id,
            "route_id":   route_id,
            "snap_ts":    snap_ts,
            "predictions": [
                {"stop_id": r[1], "stop_sequence": int(r[2]),
                 "seconds": float(sec)}
                for r, sec in zip(feature_rows, preds_sec)
            ],
        })

    return encode_trip_updates(updates, feed_ts)
