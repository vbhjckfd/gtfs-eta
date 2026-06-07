"""
Pure-Python GTFS-RT ETA inference for the worker compact data format.

Compatible with the data produced by scripts/export_worker_data.py:
  shapes        shape_id → bytes  (struct-packed float64 UTM pairs)
  trip_index    trip_id  → {route_id, shape_id, stop_times: [(stop_id, seq, sched_sec), ...]}
  stop_distances (shape_id, stop_id) → float metres along shape
  shape_lengths  shape_id → float
  route_trips    route_id → [trip_id, ...]
  model          {route_to_int, baseline, learning_rate, trees}

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
    lr           = model_data["learning_rate"]
    trees        = model_data["trees"]
    preds = []
    for row in rows:
        route_int = float(route_to_int.get(str(row[0]), -1))
        feat = [route_int] + [float(v) for v in row[1:]]
        total = baseline
        for tree_nodes in trees:
            total += lr * _traverse_tree(tree_nodes, feat)
        preds.append(total)
    return preds


# ---------------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------------

def current_stop_seq(trip_id: str, vx: float, vy: float, data: dict) -> int:
    trip = data["trip_index"].get(trip_id)
    if trip is None:
        return 0
    coords = data["shapes"].get(trip["shape_id"])
    if coords is None:
        return 0
    v_dist = poly_project(coords, vx, vy)
    last_seq = 0
    for st in trip["stop_times"]:
        if data["stop_distances"].get((trip["shape_id"], st[0]), 0.0) <= v_dist:
            last_seq = st[1]
    return last_seq


def build_features(trip_id: str, current_seq: int, current_delay: float,
                   snap_ts: datetime, data: dict) -> list:
    """Returns [(feat_row, stop_id, stop_sequence), ...] for upcoming stops."""
    trip = data["trip_index"].get(trip_id)
    if trip is None:
        return []
    shape_len = data["shape_lengths"].get(trip["shape_id"], 1.0)
    d = snap_ts.date()
    is_holiday = int((d.month, d.day) in _UA_HOLIDAYS)
    d_current = 0.0
    for st in trip["stop_times"]:
        if st[1] == current_seq:
            d_current = data["stop_distances"].get((trip["shape_id"], st[0]), 0.0)
            break
    result = []
    for i, st in enumerate(trip["stop_times"]):
        if st[1] <= current_seq:
            continue
        d_target  = data["stop_distances"].get((trip["shape_id"], st[0]), 0.0)
        seg_dist  = max(0.0, d_target - d_current)
        sched_seg = max(0.0, st[2])
        stops_rem = len(trip["stop_times"]) - i - 1
        feat_row = [
            trip["route_id"], st[1], current_seq,
            snap_ts.hour, snap_ts.weekday(), snap_ts.month,
            int(snap_ts.weekday() >= 5), is_holiday,
            current_delay, seg_dist, sched_seg, stops_rem,
            d_target / max(shape_len, 1.0),
        ]
        result.append((feat_row, st[0], st[1]))
    return result[:MAX_STOPS_AHEAD]


# ---------------------------------------------------------------------------
# Protobuf encoder
# ---------------------------------------------------------------------------

def encode_trip_updates(updates: list[dict], feed_ts: int) -> bytes:
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.header.gtfs_realtime_version = "2.0"
    feed.header.incrementality = gtfs_realtime_pb2.FeedHeader.FULL_DATASET
    feed.header.timestamp = feed_ts
    for u in updates:
        entity = feed.entity.add()
        entity.id = u["vehicle_id"]
        tu = entity.trip_update
        tu.trip.trip_id = u["trip_id"]
        if u.get("route_id"):
            tu.trip.route_id = u["route_id"]
        tu.vehicle.id = u["vehicle_id"]
        t0 = u["snap_ts"]
        cumulative = 0.0
        stop_count = 0
        for pred in u["predictions"]:
            cumulative += pred["seconds"]
            arr_ts = int(t0.timestamp() + cumulative)
            if arr_ts <= feed_ts:
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

        current_seq   = current_stop_seq(trip_id, vx, vy, gtfs_data)
        feature_rows  = build_features(trip_id, current_seq, 0.0, snap_ts, gtfs_data)
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
