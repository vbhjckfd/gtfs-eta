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
_BEARING_WRONG_DEG = 90.0   # heading vs shape-tangent diff that looks off-route

# Trip-matching thresholds (bring the compact serving path in line with
# src/trip_inference.py, which both training and scoring use).  Bearing
# disambiguates overlapping opposite-direction shape variants that pure
# nearest-distance matching cannot tell apart.
_MATCH_DIST_CAP   = 100.0   # spatial-score normaliser
_REPORTED_DIST_OK = 75.0    # trust a reported trip_id within this distance
_SPATIAL_W = 0.6
_BEARING_W = 0.4

# Recent-motion heading (issue #3 live-side direction fix). When a vehicle has
# moved at least this far since the previous snapshot, its travel direction —
# not the noisy reported bearing — drives trip matching; below it the vehicle is
# treated as stationary and its prior trip's direction is held (see
# run_inference). This is the live, look-back-only analogue of
# src/trip_inference._motion_headings (which can also look ahead).
_MOTION_MIN_MOVE_M = 25.0
_HOLD_DIST_M       = 150.0  # max shape distance to hold a stationary vehicle's prior trip

# Idling-at-origin guard: a vehicle parked at (≈) the shape start with no
# measurable forward motion is almost always waiting for its scheduled
# departure.  Predicting then yields optimistically early ETAs (the warm-start
# falls back to historical speed), so we withhold predictions until it moves.
_NOT_DEPARTED_DIST_M = 20.0

# Routes with confirmed trip-matching failures — excluded from training in
# src/train.py and suppressed here so their bad predictions don't reach riders
# or pollute the quality scorer.
_BAD_ROUTE_IDS: frozenset[str] = frozenset({"2299", "138"})


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


def _seg_bearing(ax, ay, bx, by) -> float:
    """Compass bearing (deg from North, projected coords) of segment a→b."""
    return math.degrees(math.atan2(bx - ax, by - ay)) % 360.0


def _bearing_diff(b1: float, b2: float) -> float:
    """Absolute angular difference in [0, 180]."""
    diff = abs(b1 - b2) % 360.0
    return diff if diff <= 180.0 else 360.0 - diff


def poly_match(shape_bytes: bytes, px: float, py: float) -> tuple[float, float]:
    """Nearest distance to the polyline and the shape's tangent bearing there.

    The tangent points in the trip's direction of travel (shapes are ordered
    start→end), so comparing it to the vehicle's heading tells the two
    directions of an overlapping route apart.
    """
    n = len(shape_bytes) // 16
    min_d = math.inf
    tangent = 0.0
    for i in range(n - 1):
        ax, ay = struct.unpack_from("dd", shape_bytes, i * 16)
        bx, by = struct.unpack_from("dd", shape_bytes, (i + 1) * 16)
        d, _ = _seg_nearest(px, py, ax, ay, bx, by)
        if d < min_d:
            min_d = d
            tangent = _seg_bearing(ax, ay, bx, by)
    return min_d, tangent


# ---------------------------------------------------------------------------
# Off-route tracker
# ---------------------------------------------------------------------------

def update_tracker(
    trackers: dict, vid: str, min_dist: float, bearing_diff: float | None = None
) -> bool:
    """Update hysteresis state for vehicle *vid*. Returns True if off-route.

    Mirrors src/trip_inference.py's VehicleRouteTracker: a snapshot looks off
    if it is spatially distant, *or* heading the wrong way down the shape, *or*
    steadily moving away — any of which, sustained for _OFF_CONSEC snapshots,
    flips the vehicle to off-route.
    """
    state = trackers.setdefault(vid, {"status": "on_route", "off": 0, "on": 0})
    dists = state.setdefault("dists", [])
    dists.append(min_dist)
    if len(dists) > 5:
        del dists[0]

    spatially_off = min_dist > _OFF_ROUTE_DIST
    bearing_wrong = (
        bearing_diff is not None
        and bearing_diff > _BEARING_WRONG_DEG
        and min_dist > 50.0          # ignore stationary / slow vehicles
    )
    moving_away = (
        len(dists) >= 3
        and dists[-1] > dists[-2] > dists[-3]
        and min_dist > _ON_ROUTE_DIST
    )
    looks_off = spatially_off or bearing_wrong or moving_away

    if state["status"] == "on_route":
        if looks_off:
            state["off"] += 1
            state["on"] = 0
        else:
            state["off"] = 0
            state["on"] += 1
        if state["off"] >= _OFF_CONSEC:
            state["status"] = "off_route"
            state["off"] = 0
            state["on"] = 0
    else:
        if min_dist <= _ON_ROUTE_DIST:
            state["on"] += 1
            state["off"] = 0
        else:
            state["on"] = 0
            state["off"] += 1
        if state["on"] >= _ON_CONSEC:
            state["status"] = "on_route"
            state["off"] = 0
            state["on"] = 0
    return state["status"] == "off_route"


# ---------------------------------------------------------------------------
# Trip inference
# ---------------------------------------------------------------------------

def _bearing_score(bearing: float | None, tangent: float) -> float:
    """Bearing penalty in [0, 1]; neutral (0.5) when no heading is reported."""
    if bearing is None:
        return 0.5
    return _bearing_diff(bearing, tangent) / 180.0


def infer_trip(route_id, reported_trip_id, vx, vy, bearing, data):
    """Best (trip_id, spatial_dist, tangent_bearing) for a vehicle snapshot.

    Combines spatial distance with heading alignment so opposite-direction
    shape variants — which sit on top of each other and defeat pure
    nearest-distance matching — are told apart.
    """
    candidates = data["route_trips"].get(str(route_id), [])
    if not candidates:
        return None, 9999.0, 0.0

    # Fast path: trust the reported trip only if it is both near AND not headed
    # the wrong way down its shape.
    if reported_trip_id and reported_trip_id in data["trip_index"]:
        shape_id = data["trip_index"][reported_trip_id]["shape_id"]
        coords = data["shapes"].get(shape_id)
        if coords is not None:
            d, tangent = poly_match(coords, vx, vy)
            if d < _REPORTED_DIST_OK and (
                bearing is None
                or _bearing_diff(bearing, tangent) <= _BEARING_WRONG_DEG
            ):
                return reported_trip_id, d, tangent

    best_id, best_dist, best_tangent, best_score = None, math.inf, 0.0, math.inf
    for tid in candidates:
        trip = data["trip_index"].get(tid)
        if trip is None:
            continue
        coords = data["shapes"].get(trip["shape_id"])
        if coords is None:
            continue
        d, tangent = poly_match(coords, vx, vy)
        score = (
            _SPATIAL_W * min(d / _MATCH_DIST_CAP, 5.0)
            + _BEARING_W * _bearing_score(bearing, tangent)
        )
        if score < best_score:
            best_score, best_dist, best_id, best_tangent = score, d, tid, tangent
    return best_id, best_dist, best_tangent


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

    sts = trip["stop_times"]  # [(stop_id, seq, sched_cum_sec), ...]
    n_stops_total = len(sts)
    entries = sorted(
        (data["stop_distances"].get((trip["shape_id"], st[0]), 0.0), st[0], st[1], i)
        for i, st in enumerate(sts)
    )

    # Route+hour priors for warm-started ETA and dwell-aware estimate.
    route_id = trip["route_id"]
    priors   = data.get("route_hour_priors", {})
    _fallback = priors.get("_global", (5.0, 40.0))
    hist_speed, hist_tps = priors.get(f"{route_id}:{snap_ts.hour}", _fallback)

    result = []
    stops_ahead = 0
    for d_target, stop_id, stop_seq, orig_idx in entries:
        if d_target <= v_dist:
            continue
        stops_ahead += 1
        if stops_ahead > MAX_STOPS_AHEAD:
            break
        rem_dist  = d_target - v_dist
        eff_speed = speed if speed > 0.0 else hist_speed
        feat_row = [
            route_id, stop_seq, stops_ahead,
            snap_ts.hour, snap_ts.weekday(), snap_ts.month,
            int(snap_ts.weekday() >= 5), is_holiday,
            rem_dist,                                    # idx 8  remaining_dist_m
            speed,                                       # idx 9  progress_speed_mps
            n_stops_total - 1 - orig_idx,               # idx 10 stops_remaining
            d_target / shape_len,                        # idx 11 trip_progress_frac
            rem_dist / max(1, stops_ahead),              # idx 12 dist_per_stop_m
            rem_dist / max(eff_speed, 0.1),              # idx 13 speed_eta_warm
            hist_speed,                                  # idx 14 hist_speed_mps
            stops_ahead * hist_tps,                      # idx 15 hist_travel_time_est
        ]
        result.append((feat_row, stop_id, stop_seq))
    return result


# ---------------------------------------------------------------------------
# Protobuf encoder
# ---------------------------------------------------------------------------

def _isotonic(values: list[float]) -> list[float]:
    """Least-squares non-decreasing fit (pool-adjacent-violators).

    The per-stop predictions are independent multi-horizon outputs and can come
    back slightly out of order.  A plain running max only ever pushes the
    offending stop *later*, biasing far horizons late; PAVA instead distributes
    the correction across the violating run, so the monotone sequence stays
    centred on the model's mean — no systematic late bias.
    """
    blocks: list[list[float]] = []  # each: [sum, count]
    for v in values:
        blocks.append([v, 1.0])
        while len(blocks) >= 2 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s2, c2 = blocks.pop()
            s1, c1 = blocks.pop()
            blocks.append([s1 + s2, c1 + c2])
    out: list[float] = []
    for s, c in blocks:
        out.extend([s / c] * int(c))
    return out


def _uncertainty_for(table: dict | None, horizon: int) -> int | None:
    """Per-horizon prediction uncertainty (seconds), or None when unavailable.

    *table* maps a prediction horizon (stops ahead, 1-based) to a ± band in
    seconds — the model's per-horizon test-set MAE, baked in at training time.
    Error grows with horizon, so horizons past the largest measured key reuse
    that last (widest) band rather than dropping the field.
    """
    if not table:
        return None
    v = table.get(horizon)
    if v is None:
        keys = [k for k in table if isinstance(k, int)]
        if not keys:
            return None
        v = table[min(max(keys), horizon)] if horizon < min(keys) else table[max(keys)]
    return int(v)


def encode_trip_updates(
    updates: list[dict], feed_ts: int, uncertainty_by_horizon: dict | None = None
) -> bytes:
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
        tu.trip.schedule_relationship = gtfs_realtime_pb2.TripDescriptor.SCHEDULED
        if u.get("route_id"):
            tu.trip.route_id = u["route_id"]
        tu.vehicle.id = u["vehicle_id"]
        t0 = u["snap_ts"]
        stop_count = 0
        last_arr_ts = 0
        # Enforce non-decreasing arrival times with an isotonic fit rather than a
        # one-sided running max (which biases far stops late).
        iso_secs = _isotonic([p["seconds"] for p in u["predictions"]])
        for pred, sec in zip(u["predictions"], iso_secs):
            # Direct multi-horizon seconds from the snapshot; the final max only
            # repairs integer-rounding ties (the isotonic fit is already monotone).
            arr_ts = max(int(t0.timestamp() + sec), last_arr_ts)
            last_arr_ts = arr_ts
            if arr_ts <= now_ts:
                continue
            stu = tu.stop_time_update.add()
            stu.stop_id = pred["stop_id"]
            stu.stop_sequence = pred["stop_sequence"]
            stu.schedule_relationship = gtfs_realtime_pb2.TripUpdate.StopTimeUpdate.SCHEDULED
            stu.arrival.time = arr_ts
            stu.departure.time = arr_ts + _DWELL_SECS
            # Publish the model's confidence so consumers can widen the window
            # for far-horizon stops. Keyed by the true horizon when carried,
            # else by emitted position (matches the scorer's stops_ahead proxy).
            unc = _uncertainty_for(
                uncertainty_by_horizon, pred.get("stops_ahead", stop_count + 1)
            )
            if unc is not None:
                stu.arrival.uncertainty = unc
                stu.departure.uncertainty = unc
            stop_count += 1
        if stop_count == 0:
            del feed.entity[-1]
    return feed.SerializeToString()


# Vehicle is treated as dwelling AT its next stop (rather than IN_TRANSIT_TO it)
# once it is within this many metres of it along the shape.
_STOPPED_AT_RADIUS_M = 25.0
# Vehicle is flagged INCOMING_AT (imminent arrival) within this radius; beyond
# it the status is IN_TRANSIT_TO.
_INCOMING_AT_RADIUS_M = 150.0

# Typical stop dwell so departure ≠ arrival.  Fixed per-stop; no dwell model yet.
_DWELL_SECS = 15


def encode_vehicle_positions(records: list[dict], feed_ts: int) -> bytes:
    """Encode cleaned vehicle positions as a GTFS-RT VehiclePositions feed.

    Re-publishes the upstream positions enriched with this project's corrected
    trip match, the next stop and stop status, and a congestion estimate — the
    by-products of the same inference pass that builds TripUpdates, which the
    operator's own VehiclePositions feed lacks (it reports the raw, sometimes
    wrong, trip_id and no stop progress).
    """
    now_ts = int(time.time())
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.header.gtfs_realtime_version = "2.0"
    feed.header.incrementality = gtfs_realtime_pb2.FeedHeader.FULL_DATASET
    feed.header.timestamp = now_ts
    for r in records:
        entity = feed.entity.add()
        entity.id = r["vehicle_id"]
        vp = entity.vehicle
        vp.trip.trip_id = r["trip_id"]
        vp.trip.schedule_relationship = gtfs_realtime_pb2.TripDescriptor.SCHEDULED
        if r.get("route_id"):
            vp.trip.route_id = r["route_id"]
        vp.vehicle.id = r["vehicle_id"]
        vp.position.latitude = r["lat"]
        vp.position.longitude = r["lon"]
        if r.get("bearing") is not None:
            vp.position.bearing = r["bearing"]
        if r.get("speed") is not None:
            vp.position.speed = r["speed"]
        if r.get("stop_id") is not None:
            vp.stop_id = r["stop_id"]
            vp.current_stop_sequence = r["stop_sequence"]
            vp.current_status = r["status"]
        if r.get("congestion") is not None:
            vp.congestion_level = r["congestion"]
        # The position's own capture time, not the (later) feed publish time.
        vp.timestamp = feed_ts
    return feed.SerializeToString()


def _congestion_level(speed: float, hist_speed: float):
    """Map observed-vs-historical speed to a GTFS-RT CongestionLevel.

    Returns None when speed is unknown (-1) or the vehicle is stationary, so we
    never assert "smooth" from a single missing measurement.
    """
    if speed <= 0.0 or hist_speed <= 0.0:
        return None
    ratio = speed / hist_speed
    VP = gtfs_realtime_pb2.VehiclePosition
    if ratio >= 0.7:
        return VP.RUNNING_SMOOTHLY
    if ratio >= 0.4:
        return VP.STOP_AND_GO
    if ratio >= 0.2:
        return VP.CONGESTION
    return VP.SEVERE_CONGESTION


# ---------------------------------------------------------------------------
# Main inference pass
# ---------------------------------------------------------------------------

def run_inference(gtfs_data: dict, model_data: dict, trackers: dict,
                  vp_bytes: bytes, *, with_vehicle_positions: bool = False):
    """Vehicle-positions protobuf bytes → TripUpdates protobuf bytes.

    With ``with_vehicle_positions=True`` returns a ``(trip_updates_bytes,
    vehicle_positions_bytes)`` tuple — the second feed re-publishes the cleaned
    positions (corrected trip, next stop, congestion) computed in this same
    pass. The default single-bytes return keeps existing callers unchanged.
    """
    vp_feed = gtfs_realtime_pb2.FeedMessage()
    vp_feed.ParseFromString(vp_bytes)
    feed_ts = int(vp_feed.header.timestamp) or int(time.time())
    snap_ts = datetime.fromtimestamp(feed_ts, tz=timezone.utc)
    priors = gtfs_data.get("route_hour_priors", {})

    updates = []
    vp_records: list[dict] = []
    for entity in vp_feed.entity:
        if not entity.HasField("vehicle"):
            continue
        v   = entity.vehicle
        pos = v.position if v.HasField("position") else None
        trp = v.trip     if v.HasField("trip")     else None
        if pos is None:
            continue

        lat, lon      = pos.latitude, pos.longitude
        bearing       = float(pos.bearing) if pos.HasField("bearing") else None
        gps_speed     = float(pos.speed)   if pos.HasField("speed")   else None
        route_id      = str(trp.route_id) if trp else None
        reported_tid  = str(trp.trip_id)  if trp else None
        vid           = v.vehicle.id if v.HasField("vehicle") else entity.id

        if not route_id or not lat or not lon:
            continue
        if route_id in _BAD_ROUTE_IDS:
            continue

        vx, vy = project_xy(lon, lat)

        # Recent-motion heading: derive direction from displacement since the last
        # snapshot (reliable even when a stopped vehicle reports bearing 0/noise),
        # falling back to the reported bearing only when it hasn't moved.
        state = trackers.setdefault(vid, {"status": "on_route", "off": 0, "on": 0})
        prev_xy  = state.get("xy")
        prev_tid = state["pos"][2] if state.get("pos") else None
        moving = (
            prev_xy is not None
            and math.hypot(vx - prev_xy[0], vy - prev_xy[1]) >= _MOTION_MIN_MOVE_M
        )
        heading = _seg_bearing(prev_xy[0], prev_xy[1], vx, vy) if moving else bearing
        state["xy"] = (vx, vy)

        trip_id, min_dist, tangent = infer_trip(
            route_id, reported_tid, vx, vy, heading, gtfs_data
        )
        if trip_id is None:
            continue

        # Defer direction while stationary: a stopped vehicle's bearing can't tell
        # two opposite-direction shapes apart, so don't let it flip to a reversed
        # trip — hold the previously matched trip while it stays on that shape. It
        # picks the real direction once it moves (`moving` true → motion heading).
        if not moving and prev_tid and prev_tid != trip_id:
            prev_info = gtfs_data["trip_index"].get(prev_tid)
            prev_shape = gtfs_data["shapes"].get(prev_info["shape_id"]) if prev_info else None
            if prev_shape is not None and poly_distance(prev_shape, vx, vy) <= _HOLD_DIST_M:
                trip_id = prev_tid
                min_dist, tangent = poly_match(prev_shape, vx, vy)

        bearing_diff = (
            _bearing_diff(heading, tangent) if heading is not None else None
        )

        # When a vehicle starts a new trip its previous off-route history is stale.
        # Reset the state machine so the new trip gets predictions immediately
        # rather than waiting 3 on-route snapshots to recover.  Early-morning trips
        # are especially affected: the terminus of trip N can be 150+ m from the
        # shape start of trip N+1, marking the bus "off-route" during transition.
        state = trackers.get(vid)
        if state is not None:
            prev_pos = state.get("pos")
            if prev_pos is not None and prev_pos[2] != trip_id:
                state["status"] = "on_route"
                state["off"] = 0
                state["on"] = 0
                state["dists"] = []   # old trip's distances don't carry over

        if update_tracker(trackers, vid, min_dist, bearing_diff):
            continue

        v_dist = vehicle_dist_along(trip_id, vx, vy, gtfs_data)
        speed  = progress_speed(trackers, vid, trip_id, v_dist, float(feed_ts))

        feature_rows = build_features(trip_id, v_dist, speed, snap_ts, gtfs_data)

        # Cleaned vehicle position: emitted for every on-route matched vehicle,
        # including ones whose ETAs we withhold below — the position itself is
        # still good and is what the VehiclePositions feed exists to serve.
        if with_vehicle_positions and feature_rows:
            next_feat, next_stop_id, next_stop_seq = feature_rows[0]
            next_rem = next_feat[8]  # remaining_dist_m to the next stop
            hist_speed = priors.get(
                f"{route_id}:{snap_ts.hour}", priors.get("_global", (5.0, 40.0))
            )[0]
            VP = gtfs_realtime_pb2.VehiclePosition
            vp_records.append({
                "vehicle_id": vid,
                "trip_id":    trip_id,
                "route_id":   route_id,
                "lat":        float(lat),
                "lon":        float(lon),
                "bearing":    bearing,
                "speed":      gps_speed,
                "stop_id":    next_stop_id,
                "stop_sequence": int(next_stop_seq),
                "status": (
                    VP.STOPPED_AT   if next_rem <= _STOPPED_AT_RADIUS_M
                    else VP.INCOMING_AT if next_rem <= _INCOMING_AT_RADIUS_M
                    else VP.IN_TRANSIT_TO
                ),
                "congestion": _congestion_level(speed, hist_speed),
            })

        # Idling at the origin pre-departure: withhold predictions until the
        # vehicle actually moves, rather than emit optimistically early ETAs.
        if speed <= 0.0 and v_dist < _NOT_DEPARTED_DIST_M:
            continue

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
                 "stops_ahead": int(r[0][2]), "seconds": float(sec)}
                for r, sec in zip(feature_rows, preds_sec)
            ],
        })

    tu_bytes = encode_trip_updates(
        updates, feed_ts, model_data.get("uncertainty_by_horizon")
    )
    if with_vehicle_positions:
        return tu_bytes, encode_vehicle_positions(vp_records, feed_ts)
    return tu_bytes
