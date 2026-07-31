"""
Pure-Python GTFS-RT ETA inference for the worker compact data format.

Compatible with the data produced by scripts/export_worker_data.py:
  shapes        shape_id → bytes  (struct-packed float64 UTM pairs)
  trip_index    trip_id  → {route_id, shape_id, start_sec,
                            stop_times: [(stop_id, seq, sched_cum_sec), ...]}
                where sched_cum_sec is scheduled seconds since the trip's
                first stop (cumulative) and start_sec is that first stop's
                departure in seconds since local midnight
  stop_distances (shape_id, stop_id) → float metres along shape
  shape_lengths  shape_id → float
  route_trips    route_id → [trip_id, ...]
  route_types    route_id → GTFS route_type int
  feed_timezone  IANA zone name of the feed's local time
  calendar       calendar.txt rows (service_id, weekday flags, start/end date)
  calendar_dates calendar_dates.txt rows (service_id, date, exception_type)
  model          {route_to_int, baseline, learning_rate, trees}

``start_sec``, ``route_types`` and ``feed_timezone`` are recent additions;
data exported before them simply disables the terminus schedule path below.
``calendar``/``calendar_dates`` are likewise recent; their absence disables
the active-service filter in infer_trip() and falls back to matching on
geometry alone across every trip ever defined for the route, regardless of
whether it runs today.

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

# Idling-at-origin guard: a vehicle parked at (≈) the shape start with no
# measurable forward motion is waiting for its departure.  How that departure is
# estimated depends on the mode — see the two blocks below — but a vehicle idling
# short of the terminus stop itself gets nothing either way.
_NOT_DEPARTED_DIST_M = 20.0

# Schedule-anchored terminus ETAs.
#
# GTFS stop times in Lviv are unreliable in general — that is why no schedule
# feature survived into the model (src/features.py).  They are *not* unreliable
# for an electric vehicle sitting at its terminus: it leaves on the timetable.
# Measured on 5 days of labelled snapshots, over rows the idle-at-origin guard
# above currently discards (ETA straight from the schedule vs actual arrival):
#
#     tram        MAE  94 s   median  51 s   82 % within 2 min
#     trolleybus  MAE 140 s   median  76 s   68 % within 2 min
#     bus         MAE 222 s   median 161 s   40 % within 2 min  (best bucket)
#
# The model has no notion of a pending departure and scores MAE 208 s (tram) /
# 395 s (trolleybus) on those same rows, so the schedule wins outright — but
# only for the two electric modes.  A bus at a terminus is served by the model
# instead; see _TERMINUS_MODEL_UNCERTAINTY_SEC.
_SCHEDULE_RELIABLE_ROUTE_TYPES = frozenset({0, 11})   # tram, trolleybus

# How far the scheduled departure may sit from now for the schedule to be
# believed.  Outside this window the vehicle is parked at a terminus it is not
# about to leave (layover, shift change) or — more often — the matched trip is
# the wrong instance, and accuracy collapses: beyond +30 min the tram MAE goes
# from ~100 s to 1600 s+.  A small overdue tolerance keeps the common
# "leaving a minute late" case covered.
_TERMINUS_MAX_OVERDUE_SEC = 120.0
_TERMINUS_MAX_WAIT_SEC    = 1800.0

# How far along the shape the vehicle may be from the trip's first stop and
# still count as standing at it.  Sitting near the *shape start* is not the same
# thing: 3 % of tram/trolleybus trips have their first stop a few hundred metres
# in, and waiting on the lead-in to a terminus is not waiting at it.
_TERMINUS_STOP_RADIUS_M = 50.0
#
# Sideways, the ordinary on-route threshold (_OFF_ROUTE_DIST) is the bound —
# 14 % of vehicles parked at a terminus are >50 m off the shape, but a tram or
# trolleybus cannot wander: it is rail- or wire-bound, so that offset is the
# terminus loop being drawn approximately, not a vehicle somewhere else.  The
# measurement is thin either way (23 departures observed beyond 50 m, median
# error 85 s against 60 s within it), so the physical argument decides.

# Uncertainty published with schedule-anchored ETAs.  The model's per-horizon
# table describes model error and does not apply here; this covers the measured
# MAE of both electric modes.
_TERMINUS_UNCERTAINTY_SEC = 150

# A bus at a terminus is served from the model, never from the timetable: the
# schedule is worthless for it (MAE 222 s at its best, 40 % within 2 min) and
# nothing schedule-free pins its departure down either.  Every alternative tried
# on 4 held-out days of the same rows landed in the same place — the model alone
# MAE 406 s / median 231 s; plus a learned "how long has it already been parked"
# residual-dwell term MAE 406 s; plus a flat learned offset MAE 388 s, which
# halves the -282 s earliness but costs median accuracy.  None is worth its
# machinery, so the model's own output ships uncorrected.
#
# What is *not* optional is saying so: the model's per-horizon bands (118-174 s)
# describe a moving vehicle, where it scores MAE 126 s.  Publishing those here
# would understate the error threefold, so terminus model ETAs carry a band of
# their own, close to the measured MAE and flat across horizons — as the error
# itself is (372-387 s from 1 stop ahead to 10).
_TERMINUS_MODEL_UNCERTAINTY_SEC = 400

_HALF_DAY_SEC = 12 * 3600
_DAY_SEC      = 24 * 3600

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


def _active_service_ids(gtfs_data: dict, day) -> frozenset[str] | None:
    """Service ids running on this calendar day (calendar.txt + calendar_dates.txt).

    None when the export predates these fields (see module docstring) —
    callers must treat that as "day filter unavailable", not "nothing runs
    today".
    """
    calendar = gtfs_data.get("calendar")
    calendar_dates = gtfs_data.get("calendar_dates")
    if calendar is None or calendar_dates is None:
        return None
    dow = day.strftime("%A").lower()
    date_str = day.strftime("%Y%m%d")
    active = {
        row["service_id"] for row in calendar
        if row["start_date"] <= date_str <= row["end_date"] and row.get(dow) == "1"
    }
    for row in calendar_dates:
        if row["date"] != date_str:
            continue
        if row["exception_type"] == "1":
            active.add(row["service_id"])
        elif row["exception_type"] == "2":
            active.discard(row["service_id"])
    return frozenset(active)


def infer_trip(route_id, reported_trip_id, vx, vy, bearing, data, active_service_ids=None):
    """Best (trip_id, spatial_dist, tangent_bearing) for a vehicle snapshot.

    Combines spatial distance with heading alignment so opposite-direction
    shape variants — which sit on top of each other and defeat pure
    nearest-distance matching — are told apart.

    ``active_service_ids``, when given, restricts candidates to trips actually
    scheduled today (see _active_service_ids). Without it, two routes whose
    weekday and weekend trips reuse the same shape — e.g. route 105 — are
    geometrically indistinguishable, and matching silently picks whichever
    calendar variant scores best on position alone, not the one actually
    running.
    """
    candidates = data["route_trips"].get(str(route_id), [])
    if active_service_ids is not None:
        candidates = [
            tid for tid in candidates
            if data["trip_index"].get(tid, {}).get("service_id") in active_service_ids
        ]
    if not candidates:
        return None, 9999.0, 0.0

    # Fast path: trust the reported trip only if it is both near AND not headed
    # the wrong way down its shape.
    if reported_trip_id and reported_trip_id in data["trip_index"]:
        reported_info = data["trip_index"][reported_trip_id]
        reported_active = (
            active_service_ids is None
            or reported_info.get("service_id") in active_service_ids
        )
        coords = data["shapes"].get(reported_info["shape_id"]) if reported_active else None
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
# Schedule-anchored terminus predictions
# ---------------------------------------------------------------------------

_tz_cache: dict[str | None, timezone] = {}


def _feed_tz(name: str | None):
    """Feed-local tzinfo, falling back to UTC when zoneinfo can't resolve it."""
    if name in _tz_cache:
        return _tz_cache[name]
    tz = timezone.utc
    if name:
        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(name)
        except Exception:
            tz = timezone.utc
    _tz_cache[name] = tz
    return tz


def _local_seconds_of_day(snap_ts: datetime, data: dict) -> float:
    local = snap_ts.astimezone(_feed_tz(data.get("feed_timezone")))
    return local.hour * 3600 + local.minute * 60 + local.second


def terminus_seconds_until_departure(trip_id: str, snap_ts: datetime,
                                     data: dict) -> float | None:
    """Seconds until *trip_id* is scheduled to leave its first stop.

    ``None`` when the schedule should not be trusted here: a mode whose
    timetable does not hold (bus), an export without ``start_sec``, or a
    departure too far from now to belong to this vehicle's next run.  A
    departure that is only slightly overdue returns 0.0, not a negative wait.
    """
    trip = data["trip_index"].get(trip_id)
    if trip is None:
        return None
    route_type = data.get("route_types", {}).get(str(trip["route_id"]))
    if route_type not in _SCHEDULE_RELIABLE_ROUTE_TYPES:
        return None
    start_sec = trip.get("start_sec")
    if start_sec is None:
        return None

    wait = float(start_sec) - _local_seconds_of_day(snap_ts, data)
    # Service-day wrap: a 23:50 departure seen at 00:05 is 15 min overdue,
    # not 23 h 45 min early (and the mirror case for after-midnight trips,
    # whose start_sec exceeds 86400).
    if wait < -_HALF_DAY_SEC:
        wait += _DAY_SEC
    elif wait > _HALF_DAY_SEC:
        wait -= _DAY_SEC

    if wait < -_TERMINUS_MAX_OVERDUE_SEC or wait > _TERMINUS_MAX_WAIT_SEC:
        return None
    return max(wait, 0.0)


def at_first_stop(trip_id: str, v_dist: float, off_route_dist: float,
                  data: dict) -> bool:
    """Is the vehicle physically standing at this trip's first stop?

    *v_dist* is its projection along the shape and *off_route_dist* its
    perpendicular distance from it, so the pair fixes the vehicle in both axes
    — being near the start of the shape says nothing about either.  The
    sideways tolerance is deliberately loose; see _TERMINUS_STOP_RADIUS_M.
    """
    trip = data["trip_index"].get(trip_id)
    if trip is None or not trip["stop_times"]:
        return False
    if off_route_dist > _OFF_ROUTE_DIST:
        return False
    first_stop_dist = data["stop_distances"].get(
        (trip["shape_id"], trip["stop_times"][0][0])
    )
    if first_stop_dist is None:
        return False
    return abs(v_dist - first_stop_dist) <= _TERMINUS_STOP_RADIUS_M


def terminus_schedule_predictions(trip_id: str, snap_ts: datetime, data: dict,
                                  feature_rows: list, v_dist: float,
                                  off_route_dist: float) -> list:
    """Prediction dicts for a vehicle idling at its trip's first stop.

    Same shape as the model path's predictions so the encoder stays unaware of
    where the numbers came from; empty unless the vehicle is really standing at
    the terminus (``at_first_stop``) and its departure is close enough to be
    believed (``terminus_seconds_until_departure``).
    """
    if not feature_rows:
        return []
    if not at_first_stop(trip_id, v_dist, off_route_dist, data):
        return []
    depart_in = terminus_seconds_until_departure(trip_id, snap_ts, data)
    if depart_in is None:
        return []

    trip = data["trip_index"][trip_id]
    # Keyed by stop_sequence, not stop_id — a loop route visits a stop twice.
    sched_cum = {int(seq): cum for _, seq, cum in trip["stop_times"]}

    preds = []
    for feat_row, stop_id, stop_seq in feature_rows:
        cum = sched_cum.get(int(stop_seq))
        if cum is None:
            continue
        preds.append({
            "stop_id": stop_id,
            "stop_sequence": int(stop_seq),
            "stops_ahead": int(feat_row[2]),
            "seconds": depart_in + float(cum),
        })
    return preds


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


def _bias_correction_for(table: dict | None, horizon: int) -> float:
    """Per-horizon signed bias (seconds) to subtract from a raw prediction.

    *table* maps stops_ahead to the live-pooled ``pred - actual`` mean (see
    ``src.scoring.live_bias_by_horizon``). A missing table or horizon means no
    correction (0.0), not a guessed one — an unset table must be a no-op.
    Horizons past the largest measured key reuse that key's bias, same
    out-of-range handling as ``_uncertainty_for``.
    """
    if not table:
        return 0.0
    v = table.get(horizon)
    if v is None:
        keys = [k for k in table if isinstance(k, int)]
        if not keys:
            return 0.0
        v = table[min(max(keys), horizon)] if horizon < min(keys) else table[max(keys)]
    return float(v)


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
            # An update may carry its own flat band when its numbers didn't come
            # from the model at all (schedule-anchored terminus ETAs).
            unc = u.get("uncertainty") or _uncertainty_for(
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
        # The position's own capture time, not the (later) feed publish time —
        # per-vehicle when the source reported one (see run_inference), falling
        # back to the batch feed_ts only when it didn't.
        vp.timestamp = r.get("vehicle_ts", feed_ts)
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


# A vehicle's own position report can lag the feed's batch-capture time far
# more than the batch timestamp suggests: upstream keeps serving some
# vehicles' last-known fix long after the device stopped updating (observed
# 2026-07-28: 590 vehicles, per-vehicle age median 14s / p90 62s, but 17 of
# them 1h+ stale, one over 30 days). Predicting off a stale fix shows a
# live-looking ETA for a bus that already passed the stop, or isn't there at
# all — trust nothing older than this. Generous relative to the p90 above so
# normal reporting jitter never gets caught by it.
STALE_VEHICLE_MAX_AGE_SEC = 180


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
    # Computed once per push (all vehicles in this batch share the same feed
    # day) — see _active_service_ids for why this matters at all.
    active_service_ids = _active_service_ids(gtfs_data, snap_ts.date())

    # Per-horizon bias correction, weekday/weekend split when available (falls
    # back to the flat table, then to no correction — see _bias_correction_for).
    # UTC weekday, matching the is_weekend feature build_features computes below.
    bias_weekend = model_data.get("bias_by_horizon_weekend")
    if bias_weekend:
        bucket = "weekend" if snap_ts.weekday() >= 5 else "weekday"
        bias_table = bias_weekend.get(bucket) or model_data.get("bias_by_horizon")
    else:
        bias_table = model_data.get("bias_by_horizon")

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

        vehicle_ts = int(v.timestamp) if v.HasField("timestamp") else None
        if vehicle_ts is not None and feed_ts - vehicle_ts > STALE_VEHICLE_MAX_AGE_SEC:
            continue

        vx, vy = project_xy(lon, lat)
        trip_id, min_dist, tangent = infer_trip(
            route_id, reported_tid, vx, vy, bearing, gtfs_data, active_service_ids
        )
        if trip_id is None:
            continue
        bearing_diff = (
            _bearing_diff(bearing, tangent) if bearing is not None else None
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
                # The vehicle's own reported fix time, not the batch feed_ts —
                # a consumer honestly seeing "this position is 90s old" is the
                # whole point of the staleness filter above; stamping every
                # vehicle with the same fresh batch time (the previous
                # behavior) hid that distinction entirely.
                "vehicle_ts": vehicle_ts if vehicle_ts is not None else feed_ts,
            })

        if not feature_rows:
            continue

        # Idling at the origin pre-departure.  A tram or trolleybus standing at
        # its terminus leaves on the timetable, so serve the schedule.  A bus
        # falls through to the model below — its timetable says nothing, and no
        # schedule-free estimate of its departure beats just asking the model —
        # but it is published with a band that admits how wide the error is.
        # A vehicle idling short of the terminus stop is still served nothing.
        terminus_unc = None
        if speed <= 0.0 and v_dist < _NOT_DEPARTED_DIST_M:
            if not at_first_stop(trip_id, v_dist, min_dist, gtfs_data):
                continue
            sched_preds = terminus_schedule_predictions(
                trip_id, snap_ts, gtfs_data, feature_rows, v_dist, min_dist
            )
            if sched_preds:
                updates.append({
                    "vehicle_id": vid,
                    "trip_id":    trip_id,
                    "route_id":   route_id,
                    "snap_ts":    snap_ts,
                    "predictions": sched_preds,
                    # Not model output — the per-horizon model bands don't
                    # describe these, so carry their own.
                    "uncertainty": _TERMINUS_UNCERTAINTY_SEC,
                })
                continue
            terminus_unc = _TERMINUS_MODEL_UNCERTAINTY_SEC

        preds_sec = predict_rows(model_data, [r[0] for r in feature_rows])
        updates.append({
            "vehicle_id": vid,
            "trip_id":    trip_id,
            "route_id":   route_id,
            "snap_ts":    snap_ts,
            "predictions": [
                {"stop_id": r[1], "stop_sequence": int(r[2]),
                 "stops_ahead": int(r[0][2]),
                 "seconds": float(sec) - _bias_correction_for(bias_table, int(r[0][2]))}
                for r, sec in zip(feature_rows, preds_sec)
            ],
            # None for a moving vehicle → the per-horizon model bands apply.
            "uncertainty": terminus_unc,
        })

    tu_bytes = encode_trip_updates(
        updates, feed_ts, model_data.get("uncertainty_by_horizon")
    )
    if with_vehicle_positions:
        return tu_bytes, encode_vehicle_positions(vp_records, feed_ts)
    return tu_bytes
