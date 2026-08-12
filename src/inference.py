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
  ambiguous_shapes  set of shape_ids that self-intersect (out-and-back
                    routes, tram turnarounds) — see vehicle_dist_along
  model          {route_to_int, baseline, learning_rate, trees}

``start_sec``, ``route_types`` and ``feed_timezone`` are recent additions;
data exported before them simply disables the terminus schedule path below.
``calendar``/``calendar_dates`` are likewise recent; their absence disables
the active-service filter in infer_trip() and falls back to matching on
geometry alone across every trip ever defined for the route, regardless of
whether it runs today. ``ambiguous_shapes`` absence disables the
schedule-aware tie-break in vehicle_dist_along() and falls back to raw,
unconstrained nearest-point projection everywhere.

The model predicts seconds_to_arrival directly per upcoming stop (multi-
horizon), anchored at the vehicle's *projected position along the shape* —
not at the last passed stop — so a bus sitting at a stop gets
remaining_dist ≈ 0 and a near-zero ETA instead of a full segment time.

No R2 / JS APIs — runs on standard CPython. The only consumer is the
scripts/push_feed.py daemon; worker/worker.js serves the finished blob from R2
and does not run inference, so numpy is fair game on the scoring hot path (see
predict_rows).
"""
from __future__ import annotations

import math
import struct
import time
from datetime import datetime, time as dtime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as _np
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
# nearest-distance matching cannot tell apart; progress disambiguates
# same-shape trips running at different times of day (e.g. route 105's
# back-to-back weekday runs) that bearing alone cannot — weights match
# src/trip_inference.py's score_trip exactly, since that weighting is
# already proven in production via the batch/actuals path.
_MATCH_DIST_CAP   = 100.0   # spatial-score normaliser
_REPORTED_DIST_OK = 75.0    # trust a reported trip_id within this distance
_SPATIAL_W  = 0.4
_BEARING_W  = 0.3
_PROGRESS_W = 0.3

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

# Cache-miss sentinel. The memo dicts below cache values whose legitimate
# result is often None ("this trip has no usable schedule"), so `is _MISS` is
# the only way to tell a miss from a cached negative.
_MISS = object()

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


# Two points geographically within this of each other are treated as a tie
# for poly_project_near — self-intersecting shapes revisit the same physical
# spot exactly, but real GPS/segment-sampling jitter means "the same point"
# rarely comes out at exactly 0.0 apart.
_PROJECT_TIE_TOLERANCE_M = 30.0


def poly_project_near(
    shape_bytes: bytes, px: float, py: float, expected_dist: float | None,
) -> float:
    """Distance along the polyline to the point nearest (px, py) — except
    when the shape revisits the same physical location (self-intersecting:
    out-and-back routes, tram turnarounds), where multiple far-apart
    distances-along-shape can tie for nearest. plain poly_project breaks
    that tie arbitrarily (whichever segment is walked first); this instead
    prefers whichever tied candidate sits closest to `expected_dist` — the
    schedule-implied position (see _expected_dist_along) — since the two
    occurrences of a self-intersection are rarely both schedule-plausible
    at once. Falls back to plain nearest-point when expected_dist is None
    (no usable schedule for this trip) or there's no tie to break.
    """
    n = len(shape_bytes) // 16
    cum = 0.0
    dists_alongs: list[tuple[float, float]] = []
    nearest_dist_along = 0.0
    min_d = math.inf
    for i in range(n - 1):
        ax, ay = struct.unpack_from("dd", shape_bytes, i * 16)
        bx, by = struct.unpack_from("dd", shape_bytes, (i + 1) * 16)
        seg_len = math.hypot(bx - ax, by - ay)
        d, t = _seg_nearest(px, py, ax, ay, bx, by)
        dist_along = cum + t * seg_len
        dists_alongs.append((d, dist_along))
        if d < min_d:
            min_d = d
            nearest_dist_along = dist_along
        cum += seg_len

    if expected_dist is None:
        return nearest_dist_along

    tied = [da for d, da in dists_alongs if d <= min_d + _PROJECT_TIE_TOLERANCE_M]
    return min(tied, key=lambda da: abs(da - expected_dist))


def bbox_min_distance(bbox, px: float, py: float) -> float:
    """Lower bound on the distance from (px, py) to any point of *bbox*.

    Zero inside the box. Never exceeds the true nearest-point distance, which
    is what makes it usable to rule a shape out without walking it.
    """
    minx, miny, maxx, maxy = bbox
    dx = minx - px if px < minx else (px - maxx if px > maxx else 0.0)
    dy = miny - py if py < miny else (py - maxy if py > maxy else 0.0)
    return math.hypot(dx, dy)


def _seg_bearing(ax, ay, bx, by) -> float:
    """Compass bearing (deg from North, projected coords) of segment a→b."""
    return math.degrees(math.atan2(bx - ax, by - ay)) % 360.0


def _bearing_diff(b1: float, b2: float) -> float:
    """Absolute angular difference in [0, 180]."""
    diff = abs(b1 - b2) % 360.0
    return diff if diff <= 180.0 else 360.0 - diff


def poly_match(shape_bytes: bytes, px: float, py: float) -> tuple[float, float, float]:
    """Nearest distance to the polyline, the shape's tangent bearing there,
    and the distance along the shape to that nearest point.

    The tangent points in the trip's direction of travel (shapes are ordered
    start→end), so comparing it to the vehicle's heading tells the two
    directions of an overlapping route apart. Distance-along-shape lets a
    caller compare against where the schedule expects the vehicle to be —
    telling apart same-shape trips running at different times of day, which
    position and bearing alone cannot.
    """
    n = len(shape_bytes) // 16
    min_d = math.inf
    tangent = 0.0
    dist_along = 0.0
    cum = 0.0
    for i in range(n - 1):
        ax, ay = struct.unpack_from("dd", shape_bytes, i * 16)
        bx, by = struct.unpack_from("dd", shape_bytes, (i + 1) * 16)
        seg_len = math.hypot(bx - ax, by - ay)
        d, t = _seg_nearest(px, py, ax, ay, bx, by)
        if d < min_d:
            min_d = d
            tangent = _seg_bearing(ax, ay, bx, by)
            dist_along = cum + t * seg_len
        cum += seg_len
    return min_d, tangent, dist_along


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


def _trip_sched_arrays(trip_id: str, data: dict):
    """``(dists, times)`` for a trip's scheduled stops, or None when unusable.

    ``dists`` is distance along the shape, ``times`` is local seconds since
    midnight of the service day (>= 86400 for a trip running past midnight).
    Both depend only on static export data, so the result is memoised — on the
    *data dict itself* rather than a module global, so the cache lives and dies
    with the export it describes and two different dicts (a test fixture, a
    reloaded blob) can never see each other's entries.
    """
    cache = data.setdefault("_sched_arrays", {})
    cached = cache.get(trip_id, _MISS)
    if cached is not _MISS:
        return cached

    result = None
    info = data["trip_index"].get(trip_id)
    if info is not None and info.get("start_sec") is not None:
        shape_id = info["shape_id"]
        start_sec = info["start_sec"]
        stop_distances = data["stop_distances"]
        dists, times = [], []
        for stop_id, _seq, cum_sec in info["stop_times"]:
            d = stop_distances.get((shape_id, stop_id))
            if d is None:
                continue
            dists.append(d)
            times.append(start_sec + cum_sec)
        if len(dists) >= 2:
            result = (dists, times)

    cache[trip_id] = result
    return result


def _expected_dist_along(
    trip_id: str, now_sec: float, data: dict, cache: dict | None = None,
) -> float | None:
    """Expected distance (m) along the shape at `now_sec` (local seconds
    since midnight of the service day — may exceed 86400 for a trip that
    runs past midnight), from scheduled stop times.

    Mirrors src/trip_inference.py's _expected_dist_along for the compact
    data format: trip_index[tid]["stop_times"] holds (stop_id, seq,
    sched_cum_sec) — seconds since the trip's own first stop — and
    "start_sec" anchors that to local midnight.

    ``cache`` memoises the answer by trip_id. Every vehicle in one inference
    pass shares a single ``now_sec`` (run_inference computes it once for the
    whole feed), so this returns an identical value for a given trip however
    many vehicles are scored against it — without the cache, a route's
    candidate trips are re-interpolated once per vehicle, hundreds of times
    per push. The caller owns the dict's lifetime, which is what makes the
    trip_id-only key safe: a new ``now_sec`` means a new cache.
    """
    arrays = _trip_sched_arrays(trip_id, data)
    if cache is not None:
        cached = cache.get(trip_id, _MISS)
        if cached is not _MISS:
            return cached
    result = _interp_expected_dist(arrays, now_sec)
    if cache is not None:
        cache[trip_id] = result
    return result


def _interp_expected_dist(arrays, now_sec: float) -> float | None:
    if arrays is None:
        return None
    dists, times = arrays

    # An after-midnight trip (times[-1] >= a day) is still "now" during the
    # early-morning hours before its own start_sec — shift now_sec into the
    # same >=24h numbering rather than wrongly comparing against yesterday.
    if times[-1] >= _DAY_SEC and now_sec < times[0] - _HALF_DAY_SEC:
        now_sec = now_sec + _DAY_SEC

    if now_sec <= times[0]:
        return dists[0]
    if now_sec >= times[-1]:
        return dists[-1]
    for i in range(len(times) - 1):
        if times[i] <= now_sec <= times[i + 1]:
            span = times[i + 1] - times[i]
            frac = (now_sec - times[i]) / span if span > 0 else 0.0
            return dists[i] + frac * (dists[i + 1] - dists[i])
    return None


def infer_trip(
    route_id, reported_trip_id, vx, vy, bearing, data,
    active_service_ids=None, now_sec=None, expected_cache=None,
):
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

    ``now_sec`` (local seconds since midnight — see _expected_dist_along),
    when given, adds a schedule-progress term so that among same-day
    candidates sharing a shape — e.g. route 105's back-to-back weekday runs
    on the one shape it didn't stop sharing with the weekend calendar —
    matching prefers the trip whose scheduled position is actually close to
    the vehicle, not just whichever happens to score best on position and
    heading alone.

    ``expected_cache`` is passed straight to _expected_dist_along — see there.
    """
    candidates = data["route_trips"].get(str(route_id), [])
    if active_service_ids is not None:
        candidates = [
            tid for tid in candidates
            if data["trip_index"].get(tid, {}).get("service_id") in active_service_ids
        ]
    if not candidates:
        return None, 9999.0, 0.0

    # poly_match walks every point of a shape, and depends only on the shape and
    # this one vehicle's position — but a Lviv route averages 216 trips over
    # just 2.5 distinct shapes, so scoring candidates trip-by-trip re-walked the
    # same polyline dozens of times per vehicle. Memoise it for this call.
    shape_match: dict[str, tuple | None] = {}

    def _match(shape_id):
        cached = shape_match.get(shape_id, _MISS)
        if cached is _MISS:
            coords = data["shapes"].get(shape_id)
            cached = poly_match(coords, vx, vy) if coords is not None else None
            shape_match[shape_id] = cached
        return cached

    # Fast path: trust the reported trip only if it is both near AND not headed
    # the wrong way down its shape.
    if reported_trip_id and reported_trip_id in data["trip_index"]:
        reported_info = data["trip_index"][reported_trip_id]
        reported_active = (
            active_service_ids is None
            or reported_info.get("service_id") in active_service_ids
        )
        match = _match(reported_info["shape_id"]) if reported_active else None
        if match is not None:
            d, tangent, _ = match
            if d < _REPORTED_DIST_OK and (
                bearing is None
                or _bearing_diff(bearing, tangent) <= _BEARING_WRONG_DEG
            ):
                return reported_trip_id, d, tangent

    # Bounding boxes let a shape be ruled out without walking it. The bearing
    # and progress terms are each in [0, 1] and non-negative, so a shape whose
    # *nearest possible* point already scores worse than the best full score so
    # far cannot win, whatever its heading or schedule. Skipping it is therefore
    # exact, not approximate — the chosen trip and its reported distance are
    # unchanged. Absent from an older export, in which case nothing is pruned.
    bboxes = data.get("shape_bboxes") or {}

    best_id, best_dist, best_tangent, best_score = None, math.inf, 0.0, math.inf
    for tid in candidates:
        trip = data["trip_index"].get(tid)
        if trip is None:
            continue
        shape_id = trip["shape_id"]
        if shape_id not in shape_match:
            bbox = bboxes.get(shape_id)
            if bbox is not None and best_score < math.inf:
                floor = _SPATIAL_W * min(bbox_min_distance(bbox, vx, vy) / _MATCH_DIST_CAP, 5.0)
                if floor >= best_score:
                    continue
        match = _match(shape_id)
        if match is None:
            continue
        d, tangent, dist_along = match

        if now_sec is not None:
            expected = _expected_dist_along(tid, now_sec, data, expected_cache)
        else:
            expected = None
        if expected is not None:
            shape_len = data["shape_lengths"].get(trip["shape_id"], 0.0)
            progress_score = min(abs(dist_along - expected) / max(shape_len, 1.0), 1.0)
        else:
            progress_score = 0.5

        score = (
            _SPATIAL_W * min(d / _MATCH_DIST_CAP, 5.0)
            + _BEARING_W * _bearing_score(bearing, tangent)
            + _PROGRESS_W * progress_score
        )
        if score < best_score:
            best_score, best_dist, best_id, best_tangent = score, d, tid, tangent
    return best_id, best_dist, best_tangent


# ---------------------------------------------------------------------------
# ML inference (pure-Python GBT tree traversal)
# ---------------------------------------------------------------------------

def _traverse_tree(nodes, feat):
    """Scalar traversal of one exported tree. Defines the semantics the
    vectorised path in predict_rows must reproduce exactly (go left on
    ``<= threshold``); kept as the readable reference for that contract."""
    idx = 0
    while True:
        f_idx, threshold, left, right, is_leaf, value = nodes[idx]
        if is_leaf:
            return value
        idx = left if feat[f_idx] <= threshold else right


def _flat_trees(model_data: dict) -> dict:
    """Flatten every tree's nodes into shared arrays, once per loaded model.

    Node indices are rebased onto a single concatenated array so all trees can
    be walked at the same time, with ``roots`` holding each tree's entry point.
    Memoised on the model dict: it is loaded once per daemon run and reused by
    every push.
    """
    cached = model_data.get("_flat_trees")
    if cached is not None:
        return cached

    f_idx, thr, left, right, is_leaf, value, roots = [], [], [], [], [], [], []
    offset = 0
    for nodes in model_data["trees"]:
        roots.append(offset)
        for fi, th, lf, rt, leaf, val in nodes:
            # A leaf's feature index is never read by the scalar path; the
            # vectorised one gathers it unconditionally, so pin it to a column
            # that always exists and discard the result via the leaf mask.
            f_idx.append(0 if leaf else int(fi))
            thr.append(th)
            left.append(offset + lf)
            right.append(offset + rt)
            is_leaf.append(leaf)
            value.append(val)
        offset += len(nodes)

    flat = {
        "f_idx":   _np.asarray(f_idx, dtype=_np.int32),
        "thr":     _np.asarray(thr, dtype=_np.float64),
        "left":    _np.asarray(left, dtype=_np.int32),
        "right":   _np.asarray(right, dtype=_np.int32),
        "is_leaf": _np.asarray(is_leaf, dtype=bool),
        "value":   _np.asarray(value, dtype=_np.float64),
        "roots":   _np.asarray(roots, dtype=_np.int32),
    }
    model_data["_flat_trees"] = flat
    return flat


# Rows walked at once. The working set is rows x trees, so this bounds peak
# memory (~10 MB per 1000 rows at 1200 trees) without giving up vectorisation.
_PREDICT_CHUNK_ROWS = 1024


def predict_rows(model_data: dict, rows: list) -> list:
    """Sum every tree's leaf value for every row.

    Walks all trees for a chunk of rows simultaneously rather than one
    (row, tree) pair at a time: the scalar version costs rows x trees x depth
    interpreter steps, which caps how large a model the 10 s push loop can
    afford — measured at 4.3x slower on a realistic push when trees went from
    500x63 to 1200x127. Numerically identical to _traverse_tree, which the
    sklearn-parity test pins.
    """
    if not rows:
        return []

    route_to_int = model_data["route_to_int"]
    baseline     = model_data["baseline"]
    flat         = _flat_trees(model_data)

    feats = _np.empty((len(rows), len(rows[0])), dtype=_np.float64)
    for i, row in enumerate(rows):
        feats[i, 0] = route_to_int.get(str(row[0]), -1)
        feats[i, 1:] = row[1:]

    roots = flat["roots"]
    f_idx, thr = flat["f_idx"], flat["thr"]
    left, right, leaf_of, value = flat["left"], flat["right"], flat["is_leaf"], flat["value"]

    n_trees = len(roots)
    out = _np.empty(len(rows), dtype=_np.float64)
    for start in range(0, len(rows), _PREDICT_CHUNK_ROWS):
        chunk = feats[start:start + _PREDICT_CHUNK_ROWS]
        n_chunk = len(chunk)

        # One flat (row, tree) walker per pair, row-major.
        state = _np.tile(roots, n_chunk)
        row_of = _np.repeat(_np.arange(n_chunk), n_trees)
        active = _np.arange(n_chunk * n_trees)

        # Drop walkers as they land, instead of stepping every pair until the
        # deepest one finishes: these trees are unbalanced (median depth 14
        # against 6 for a balanced 63-leaf tree, max 27), so most pairs are
        # done long before the last, and carrying them costs full-width passes.
        while active.size:
            at = state[active]
            landed = leaf_of[at]
            if landed.any():
                active = active[~landed]
                if not active.size:
                    break
                at = state[active]
            # Same rule as the scalar path: <= threshold goes left.
            go_right = chunk[row_of[active], f_idx[at]] > thr[at]
            state[active] = _np.where(go_right, right[at], left[at])

        # HistGradientBoosting leaf values already include the learning rate
        # (shrinkage is applied at fit time), so they are summed raw.
        out[start:start + n_chunk] = baseline + value[state].reshape(n_chunk, n_trees).sum(axis=1)

    return out.tolist()


# ---------------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------------

# Progress-speed measurement between consecutive pushes (mirrors the training
# computation in src/labeling.py — keep the constants in sync).
SPEED_UNKNOWN = -1.0
_SPEED_MIN_GAP_SEC = 3.0
_SPEED_MAX_GAP_SEC = 120.0
_SPEED_MAX_BACKWARD_M = 30.0
_MOVE_EPS_M = 25.0  # below this a vehicle has not advanced — jitter, not progress


def vehicle_dist_along(
    trip_id: str, vx: float, vy: float, data: dict, now_sec: float | None = None,
    expected_cache: dict | None = None,
) -> float:
    """Vehicle's projected distance (m) along the trip's shape.

    Self-intersecting shapes (out-and-back routes, tram turnarounds — see
    GTFSStatic.is_ambiguous_shape, exported as data["ambiguous_shapes"])
    make raw nearest-point projection ambiguous: two far-apart points along
    the shape can sit at the same physical location, so a snapshot can snap
    onto the wrong one just because it's geographically nearby — corrupting
    remaining-distance-to-stop and everything downstream (route 122's
    chronic flat bias was this).

    Earlier attempt: clamp a projection that jumps too far, confirming it
    over one push before trusting it. That works for a GPS glitch, but not
    for a real out-and-back trip — a vehicle genuinely re-approaching a
    self-intersection mid-route also "jumps" and then "confirms" it on the
    next push, since confirmation only checks consistency between pushes,
    not correctness. Measured live: made route 122 worse (887s MAE), not
    better. This version resolves the ambiguity at its source instead —
    poly_project_near() picks whichever tied-nearest occurrence is closer to
    where the schedule says the trip should be right now (_expected_dist_along,
    the same signal infer_trip's progress term already uses to pick between
    candidate trips, just applied here to pick between positions on one
    shape). No per-vehicle state needed: unlike a jump clamp, this needs no
    trajectory history and gets the very first snapshot of a trip right too.
    ``now_sec`` missing, or no schedule for this trip, falls back to plain
    nearest-point projection — same as an ordinary, non-ambiguous shape.
    """
    trip = data["trip_index"].get(trip_id)
    if trip is None:
        return 0.0
    shape_id = trip["shape_id"]
    coords = data["shapes"].get(shape_id)
    if coords is None:
        return 0.0

    if now_sec is None or shape_id not in data.get("ambiguous_shapes", ()):
        return poly_project(coords, vx, vy)

    expected = _expected_dist_along(trip_id, now_sec, data, expected_cache)
    return poly_project_near(coords, vx, vy, expected)


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


def stationary_seconds(trackers: dict, vid: str, trip_id: str, v_dist: float,
                       ts_sec: float) -> float:
    """Seconds since this vehicle last advanced more than ``_MOVE_EPS_M``.

    The live counterpart of ``labeling._stationary_seconds`` (feature index 16).
    ``progress_speed`` already reports ~0 for a stopped vehicle, but says nothing
    about *how long* it has been stopped — and a bus 20 minutes into a terminus
    layover reaches its next stop nothing like a bus paused at a light.

    Anchored state, so a vehicle that never moves keeps accumulating rather than
    resetting each push. Returns 0.0 ("moving") on a first sighting or a trip
    change — the neutral value, matching what a moving vehicle reports.

    The anchor has to outlive the process: the daemon is restarted by every cron
    tick, which would otherwise cap this at the ~5.5 min a single run lasts even
    though training measures it over whole-day trajectories. push_feed carries
    the anchors across runs through R2 (see its TRACKER_STATE_KEY); if that
    carryover is unavailable the daemon simply starts cold, costing accuracy on
    stopped vehicles rather than producing a wrong value.
    """
    state = trackers.setdefault(vid, {"status": "on_route", "off": 0, "on": 0})
    anchor = state.get("still")
    if anchor is None or anchor[2] != trip_id or v_dist - anchor[1] > _MOVE_EPS_M:
        state["still"] = (ts_sec, v_dist, trip_id)
        return 0.0
    return max(0.0, ts_sec - anchor[0])


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


def _trip_stop_entries(trip_id: str, data: dict) -> list:
    """A trip's stops as ``[(dist_along, stop_id, stop_sequence, orig_idx), ...]``,
    sorted by distance along the shape.

    Static per trip, so memoised on the data dict for the same reason (and with
    the same lifetime) as _trip_sched_arrays — otherwise every vehicle re-sorts
    its trip's stop list on every push.
    """
    cache = data.setdefault("_trip_entries", {})
    entries = cache.get(trip_id)
    if entries is None:
        trip = data["trip_index"].get(trip_id)
        if trip is None:
            return []
        sts = trip["stop_times"]  # [(stop_id, seq, sched_cum_sec), ...]
        entries = sorted(
            (data["stop_distances"].get((trip["shape_id"], st[0]), 0.0), st[0], st[1], i)
            for i, st in enumerate(sts)
        )
        cache[trip_id] = entries
    return entries


def stop_after_horizon(trip_id: str, v_dist: float, data: dict) -> dict | None:
    """The first upcoming stop this pass does *not* predict, or None.

    ``None`` when the trip ends within MAX_STOPS_AHEAD — there is nothing past
    the horizon to say anything about. See encode_trip_updates for why the
    caller wants it.
    """
    ahead = [e for e in _trip_stop_entries(trip_id, data) if e[0] > v_dist]
    if len(ahead) <= MAX_STOPS_AHEAD:
        return None
    _d, stop_id, stop_seq, _idx = ahead[MAX_STOPS_AHEAD]
    return {"stop_id": stop_id, "stop_sequence": int(stop_seq)}


def build_features(trip_id: str, v_dist: float, speed: float,
                   snap_ts: datetime, data: dict, stationary_sec: float = 0.0) -> list:
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

    n_stops_total = len(trip["stop_times"])
    entries = _trip_stop_entries(trip_id, data)

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
            stationary_sec,                              # idx 16 stationary_sec
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


def trip_instance_context(trip_id: str, snap_ts: datetime, data: dict,
                          now_sec: float, feed_tz, cache: dict | None = None):
    """Instance identity and scheduled arrival epochs for one trip.

    Returns ``(start_date, start_time, sched_epoch_by_seq)``: the GTFS-RT
    TripDescriptor pair that binds an update to a specific *run* of the trip,
    plus each stop's scheduled arrival as epoch seconds (for TripUpdate.delay).
    ``(None, None, {})`` when the trip carries no parseable schedule — which is
    also what an export predating ``start_sec`` looks like, so the fields simply
    stay off the feed rather than being guessed.

    Cached by trip_id, with the same one-push lifetime and for the same reason
    as _expected_dist_along's cache.
    """
    if cache is not None:
        hit = cache.get(trip_id, _MISS)
        if hit is not _MISS:
            return hit

    result = (None, None, {})
    trip = data["trip_index"].get(trip_id)
    arrays = _trip_sched_arrays(trip_id, data)
    if trip is not None and arrays is not None:
        _dists, times = arrays
        start_sec = float(trip["start_sec"])
        service_date = snap_ts.astimezone(feed_tz).date()
        # Same service-day-wrap rule as _interp_expected_dist: a trip whose
        # schedule runs past midnight is still *yesterday's* run during the
        # early-morning hours before its own scheduled start.
        if times[-1] >= _DAY_SEC and now_sec < times[0] - _HALF_DAY_SEC:
            service_date -= timedelta(days=1)
        midnight = datetime.combine(service_date, dtime.min, tzinfo=feed_tz).timestamp()
        hh, rem = divmod(int(start_sec), 3600)
        mm, ss = divmod(rem, 60)
        result = (
            service_date.strftime("%Y%m%d"),
            # GTFS start_time may legitimately exceed 24:00:00 for an
            # after-midnight trip; it is left in that numbering, as in the
            # static feed it came from.
            f"{hh:02d}:{mm:02d}:{ss:02d}",
            {int(seq): midnight + start_sec + cum
             for _sid, seq, cum in trip["stop_times"]},
        )

    if cache is not None:
        cache[trip_id] = result
    return result


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

def _delay_for(predictions: list, snap_ts: datetime,
               sched_epoch: dict) -> int | None:
    """Schedule deviation in seconds at the nearest predicted stop, late positive.

    ``None`` when there is nothing to compare against — no predictions, or a
    trip with no parseable schedule — so the field is omitted rather than
    reported as an on-time zero.
    """
    if not predictions or not sched_epoch:
        return None
    first = predictions[0]
    sched = sched_epoch.get(int(first["stop_sequence"]))
    if sched is None:
        return None
    return int(snap_ts.timestamp() + float(first["seconds"]) - sched)


def publishes_delay(route_type) -> bool:
    """Whether a trip-level delay is meaningful for this mode.

    Delay is deviation from the published timetable, so it is only worth as
    much as that timetable. Lviv's is unreliable enough that no schedule
    feature survived into the model at all (src/features.py) — and measured on
    the live feed, bus delay spans roughly ±30 min at the 10th/90th percentile,
    which is noise dressed as a number. Consumers use TripUpdate.delay to
    extrapolate stops we don't list, so publishing that would actively mislead.

    The two electric modes are the same exception the terminus path already
    makes, and for the same measured reason (see
    _SCHEDULE_RELIABLE_ROUTE_TYPES): rail- and wire-bound vehicles do keep
    their timetable. An unknown route_type is treated as unreliable.
    """
    return route_type in _SCHEDULE_RELIABLE_ROUTE_TYPES


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
        # Trip-instance binding. trip_id alone does not say *which run* an update
        # describes: a loop route repeats its trip through the day and an
        # after-midnight trip belongs to the previous service day. Consumers that
        # key their own state on (trip_id, start_date) had nothing to key on.
        if u.get("start_date"):
            tu.trip.start_date = u["start_date"]
        if u.get("start_time"):
            tu.trip.start_time = u["start_time"]
        if u.get("direction_id") is not None:
            tu.trip.direction_id = int(u["direction_id"])
        tu.vehicle.id = u["vehicle_id"]
        # When these predictions were computed — i.e. how old the vehicle fix
        # behind them is. The header timestamp is the *publish* time and is
        # always fresh, so without this a prediction made from a 170 s-old
        # position (the staleness filter tolerates up to
        # STALE_VEHICLE_MAX_AGE_SEC) looked exactly as current as one made from
        # a 2 s-old fix. The VehiclePositions feed has carried this per vehicle
        # since it existed; TripUpdates simply never did.
        if u.get("trip_ts"):
            tu.timestamp = int(u["trip_ts"])
        # Schedule deviation at the nearest predicted stop, positive = late.
        # Still widely consumed at trip level (OTP and friends) as a summary of
        # how the run is doing, which per-stop absolute times don't give directly.
        if u.get("delay") is not None:
            tu.delay = int(u["delay"])
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
            # A stop whose predicted arrival has already elapsed is published as
            # "arriving now" rather than dropped.  Dropping it removed the one
            # arrival a waiting rider cares about most, and — because the
            # horizon it served vanished from the emitted list — silently
            # renumbered every later stop's position in the feed.  The quality
            # scorer reads stops_ahead from exactly that position
            # (src/scoring._parse_prediction_feed), so the per-horizon
            # uncertainty and bias tables were being calibrated against
            # horizon N-1 and then served at horizon N.
            arr_ts = max(arr_ts, now_ts + 1)
            last_arr_ts = arr_ts
            stu = tu.stop_time_update.add()
            stu.stop_id = pred["stop_id"]
            stu.stop_sequence = pred["stop_sequence"]
            stu.schedule_relationship = gtfs_realtime_pb2.TripUpdate.StopTimeUpdate.SCHEDULED
            stu.arrival.time = arr_ts
            stu.departure.time = arr_ts + int(u.get("dwell") or _DWELL_SECS)
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

        # Fence off the stops beyond our horizon.  GTFS-RT says a consumer
        # applies the last StopTimeUpdate's deviation to every *later* stop of
        # the trip that isn't listed — so with a MAX_STOPS_AHEAD cap, our 10th
        # stop's error was being extrapolated across the entire remainder of the
        # route by any spec-following consumer.  A single NO_DATA sentinel at
        # the next stop ends the propagation and says plainly "no prediction
        # past here".  Carries no arrival/departure, as the spec requires.
        term = u.get("horizon_end")
        if stop_count and term is not None:
            stu = tu.stop_time_update.add()
            stu.stop_id = term["stop_id"]
            stu.stop_sequence = int(term["stop_sequence"])
            stu.schedule_relationship = (
                gtfs_realtime_pb2.TripUpdate.StopTimeUpdate.NO_DATA
            )

        if stop_count == 0:
            del feed.entity[-1]
    return feed.SerializeToString()


# Vehicle is treated as dwelling AT its next stop (rather than IN_TRANSIT_TO it)
# once it is within this many metres of it along the shape.
_STOPPED_AT_RADIUS_M = 25.0
# Vehicle is flagged INCOMING_AT (imminent arrival) within this radius; beyond
# it the status is IN_TRANSIT_TO.
_INCOMING_AT_RADIUS_M = 150.0

# Fallback stop dwell, used only when the export carries no measured table.
# The measured medians are 27-32 s (see _dwell_for), so this legacy 15 s is
# kept purely so an older model blob behaves exactly as it did before.
_DWELL_SECS = 15


def _dwell_for(table: dict | None, route_type) -> int:
    """Stop dwell in seconds for a route type.

    *table* maps GTFS route_type to the measured median dwell, with a
    ``_global`` catch-all (scripts/measure_dwell.py builds it; `make export`
    bakes it into the model blob). A missing table means an export that
    predates the measurement, and falls back to the old flat constant.

    The measurement conditions on the vehicle actually having stopped — a
    stationary run is what it detects — so this is the dwell at stops that are
    served, not an average diluted by drive-throughs. That is the right
    quantity for a published departure time: a stop the vehicle sails through
    has arrival ≈ departure anyway, and overstating departure slightly is the
    safe direction for someone deciding whether to run for it. Note also that
    dwells shorter than one sampling gap (~11 s) are invisible to the detector,
    so these medians are, if anything, mildly high.
    """
    if not table:
        return _DWELL_SECS
    value = table.get(route_type)
    if value is None:
        value = table.get("_global", _DWELL_SECS)
    return int(value)


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

# How far the *whole fleet's* clock may lag the feed header before we stop
# trusting the feed at all.  Upstream can skew the two clocks against each
# other while the positions themselves stay live (observed 2026-07-31 09:16Z:
# every one of 590 vehicles suddenly 9+ min behind the header, yet 221 of them
# moved between two samples 46 s apart).  Measuring staleness against the
# header then drops 100 % of the fleet and we serve an empty feed — see
# staleness_reference for how the anchor slides instead.  The slide is capped
# here because a fleet-wide lag is also what a genuinely frozen upstream looks
# like, and past some age the positions are worthless however live the clock
# claims they are.
MAX_FEED_CLOCK_SKEW_SEC = 30 * 60


def staleness_reference(vp_feed, feed_ts: int) -> tuple[int, int]:
    """Timestamp to measure per-vehicle staleness against, plus the feed's skew.

    Returns ``(reference_ts, skew_sec)``, where the skew is how far the feed's
    *newest* fix trails the header.  A healthy feed's newest fix is seconds old
    by construction (600 vehicles, median age ~12 s), so the reference stays the
    header and behavior is unchanged.

    Only once the whole fleet trails the header by more than a vehicle is
    allowed to be stale — impossible for real staleness, since that would mean
    every device stopped reporting at once — does the reference slide to the
    newest per-vehicle timestamp, judging the fleet against itself.  Vehicles
    whose own fix trails *that* are still dropped, so the genuine multi-hour
    stragglers ca6bde4 was written for keep getting dropped.

    The slide stops at MAX_FEED_CLOCK_SKEW_SEC: beyond it the reference falls
    back to the header, every vehicle fails the age check, and we publish
    nothing rather than pretend a long-dead feed is current.
    """
    newest = 0
    for entity in vp_feed.entity:
        if entity.HasField("vehicle") and entity.vehicle.HasField("timestamp"):
            ts = int(entity.vehicle.timestamp)
            if ts > newest:
                newest = ts
    if newest == 0:
        return feed_ts, 0
    skew = feed_ts - newest
    if skew <= STALE_VEHICLE_MAX_AGE_SEC or skew > MAX_FEED_CLOCK_SKEW_SEC:
        return feed_ts, max(skew, 0)
    return newest, skew


# ---------------------------------------------------------------------------
# Main inference pass
# ---------------------------------------------------------------------------

def run_inference(gtfs_data: dict, model_data: dict, trackers: dict,
                  vp_bytes: bytes, *, with_vehicle_positions: bool = False,
                  stats: dict | None = None):
    """Vehicle-positions protobuf bytes → TripUpdates protobuf bytes.

    With ``with_vehicle_positions=True`` returns a ``(trip_updates_bytes,
    vehicle_positions_bytes)`` tuple — the second feed re-publishes the cleaned
    positions (corrected trip, next stop, congestion) computed in this same
    pass. The default single-bytes return keeps existing callers unchanged.

    Pass a dict as *stats* to have this pass record what it saw about the
    upstream feed — ``vehicles_in``, ``vehicles_stale`` and ``feed_skew_sec``
    (see staleness_reference).  The daemon publishes these as R2 object
    metadata so /health can tell "upstream went stale" apart from "inference
    broke", which are indistinguishable from an empty feed alone.
    """
    vp_feed = gtfs_realtime_pb2.FeedMessage()
    vp_feed.ParseFromString(vp_bytes)
    feed_ts = int(vp_feed.header.timestamp) or int(time.time())
    snap_ts = datetime.fromtimestamp(feed_ts, tz=timezone.utc)
    priors = gtfs_data.get("route_hour_priors", {})
    # Computed once per push (all vehicles in this batch share the same feed
    # day) — see _active_service_ids for why this matters at all.
    active_service_ids = _active_service_ids(gtfs_data, snap_ts.date())

    # Local wall-clock seconds since midnight, for _expected_dist_along's
    # schedule-progress term — GTFS stop times are local, not UTC.
    try:
        feed_tz = ZoneInfo(gtfs_data.get("feed_timezone") or "Europe/Kiev")
    except Exception:
        feed_tz = ZoneInfo("Europe/Kiev")
    local_ts = snap_ts.astimezone(feed_tz)
    now_sec = local_ts.hour * 3600 + local_ts.minute * 60 + local_ts.second

    # Per-push memo dicts. Both cache values that are a pure function of a trip
    # and this pass's single now_sec, so they are keyed by trip_id alone and
    # must not outlive the push — see _expected_dist_along.
    expected_cache: dict = {}
    instance_cache: dict = {}

    # Per-horizon bias correction, weekday/weekend split when available (falls
    # back to the flat table, then to no correction — see _bias_correction_for).
    # UTC weekday, matching the is_weekend feature build_features computes below.
    bias_weekend = model_data.get("bias_by_horizon_weekend")
    if bias_weekend:
        bucket = "weekend" if snap_ts.weekday() >= 5 else "weekday"
        bias_table = bias_weekend.get(bucket) or model_data.get("bias_by_horizon")
    else:
        bias_table = model_data.get("bias_by_horizon")

    # Measured per-route-type stop dwell; absent on an older export, in which
    # case _dwell_for falls back to the legacy flat constant.
    dwell_table = model_data.get("dwell_by_route_type")

    # Judge per-vehicle staleness against the feed's own newest fix, not blindly
    # against the header — the two clocks can drift apart upstream.
    stale_ref, feed_skew = staleness_reference(vp_feed, feed_ts)
    if feed_skew > STALE_VEHICLE_MAX_AGE_SEC:
        print(
            f"[warn] upstream per-vehicle clock lags the feed header by "
            f"{feed_skew}s"
            + (
                " — beyond the trusted skew, dropping every vehicle"
                if stale_ref == feed_ts
                else " — anchoring staleness to the newest fix"
            ),
            flush=True,
        )

    vehicles_in = 0
    vehicles_stale = 0
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

        vehicles_in += 1
        vehicle_ts = int(v.timestamp) if v.HasField("timestamp") else None
        if vehicle_ts is not None and stale_ref - vehicle_ts > STALE_VEHICLE_MAX_AGE_SEC:
            vehicles_stale += 1
            continue

        vx, vy = project_xy(lon, lat)
        trip_id, min_dist, tangent = infer_trip(
            route_id, reported_tid, vx, vy, bearing, gtfs_data,
            active_service_ids, now_sec, expected_cache,
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

        v_dist = vehicle_dist_along(
            trip_id, vx, vy, gtfs_data, now_sec, expected_cache
        )
        speed  = progress_speed(trackers, vid, trip_id, v_dist, float(feed_ts))
        still  = stationary_seconds(trackers, vid, trip_id, v_dist, float(feed_ts))

        feature_rows = build_features(
            trip_id, v_dist, speed, snap_ts, gtfs_data, still
        )

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

        # Trip-level fields shared by both prediction paths below. horizon_end
        # is the first stop we deliberately say nothing about; trip_ts is the
        # age of the fix these numbers rest on (see encode_trip_updates).
        start_date, start_time, sched_epoch = trip_instance_context(
            trip_id, snap_ts, gtfs_data, now_sec, feed_tz, instance_cache
        )
        route_type = gtfs_data.get("route_types", {}).get(route_id)
        instance_fields = {
            "start_date":   start_date,
            "start_time":   start_time,
            "direction_id": gtfs_data["trip_index"][trip_id].get("direction_id"),
            "horizon_end":  stop_after_horizon(trip_id, v_dist, gtfs_data),
            "trip_ts":      vehicle_ts if vehicle_ts is not None else feed_ts,
            "dwell":        _dwell_for(dwell_table, route_type),
        }
        delay_ok = publishes_delay(route_type)

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
                    "delay": _delay_for(sched_preds, snap_ts, sched_epoch) if delay_ok else None,
                    **instance_fields,
                })
                continue
            terminus_unc = _TERMINUS_MODEL_UNCERTAINTY_SEC

        preds_sec = predict_rows(model_data, [r[0] for r in feature_rows])
        model_preds = [
            {"stop_id": r[1], "stop_sequence": int(r[2]),
             "stops_ahead": int(r[0][2]),
             "seconds": float(sec) - _bias_correction_for(bias_table, int(r[0][2]))}
            for r, sec in zip(feature_rows, preds_sec)
        ]
        updates.append({
            "vehicle_id": vid,
            "trip_id":    trip_id,
            "route_id":   route_id,
            "snap_ts":    snap_ts,
            "predictions": model_preds,
            # None for a moving vehicle → the per-horizon model bands apply.
            "uncertainty": terminus_unc,
            "delay": _delay_for(model_preds, snap_ts, sched_epoch) if delay_ok else None,
            **instance_fields,
        })

    if stats is not None:
        stats.update({
            # Vehicles that reached the staleness check — i.e. already past the
            # missing-position and bad-route filters, so the two counts below
            # are directly comparable.
            "vehicles_in": vehicles_in,
            "vehicles_stale": vehicles_stale,
            "feed_skew_sec": feed_skew,
        })

    tu_bytes = encode_trip_updates(
        updates, feed_ts, model_data.get("uncertainty_by_horizon")
    )
    if with_vehicle_positions:
        return tu_bytes, encode_vehicle_positions(vp_records, feed_ts)
    return tu_bytes
