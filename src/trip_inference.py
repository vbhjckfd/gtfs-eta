"""
Correct erroneous trip_id values reported in GTFS-RT, and detect off-route vehicles.

Trip matching uses two signals per candidate:
  1. Spatial distance from vehicle to shape (metres, EPSG:32635).
  2. Bearing alignment: vehicle heading vs. shape tangent at nearest point.

A schedule-progress signal was dropped (issue #3): a ground-truth adjudication
showed it (with prev-trip stickiness) holds the *reverse* direction at termini —
this batch matcher picked the wrong direction ~1.6x as often as the live
serving matcher, which uses spatial+bearing only. Since this module also labels
training data and scoring actuals, that mislabeling fed the model. Weights now
mirror src/inference.py (the live path), and stickiness is bearing-gated.

This weight/stickiness change is only a marginal direction fix (~+0.4pp absolute
accuracy on a measured window): the residual wrong-direction picks happen on
*stationary* vehicles at termini, where bearing≈0 and the two opposite shapes are
equidistant, so no per-snapshot scoring can disambiguate.
TODO(direction, issue #3): defer committing a direction while a vehicle is
stationary (extend the src/inference.py `_NOT_DEPARTED` idling guard to direction)
and lock it from the post-departure heading. That is the genuine lever — a
temporal/motion-history change, not a scoring-weight one.

Off-route detection uses a per-vehicle state machine:
  ON_ROUTE  →  OFF_ROUTE  when dist > OFF_ROUTE_DIST_M for OFF_ROUTE_CONSEC snapshots,
                           or bearing is wrong (> 90°) for the same window.
  OFF_ROUTE →  ON_ROUTE   when dist ≤ ON_ROUTE_DIST_M for ON_ROUTE_CONSEC snapshots.

When a vehicle is off-route, inferred_trip_id is set to None.
"""

from __future__ import annotations

import math
from collections import deque
from datetime import datetime
from enum import Enum
from typing import NamedTuple

import pandas as pd
from shapely.geometry import LineString, Point

from src.gtfs_static import GTFSStatic, TripInfo, _project_xy

# --- Trip matching constants (mirror src/inference.py: _SPATIAL_W / _BEARING_W) ---
MAX_DIST_M = 100.0          # hard cut-off for trip scoring
SPATIAL_WEIGHT = 0.6
BEARING_WEIGHT = 0.4
HYSTERESIS_DIST_M = 150.0

# --- Off-route detection constants ---
OFF_ROUTE_DIST_M = 150.0    # distance threshold to flag off-route
ON_ROUTE_DIST_M = 100.0     # distance to consider returned on-route
OFF_ROUTE_CONSEC = 3        # consecutive snapshots to transition → OFF_ROUTE
ON_ROUTE_CONSEC = 3         # consecutive snapshots to transition → ON_ROUTE
BEARING_WRONG_DEG = 90.0    # bearing diff > this considered wrong direction


# ---------------------------------------------------------------------------
# Off-route state machine
# ---------------------------------------------------------------------------

class RouteStatus(str, Enum):
    ON_ROUTE = "on_route"
    OFF_ROUTE = "off_route"


class VehicleRouteTracker:
    """
    Maintains off-route state for one vehicle across successive snapshots.

    Call `update(min_route_dist, bearing_diff)` for each snapshot in
    chronological order.  Returns the current RouteStatus.
    """

    def __init__(self) -> None:
        self.status = RouteStatus.ON_ROUTE
        self._consec_off = 0    # consecutive snapshots looking off-route
        self._consec_on = 0     # consecutive snapshots looking on-route
        self._recent_dists: deque[float] = deque(maxlen=5)

    def update(self, min_route_dist: float, bearing_diff: float | None) -> RouteStatus:
        """
        Parameters
        ----------
        min_route_dist : minimum distance (m) to ANY active shape for the route.
        bearing_diff   : degrees between vehicle bearing and best shape tangent,
                         or None if unavailable.
        """
        self._recent_dists.append(min_route_dist)

        # Individual snapshot signals
        spatially_off = min_route_dist > OFF_ROUTE_DIST_M
        bearing_wrong = (
            bearing_diff is not None
            and bearing_diff > BEARING_WRONG_DEG
            and min_route_dist > 50.0      # ignore stationary / slow vehicles
        )

        # "Moving away": last 3 distances strictly increasing AND already distant
        dists = list(self._recent_dists)
        moving_away = (
            len(dists) >= 3
            and all(dists[-i] > dists[-i - 1] for i in range(1, min(3, len(dists))))
            and min_route_dist > ON_ROUTE_DIST_M
        )

        snapshot_looks_off = spatially_off or bearing_wrong or moving_away

        if self.status == RouteStatus.ON_ROUTE:
            if snapshot_looks_off:
                self._consec_off += 1
                self._consec_on = 0
            else:
                self._consec_off = 0
                self._consec_on += 1
            if self._consec_off >= OFF_ROUTE_CONSEC:
                self.status = RouteStatus.OFF_ROUTE
                self._consec_off = 0
                self._consec_on = 0

        else:  # OFF_ROUTE
            if min_route_dist <= ON_ROUTE_DIST_M:
                self._consec_on += 1
                self._consec_off = 0
            else:
                self._consec_on = 0
                self._consec_off += 1
            if self._consec_on >= ON_ROUTE_CONSEC:
                self.status = RouteStatus.ON_ROUTE
                self._consec_off = 0
                self._consec_on = 0

        return self.status


# ---------------------------------------------------------------------------
# Trip scoring helpers
# ---------------------------------------------------------------------------

class InferenceResult(NamedTuple):
    vehicle_id: str
    timestamp: datetime
    reported_trip_id: str | None
    inferred_trip_id: str | None   # None when off_route=True
    inferred_route_id: str | None
    score: float
    high_confidence: bool
    off_route: bool
    route_status: RouteStatus


def _bearing_diff(b1: float, b2: float) -> float:
    """Absolute angular difference in [0, 180]."""
    diff = abs(b1 - b2) % 360
    return diff if diff <= 180 else 360 - diff


def _shape_tangent_bearing(shape: LineString, dist: float) -> float:
    """Approximate bearing of the shape at *dist* metres along it."""
    delta = min(5.0, shape.length * 0.01)
    d1 = max(0.0, dist - delta)
    d2 = min(shape.length, dist + delta)
    p1 = shape.interpolate(d1)
    p2 = shape.interpolate(d2)
    dx, dy = p2.x - p1.x, p2.y - p1.y
    return math.degrees(math.atan2(dx, dy)) % 360


def score_trip(
    trip: TripInfo,
    vx: float,
    vy: float,
    bearing: float | None,
    gtfs: GTFSStatic,
) -> tuple[float, float]:
    """
    Return (combined_score, spatial_dist_m) from spatial distance + bearing
    alignment (no schedule term — see module docstring).
    combined_score = inf when spatial_dist > MAX_DIST_M.
    spatial_dist is always the real distance (never inf unless shape is missing).
    """
    shape = gtfs.get_shape_linestring(trip.shape_id)
    if shape is None:
        return math.inf, math.inf

    pt = Point(vx, vy)
    spatial_dist = shape.distance(pt)

    if spatial_dist > MAX_DIST_M:
        return math.inf, spatial_dist   # real dist returned for off-route tracking

    dist_along = shape.project(pt)
    spatial_score = spatial_dist / MAX_DIST_M

    if bearing is not None:
        tangent = _shape_tangent_bearing(shape, dist_along)
        bearing_score = _bearing_diff(bearing, tangent) / 180.0
    else:
        bearing_score = 0.5

    combined = SPATIAL_WEIGHT * spatial_score + BEARING_WEIGHT * bearing_score
    return combined, spatial_dist


def _bearing_diff_for_shape(
    shape: LineString,
    vx: float,
    vy: float,
    bearing: float,
) -> float:
    """Bearing difference between vehicle and shape tangent at nearest point."""
    pt = Point(vx, vy)
    dist_along = shape.project(pt)
    tangent = _shape_tangent_bearing(shape, dist_along)
    return _bearing_diff(bearing, tangent)


def _stickiness_dist(
    prev_shape: LineString,
    vx: float,
    vy: float,
    bearing: float | None,
    best_dist: float,
) -> float | None:
    """Distance to the previous trip's shape if hysteresis should hold it, else None.

    Holds the previous trip when it is within HYSTERESIS_DIST_M and no farther
    than the fresh winner — but NOT when its tangent opposes the vehicle's bearing
    (> BEARING_WRONG_DEG). The bearing gate stops stickiness from pinning a vehicle
    to the outbound trip after it has reversed at a terminus for the return trip,
    which the issue #3 adjudication identified as the dominant batch direction error.
    """
    prev_dist = prev_shape.distance(Point(vx, vy))
    if prev_dist > HYSTERESIS_DIST_M or prev_dist > best_dist:
        return None
    if bearing is not None and _bearing_diff_for_shape(prev_shape, vx, vy, bearing) > BEARING_WRONG_DEG:
        return None
    return prev_dist


# ---------------------------------------------------------------------------
# Per-snapshot inference (single vehicle)
# ---------------------------------------------------------------------------

def infer_trip_for_vehicle(
    vehicle_id: str,
    route_id: str,
    reported_trip_id: str | None,
    lat: float,
    lon: float,
    bearing: float | None,
    now: datetime,
    gtfs: GTFSStatic,
    previous_trip_id: str | None = None,
) -> InferenceResult:
    """Infer the best trip for a single vehicle snapshot (no off-route tracking)."""
    vx, vy = _project_xy(lon, lat)
    candidates = gtfs.active_trips_for_route(route_id, now)
    if not candidates:
        return InferenceResult(
            vehicle_id=vehicle_id, timestamp=now,
            reported_trip_id=reported_trip_id, inferred_trip_id=reported_trip_id,
            inferred_route_id=route_id, score=math.inf, high_confidence=False,
            off_route=False, route_status=RouteStatus.ON_ROUTE,
        )

    best_tid, best_score, best_dist = None, math.inf, math.inf
    for tid in candidates:
        trip = gtfs.get_trip(tid)
        if trip is None:
            continue
        s, d = score_trip(trip, vx, vy, bearing, gtfs)
        if s < best_score:
            best_score, best_dist, best_tid = s, d, tid

    if previous_trip_id is not None and best_tid != previous_trip_id:
        prev_trip = gtfs.get_trip(previous_trip_id)
        if prev_trip is not None:
            prev_shape = gtfs.get_shape_linestring(prev_trip.shape_id)
            if prev_shape is not None and _stickiness_dist(
                prev_shape, vx, vy, bearing, best_dist
            ) is not None:
                best_tid = previous_trip_id
                best_score, best_dist = score_trip(prev_trip, vx, vy, bearing, gtfs)

    inferred_route = None
    if best_tid:
        t = gtfs.get_trip(best_tid)
        inferred_route = t.route_id if t else route_id

    return InferenceResult(
        vehicle_id=vehicle_id, timestamp=now,
        reported_trip_id=reported_trip_id, inferred_trip_id=best_tid,
        inferred_route_id=inferred_route, score=best_score,
        high_confidence=(best_dist < MAX_DIST_M / 2),
        off_route=False, route_status=RouteStatus.ON_ROUTE,
    )


# ---------------------------------------------------------------------------
# Batch inference with off-route state machine
# ---------------------------------------------------------------------------

def infer_trips(df: pd.DataFrame, gtfs: GTFSStatic) -> pd.DataFrame:
    """
    Apply trip inference + off-route detection to a snapshot DataFrame.

    Input columns: vehicle_id, route_id, trip_id, lat, lon, bearing, timestamp.

    Output adds: reported_trip_id, inferred_trip_id (None when off_route),
                 inferred_route_id, inference_score, high_confidence,
                 off_route, route_status.
    """
    df = df.copy()
    df["reported_trip_id"] = df["trip_id"]
    df.sort_values(["vehicle_id", "timestamp"], inplace=True, ignore_index=True)

    known_trips = set(gtfs._trip_index.keys())

    inferred_tids: list[str | None] = []
    inferred_routes: list[str | None] = []
    scores: list[float] = []
    confidences: list[bool] = []
    off_routes: list[bool] = []
    route_statuses: list[str] = []

    prev_trip: dict[str, str | None] = {}
    trackers: dict[str, VehicleRouteTracker] = {}
    candidates_cache: dict[tuple, list[str]] = {}

    for _, row in df.iterrows():
        vid = str(row["vehicle_id"])
        route_id = str(row["route_id"]) if pd.notna(row.get("route_id")) else ""

        if not route_id or pd.isna(row.get("lat")) or pd.isna(row.get("lon")):
            inferred_tids.append(None)
            inferred_routes.append(None)
            scores.append(math.inf)
            confidences.append(False)
            off_routes.append(False)
            route_statuses.append(RouteStatus.ON_ROUTE)
            continue

        ts = row["timestamp"]
        now = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        vx, vy = _project_xy(float(row["lon"]), float(row["lat"]))
        bearing = float(row["bearing"]) if pd.notna(row.get("bearing")) else None
        reported_tid = row.get("trip_id")
        prev_tid = prev_trip.get(vid)
        tracker = trackers.setdefault(vid, VehicleRouteTracker())

        # ----------------------------------------------------------------
        # Fast path: known valid trip_id within MAX_DIST_M/2
        # ----------------------------------------------------------------
        if reported_tid and reported_tid in known_trips:
            trip = gtfs.get_trip(reported_tid)
            shape = gtfs.get_shape_linestring(trip.shape_id) if trip else None
            if shape is not None:
                dist = shape.distance(Point(vx, vy))
                if dist < MAX_DIST_M / 2:
                    b_diff = _bearing_diff_for_shape(shape, vx, vy, bearing) if bearing else None
                    status = tracker.update(dist, b_diff)
                    prev_trip[vid] = reported_tid if status == RouteStatus.ON_ROUTE else None
                    is_off = status == RouteStatus.OFF_ROUTE
                    inferred_tids.append(None if is_off else reported_tid)
                    inferred_routes.append(None if is_off else trip.route_id)
                    scores.append(dist / MAX_DIST_M)
                    confidences.append(not is_off)
                    off_routes.append(is_off)
                    route_statuses.append(status)
                    continue

        # ----------------------------------------------------------------
        # Slow path: score all active candidates
        # ----------------------------------------------------------------
        cache_key = (route_id, now.strftime("%Y%m%d"), now.hour)
        if cache_key not in candidates_cache:
            candidates_cache[cache_key] = gtfs.active_trips_for_route(route_id, now)
        candidates = candidates_cache[cache_key]

        best_tid = None
        best_score = math.inf
        best_dist = math.inf
        min_route_dist = math.inf   # nearest shape dist, no cap — for off-route tracking

        for tid in candidates:
            trip = gtfs.get_trip(tid)
            if trip is None:
                continue
            s, d = score_trip(trip, vx, vy, bearing, gtfs)
            if d != math.inf:
                min_route_dist = min(min_route_dist, d)
            if s < best_score:
                best_score, best_dist, best_tid = s, d, tid

        # Hysteresis (bearing-gated: never hold a reverse-direction trip)
        if prev_tid is not None and best_tid != prev_tid:
            prev_trip_obj = gtfs.get_trip(prev_tid)
            if prev_trip_obj is not None:
                prev_shape = gtfs.get_shape_linestring(prev_trip_obj.shape_id)
                if prev_shape is not None:
                    held = _stickiness_dist(prev_shape, vx, vy, bearing, best_dist)
                    if held is not None:
                        best_tid = prev_tid
                        best_score, best_dist = score_trip(prev_trip_obj, vx, vy, bearing, gtfs)
                        min_route_dist = min(min_route_dist, held)

        # Bearing diff against winning shape (for off-route tracker)
        b_diff_for_tracker: float | None = None
        if best_tid and bearing is not None:
            win_trip = gtfs.get_trip(best_tid)
            if win_trip:
                win_shape = gtfs.get_shape_linestring(win_trip.shape_id)
                if win_shape:
                    b_diff_for_tracker = _bearing_diff_for_shape(win_shape, vx, vy, bearing)

        # Update off-route state machine
        status = tracker.update(min_route_dist, b_diff_for_tracker)
        is_off = status == RouteStatus.OFF_ROUTE

        inferred_route = None
        if best_tid and not is_off:
            t = gtfs.get_trip(best_tid)
            inferred_route = t.route_id if t else route_id

        prev_trip[vid] = (best_tid if not is_off else None)
        inferred_tids.append(None if is_off else best_tid)
        inferred_routes.append(inferred_route)
        scores.append(best_score)
        confidences.append(best_dist < MAX_DIST_M / 2 and not is_off)
        off_routes.append(is_off)
        route_statuses.append(status)

    df["inferred_trip_id"] = inferred_tids
    df["inferred_route_id"] = inferred_routes
    df["inference_score"] = scores
    df["high_confidence"] = confidences
    df["off_route"] = off_routes
    df["route_status"] = [s.value if isinstance(s, RouteStatus) else s for s in route_statuses]
    return df
