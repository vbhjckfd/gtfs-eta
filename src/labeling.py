"""
Build labeled training data from GTFS-RT snapshots.

For each (vehicle_id, inferred_trip_id) trajectory:
  1. Project every position onto the trip's shape → distance along route (m).
  2. Detect when the vehicle passes each scheduled stop.
  3. Record the actual passing timestamp.

Two outputs:
  build_labels        — arrival events per stop (one row per stop crossing).
  build_training_rows — snapshot-anchored examples: one row per
                        (snapshot, upcoming stop) pair, so the model sees the
                        vehicle at every point along a segment (including
                        sitting right at a stop), not only at stop crossings.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from shapely.geometry import Point

from src.gtfs_static import GTFSStatic, _parse_gtfs_time_utc, _project_xy

# Vehicles essentially never sustain speeds above this (GPS jitter aside).
# Used to reject spurious jumps in raw per-point shape projection: on a
# shape that loops or comes close to itself (e.g. route 122's out-and-back
# to a village), a snapshot can snap onto a distant, wrong occurrence of the
# shape just because it's geographically nearby, which _detect_stop_crossings
# then reads as an impossibly fast arrival.
_MAX_VEHICLE_SPEED_MPS = 40.0


def _project_vehicle_positions(
    traj: pd.DataFrame,
    shape,
    constrain: bool = False,
) -> pd.Series:
    """Return a Series of distances along *shape* for each row in *traj*.

    The projections are batched through shapely's vectorised
    ``line_locate_point`` (C-level over the whole trajectory at once) instead of
    a per-row ``shape.project`` call, which dominated pipeline runtime.

    Raw per-point projection is unconstrained and ambiguous on
    self-intersecting shapes: a snapshot can snap onto a distant, wrong
    occurrence just because it's geographically nearby, which
    _detect_stop_crossings then reads as an impossibly fast arrival.
    ``constrain=True`` (pass only for shapes flagged by
    GTFSStatic.is_ambiguous_shape) corrects for this: assuming rows are
    chronologically ordered (callers sort by timestamp), the result is
    monotonic and speed-capped — clamped to
    [previous, previous + _MAX_VEHICLE_SPEED_MPS * elapsed_seconds] — UNLESS
    the next snapshot confirms the jump (stays past the midpoint between the
    old and new distance), in which case it's accepted outright. A single
    misprojection is almost always followed by the next snapshot reverting
    back near the true trajectory; genuine fast progress (e.g. catching up
    after a snapshot gap) keeps going.

    This correction is scoped to ambiguous shapes only: on ordinary shapes it
    costs accuracy rather than adding it — false-positive clamps on
    legitimate fast segments or gappy snapshot coverage create a small
    systematic lag with nothing to do with self-intersecting geometry.
    """
    import shapely

    lon = traj["lon"].to_numpy(dtype=float)
    lat = traj["lat"].to_numpy(dtype=float)
    valid = ~(np.isnan(lon) | np.isnan(lat))

    raw = np.full(len(traj), np.nan)
    if valid.any():
        xs = np.empty(valid.sum())
        ys = np.empty(valid.sum())
        for i, (lo, la) in enumerate(zip(lon[valid], lat[valid])):
            xs[i], ys[i] = _project_xy(lo, la)
        raw[valid] = shapely.line_locate_point(shape, shapely.points(xs, ys))

    if not constrain:
        return pd.Series(raw, index=traj.index)

    out = np.full(len(traj), np.nan)
    times = traj["timestamp"].to_numpy()
    idx = np.flatnonzero(valid & ~np.isnan(raw))

    prev_dist = None
    prev_time = None
    k = 0
    while k < len(idx):
        i = idx[k]
        if prev_dist is None:
            out[i] = raw[i]
            prev_dist, prev_time = raw[i], times[i]
            k += 1
            continue

        elapsed = max(
            (pd.Timestamp(times[i]) - pd.Timestamp(prev_time)).total_seconds(), 0.0
        )
        max_forward = prev_dist + _MAX_VEHICLE_SPEED_MPS * elapsed

        if raw[i] <= max_forward:
            d = max(raw[i], prev_dist)
        else:
            confirmed = False
            if k + 1 < len(idx):
                j = idx[k + 1]
                midpoint = (prev_dist + raw[i]) / 2
                if raw[j] >= midpoint:
                    confirmed = True
            d = raw[i] if confirmed else max(min(raw[i], max_forward), prev_dist)

        out[i] = d
        prev_dist, prev_time = d, times[i]
        k += 1

    return pd.Series(out, index=traj.index)


def _detect_stop_crossings(
    traj: pd.DataFrame,
    dist_col: str,
    stop_dists: list[tuple[str, int, float]],  # (stop_id, stop_seq, dist_along)
) -> list[dict]:
    """
    For each stop, find the first timestamp where the vehicle's dist_along
    crosses (reaches or passes) the stop's dist_along.

    Returns a list of dicts with keys:
        stop_id, stop_sequence, actual_arrival (UTC datetime or None)
    """
    traj = traj.dropna(subset=[dist_col]).sort_values("timestamp")
    vehicle_dists = traj[dist_col].values
    vehicle_times = traj["timestamp"].values

    results = []
    for stop_id, stop_seq, stop_dist in stop_dists:
        # First index where vehicle dist >= stop dist
        idxs = np.where(vehicle_dists >= stop_dist)[0]
        if len(idxs) == 0:
            actual = None
        else:
            i = idxs[0]
            # Interpolate between the snapshot before and after the crossing
            if i > 0:
                d_before = vehicle_dists[i - 1]
                d_after = vehicle_dists[i]
                t_before = pd.Timestamp(vehicle_times[i - 1]).timestamp()
                t_after = pd.Timestamp(vehicle_times[i]).timestamp()
                span = d_after - d_before
                if span > 0:
                    frac = (stop_dist - d_before) / span
                    t_cross = t_before + frac * (t_after - t_before)
                    actual = datetime.fromtimestamp(t_cross, tz=timezone.utc)
                else:
                    actual = datetime.fromtimestamp(
                        pd.Timestamp(vehicle_times[i]).timestamp(), tz=timezone.utc
                    )
            else:
                actual = datetime.fromtimestamp(
                    pd.Timestamp(vehicle_times[i]).timestamp(), tz=timezone.utc
                )

        results.append({
            "stop_id": stop_id,
            "stop_sequence": stop_seq,
            "actual_arrival": actual,
        })

    return results


def label_trajectory(
    vehicle_id: str,
    trip_id: str,
    traj: pd.DataFrame,
    gtfs: GTFSStatic,
) -> pd.DataFrame | None:
    """
    Build labeled rows for one (vehicle, trip) trajectory.

    *traj* must be sorted by timestamp and contain lat/lon columns.
    Off-route segments are split out: labeling stops at the first off-route
    snapshot and resumes only after the vehicle returns on-route.
    Returns None if the trip or shape cannot be resolved.
    """
    trip = gtfs.get_trip(trip_id)
    if trip is None:
        return None

    shape = gtfs.get_shape_linestring(trip.shape_id)
    if shape is None:
        return None

    if traj.empty or traj["lat"].isna().all():
        return None

    traj = traj.copy()

    # Drop off-route snapshots. This correctly handles vehicles that leave
    # mid-trip and return: only on-route windows are labeled.
    if "off_route" in traj.columns:
        traj = traj[~traj["off_route"]].copy()
        if traj.empty:
            return None

    traj["dist_along"] = _project_vehicle_positions(
        traj, shape, constrain=gtfs.is_ambiguous_shape(trip.shape_id)
    )

    # Collect stop distances
    stop_dists: list[tuple[str, int, float]] = []
    base_date = traj["timestamp"].iloc[0].date()
    for st in trip.stop_times:
        d = gtfs.get_stop_distance_along_shape(trip.shape_id, st.stop_id)
        if d is None:
            stop_info = gtfs.get_stop(st.stop_id)
            if stop_info is not None:
                d = shape.project(Point(stop_info.x, stop_info.y))
            else:
                continue
        stop_dists.append((st.stop_id, st.stop_sequence, d))

    crossings = _detect_stop_crossings(traj, "dist_along", stop_dists)
    if not crossings:
        return None

    rows = []
    for crossing, st in zip(crossings, trip.stop_times):
        sched_utc = _parse_gtfs_time_utc(
            st.arrival_time or st.departure_time, base_date, gtfs.feed_tz
        )
        actual = crossing["actual_arrival"]
        if actual is None:
            continue

        delay_sec = (actual - sched_utc).total_seconds() if sched_utc is not None else None

        rows.append({
            "vehicle_id": vehicle_id,
            "trip_id": trip_id,
            "route_id": trip.route_id,
            "stop_id": st.stop_id,
            "stop_sequence": st.stop_sequence,
            "scheduled_arrival": sched_utc,
            "actual_arrival": actual,
            "delay_sec": delay_sec,
            "date": base_date,
            "hour": actual.astimezone(gtfs.feed_tz).hour,
            "day_of_week": actual.weekday(),
        })

    if not rows:
        return None

    return pd.DataFrame(rows)


# Snapshot-anchored training rows ------------------------------------------

MAX_STOPS_AHEAD = 10

# Progress speed is only trusted when the gap between consecutive snapshots is
# sane; otherwise the row gets SPEED_UNKNOWN and the model treats it as its
# own category (no NaN handling needed in the exported pure-Python trees).
SPEED_UNKNOWN = -1.0
_SPEED_MIN_GAP_SEC = 3.0
_SPEED_MAX_GAP_SEC = 120.0
_SPEED_MAX_BACKWARD_M = 30.0  # projection jitter tolerance

MAX_HORIZON_SEC = 3600.0


def _progress_speeds(dists: np.ndarray, times_sec: np.ndarray) -> np.ndarray:
    """Per-snapshot speed (m/s) from consecutive shape projections."""
    speeds = np.full(len(dists), SPEED_UNKNOWN)
    for i in range(1, len(dists)):
        dt = times_sec[i] - times_sec[i - 1]
        dd = dists[i] - dists[i - 1]
        if not (_SPEED_MIN_GAP_SEC <= dt <= _SPEED_MAX_GAP_SEC):
            continue
        if dd < -_SPEED_MAX_BACKWARD_M:
            continue  # projection glitch (jumped backwards) — don't trust
        speeds[i] = max(0.0, dd) / dt
    return speeds


def training_rows_for_trajectory(
    vehicle_id: str,
    trip_id: str,
    traj: pd.DataFrame,
    gtfs: GTFSStatic,
    max_stops_ahead: int = MAX_STOPS_AHEAD,
) -> pd.DataFrame | None:
    """
    Snapshot-anchored rows for one (vehicle, trip) trajectory.

    For every snapshot, emit one row per upcoming stop (up to max_stops_ahead)
    with the observed arrival time at that stop as the target. This matches
    the live-serving question exactly: "given the vehicle *here*, when does it
    reach stop S?" — including remaining distances near zero.
    """
    trip = gtfs.get_trip(trip_id)
    if trip is None:
        return None

    shape = gtfs.get_shape_linestring(trip.shape_id)
    if shape is None:
        return None

    if traj.empty or traj["lat"].isna().all():
        return None

    traj = traj.copy()
    if "off_route" in traj.columns:
        traj = traj[~traj["off_route"]].copy()
        if traj.empty:
            return None

    traj["dist_along"] = _project_vehicle_positions(
        traj, shape, constrain=gtfs.is_ambiguous_shape(trip.shape_id)
    )
    traj = traj.dropna(subset=["dist_along"]).sort_values("timestamp")
    if traj.empty:
        return None

    # Stop distances along the shape
    stop_dists: list[tuple[str, int, float]] = []
    base_date = traj["timestamp"].iloc[0].date()
    for st in trip.stop_times:
        d = gtfs.get_stop_distance_along_shape(trip.shape_id, st.stop_id)
        if d is None:
            stop_info = gtfs.get_stop(st.stop_id)
            if stop_info is None:
                continue
            d = shape.project(Point(stop_info.x, stop_info.y))
        stop_dists.append((st.stop_id, st.stop_sequence, d))

    crossings = _detect_stop_crossings(traj, "dist_along", stop_dists)
    if not crossings:
        return None

    # Stops that were actually reached, ordered by distance along the shape
    arrived = [
        (sd[0], sd[1], sd[2], c["actual_arrival"])
        for sd, c in zip(stop_dists, crossings)
        if c["actual_arrival"] is not None
    ]
    arrived.sort(key=lambda t: t[2])
    if not arrived:
        return None

    n_stops_total = len(trip.stop_times)
    seq_to_index = {st.stop_sequence: i for i, st in enumerate(trip.stop_times)}

    dists = traj["dist_along"].to_numpy(dtype=float)
    times = traj["timestamp"].to_numpy()
    times_sec = np.array([pd.Timestamp(t).timestamp() for t in times])
    speeds = _progress_speeds(dists, times_sec)
    arr_dists = np.array([a[2] for a in arrived])

    rows = []
    for i in range(len(traj)):
        d_vehicle = dists[i]
        t_vehicle = times_sec[i]
        # Upcoming = stops strictly ahead of the vehicle's projected position
        start = int(np.searchsorted(arr_dists, d_vehicle, side="right"))
        emitted = 0
        for stop_id, stop_seq, stop_dist, arrival in arrived[start:]:
            if emitted >= max_stops_ahead:
                break
            seconds_to_arrival = arrival.timestamp() - t_vehicle
            if seconds_to_arrival < 0:
                continue  # already passed in time despite distance jitter
            if seconds_to_arrival > MAX_HORIZON_SEC:
                break
            emitted += 1
            rows.append({
                "vehicle_id": vehicle_id,
                "trip_id": trip_id,
                "route_id": trip.route_id,
                "date": base_date,
                "snapshot_ts": datetime.fromtimestamp(t_vehicle, tz=timezone.utc),
                "dist_along_m": d_vehicle,
                "progress_speed_mps": speeds[i],
                "stop_id": stop_id,
                "stop_sequence": stop_seq,
                "stop_dist_along_m": stop_dist,
                "stops_ahead": emitted,
                "stops_remaining": n_stops_total - 1 - seq_to_index.get(stop_seq, 0),
                "actual_arrival": arrival,
                "seconds_to_arrival": seconds_to_arrival,
            })

    if not rows:
        return None
    return pd.DataFrame(rows)


def build_training_rows(
    df: pd.DataFrame,
    gtfs: GTFSStatic,
    trip_col: str = "inferred_trip_id",
    max_stops_ahead: int = MAX_STOPS_AHEAD,
) -> pd.DataFrame:
    """
    Build the snapshot-anchored training dataset from a snapshot DataFrame.

    Input: snapshot df with columns vehicle_id, trip_id / inferred_trip_id,
           lat, lon, timestamp (and optionally off_route).

    Output: one row per (snapshot, upcoming stop) — see
    training_rows_for_trajectory for the schema.
    """
    if "off_route" in df.columns:
        df = df[~df["off_route"]].copy()

    groups = df.groupby(["vehicle_id", trip_col], sort=False)
    pieces: list[pd.DataFrame] = []

    for (vehicle_id, trip_id), traj in groups:
        if not trip_id or pd.isna(trip_id):
            continue
        result = training_rows_for_trajectory(
            vehicle_id=str(vehicle_id),
            trip_id=str(trip_id),
            traj=traj.sort_values("timestamp"),
            gtfs=gtfs,
            max_stops_ahead=max_stops_ahead,
        )
        if result is not None:
            pieces.append(result)

    if not pieces:
        return pd.DataFrame()

    out = pd.concat(pieces, ignore_index=True)
    out.sort_values(["date", "trip_id", "snapshot_ts", "stop_sequence"],
                    inplace=True, ignore_index=True)
    return out


def build_labels(
    df: pd.DataFrame,
    gtfs: GTFSStatic,
    trip_col: str = "inferred_trip_id",
) -> pd.DataFrame:
    """
    Build the full labeled dataset from a snapshot DataFrame.

    Input: snapshot df with columns vehicle_id, trip_id / inferred_trip_id,
           lat, lon, timestamp.

    Output: long-format DataFrame suitable for feature engineering.
    """
    # Pre-filter: rows where off_route=True have inferred_trip_id=None already,
    # but be explicit so label_trajectory receives only on-route snapshots.
    if "off_route" in df.columns:
        df = df[~df["off_route"]].copy()

    groups = df.groupby(["vehicle_id", trip_col], sort=False)
    pieces: list[pd.DataFrame] = []

    for (vehicle_id, trip_id), traj in groups:
        if not trip_id or pd.isna(trip_id):
            continue
        result = label_trajectory(
            vehicle_id=str(vehicle_id),
            trip_id=str(trip_id),
            traj=traj.sort_values("timestamp"),
            gtfs=gtfs,
        )
        if result is not None:
            pieces.append(result)

    if not pieces:
        return pd.DataFrame()

    labeled = pd.concat(pieces, ignore_index=True)
    labeled.sort_values(["date", "trip_id", "stop_sequence"], inplace=True, ignore_index=True)
    return labeled
