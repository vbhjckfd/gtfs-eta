"""
Build labeled training data from GTFS-RT snapshots.

For each (vehicle_id, inferred_trip_id) trajectory:
  1. Project every position onto the trip's shape → distance along route (m).
  2. Detect when the vehicle passes each scheduled stop.
  3. Record the actual passing timestamp.

Output: long-format DataFrame with actual arrival times per stop.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from shapely.geometry import Point

from src.gtfs_static import GTFSStatic, _parse_gtfs_time_utc, _project_xy


def _project_vehicle_positions(
    traj: pd.DataFrame,
    shape,
) -> pd.Series:
    """Return a Series of distances along *shape* for each row in *traj*."""
    dists = []
    for _, row in traj.iterrows():
        if pd.isna(row["lat"]) or pd.isna(row["lon"]):
            dists.append(float("nan"))
            continue
        vx, vy = _project_xy(float(row["lon"]), float(row["lat"]))
        dists.append(shape.project(Point(vx, vy)))
    return pd.Series(dists, index=traj.index)


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

    traj["dist_along"] = _project_vehicle_positions(traj, shape)

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
