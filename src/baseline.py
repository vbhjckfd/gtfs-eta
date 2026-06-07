"""
Baseline ETA model: schedule + current delay propagation.

Given a vehicle's position and corrected trip_id, finds the last passed stop,
computes the current delay, and applies it uniformly to future scheduled times.

This is the benchmark that the ML model must beat.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import NamedTuple

import pandas as pd
from shapely.geometry import Point

from src.gtfs_static import GTFSStatic, StopTime, TripInfo, _parse_gtfs_time_utc, _project_xy


class ETAResult(NamedTuple):
    stop_id: str
    stop_sequence: int
    scheduled_arrival: datetime | None
    estimated_arrival: datetime | None
    delay_sec: float | None


def compute_eta(
    trip_id: str,
    lat: float,
    lon: float,
    now: datetime,
    gtfs: GTFSStatic,
) -> list[ETAResult]:
    """
    Return ETA for every upcoming stop on the trip.

    Steps:
    1. Project vehicle position onto the trip shape.
    2. Find the last passed stop (vehicle is past its shape-distance).
    3. Compute current delay at that stop.
    4. Propagate delay to all remaining stops.
    """
    trip = gtfs.get_trip(trip_id)
    if trip is None:
        return []

    shape = gtfs.get_shape_linestring(trip.shape_id)
    if shape is None:
        return _schedule_only(trip, now, gtfs)

    vx, vy = _project_xy(lon, lat)
    vehicle_dist = shape.project(Point(vx, vy))

    base_date = now.date()

    # Build (stop_time, dist_along_shape) pairs
    enriched: list[tuple[StopTime, float]] = []
    for st in trip.stop_times:
        d = gtfs.get_stop_distance_along_shape(trip.shape_id, st.stop_id)
        if d is None:
            # Fall back: project the stop's coordinates
            stop_info = gtfs.get_stop(st.stop_id)
            if stop_info is not None:
                d = shape.project(Point(stop_info.x, stop_info.y))
            else:
                d = 0.0
        enriched.append((st, d))

    # Last passed stop = highest dist_along that is ≤ vehicle_dist
    last_passed_idx: int | None = None
    for i, (st, d) in enumerate(enriched):
        if d <= vehicle_dist:
            last_passed_idx = i

    delay_sec = _compute_delay(enriched, last_passed_idx, now, base_date, gtfs.feed_tz)

    results: list[ETAResult] = []
    for st, d in enriched:
        sched = _parse_gtfs_time_utc(st.arrival_time or st.departure_time, base_date, gtfs.feed_tz)
        if d > vehicle_dist:  # future stop
            eta = (sched + timedelta(seconds=delay_sec)) if (sched and delay_sec is not None) else sched
            results.append(ETAResult(
                stop_id=st.stop_id,
                stop_sequence=st.stop_sequence,
                scheduled_arrival=sched,
                estimated_arrival=eta,
                delay_sec=delay_sec,
            ))

    return results


def _compute_delay(
    enriched: list[tuple[StopTime, float]],
    last_passed_idx: int | None,
    now: datetime,
    base_date,
    feed_tz=None,
) -> float | None:
    """Delay in seconds at the last passed stop. Positive = late."""
    if last_passed_idx is None:
        return 0.0

    st, _ = enriched[last_passed_idx]
    from zoneinfo import ZoneInfo
    tz = feed_tz or ZoneInfo("Europe/Kiev")
    from src.gtfs_static import _parse_gtfs_time_utc
    sched = _parse_gtfs_time_utc(st.departure_time or st.arrival_time, base_date, tz)
    if sched is None:
        return 0.0

    now_utc = now if now.tzinfo is not None else now.replace(tzinfo=ZoneInfo("UTC"))
    return (now_utc - sched).total_seconds()


def _schedule_only(trip: TripInfo, now: datetime, gtfs: GTFSStatic) -> list[ETAResult]:
    """Fallback when no shape is available: return scheduled times with zero delay."""
    base_date = now.date()
    results = []
    for st in trip.stop_times:
        sched = _parse_gtfs_time(st.arrival_time or st.departure_time, base_date)
        results.append(ETAResult(
            stop_id=st.stop_id,
            stop_sequence=st.stop_sequence,
            scheduled_arrival=sched,
            estimated_arrival=sched,
            delay_sec=0.0,
        ))
    return results


def compute_eta_batch(
    df: pd.DataFrame,
    gtfs: GTFSStatic,
    use_inferred: bool = True,
) -> pd.DataFrame:
    """
    Compute baseline ETAs for a snapshot DataFrame.

    Input columns: vehicle_id, lat, lon, timestamp,
                   inferred_trip_id (or trip_id if use_inferred=False).

    Returns long-format DataFrame:
        vehicle_id, timestamp, stop_id, stop_sequence,
        scheduled_arrival, estimated_arrival, delay_sec
    """
    trip_col = "inferred_trip_id" if use_inferred else "trip_id"
    rows = []
    for _, row in df.iterrows():
        tid = row.get(trip_col)
        if not tid or pd.isna(tid):
            continue
        ts = row["timestamp"]
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
        etas = compute_eta(
            trip_id=str(tid),
            lat=float(row["lat"]),
            lon=float(row["lon"]),
            now=ts,
            gtfs=gtfs,
        )
        for eta in etas:
            rows.append({
                "vehicle_id": row["vehicle_id"],
                "timestamp": row["timestamp"],
                "stop_id": eta.stop_id,
                "stop_sequence": eta.stop_sequence,
                "scheduled_arrival": eta.scheduled_arrival,
                "estimated_arrival": eta.estimated_arrival,
                "delay_sec": eta.delay_sec,
            })

    return pd.DataFrame(rows)
