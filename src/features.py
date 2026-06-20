"""
Feature engineering for the gradient-boosted ETA model.

Target: seconds_to_arrival
  = actual_arrival_at_target_stop - snapshot_timestamp

Training rows are snapshot-anchored (see labeling.build_training_rows): the
vehicle can be anywhere along the route — mid-segment, or sitting right at a
stop — which makes the training distribution identical to what live inference
asks. The key features:
  - remaining_dist_m: shape distance from the vehicle's projected position to
    the target stop (near zero when the bus is at the stop)
  - sched_remaining_sec: the schedule's own expectation for that remaining
    distance, interpolated at the vehicle's position (a strong prior)
  - progress_speed_mps: observed speed over the last snapshot interval
    (SPEED_UNKNOWN sentinel when not measurable)
  - stops_ahead: prediction horizon in stops (1 = next stop)
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.gtfs_static import GTFSStatic, TripInfo, _parse_gtfs_time
from src.labeling import SPEED_UNKNOWN  # re-exported for inference callers

_UA_HOLIDAYS: set[tuple[int, int]] = {
    (1, 1),   # New Year
    (1, 7),   # Christmas (Julian)
    (3, 8),   # International Women's Day
    (5, 1),   # Labour Day
    (5, 9),   # Victory Day
    (6, 28),  # Constitution Day
    (8, 24),  # Independence Day
    (10, 14), # Defender of Ukraine Day
    (12, 25), # Christmas (Gregorian)
}

_DUMMY_DATE = date(2000, 1, 1)  # only schedule *deltas* are used


def _is_holiday(d: date) -> bool:
    return (d.month, d.day) in _UA_HOLIDAYS


def build_sched_profile(gtfs: GTFSStatic, trip: TripInfo) -> list[tuple[str, int, float, float]]:
    """
    Per-stop schedule profile for a trip:
        [(stop_id, stop_sequence, dist_along_m, sched_cum_sec), ...]

    sched_cum_sec is the scheduled seconds elapsed since the trip's first stop
    (cumulative, so differences give scheduled segment times and absolute
    clock/timezone handling is unnecessary). Stops with unparseable times carry
    the previous cumulative value. Sorted by distance along the shape.
    """
    profile = []
    t0 = None
    cum = 0.0
    for st in trip.stop_times:
        d = gtfs.get_stop_distance_along_shape(trip.shape_id, st.stop_id)
        if d is None:
            stop_info = gtfs.get_stop(st.stop_id)
            if stop_info is None:
                continue
            shape = gtfs.get_shape_linestring(trip.shape_id)
            if shape is None:
                continue
            from shapely.geometry import Point
            d = shape.project(Point(stop_info.x, stop_info.y))

        t = _parse_gtfs_time(st.arrival_time or st.departure_time, _DUMMY_DATE)
        if t is not None:
            if t0 is None:
                t0 = t
            cum = max(cum, (t - t0).total_seconds())
        profile.append((st.stop_id, st.stop_sequence, d, cum))

    profile.sort(key=lambda p: p[2])
    return profile


def sched_sec_at_dist(profile: list[tuple[str, int, float, float]], d: float) -> float:
    """Interpolated cumulative scheduled seconds at shape distance *d*."""
    if not profile:
        return 0.0
    if d <= profile[0][2]:
        return profile[0][3]
    for i in range(1, len(profile)):
        d_prev, s_prev = profile[i - 1][2], profile[i - 1][3]
        d_next, s_next = profile[i][2], profile[i][3]
        if d <= d_next:
            span = d_next - d_prev
            if span <= 0:
                return s_next
            frac = (d - d_prev) / span
            return s_prev + frac * (s_next - s_prev)
    return profile[-1][3]


def compute_features_for_training(
    training_rows: pd.DataFrame,
    gtfs: GTFSStatic,
) -> pd.DataFrame:
    """
    Build the model feature matrix from labeling.build_training_rows output.

    Vectorised per trip group (the input easily reaches millions of rows).
    Returns a DataFrame with FEATURE_COLS + TARGET_COL + date, plus
    `sched_remaining_sec` doubling as the baseline in train.py.
    """
    rows = training_rows.copy()
    rows["snapshot_ts"] = pd.to_datetime(rows["snapshot_ts"], utc=True)
    if "date" in rows.columns:
        rows["date"] = pd.to_datetime(rows["date"]).dt.date

    pieces = []
    for trip_id, grp in rows.groupby("trip_id", sort=False):
        trip = gtfs.get_trip(str(trip_id))
        if trip is None:
            continue
        shape_len = max(gtfs.get_shape_length(trip.shape_id), 1.0)
        profile = build_sched_profile(gtfs, trip)
        if not profile:
            continue
        prof_dists = np.array([p[2] for p in profile])
        prof_cums = np.array([p[3] for p in profile])
        sched_by_seq = {seq: cum for (_, seq, _, cum) in profile}

        d_vehicle = grp["dist_along_m"].to_numpy(dtype=float)
        d_target = grp["stop_dist_along_m"].to_numpy(dtype=float)
        sched_at_pos = np.interp(d_vehicle, prof_dists, prof_cums)
        sched_target = (
            grp["stop_sequence"].astype(int).map(sched_by_seq)
            .fillna(pd.Series(sched_at_pos, index=grp.index))
            .to_numpy(dtype=float)
        )

        snap = grp["snapshot_ts"].dt
        dow = snap.weekday.to_numpy()
        month_day = snap.month * 100 + snap.day
        holiday = month_day.isin({m * 100 + d for m, d in _UA_HOLIDAYS}).astype(int)

        stops_ahead_arr = grp["stops_ahead"].astype(int).to_numpy()
        sched_rem = np.maximum(0.0, sched_target - sched_at_pos)
        rem_dist = np.maximum(0.0, d_target - d_vehicle)
        stops_ahead_safe = np.maximum(1, stops_ahead_arr)
        speed = grp["progress_speed_mps"].to_numpy(dtype=float)

        pieces.append(pd.DataFrame({
            "route_id": trip.route_id,
            "stop_sequence": grp["stop_sequence"].astype(int).to_numpy(),
            "stops_ahead": stops_ahead_arr,
            "hour": snap.hour.to_numpy(),
            "day_of_week": dow,
            "month": snap.month.to_numpy(),
            "is_weekend": (dow >= 5).astype(int),
            "is_holiday": holiday.to_numpy(),
            "remaining_dist_m": rem_dist,
            "progress_speed_mps": speed,
            "stops_remaining": grp["stops_remaining"].astype(int).to_numpy(),
            "trip_progress_frac": d_target / shape_len,
            "dist_per_stop_m": rem_dist / stops_ahead_safe,
            # Reference-only columns (not model features):
            "sched_remaining_sec": sched_rem,
            "date": grp["date"].to_numpy(),
            TARGET_COL: grp["seconds_to_arrival"].to_numpy(dtype=float),
        }))

    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def compute_features_for_inference(
    trip_id: str,
    vehicle_dist_m: float,
    snapshot_time,
    progress_speed_mps: float,
    gtfs: GTFSStatic,
    max_stops_ahead: int = 10,
    priors: dict | None = None,
) -> pd.DataFrame:
    """One feature row per upcoming stop for live inference."""
    trip = gtfs.get_trip(trip_id)
    if trip is None:
        return pd.DataFrame()

    shape_len = max(gtfs.get_shape_length(trip.shape_id), 1.0)
    snap_ts = pd.Timestamp(snapshot_time)
    profile = build_sched_profile(gtfs, trip)
    seq_to_index = {st.stop_sequence: i for i, st in enumerate(trip.stop_times)}
    n_stops_total = len(trip.stop_times)

    sched_at_pos = sched_sec_at_dist(profile, vehicle_dist_m)
    is_holiday = int(_is_holiday(snap_ts.date()))

    rows = []
    stops_ahead = 0
    for stop_id, stop_seq, d_target, sched_cum in profile:
        if d_target <= vehicle_dist_m:
            continue
        stops_ahead += 1
        if stops_ahead > max_stops_ahead:
            break
        sched_rem = max(0.0, sched_cum - sched_at_pos)
        rem_dist = d_target - vehicle_dist_m
        rows.append({
            "trip_id": trip_id,
            "stop_id": stop_id,
            "route_id": trip.route_id,
            "stop_sequence": stop_seq,
            "stops_ahead": stops_ahead,
            "hour": snap_ts.hour,
            "day_of_week": snap_ts.weekday(),
            "month": snap_ts.month,
            "is_weekend": int(snap_ts.weekday() >= 5),
            "is_holiday": is_holiday,
            "remaining_dist_m": rem_dist,
            "progress_speed_mps": progress_speed_mps,
            "stops_remaining": n_stops_total - 1 - seq_to_index.get(stop_seq, 0),
            "trip_progress_frac": d_target / shape_len,
            "dist_per_stop_m": rem_dist / max(1, stops_ahead),
            # Reference-only: schedule interpolation kept for sanity checks.
            "sched_remaining_sec": sched_rem,
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = apply_priors(result, priors)
    return result


# Fallbacks used when priors are unavailable (no lookup match or no priors at all).
_FALLBACK_SPEED_MPS = 5.0   # ~18 km/h urban bus
_FALLBACK_TPS_SEC   = 40.0  # seconds per stop including dwell


def apply_priors(df: pd.DataFrame, priors: dict | None) -> pd.DataFrame:
    """Enrich a feature DataFrame with route+hour speed/dwell priors.

    Adds three columns:
      speed_eta_warm       — remaining_dist / effective_speed (warm-started)
      hist_speed_mps       — route+hour historical median speed
      hist_travel_time_est — stops_ahead * historical seconds-per-stop (dwell-aware ETA)

    When priors is None or a lookup is missing, global fallback constants are used.
    Called by train.py after the time split (so priors are computed on train data only)
    and by compute_features_for_inference when priors are available.
    """
    df = df.copy()

    if priors:
        lookup = priors["lookup"]
        g_speed = priors["global_speed"]
        g_tps   = priors["global_tps"]
        prior_rows = [
            {"route_id": str(rh[0]), "hour": int(rh[1]),
             "_hist_speed": float(v[0]), "_hist_tps": float(v[1])}
            for rh, v in lookup.items()
        ]
        prior_df = pd.DataFrame(prior_rows)
        df = df.merge(prior_df, on=["route_id", "hour"], how="left")
        df["_hist_speed"] = df["_hist_speed"].fillna(g_speed)
        df["_hist_tps"]   = df["_hist_tps"].fillna(g_tps)
    else:
        df["_hist_speed"] = _FALLBACK_SPEED_MPS
        df["_hist_tps"]   = _FALLBACK_TPS_SEC

    speed     = df["progress_speed_mps"].to_numpy(dtype=float)
    hist_spd  = df["_hist_speed"].to_numpy(dtype=float)
    eff_speed = np.where(speed > 0, speed, hist_spd)

    # Full per-stop dwell across every stop ahead.  The (stops_ahead - 1) cut
    # tried in 7e3abdd to fix stops_ahead=1 pessimism dragged the whole horizon
    # curve into systematic optimism (issue #5); reverted to stops_ahead.
    df["hist_speed_mps"]        = hist_spd
    df["speed_eta_warm"]        = df["remaining_dist_m"].to_numpy() / np.maximum(eff_speed, 0.1)
    df["hist_travel_time_est"]  = df["stops_ahead"].to_numpy() * df["_hist_tps"].to_numpy()
    df = df.drop(columns=["_hist_speed", "_hist_tps"])
    return df


# Order matters: the worker export (scripts/export_worker_data.py) and the
# pure-Python tree traversal (src/inference.py) index features positionally.
# GTFS schedule features removed — stop departure times are unreliable in Lviv
# transit. Only GPS-derived signals are used.
# Prior-derived features (speed_eta_warm, hist_speed_mps, hist_travel_time_est)
# are added by apply_priors() after compute_features_for_training returns.
BASE_FEATURE_COLS = [
    "route_id",           # 0
    "stop_sequence",      # 1
    "stops_ahead",        # 2
    "hour",               # 3
    "day_of_week",        # 4
    "month",              # 5
    "is_weekend",         # 6
    "is_holiday",         # 7
    "remaining_dist_m",   # 8
    "progress_speed_mps", # 9
    "stops_remaining",    # 10
    "trip_progress_frac", # 11
    "dist_per_stop_m",    # 12  remaining_dist_m / stops_ahead
]

PRIOR_FEATURE_COLS = [
    "speed_eta_warm",        # 13  remaining_dist / effective_speed (warm-started)
    "hist_speed_mps",        # 14  route+hour historical median speed
    "hist_travel_time_est",  # 15  stops_ahead * hist seconds-per-stop (dwell-aware)
]

FEATURE_COLS = BASE_FEATURE_COLS + PRIOR_FEATURE_COLS

TARGET_COL = "seconds_to_arrival"
