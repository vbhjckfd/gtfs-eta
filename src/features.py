"""
Feature engineering for the LightGBM ETA model.

Target: seconds_to_target
  = actual_arrival_at_target_stop - actual_arrival_at_previous_stop

The key insight is that we're predicting inter-stop travel time given:
  - how late/early the vehicle currently is (current_delay_sec)
  - time of day, day of week
  - segment geometry (distance, position along route)
  - scheduled segment time (the schedule's own expectation, a strong prior)
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.gtfs_static import GTFSStatic, _parse_gtfs_time_utc

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


def _is_holiday(d: date) -> bool:
    return (d.month, d.day) in _UA_HOLIDAYS


def compute_features_for_training(
    labeled: pd.DataFrame,
    gtfs: GTFSStatic,
) -> pd.DataFrame:
    """
    Build feature rows from the labeled dataset.

    Groups by (vehicle_id, trip_id, date) so that each vehicle-trip-day run
    is treated as an independent sequence.  For each stop, the prediction
    context is the previous stop's actual arrival time.

    Returns a DataFrame with FEATURE_COLS + TARGET_COL columns, plus
    `scheduled_segment_sec` (used as the baseline in train.py).
    """
    labeled = labeled.copy()
    labeled["actual_arrival"] = pd.to_datetime(labeled["actual_arrival"], utc=True)
    labeled["scheduled_arrival"] = pd.to_datetime(labeled["scheduled_arrival"], utc=True, errors="coerce")

    feature_rows = []

    group_cols = ["vehicle_id", "trip_id", "date"]
    # date may be a date object or string — normalise
    if "date" in labeled.columns:
        labeled["date"] = pd.to_datetime(labeled["date"]).dt.date

    for keys, grp in labeled.groupby(group_cols, sort=False):
        vehicle_id, trip_id, row_date = keys
        grp = grp.sort_values("stop_sequence").reset_index(drop=True)

        trip = gtfs.get_trip(str(trip_id))
        if trip is None:
            continue
        shape_len = gtfs.get_shape_length(trip.shape_id)

        for target_idx in range(len(grp)):
            target = grp.iloc[target_idx]
            if pd.isna(target["actual_arrival"]):
                continue

            if target_idx == 0:
                pred_time = target["scheduled_arrival"] if pd.notna(target["scheduled_arrival"]) else target["actual_arrival"]
                current_delay_sec = 0.0
                scheduled_segment_sec = 0.0
            else:
                prev = grp.iloc[target_idx - 1]
                if pd.isna(prev["actual_arrival"]):
                    continue
                pred_time = prev["actual_arrival"]
                current_delay_sec = float(prev["delay_sec"]) if pd.notna(prev["delay_sec"]) else 0.0

                # Scheduled segment: how long the schedule says this segment takes
                sched_prev = prev["scheduled_arrival"]
                sched_tgt = target["scheduled_arrival"]
                if pd.notna(sched_prev) and pd.notna(sched_tgt):
                    scheduled_segment_sec = max(
                        0.0, (pd.Timestamp(sched_tgt) - pd.Timestamp(sched_prev)).total_seconds()
                    )
                else:
                    scheduled_segment_sec = 0.0

            pred_time = pd.Timestamp(pred_time)
            target_time = pd.Timestamp(target["actual_arrival"])
            seconds_to_target = (target_time - pred_time).total_seconds()

            if seconds_to_target < 0:
                continue

            # Segment distance
            if target_idx == 0:
                segment_dist_m = 0.0
            else:
                pred_stop_id = grp.iloc[target_idx - 1]["stop_id"]
                d_pred = gtfs.get_stop_distance_along_shape(trip.shape_id, pred_stop_id) or 0.0
                d_tgt = gtfs.get_stop_distance_along_shape(trip.shape_id, target["stop_id"]) or 0.0
                segment_dist_m = max(0.0, d_tgt - d_pred)

            d = pred_time.date()
            feature_rows.append({
                "route_id": trip.route_id,
                "stop_sequence": int(target["stop_sequence"]),
                "prediction_stop_sequence": int(grp.iloc[target_idx - 1]["stop_sequence"]) if target_idx > 0 else 0,
                "hour": pred_time.hour,
                "day_of_week": pred_time.weekday(),
                "month": pred_time.month,
                "is_weekend": int(pred_time.weekday() >= 5),
                "is_holiday": int(_is_holiday(d)),
                "current_delay_sec": current_delay_sec,
                "segment_distance_m": segment_dist_m,
                "scheduled_segment_sec": scheduled_segment_sec,
                "stops_remaining": len(grp) - 1 - target_idx,
                "trip_progress_frac": (
                    (gtfs.get_stop_distance_along_shape(trip.shape_id, target["stop_id"]) or 0.0)
                    / max(shape_len, 1.0)
                ),
                # For train/test split
                "date": row_date,
                TARGET_COL: seconds_to_target,
            })

    return pd.DataFrame(feature_rows)


def compute_features_for_inference(
    vehicle_id: str,
    trip_id: str,
    current_stop_sequence: int,
    current_delay_sec: float,
    snapshot_time,
    recent_speed_mps: float | None,
    gtfs: GTFSStatic,
    vehicle_dist_along_shape: float = 0.0,
) -> pd.DataFrame:
    """Build one feature row per future stop for live inference."""
    trip = gtfs.get_trip(trip_id)
    if trip is None:
        return pd.DataFrame()

    shape_len = gtfs.get_shape_length(trip.shape_id)
    snap_ts = pd.Timestamp(snapshot_time)
    d = snap_ts.date()

    # Find current stop's info
    current_st = None
    for st in trip.stop_times:
        if st.stop_sequence == current_stop_sequence:
            current_st = st
            break

    # d_last_stop: shape distance of the last passed stop (reference for sched scaling)
    d_last_stop = 0.0
    if current_st:
        d_last_stop = gtfs.get_stop_distance_along_shape(trip.shape_id, current_st.stop_id) or 0.0

    # Use vehicle's actual projected position along shape so segment_dist_m reflects
    # the true remaining distance rather than the full stop-to-stop segment. This
    # prevents the model from predicting "1 min" when the vehicle is already at the stop.
    d_current = vehicle_dist_along_shape if vehicle_dist_along_shape > d_last_stop else d_last_stop

    sched_current_utc = None
    if current_st:
        sched_current_utc = _parse_gtfs_time_utc(
            current_st.arrival_time or current_st.departure_time, d, gtfs.feed_tz
        )

    rows = []
    for st in trip.stop_times:
        if st.stop_sequence <= current_stop_sequence:
            continue

        d_target = gtfs.get_stop_distance_along_shape(trip.shape_id, st.stop_id) or 0.0
        segment_dist_m = max(0.0, d_target - d_current)

        sched_target_utc = _parse_gtfs_time_utc(
            st.arrival_time or st.departure_time, d, gtfs.feed_tz
        )
        if sched_current_utc and sched_target_utc:
            sched_seg_full = max(0.0, (sched_target_utc - sched_current_utc).total_seconds())
            # Scale scheduled time proportionally to remaining distance from vehicle position
            seg_full_m = max(1.0, d_target - d_last_stop)
            remaining_frac = min(1.0, max(0.0, (d_target - d_current) / seg_full_m))
            sched_seg = sched_seg_full * remaining_frac
        else:
            sched_seg = 0.0

        rows.append({
            "vehicle_id": vehicle_id,
            "trip_id": trip_id,
            "route_id": trip.route_id,
            "stop_id": st.stop_id,
            "stop_sequence": st.stop_sequence,
            "prediction_stop_sequence": current_stop_sequence,
            "hour": snap_ts.hour,
            "day_of_week": snap_ts.weekday(),
            "month": snap_ts.month,
            "is_weekend": int(snap_ts.weekday() >= 5),
            "is_holiday": int(_is_holiday(d)),
            "current_delay_sec": current_delay_sec,
            "segment_distance_m": segment_dist_m,
            "scheduled_segment_sec": sched_seg,
            "stops_remaining": sum(1 for s in trip.stop_times if s.stop_sequence > st.stop_sequence),
            "trip_progress_frac": d_target / max(shape_len, 1.0),
        })

    return pd.DataFrame(rows)


FEATURE_COLS = [
    "route_id",
    "stop_sequence",
    "prediction_stop_sequence",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",
    "current_delay_sec",
    "segment_distance_m",
    "scheduled_segment_sec",
    "stops_remaining",
    "trip_progress_frac",
]

TARGET_COL = "seconds_to_target"
