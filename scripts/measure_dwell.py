"""
Measure per-stop dwell time from the labelled training parquets.

The served feed publishes ``departure = arrival + dwell`` for every stop, and
that dwell has always been a flat guess (``_DWELL_SECS = 15`` in
src/inference.py) with nothing behind it.  This measures the real thing.

Method — stationary-run detection on the cleaned along-shape projection:

  * dedupe the snapshot-anchored rows back to one position per
    (vehicle, trip, snapshot);
  * walk each trajectory and cut it into maximal runs where the vehicle
    advances less than STILL_ADVANCE_M between consecutive snapshots;
  * keep runs that sit within ATTRIBUTE_RADIUS_M of a stop on that trip;
  * the run's duration, plus one sampling interval (the vehicle stopped and
    started somewhere inside the gaps at either end), is the dwell.

Bronze snapshots land every ~11 s (median), so a 10-40 s dwell is resolvable —
coarsely per event, but the medians over millions of events are solid.  Runs
that never end inside the data (the trajectory stops while the vehicle is
still stationary — end of trip, terminus layover) are discarded rather than
truncated, and so are runs longer than MAX_DWELL_SEC, which are layovers and
traffic jams, not stop dwell.

Usage:
    python scripts/measure_dwell.py                 # every local parquet
    python scripts/measure_dwell.py --days 14       # most recent 14
    python scripts/measure_dwell.py --out models/dwell.joblib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import pandas as pd

TRAINING_DIR = Path("data/training")

# A vehicle advancing less than this between two snapshots is standing still.
# Generous against GPS jitter on the projection (a parked bus wanders a few
# metres) but well under the ~40 m a moving bus covers in one sampling gap.
STILL_ADVANCE_M = 8.0

# How close a stationary run must sit to a stop to count as dwelling *at* it
# rather than at a red light. Lviv's stop spacing is ~400 m, so this cannot
# capture the wrong stop; it can only reject.
ATTRIBUTE_RADIUS_M = 40.0

# Beyond this a "dwell" is a terminus layover or a jam, not passenger service.
MAX_DWELL_SEC = 180.0

# Sub-threshold runs of a single snapshot carry no duration information.
MIN_RUN_SNAPSHOTS = 2


def _positions(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (vehicle, trip, snapshot) — the trajectory, not the labels."""
    pos = (
        df[["vehicle_id", "trip_id", "route_id", "snapshot_ts", "dist_along_m"]]
        .drop_duplicates(["vehicle_id", "trip_id", "snapshot_ts"])
        .sort_values(["vehicle_id", "trip_id", "snapshot_ts"])
    )
    pos["snapshot_ts"] = pd.to_datetime(pos["snapshot_ts"], utc=True)
    return pos


def _stop_positions(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """trip_id → sorted array of stop distances along the shape."""
    stops = df[["trip_id", "stop_dist_along_m"]].drop_duplicates()
    return {
        str(tid): np.sort(g["stop_dist_along_m"].to_numpy(dtype=float))
        for tid, g in stops.groupby("trip_id", sort=False)
    }


def _dwells_for_trajectory(times: np.ndarray, dists: np.ndarray,
                           stop_dists: np.ndarray) -> list[tuple[float, float]]:
    """[(dwell_sec, distance_to_nearest_stop_m), ...] for one trajectory."""
    if len(times) < MIN_RUN_SNAPSHOTS + 1 or stop_dists.size == 0:
        return []

    out: list[tuple[float, float]] = []
    run_start = 0
    for i in range(1, len(dists)):
        if dists[i] - dists[i - 1] < STILL_ADVANCE_M:
            continue
        # The vehicle moved at i, so the run [run_start, i-1] just ended.
        length = i - run_start
        if length >= MIN_RUN_SNAPSHOTS:
            # One sampling interval of credit: the vehicle came to rest
            # somewhere between the previous snapshot and run_start, and pulled
            # away somewhere between i-1 and i. Charging one gap total splits
            # that difference rather than assuming either extreme.
            gap = (times[i] - times[i - 1]) if i > 0 else 0.0
            dwell = float(times[i - 1] - times[run_start] + gap)
            here = float(np.mean(dists[run_start:i]))
            nearest = float(np.min(np.abs(stop_dists - here)))
            if 0.0 < dwell <= MAX_DWELL_SEC:
                out.append((dwell, nearest))
        run_start = i
    # A run still open at the end of the trajectory is discarded: the vehicle
    # is parked past the data, so its duration is a lower bound, not a dwell.
    return out


def measure(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for p in paths:
        df = pd.read_parquet(
            p,
            columns=["vehicle_id", "trip_id", "route_id", "snapshot_ts",
                     "dist_along_m", "stop_dist_along_m"],
        )
        pos = _positions(df)
        stop_map = _stop_positions(df)
        route_of = dict(zip(pos["trip_id"].astype(str), pos["route_id"].astype(str)))

        n_before = len(rows)
        for (_vid, tid), g in pos.groupby(["vehicle_id", "trip_id"], sort=False):
            stop_dists = stop_map.get(str(tid))
            if stop_dists is None:
                continue
            times = g["snapshot_ts"].to_numpy(dtype="datetime64[ns]").astype("int64") / 1e9
            dists = g["dist_along_m"].to_numpy(dtype=float)
            for dwell, nearest in _dwells_for_trajectory(times, dists, stop_dists):
                if nearest <= ATTRIBUTE_RADIUS_M:
                    rows.append({
                        "date": p.stem,
                        "route_id": route_of.get(str(tid), ""),
                        "dwell_sec": dwell,
                    })
        print(f"  {p.stem}: {len(rows) - n_before:,} dwell events", flush=True)

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=None,
                    help="use only the N most recent parquets")
    ap.add_argument("--out", default=None,
                    help="write the route_type → dwell table to this joblib path")
    args = ap.parse_args()

    paths = sorted(TRAINING_DIR.glob("*.parquet"))
    if not paths:
        raise SystemExit(f"no parquets in {TRAINING_DIR}")
    if args.days:
        paths = paths[-args.days:]
    print(f"Measuring dwell over {len(paths)} day(s)…", flush=True)

    events = measure(paths)
    if events.empty:
        raise SystemExit("no dwell events detected")

    print(f"\n{len(events):,} dwell events")
    print(events["dwell_sec"].describe(
        percentiles=[0.25, 0.5, 0.75, 0.9]).round(1).to_string())

    from src.gtfs_static import get_gtfs
    gtfs = get_gtfs()
    sys.path.insert(0, "scripts")
    from export_worker_data import _build_route_types
    route_types = _build_route_types(gtfs)
    events["route_type"] = events["route_id"].map(route_types)

    print("\nBy route_type (0=tram, 3=bus, 11=trolleybus):")
    by_type = events.groupby("route_type")["dwell_sec"].agg(["count", "median", "mean"])
    print(by_type.round(1).to_string())

    table = {
        int(rt): int(round(row["median"]))
        for rt, row in by_type.iterrows()
        if pd.notna(rt) and row["count"] >= 500
    }
    table["_global"] = int(round(events["dwell_sec"].median()))
    print(f"\nDwell table: {table}")

    if args.out:
        import joblib
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(table, out)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
