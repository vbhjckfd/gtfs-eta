"""Backfill the `stationary_sec` column into training parquets built before it existed.

`src.labeling` emits the column for every new pipeline run, but `run_pipeline
--all` skips days it has already processed, so without this the existing archive
would keep training the model on an all-zero column (features.py substitutes
zeros when the column is missing — safe, but useless).

No re-derivation from raw snapshots is needed: the parquets already carry the
trajectory (vehicle_id, trip_id, snapshot_ts, dist_along_m), which is exactly
`_stationary_seconds`' input, so this reproduces what labeling would have
written. The one difference is snapshots that emitted no rows at all (past the
horizon, no upcoming stops) are absent here — that can only delay an anchor
reset by one snapshot, never change whether a vehicle reads as stopped.

Writes via a temp file + atomic replace, and skips files that already have the
column, so it is safe to interrupt and re-run.

    python scripts/backfill_stationary.py [data/training/]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.labeling import _stationary_seconds  # noqa: E402

KEYS = ["vehicle_id", "trip_id", "snapshot_ts"]


def backfill_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add stationary_sec, computed on the unique trajectory then broadcast."""
    traj = (
        df[KEYS + ["dist_along_m"]]
        .drop_duplicates(KEYS)
        .sort_values(KEYS, kind="mergesort")
        .reset_index(drop=True)
    )
    tsec = (
        (pd.to_datetime(traj["snapshot_ts"], utc=True) - pd.Timestamp("1970-01-01", tz="UTC"))
        // pd.Timedelta(seconds=1)
    ).to_numpy()
    dist = traj["dist_along_m"].to_numpy(dtype=float)

    vals = np.zeros(len(traj), dtype=float)
    for idx in traj.groupby(["vehicle_id", "trip_id"], sort=False).indices.values():
        vals[idx] = _stationary_seconds(dist[idx], tsec[idx])
    traj["stationary_sec"] = vals

    return df.merge(traj[KEYS + ["stationary_sec"]], on=KEYS, how="left")


def main() -> int:
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "data/training")
    files = sorted(target.glob("*.parquet"))
    if not files:
        print(f"No parquets in {target}")
        return 1

    print(f"Backfilling stationary_sec across {len(files)} files in {target}")
    for f in files:
        if "stationary_sec" in pq.ParquetFile(f).schema_arrow.names:
            print(f"  {f.name}: already present — skipped")
            continue
        df = backfill_frame(pd.read_parquet(f))
        tmp = f.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(f)
        share = 100.0 * float((df["stationary_sec"] > 300).mean())
        print(f"  {f.name}: {len(df):,} rows  stationary>300s {share:.1f}%", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
