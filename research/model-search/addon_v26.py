"""
Run 26 add-on: V26 position-age cols appended to existing ms_features_v10 day
files (written to ms_features_v26, same rows).

The raw feed carries a per-vehicle GPS fix time next to the feed header time,
and training has always ignored it: every row is anchored at the feed time, so
the position may already be tens of seconds old. On 2026-09-29 the age
(feed ts - vehicle ts) was p50 14 s, p75 29 s, p90 62 s; 24% of rows > 30 s.
A stale fix means the vehicle is further along than the row says (moving) or
that "stationary" is only a frozen report.

  pos_age_s       feed ts - vehicle ts at the snapshot (s)
  pos_age_med600  this vehicle's median age over its snapshots in the last 600 s
                  (the device's reporting cadence)
  pos_age_dist    pos_age_s * own_speed_180 (m the vehicle probably moved since the fix)
Serving: src/inference.py already reads vehicle_ts per vehicle; the 600 s median
needs a small per-vehicle ring (one int per snapshot).

Needs data/ms_lite/<day>.age.parquet from pipeline_lite.py (run 26+).
Usage: python research/model-search/addon_v26.py --days 2026-09-15..2026-09-29
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

from features_live import _epoch  # noqa: E402
from harness import TRAIN_DIR, expand_days  # noqa: E402

import os  # noqa: E402
SRC = REPO / "data" / os.environ.get("MS_ADDON_SRC", "ms_features_v10")   # run 27: overridable
DST = REPO / "data" / os.environ.get("MS_ADDON_DST", "ms_features_v26")
V26_COLS = ["pos_age_s", "pos_age_med600", "pos_age_dist"]


def age_table(day: str) -> pd.DataFrame:
    a = pd.read_parquet(TRAIN_DIR / f"{day}.age.parquet")
    a = pd.DataFrame({"vehicle_id": a["vehicle_id"].astype(str).to_numpy(),
                      "t": _epoch(a["timestamp"]),
                      "age": _epoch(a["timestamp"]) - pd.to_numeric(a["vehicle_ts"]).to_numpy(dtype=float)})
    a = a.drop_duplicates(["vehicle_id", "t"]).sort_values(["vehicle_id", "t"]).reset_index(drop=True)
    a.loc[a["age"] < 0, "age"] = 0.0
    idx = pd.to_datetime(a["t"], unit="s")
    med = (a.set_index(idx).groupby("vehicle_id", sort=False)["age"]
           .rolling("600s", min_periods=1).median())
    a["med600"] = med.to_numpy()   # groupby keeps the (vehicle, t) order since a is sorted
    return a


def add_day(day: str) -> None:
    out = DST / f"{day}_p3.parquet"
    if out.exists():
        return
    f = pd.read_parquet(SRC / f"{day}_p3.parquet")
    a = age_table(day)
    q = pd.DataFrame({"vehicle_id": f["vehicle_id"].astype(str).to_numpy(), "t": _epoch(f["snapshot_ts"])})
    m = q.merge(a, on=["vehicle_id", "t"], how="left")
    f["pos_age_s"] = m["age"].to_numpy()
    f["pos_age_med600"] = m["med600"].to_numpy()
    v = f["own_speed_180"].to_numpy(dtype=float)
    f["pos_age_dist"] = np.where(np.isfinite(v) & (v >= 0), f["pos_age_s"].to_numpy() * v, np.nan)
    DST.mkdir(parents=True, exist_ok=True)
    f.to_parquet(out, index=False)
    age = f["pos_age_s"]
    print(f"  {day}: {len(f):,} rows, age cov {age.notna().mean():.3f}, p50/p90 "
          f"{age.quantile(.5):.0f}/{age.quantile(.9):.0f}s, >30s {(age > 30).mean():.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", required=True)
    a = ap.parse_args()
    for d in expand_days(a.days):
        add_day(d)


if __name__ == "__main__":
    main()
