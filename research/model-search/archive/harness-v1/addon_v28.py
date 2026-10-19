"""
Run 28 add-on: V28 feed-speed / odometer cols appended to ms_features_v26 day
files (written to ms_features_v28, same rows).

The raw feed carries, per vehicle, a reported GPS speed (m/s, quantised to
1 km/h) and an odometer (km). Training and src/features have always ignored both.
On 2026-09-29 both were present on 100% of rows; odometer deltas match the GPS
chord distance (ratio p50 1.007), and it never moves while stopped. Reported speed
correlates only 0.61 with the 60 s average speed, so it is an instantaneous
reading (braking / pulling away), which the position-derived own_speed_* can't see.

  fs_now        reported speed at the snapshot (m/s)
  fs_mean60     mean reported speed over this vehicle's snapshots in the last 60 s
  odo_spd_60    odometer m / fix-clock s over the last 60 s
  odo_spd_180   same over 180 s
  odo_m_300     odometer m in the last 300 s
Serving: read straight from the feed entity; the windows need a small
per-vehicle ring of (feed ts, vehicle ts, speed, odometer).

Needs data/ms_lite/<day>.age.parquet with speed + odometer (pipeline_lite, run 28+).
Usage: python research/model-search/addon_v28.py --days 2026-09-15..2026-09-29
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

from features_live import _epoch  # noqa: E402
from harness import TRAIN_DIR, expand_days  # noqa: E402

SRC = REPO / "data" / os.environ.get("MS_ADDON28_SRC", "ms_features_v26")
DST = REPO / "data" / os.environ.get("MS_ADDON28_DST", "ms_features_v28")
V28_COLS = ["fs_now", "fs_mean60", "odo_spd_60", "odo_spd_180", "odo_m_300"]


def feed_table(day: str) -> pd.DataFrame:
    a = pd.read_parquet(TRAIN_DIR / f"{day}.age.parquet")
    a = pd.DataFrame({"vehicle_id": a["vehicle_id"].astype(str).to_numpy(),
                      "t": _epoch(a["timestamp"]),
                      "vts": pd.to_numeric(a["vehicle_ts"]).to_numpy(dtype=float),
                      "spd": pd.to_numeric(a["speed"]).to_numpy(dtype=float),
                      "odo": pd.to_numeric(a["odometer"]).to_numpy(dtype=float) * 1000.0})
    a = a.drop_duplicates(["vehicle_id", "t"]).sort_values(["vehicle_id", "t"]).reset_index(drop=True)
    idx = pd.to_datetime(a["t"], unit="s")
    a["fs_mean60"] = (a.set_index(idx).groupby("vehicle_id", sort=False)["spd"]
                      .rolling("60s", min_periods=1).mean()).to_numpy()
    for w in (60, 180, 300):
        # the vehicle's latest snapshot at or before t - w (within 2w, else missing)
        q = pd.DataFrame({"vehicle_id": a["vehicle_id"], "tq": a["t"] - w, "row": np.arange(len(a))})
        q = q.sort_values("tq")
        b = a[["vehicle_id", "t", "vts", "odo"]].rename(columns={"t": "tq", "vts": "vts0", "odo": "odo0"})
        b = b.sort_values("tq")
        b["t0"] = b["tq"]
        m = pd.merge_asof(q, b, on="tq", by="vehicle_id", direction="backward",
                          tolerance=float(w)).sort_values("row")
        dodo = a["odo"].to_numpy() - m["odo0"].to_numpy()
        dodo[dodo < 0] = np.nan                 # odometer reset
        if w == 300:
            a["odo_m_300"] = dodo
        else:
            dv = a["vts"].to_numpy() - m["vts0"].to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                a[f"odo_spd_{w}"] = np.where(dv > 0, dodo / dv, np.nan)
    a["fs_now"] = a["spd"]
    return a


def add_day(day: str) -> None:
    out = DST / f"{day}_p3.parquet"
    if out.exists():
        return
    f = pd.read_parquet(SRC / f"{day}_p3.parquet")
    a = feed_table(day)
    q = pd.DataFrame({"vehicle_id": f["vehicle_id"].astype(str).to_numpy(), "t": _epoch(f["snapshot_ts"])})
    m = q.merge(a[["vehicle_id", "t"] + V28_COLS], on=["vehicle_id", "t"], how="left")
    for c in V28_COLS:
        f[c] = m[c].to_numpy()
    DST.mkdir(parents=True, exist_ok=True)
    f.to_parquet(out, index=False)
    cov = " ".join(f"{c} {f[c].notna().mean():.3f}" for c in V28_COLS)
    print(f"  {day}: {len(f):,} rows, cov {cov}; odo_spd_60 p50 {f['odo_spd_60'].median():.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", required=True)
    a = ap.parse_args()
    for d in expand_days(a.days):
        add_day(d)


if __name__ == "__main__":
    main()
