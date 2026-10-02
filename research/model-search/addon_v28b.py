"""
Run 28 add-on, part 2: more feed-speed / odometer shape on top of ms_features_v28
(written to ms_features_v28b, same rows). V28 (instant speed + odometer windows)
gave -1.1% / -1.6% vs mf3_age; this asks whether the recent speed trajectory adds more.

  fs_lag1       reported speed at the vehicle's previous snapshot (m/s)
  fs_acc30      (fs_now - reported speed at the snapshot at or before t - 30 s) / 30
  fs_max120     max reported speed over the last 120 s
  fs_zero_sec   seconds since the last snapshot with reported speed > 0 (capped 3600)
  odo_spd_30    odometer m / fix-clock s over the last 30 s
Usage: python research/model-search/addon_v28b.py --days 2026-09-15..2026-09-29
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from addon_v28 import REPO, TRAIN_DIR, _epoch, expand_days  # noqa: E402

SRC = REPO / "data" / "ms_features_v28"
DST = REPO / "data" / "ms_features_v28b"
V28B_COLS = ["fs_lag1", "fs_acc30", "fs_max120", "fs_zero_sec", "odo_spd_30"]


def _back(a: pd.DataFrame, w: int, cols: list[str]) -> pd.DataFrame:
    """Values of `cols` at this vehicle's latest snapshot at or before t - w (within 2w)."""
    q = pd.DataFrame({"vehicle_id": a["vehicle_id"], "tq": a["t"] - w, "row": np.arange(len(a))}).sort_values("tq")
    b = a[["vehicle_id", "t"] + cols].rename(columns={"t": "tq"}).sort_values("tq")
    return pd.merge_asof(q, b, on="tq", by="vehicle_id", direction="backward",
                         tolerance=float(w)).sort_values("row").reset_index(drop=True)


def feed_table(day: str) -> pd.DataFrame:
    a = pd.read_parquet(TRAIN_DIR / f"{day}.age.parquet")
    a = pd.DataFrame({"vehicle_id": a["vehicle_id"].astype(str).to_numpy(),
                      "t": _epoch(a["timestamp"]),
                      "vts": pd.to_numeric(a["vehicle_ts"]).to_numpy(dtype=float),
                      "spd": pd.to_numeric(a["speed"]).to_numpy(dtype=float),
                      "odo": pd.to_numeric(a["odometer"]).to_numpy(dtype=float) * 1000.0})
    a = a.drop_duplicates(["vehicle_id", "t"]).sort_values(["vehicle_id", "t"]).reset_index(drop=True)
    g = a.groupby("vehicle_id", sort=False)
    a["fs_lag1"] = g["spd"].shift(1).to_numpy()
    idx = pd.to_datetime(a["t"], unit="s")
    a["fs_max120"] = (a.set_index(idx).groupby("vehicle_id", sort=False)["spd"]
                      .rolling("120s", min_periods=1).max()).to_numpy()
    tmov = pd.Series(np.where(a["spd"].to_numpy() > 0, a["t"].to_numpy(dtype=float), np.nan))
    tmov = tmov.groupby(a["vehicle_id"].to_numpy(), sort=False).ffill().to_numpy()
    a["fs_zero_sec"] = np.minimum(a["t"].to_numpy(dtype=float) - tmov, 3600.0)
    m = _back(a, 30, ["spd", "vts", "odo"])
    a["fs_acc30"] = (a["spd"].to_numpy() - m["spd"].to_numpy()) / 30.0
    dodo = a["odo"].to_numpy() - m["odo"].to_numpy()
    dodo[dodo < 0] = np.nan
    dv = a["vts"].to_numpy() - m["vts"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        a["odo_spd_30"] = np.where(dv > 0, dodo / dv, np.nan)
    return a


def add_day(day: str) -> None:
    out = DST / f"{day}_p3.parquet"
    if out.exists():
        return
    f = pd.read_parquet(SRC / f"{day}_p3.parquet")
    a = feed_table(day)
    q = pd.DataFrame({"vehicle_id": f["vehicle_id"].astype(str).to_numpy(), "t": _epoch(f["snapshot_ts"])})
    m = q.merge(a[["vehicle_id", "t"] + V28B_COLS], on=["vehicle_id", "t"], how="left")
    for c in V28B_COLS:
        f[c] = m[c].to_numpy()
    DST.mkdir(parents=True, exist_ok=True)
    f.to_parquet(out, index=False)
    print(f"  {day}: {len(f):,} rows, cov " + " ".join(f"{c} {f[c].notna().mean():.3f}" for c in V28B_COLS), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", required=True)
    for d in expand_days(ap.parse_args().days):
        add_day(d)


if __name__ == "__main__":
    main()
