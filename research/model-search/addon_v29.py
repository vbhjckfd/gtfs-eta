"""
Run 29 add-on: V29 odometer cols appended to ms_features_v28 day files (written
to ms_features_v29, same rows). V28 (instant speed + odometer over <= 300 s) won
-1.1% / -1.6% vs mf3_age; the speed trajectory (V28b) added nothing. This asks
two other questions of the odometer:

1. Longer windows (traffic state on a slower scale than the 300 s own-speed ring):
  odo_spd_600    odometer m / fix-clock s over the last 600 s
  odo_spd_1200   same over 1200 s
  fs_zero_frac600  share of the vehicle's snapshots in the last 600 s with reported speed 0
2. Odometer vs shape progress (the odometer is a projection-free distance, so a gap
   flags detours, loops or projection jumps that remaining_dist can't see):
  odo_shape_gap300  odometer m - shape (dist_along_m) progress over the last 300 s, same trip
  odo_trip_spd      odometer m / fix-clock s since the vehicle's first snapshot on this trip
Serving: the per-vehicle ring of (feed ts, vehicle ts, speed, odometer, dist_along)
grows from 300 s to 1200 s (~120 entries per vehicle), plus the odometer at trip start.

Needs data/ms_lite/<day>.age.parquet (pipeline_lite run 28+) and <day>.pos.parquet.
Usage: python research/model-search/addon_v29.py --days 2026-09-15..2026-09-29
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from addon_v28 import REPO, TRAIN_DIR, _epoch, expand_days  # noqa: E402

SRC = REPO / "data" / os.environ.get("MS_ADDON29_SRC", "ms_features_v28")
DST = REPO / "data" / os.environ.get("MS_ADDON29_DST", "ms_features_v29")
V29_COLS = ["odo_spd_600", "odo_spd_1200", "fs_zero_frac600", "odo_shape_gap300", "odo_trip_spd"]


def _back(a: pd.DataFrame, by: str, w: int, cols: list[str]) -> pd.DataFrame:
    """Values of `cols` at the latest row of the same `by` at or before t - w (within w)."""
    q = pd.DataFrame({by: a[by], "tq": a["t"] - w, "row": np.arange(len(a))}).sort_values("tq")
    b = a[[by, "t"] + cols].rename(columns={"t": "tq"}).sort_values("tq")
    return pd.merge_asof(q, b, on="tq", by=by, direction="backward",
                         tolerance=float(w)).sort_values("row").reset_index(drop=True)


def feed_table(day: str) -> pd.DataFrame:
    a = pd.read_parquet(TRAIN_DIR / f"{day}.age.parquet")
    a = pd.DataFrame({"vehicle_id": a["vehicle_id"].astype(str).to_numpy(),
                      "t": _epoch(a["timestamp"]),
                      "vts": pd.to_numeric(a["vehicle_ts"]).to_numpy(dtype=float),
                      "spd": pd.to_numeric(a["speed"]).to_numpy(dtype=float),
                      "odo": pd.to_numeric(a["odometer"]).to_numpy(dtype=float) * 1000.0})
    a = a.drop_duplicates(["vehicle_id", "t"]).sort_values(["vehicle_id", "t"]).reset_index(drop=True)
    idx = pd.to_datetime(a["t"], unit="s")
    a["z"] = (a["spd"] <= 0).astype(float)
    a["fs_zero_frac600"] = (a.set_index(idx).groupby("vehicle_id", sort=False)["z"]
                            .rolling("600s", min_periods=1).mean()).to_numpy()
    for w in (600, 1200):
        m = _back(a, "vehicle_id", w, ["vts", "odo"])
        dodo = a["odo"].to_numpy() - m["odo"].to_numpy()
        dodo[dodo < 0] = np.nan
        dv = a["vts"].to_numpy() - m["vts"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            a[f"odo_spd_{w}"] = np.where(dv > 0, dodo / dv, np.nan)
    return a


def trip_table(day: str, a: pd.DataFrame) -> pd.DataFrame:
    """Per (vehicle, trip, feed t): odometer-vs-shape gap over 300 s and trip-average odometer speed."""
    p = pd.read_parquet(TRAIN_DIR / f"{day}.pos.parquet")
    p = pd.DataFrame({"vehicle_id": p["vehicle_id"].astype(str).to_numpy(),
                      "trip_id": p["trip_id"].astype(str).to_numpy(),
                      "t": _epoch(p["snapshot_ts"]),
                      "d": p["dist_along_m"].to_numpy(dtype=float)})
    p = p.drop_duplicates(["vehicle_id", "trip_id", "t"])
    p = p.merge(a[["vehicle_id", "t", "vts", "odo"]], on=["vehicle_id", "t"], how="inner")
    p["vt"] = p["vehicle_id"] + "|" + p["trip_id"]
    p = p.sort_values(["vt", "t"]).reset_index(drop=True)
    m = _back(p, "vt", 300, ["d", "odo"])
    dodo = p["odo"].to_numpy() - m["odo"].to_numpy()
    dodo[dodo < 0] = np.nan
    p["odo_shape_gap300"] = dodo - (p["d"].to_numpy() - m["d"].to_numpy())
    g = p.groupby("vt", sort=False)
    o0, v0 = g["odo"].transform("first").to_numpy(), g["vts"].transform("first").to_numpy()
    do, dv = p["odo"].to_numpy() - o0, p["vts"].to_numpy() - v0
    with np.errstate(divide="ignore", invalid="ignore"):
        p["odo_trip_spd"] = np.where((dv >= 300) & (do >= 0), do / dv, np.nan)
    return p[["vehicle_id", "trip_id", "t", "odo_shape_gap300", "odo_trip_spd"]]


def add_day(day: str) -> None:
    out = DST / f"{day}_p3.parquet"
    if out.exists():
        return
    f = pd.read_parquet(SRC / f"{day}_p3.parquet")
    a = feed_table(day)
    tt = trip_table(day, a)
    q = pd.DataFrame({"vehicle_id": f["vehicle_id"].astype(str).to_numpy(),
                      "trip_id": f["trip_id"].astype(str).to_numpy(), "t": _epoch(f["snapshot_ts"])})
    m = q.merge(a[["vehicle_id", "t", "odo_spd_600", "odo_spd_1200", "fs_zero_frac600"]],
                on=["vehicle_id", "t"], how="left")
    m = m.merge(tt, on=["vehicle_id", "trip_id", "t"], how="left")
    assert len(m) == len(f)
    for c in V29_COLS:
        f[c] = m[c].to_numpy()
    DST.mkdir(parents=True, exist_ok=True)
    f.to_parquet(out, index=False)
    cov = " ".join(f"{c} {f[c].notna().mean():.3f}" for c in V29_COLS)
    gap = f["odo_shape_gap300"]
    print(f"  {day}: {len(f):,} rows, cov {cov}; gap300 p10/p50/p90 "
          f"{gap.quantile(.1):.0f}/{gap.median():.0f}/{gap.quantile(.9):.0f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", required=True)
    a = ap.parse_args()
    for d in expand_days(a.days):
        add_day(d)


if __name__ == "__main__":
    main()
