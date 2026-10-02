"""
Run 31 add-on: V31 cols appended to ms_features_v30 day files (written to
ms_features_v31, same rows). Two groups, both extensions of runs 29 / 30:

"Now" path time from the vehicles on the path (`_PAN`, run 30 next step 2):
  pa_now_t      sum over the occupied path links of (metres of that link still ahead
                of this vehicle) / max(median 120 s odometer speed of the vehicles on
                it, 0.5 m/s)
  pa_now_len    metres covered by those occupied links
  pa_slow_gap   path distance to the nearest vehicle ahead moving < 1 m/s over 120 s
                (NaN = none)
Very long odometer windows (`_ODOX`, run 29 next step 2):
  odo_spd_2400, odo_spd_3600  odometer m / fix-clock s over the last 40 / 60 min
Serving: the same link -> vehicles dict as V30 and a 60 min per-vehicle odometer
ring (one entry per minute is enough). Cheap.

Usage: python research/model-search/addon_v31.py --days 2026-09-15..2026-09-29
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from addon_v28 import REPO, TRAIN_DIR, _epoch, expand_days  # noqa: E402
from addon_v29 import _back  # noqa: E402
from addon_v30 import feed_speeds, occupancy, path_links  # noqa: E402
from features_live import _profiles  # noqa: E402

SRC = REPO / "data" / os.environ.get("MS_ADDON31_SRC", "ms_features_v30")
DST = REPO / "data" / os.environ.get("MS_ADDON31_DST", "ms_features_v31")
V31_COLS = ["pa_now_t", "pa_now_len", "pa_slow_gap", "odo_spd_2400", "odo_spd_3600"]


def long_odo(day: str) -> pd.DataFrame:
    a = pd.read_parquet(TRAIN_DIR / f"{day}.age.parquet")
    a = pd.DataFrame({"vehicle_id": a["vehicle_id"].astype(str).to_numpy(),
                      "t": _epoch(a["timestamp"]),
                      "vts": pd.to_numeric(a["vehicle_ts"]).to_numpy(dtype=float),
                      "odo": pd.to_numeric(a["odometer"]).to_numpy(dtype=float) * 1000.0})
    a = a.drop_duplicates(["vehicle_id", "t"]).sort_values(["vehicle_id", "t"]).reset_index(drop=True)
    for w in (2400, 3600):
        m = _back(a, "vehicle_id", w, ["vts", "odo"])
        do = a["odo"].to_numpy() - m["odo"].to_numpy()
        do[do < 0] = np.nan
        dv = a["vts"].to_numpy() - m["vts"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            a[f"odo_spd_{w}"] = np.where(dv > 0, do / dv, np.nan)
    return a[["vehicle_id", "t", "odo_spd_2400", "odo_spd_3600"]]


def add_day(day: str) -> None:
    out = DST / f"{day}_p3.parquet"
    if out.exists():
        return
    t0 = time.time()
    from src.gtfs_static import get_gtfs_for_date
    f = pd.read_parquet(SRC / f"{day}_p3.parquet")
    pos_trips = pd.unique(pd.read_parquet(TRAIN_DIR / f"{day}.pos.parquet", columns=["trip_id"])["trip_id"].astype(str))
    gtfs = get_gtfs_for_date(day)
    profs = _profiles(gtfs, np.union1d(pos_trips, pd.unique(f["trip_id"].astype(str))))
    occ = occupancy(day, profs, feed_speeds(day))
    ex = path_links(f, profs)
    ft = _epoch(f["snapshot_ts"])
    fv = f["vehicle_id"].astype(str).to_numpy()
    dvall = f["dist_along_m"].to_numpy(dtype=float)
    ex["t"] = ft[ex["rid"].to_numpy()]
    # metres of each path link still ahead of this vehicle
    own = dvall[ex["rid"].to_numpy()]
    ex["rem"] = np.where(ex["first"], np.clip(ex["start"] + ex["span"] - own, 0, None), ex["span"])
    m = ex.merge(occ, on=["key", "t"], how="inner")
    dv = dvall[m["rid"].to_numpy()]
    m["gap"] = m["start"] + m["frac"] * m["span"] - dv
    m = m[(m["vehicle_id"].to_numpy() != fv[m["rid"].to_numpy()])
          & (~m["first"] | (m["gap"] > 0))]
    lk = (m[m["o120"].notna()].groupby(["rid", "key"], sort=False)
          .agg(v=("o120", "median"), rem=("rem", "first")).reset_index())
    lk["tt"] = lk["rem"] / np.maximum(lk["v"], 0.5)
    g = lk.groupby("rid")
    res = pd.DataFrame({"pa_now_t": g["tt"].sum(), "pa_now_len": g["rem"].sum()})
    slow = m[m["o120"] < 1.0].groupby("rid")["gap"].min()
    res["pa_slow_gap"] = slow.reindex(res.index.union(slow.index))
    res = res.reindex(np.arange(len(f)))
    has_path = np.zeros(len(f), bool)
    has_path[ex["rid"].unique()] = True
    res.loc[has_path & res["pa_now_t"].isna().to_numpy(), ["pa_now_t", "pa_now_len"]] = 0.0
    for c in ["pa_now_t", "pa_now_len", "pa_slow_gap"]:
        f[c] = res[c].to_numpy()
    lo = long_odo(day)
    q = pd.DataFrame({"vehicle_id": fv, "t": ft, "row": np.arange(len(f))})
    q = q.merge(lo, on=["vehicle_id", "t"], how="left").sort_values("row")
    for c in ["odo_spd_2400", "odo_spd_3600"]:
        f[c] = q[c].to_numpy()
    DST.mkdir(parents=True, exist_ok=True)
    f.to_parquet(out, index=False)
    print(f"  {day}: {len(f):,} rows, now_t>0 {(f['pa_now_t'] > 0).mean():.3f} p50 {f.loc[f['pa_now_t'] > 0, 'pa_now_t'].median():.0f}s, "
          f"slow_gap {f['pa_slow_gap'].notna().mean():.3f}, odo3600 {f['odo_spd_3600'].notna().mean():.3f} "
          f"({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", required=True)
    a = ap.parse_args()
    for d in expand_days(a.days):
        add_day(d)


if __name__ == "__main__":
    main()
