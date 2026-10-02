"""
Run 30 add-on: V30 "vehicles on the path right now" cols appended to ms_features_v29
day files (written to ms_features_v30, same rows).

The live link store only sees COMPLETED traversals (30 s detect lag, up to 30 min
old). Other vehicles that are on the links between this vehicle and its target
stop right now are fresher evidence: a bus stuck 400 m ahead shows up here minutes
before any traversal of that link completes. Links are keyed by stop-id pair, so
any route sharing the street counts (same rule as the link store).

At the snapshot time t, over the other vehicles whose current link (last passed
stop > next stop on their own trip) is on this row's path, ahead of this vehicle:
  pa_n          how many
  pa_spd_mean   mean odometer speed over their last 120 s (m / fix-clock s)
  pa_spd_min    min of the same
  pa_stop_frac  share with reported speed 0
  pa_gap_m      path distance to the nearest of them
  pa_gap_spd    its 120 s odometer speed
  pa_cov        share of the path length (link lengths) with at least one vehicle on it
Serving: the daemon already places every vehicle on its trip (next stop + distance
along); this needs a link -> [(vehicle, fraction, speed)] dict rebuilt each cycle,
plus the 120 s odometer ring that run 28/29 already need. Cheap.

Needs data/ms_lite/<day>.pos.parquet and <day>.age.parquet (pipeline_lite run 28+).
Usage: python research/model-search/addon_v30.py --days 2026-09-15..2026-09-29
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
from features_live import _nearest_idx, _profiles  # noqa: E402

SRC = REPO / "data" / os.environ.get("MS_ADDON30_SRC", "ms_features_v29")
DST = REPO / "data" / os.environ.get("MS_ADDON30_DST", "ms_features_v30")
V30_COLS = ["pa_n", "pa_spd_mean", "pa_spd_min", "pa_stop_frac", "pa_gap_m", "pa_gap_spd", "pa_cov"]


def feed_speeds(day: str) -> pd.DataFrame:
    a = pd.read_parquet(TRAIN_DIR / f"{day}.age.parquet")
    a = pd.DataFrame({"vehicle_id": a["vehicle_id"].astype(str).to_numpy(),
                      "t": _epoch(a["timestamp"]),
                      "vts": pd.to_numeric(a["vehicle_ts"]).to_numpy(dtype=float),
                      "fs": pd.to_numeric(a["speed"]).to_numpy(dtype=float),
                      "odo": pd.to_numeric(a["odometer"]).to_numpy(dtype=float) * 1000.0})
    a = a.drop_duplicates(["vehicle_id", "t"]).sort_values(["vehicle_id", "t"]).reset_index(drop=True)
    m = _back(a, "vehicle_id", 120, ["vts", "odo"])
    do = a["odo"].to_numpy() - m["odo"].to_numpy()
    do[do < 0] = np.nan
    dv = a["vts"].to_numpy() - m["vts"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        a["o120"] = np.where(dv > 0, do / dv, np.nan)
    return a[["vehicle_id", "t", "fs", "o120"]]


def occupancy(day: str, profs: dict, feed: pd.DataFrame) -> pd.DataFrame:
    """Every vehicle's current link (stop pair) and fraction along it, per snapshot."""
    p = pd.read_parquet(TRAIN_DIR / f"{day}.pos.parquet")
    p = pd.DataFrame({"vehicle_id": p["vehicle_id"].astype(str).to_numpy(),
                      "trip_id": p["trip_id"].astype(str).to_numpy(),
                      "t": _epoch(p["snapshot_ts"]),
                      "d": p["dist_along_m"].to_numpy(dtype=float)})
    p = p.drop_duplicates(["vehicle_id", "t"], keep="last").reset_index(drop=True)
    key = np.full(len(p), None, dtype=object)
    frac = np.full(len(p), np.nan)
    for tid, ix in pd.Series(np.arange(len(p))).groupby(p["trip_id"].to_numpy(), sort=False):
        if tid not in profs:
            continue
        dists, sids = profs[tid]
        ix = ix.to_numpy()
        a = np.searchsorted(dists, p["d"].to_numpy()[ix], side="right")
        ok = (a >= 1) & (a < len(dists))
        ix, a = ix[ok], a[ok]
        sa = np.asarray(sids, dtype=object)
        key[ix] = sa[a - 1] + ">" + sa[a]
        span = dists[a] - dists[a - 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            frac[ix] = np.where(span > 0, (p["d"].to_numpy()[ix] - dists[a - 1]) / span, 0.0)
    p["key"], p["frac"] = key, frac
    p = p[p["key"].notna()].drop(columns=["trip_id", "d"])
    p["key"] = p["key"].astype(str)
    return p.merge(feed, on=["vehicle_id", "t"], how="left")


def path_links(f: pd.DataFrame, profs: dict) -> pd.DataFrame:
    """Explode rows into the links between the vehicle and its target stop."""
    n = len(f)
    rt = f["trip_id"].astype(str).to_numpy()
    dv = f["dist_along_m"].to_numpy(dtype=float)
    dtg = f["stop_dist_along_m"].to_numpy(dtype=float)
    parts = []
    for tid, ix in pd.Series(np.arange(n)).groupby(rt, sort=False):
        if tid not in profs:
            continue
        dists, sids = profs[tid]
        sa = np.asarray(sids, dtype=object)
        ix = ix.to_numpy()
        a = np.searchsorted(dists, dv[ix], side="right")
        b = _nearest_idx(dists, dtg[ix])
        ok = (a >= 1) & (b >= a) & (a < len(dists))
        ix, a, b = ix[ok], a[ok], b[ok]
        cnt = b - a + 1
        rid = np.repeat(ix, cnt)
        j = np.repeat(a, cnt) + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
        parts.append(pd.DataFrame({"rid": rid, "key": sa[j - 1] + ">" + sa[j],
                                   "start": dists[j - 1], "span": dists[j] - dists[j - 1],
                                   "first": j == np.repeat(a, cnt)}))
    ex = pd.concat(parts, ignore_index=True)
    ex["key"] = ex["key"].astype(str)
    return ex


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
    ex["t"] = ft[ex["rid"].to_numpy()]
    m = ex.merge(occ, on=["key", "t"], how="inner")
    dv = f["dist_along_m"].to_numpy(dtype=float)[m["rid"].to_numpy()]
    m["opos"] = m["start"] + m["frac"] * m["span"]
    m["gap"] = m["opos"] - dv
    m = m[(m["vehicle_id"].to_numpy() != fv[m["rid"].to_numpy()])
          & (~m["first"] | (m["gap"] > 0))]
    m = m.assign(stopped=(m["fs"] <= 0).astype(float).where(m["fs"].notna()))
    g = m.groupby("rid")
    res = pd.DataFrame({"pa_n": g.size(), "pa_spd_mean": g["o120"].mean(), "pa_spd_min": g["o120"].min(),
                        "pa_stop_frac": g["stopped"].mean()})
    near = m.sort_values("gap").drop_duplicates("rid").set_index("rid")
    res["pa_gap_m"] = near["gap"]
    res["pa_gap_spd"] = near["o120"]
    occ_len = m.drop_duplicates(["rid", "key"]).groupby("rid")["span"].sum()
    tot_len = ex.groupby("rid")["span"].sum()
    res["pa_cov"] = (occ_len / tot_len).reindex(res.index)
    res = res.reindex(np.arange(len(f)))
    has_path = np.zeros(len(f), bool)
    has_path[ex["rid"].unique()] = True
    res.loc[has_path & res["pa_n"].isna().to_numpy(), ["pa_n", "pa_cov"]] = 0.0
    for c in V30_COLS:
        f[c] = res[c].to_numpy()
    DST.mkdir(parents=True, exist_ok=True)
    f.to_parquet(out, index=False)
    print(f"  {day}: {len(f):,} rows, path {has_path.mean():.3f}, pa_n>0 {(f['pa_n'] > 0).mean():.3f}, "
          f"pa_n mean {f['pa_n'].mean():.2f}, gap p50 {f['pa_gap_m'].median():.0f} m, "
          f"spd p50 {f['pa_spd_mean'].median():.1f}, occ {len(occ):,} ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", required=True)
    a = ap.parse_args()
    for d in expand_days(a.days):
        add_day(d)


if __name__ == "__main__":
    main()
