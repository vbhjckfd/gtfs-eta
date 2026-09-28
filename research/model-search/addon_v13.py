"""
Run 20 add-on: V13 cols (trip-keyed holds) appended to ms_features_v10 day files
(written to ms_features_v13, same rows), so no full re-prep is needed.

Diagnostic behind it (diag_holds.py): route 133's one vehicle stops 12-46 min at the same
place (400 m into the trip) at ~10:03 / ~12:15 / ~15:45 on every weekday since its trip ids
changed on 09-16; route 122 likewise (~12:45 at 2.7 km, ~09:30 at 900 m). Trip ids
(block_trip_dir) recur day to day, so these are timetabled breaks that a table keyed by
trip id can learn, while the V10 location table mixes them with ordinary stops.

  dwell_t_rem_med / dwell_t_p_more120 / dwell_t_n   V10's conditional remaining dwell keyed by
      trip_id + (last passed stop, 50 m bin), prior 14 days (min DWELL_MIN_N = 5 runs)
  hold_ahead_med    median over the prior days on which this trip_id ran of the total time in
      stationary runs >= 120 s between the vehicle's position (+25 m) and the target stop
  hold_ahead_days   number of those prior days (0 = trip unseen)
Serving: two static tables at export (trip x location sorted durations; trip -> list of
(dist, duration) holds per prior day), like the V10 dwell table; ~0.5 day.

Usage: python research/model-search/addon_v13.py --days 2026-09-07..2026-09-21
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

from features_live import _epoch, _loc_keys, _profiles, dwell_features  # noqa: E402
from harness import HIST_DAYS, TRAIN_DIR, expand_days  # noqa: E402

SRC = REPO / "data" / "ms_features_v10"
DST = REPO / "data" / "ms_features_v13"
HOLD_MIN_SEC = 120
V13_COLS = ["dwell_t_rem_med", "dwell_t_p_more120", "dwell_t_n", "hold_ahead_med", "hold_ahead_days"]


def trip_runs(pos: pd.DataFrame, profs) -> pd.DataFrame:
    """stationary_runs (features_live) with the trip id and the anchor distance kept."""
    p = pd.DataFrame({"vt": pos["vehicle_id"].astype(str).to_numpy() + "|" + pos["trip_id"].astype(str).to_numpy(),
                      "tid": pos["trip_id"].astype(str).to_numpy(), "t": _epoch(pos["snapshot_ts"]),
                      "d": pos["dist_along_m"].to_numpy(dtype=float)}
                     ).drop_duplicates(["vt", "t"]).sort_values(["vt", "t"]).reset_index(drop=True)
    vt, t, d = p["vt"].to_numpy(), p["t"].to_numpy(), p["d"].to_numpy()
    new = np.r_[True, (vt[1:] != vt[:-1]) | (np.diff(t) > 120)]
    anc = np.empty(len(p), dtype=np.int64)
    a = 0
    for i in range(len(p)):   # same anchor rule as features_live.stationary_runs
        if new[i] or d[i] - d[a] > 25.0:
            a = i
        anc[i] = a
    last = np.r_[anc[1:] != anc[:-1], True]
    ai = anc[last]
    D = t[last] - t[ai]
    ok = D > 0
    ai, D = ai[ok], D[ok]
    tid = p["tid"].to_numpy()[ai]
    keys = _loc_keys(profs, tid, d[ai])
    r = pd.DataFrame({"tid": tid, "key": keys, "d": d[ai], "t": t[ai], "D": D})
    return r[r["key"] != ""].reset_index(drop=True)


def _runs(day: str, client) -> pd.DataFrame:
    fp = DST / f"truns_{day}.parquet"
    if fp.exists():
        return pd.read_parquet(fp)
    from src.gtfs_static import get_gtfs_for_date
    pos = pd.read_parquet(TRAIN_DIR / f"{day}.pos.parquet")
    gtfs = get_gtfs_for_date(day, client=client)
    r = trip_runs(pos, _profiles(gtfs, pd.unique(pos["trip_id"])))
    r["day"] = day
    DST.mkdir(parents=True, exist_ok=True)
    r.to_parquet(fp, index=False)
    return r


def hold_ahead(prior: pd.DataFrame, tids: np.ndarray, d0: np.ndarray, d1: np.ndarray):
    """Median over prior days (on which the trip ran) of the summed long holds in (d0 + 25, d1]."""
    n = len(tids)
    med = np.full(n, np.nan)
    ndays = np.zeros(n)
    ran = prior.groupby("tid")["day"].unique()
    holds = prior[prior["D"] >= HOLD_MIN_SEC]
    hg = {k: g for k, g in holds.groupby("tid")}
    for tid, ix in pd.Series(np.arange(n)).groupby(tids, sort=False):
        if tid not in ran.index:
            continue
        ix = ix.to_numpy()
        days = ran[tid]
        ndays[ix] = len(days)
        tot = np.zeros((len(ix), len(days)))
        g = hg.get(tid)
        if g is not None:
            for j, dy in enumerate(days):
                h = g[g["day"] == dy]
                if h.empty:
                    continue
                o = np.argsort(h["d"].to_numpy())
                hd, hD = h["d"].to_numpy()[o], np.r_[0.0, np.cumsum(h["D"].to_numpy()[o])]
                lo = np.searchsorted(hd, d0[ix] + 25.0, side="right")
                hi = np.searchsorted(hd, d1[ix], side="right")
                tot[:, j] = hD[np.maximum(hi, lo)] - hD[lo]
        med[ix] = np.median(tot, axis=1)
    return med, ndays


def add_day(day: str, client) -> None:
    out = DST / f"{day}_p3.parquet"
    if out.exists():
        return
    f = pd.read_parquet(SRC / f"{day}_p3.parquet")
    d0 = date.fromisoformat(day)
    prior = [_runs(p, client) for k in range(1, HIST_DAYS + 1)
             if (TRAIN_DIR / f"{(p := (d0 - timedelta(days=k)).isoformat())}.pos.parquet").exists()]
    prior = pd.concat(prior, ignore_index=True)
    from src.gtfs_static import get_gtfs_for_date
    gtfs = get_gtfs_for_date(day, client=client)
    tids = f["trip_id"].astype(str).to_numpy()
    dv = f["dist_along_m"].to_numpy(dtype=float)
    loc = _loc_keys(_profiles(gtfs, pd.unique(tids)), tids, dv)
    tk = np.where(loc != "", tids.astype(object) + "#" + loc, "")
    pr = pd.DataFrame({"key": prior["tid"].astype(object) + "#" + prior["key"], "D": prior["D"]})
    dw = dwell_features([pr], tk, f["stationary_sec"].to_numpy(dtype=float))
    f["dwell_t_rem_med"] = np.where(tk != "", dw["dwell_rem_med"].to_numpy(), np.nan)
    f["dwell_t_p_more120"] = np.where(tk != "", dw["dwell_p_more120"].to_numpy(), np.nan)
    f["dwell_t_n"] = np.where(tk != "", dw["dwell_n"].to_numpy(), np.nan)
    med, nd = hold_ahead(prior, tids, dv, f["stop_dist_along_m"].to_numpy(dtype=float))
    f["hold_ahead_med"] = med
    f["hold_ahead_days"] = nd
    f.to_parquet(out, index=False)
    st = f["stationary_sec"].to_numpy(dtype=float) > 60
    print(f"  {day}: {len(f):,} rows; trip seen before {np.mean(nd > 0):.2f}; hold_ahead > 0 "
          f"{np.nanmean(med > 0):.3f}; stopped>60s dwell cov v10={f.loc[st, 'dwell_rem_med'].notna().mean():.2f}"
          f" trip={f.loc[st, 'dwell_t_rem_med'].notna().mean():.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", required=True)
    a = ap.parse_args()
    from src.snapshots import _make_client
    c = _make_client()
    for d in expand_days(a.days):
        add_day(d, c)


if __name__ == "__main__":
    main()
