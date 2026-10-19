"""
Run 15 add-on: V12 cols appended to existing ms_features_v10 day files
(written to ms_features_v12, same rows), so no full re-prep is needed.

Same-route headway / bunching (from the day's `.cross` stop arrivals, only
crossings completed <= t - DETECT_LAG_SEC, within the last 2 h):
  hw_ahead_cur   s since the previous OTHER vehicle of the same route+direction
                 arrived at the stop this vehicle passed last (the gap to the
                 leader at our position: a big gap = more boarding / a long
                 dwell ahead, a small gap = bunching, we run fast behind it)
  hw_ahead_tgt   s since the last other same-route+direction vehicle arrived
                 at the target stop
  hw_ahead_prev  the leader's own gap to ITS leader at that stop (regularity)
Serving: a dict (route, dir, stop) -> last 2 (t, vehicle) arrivals, fed by the
crossing events the live link store already consumes.

Coarse dwell fallback (V10's 50 m location key has < 5 runs for 28% of the
stopped > 60 s rows): dwell_c_rem_med / dwell_c_p_more120 / dwell_c_n = the
same conditional remaining dwell keyed by the last passed stop only.

Usage: python research/model-search/addon_v12.py --days 2026-09-07..2026-09-21
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

from features_live import (DETECT_LAG_SEC, _epoch, _loc_keys, _profiles,  # noqa: E402
                           dwell_features)
from harness import HIST_DAYS, TRAIN_DIR, expand_days  # noqa: E402

SRC = REPO / "data" / "ms_features_v10"
DST = REPO / "data" / "ms_features_v12"
HW_WINDOW = 7200
V12_COLS = ["hw_ahead_cur", "hw_ahead_tgt", "hw_ahead_prev",
            "dwell_c_rem_med", "dwell_c_p_more120", "dwell_c_n"]


def _rd(gtfs, tids) -> dict:
    out = {}
    for tid in pd.unique(tids):
        tr = gtfs.get_trip(str(tid))
        if tr is not None:
            out[tid] = f"{tr.route_id}|{tr.direction_id}"
    return out


def _arrivals(cross: pd.DataFrame, rd: dict) -> pd.DataFrame:
    """Per (route|dir|stop) arrivals with the previous OTHER vehicle's arrival."""
    c = pd.DataFrame({"k": cross["trip_id"].map(rd).astype(object) + "|" + cross["stop_id"].astype(str),
                      "veh": cross["vehicle_id"].astype(str).to_numpy(),
                      "t": _epoch(cross["actual_arrival"])})
    c = c.dropna(subset=["k", "t"]).sort_values(["k", "t"]).reset_index(drop=True)
    g = c.groupby("k", sort=False)
    tp1, tp2 = g["t"].shift(1), g["t"].shift(2)
    vp1, vp2 = g["veh"].shift(1), g["veh"].shift(2)
    # lt: the previous OTHER vehicle's arrival (the leader), lead: 1 or 2 rows back
    lead = np.where(vp1.notna() & (vp1 != c["veh"]), 1,
                    np.where(vp2.notna() & (vp2 != c["veh"]), 2, 0))
    c["lt"] = np.where(lead == 1, tp1, np.where(lead == 2, tp2, np.nan))
    gap = c["t"] - c["lt"]
    c["gap"] = np.where(gap <= HW_WINDOW, gap, np.nan)
    # the leader's own gap to its leader
    c["lgap"] = np.where(lead == 1, c.groupby("k")["gap"].shift(1),
                         np.where(lead == 2, c.groupby("k")["gap"].shift(2), np.nan))
    return c


def _last_other(arr: pd.DataFrame, keys: np.ndarray, veh: np.ndarray, tq: np.ndarray):
    """Latest arrival at key by a vehicle != veh, <= tq, within HW_WINDOW:
    returns (t_arrival, gap of that arrival to its own leader)."""
    n = len(keys)
    allk = pd.Index(pd.unique(np.concatenate([arr["k"].to_numpy(dtype=object), keys.astype(object)])))
    a = arr.assign(ki=allk.get_indexer(arr["k"])).sort_values("t")
    q = pd.DataFrame({"rid": np.arange(n), "ki": allk.get_indexer(pd.Series(keys, dtype=object)),
                      "veh": veh, "tq": tq}).sort_values("tq")
    m = pd.merge_asof(q, a[["ki", "t", "veh", "lt", "gap", "lgap"]].rename(columns={"veh": "veh_a"}),
                      left_on="tq", right_on="t", by="ki", direction="backward", tolerance=HW_WINDOW)
    m = m.set_index("rid").reindex(np.arange(n))
    own = (m["veh_a"] == m["veh"]).to_numpy()
    # latest arrival is our own -> the leader is that arrival's leader
    t_last = np.where(own, m["lt"].to_numpy(dtype=float), m["t"].to_numpy(dtype=float))
    gap = np.where(own, m["lgap"].to_numpy(dtype=float), m["gap"].to_numpy(dtype=float))
    ok = tq - t_last <= HW_WINDOW
    return np.where(ok, t_last, np.nan), np.where(ok, gap, np.nan)


def add_day(day: str, client) -> None:
    from src.gtfs_static import get_gtfs_for_date
    out = DST / f"{day}_p3.parquet"
    if out.exists():
        return
    f = pd.read_parquet(SRC / f"{day}_p3.parquet")
    gtfs = get_gtfs_for_date(day, client=client)
    tids = f["trip_id"].to_numpy()
    profs = _profiles(gtfs, pd.unique(tids))
    loc = _loc_keys(profs, tids, f["dist_along_m"].to_numpy(dtype=float))
    last_stop = np.array([k.split("@")[0] for k in loc], dtype=object)
    t = _epoch(f["snapshot_ts"])
    tq = t - DETECT_LAG_SEC
    veh = f["vehicle_id"].astype(str).to_numpy()

    cross = pd.read_parquet(TRAIN_DIR / f"{day}.cross.parquet")
    rd = _rd(gtfs, np.concatenate([pd.unique(tids), pd.unique(cross["trip_id"])]))
    arr = _arrivals(cross, rd)
    rdr = pd.Series(tids).map(rd).to_numpy(dtype=object)
    have = pd.notna(rdr) & (last_stop != "")
    kc = np.where(have, rdr.astype(str) + "|" + last_stop.astype(str), "")
    kt = np.where(pd.notna(rdr), rdr.astype(str) + "|" + f["stop_id"].astype(str).to_numpy(), "")
    t_cur, gap_cur = _last_other(arr, kc, veh, tq)
    t_tgt, _ = _last_other(arr, kt, veh, tq)
    f["hw_ahead_cur"] = np.where(kc != "", t - t_cur, np.nan)
    f["hw_ahead_tgt"] = np.where(kt != "", t - t_tgt, np.nan)
    f["hw_ahead_prev"] = np.where(kc != "", gap_cur, np.nan)

    d0 = date.fromisoformat(day)
    prior = []
    for k in range(1, HIST_DAYS + 1):
        fp = SRC / f"runs_{(d0 - timedelta(days=k)).isoformat()}.parquet"
        if fp.exists():
            r = pd.read_parquet(fp)
            r["key"] = r["key"].str.split("@").str[0]
            prior.append(r)
    dw = dwell_features(prior, last_stop, f["stationary_sec"].to_numpy(dtype=float))
    f["dwell_c_rem_med"] = np.where(last_stop != "", dw["dwell_rem_med"].to_numpy(), np.nan)
    f["dwell_c_p_more120"] = np.where(last_stop != "", dw["dwell_p_more120"].to_numpy(), np.nan)
    f["dwell_c_n"] = np.where(last_stop != "", dw["dwell_n"].to_numpy(), np.nan)
    DST.mkdir(parents=True, exist_ok=True)
    f.to_parquet(out, index=False)
    st = f["stationary_sec"].to_numpy(dtype=float) > 60
    print(f"  {day}: {len(f):,} rows, cov " +
          ", ".join(f"{c}={f[c].notna().mean():.2f}" for c in ("hw_ahead_cur", "hw_ahead_tgt", "hw_ahead_prev")) +
          f", hw_cur med {np.nanmedian(f['hw_ahead_cur']):.0f}s" +
          f"; stopped>60s dwell cov v10={f.loc[st, 'dwell_rem_med'].notna().mean():.2f}"
          f" coarse={f.loc[st, 'dwell_c_rem_med'].notna().mean():.2f}", flush=True)


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
