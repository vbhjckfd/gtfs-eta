"""
Run 14 add-on: V11 dwell cols appended to existing ms_features_v10 day files
(written to ms_features_v11, same rows), so no full re-prep is needed.

  dwell_h_rem_med / dwell_h_p_more120 / dwell_h_n
        as V10's conditional remaining dwell, keyed by (location, 3-hour local
        band) — terminus layovers depend on the timetable of the hour
  dwell_live_last / dwell_live_age / dwell_live_m3
        today's most recent completed stop (run >= 30 s) by ANY vehicle at the
        same location (ended <= t - 30 s, within the last hour): its duration,
        its age, and the mean of the last 3 such runs. Live state at serving:
        a per-location ring of the last 3 stop durations.

Usage: python research/model-search/addon_v11.py --days 2026-08-31..2026-09-21
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
                           dwell_features, local_hour)
from harness import HIST_DAYS, expand_days  # noqa: E402

SRC = REPO / "data" / "ms_features_v10"
DST = REPO / "data" / "ms_features_v11"
V11_COLS = ["dwell_h_rem_med", "dwell_h_p_more120", "dwell_h_n",
            "dwell_live_last", "dwell_live_age", "dwell_live_m3"]


def add_day(day: str, client) -> None:
    from src.gtfs_static import get_gtfs_for_date
    out = DST / f"{day}_p3.parquet"
    if out.exists():
        return
    f = pd.read_parquet(SRC / f"{day}_p3.parquet")
    gtfs = get_gtfs_for_date(day, client=client)
    tids = f["trip_id"].to_numpy()
    profs = _profiles(gtfs, pd.unique(tids))
    keys = _loc_keys(profs, tids, f["dist_along_m"].to_numpy(dtype=float))
    t = _epoch(f["snapshot_ts"])
    s = f["stationary_sec"].to_numpy(dtype=float)
    band = (local_hour(t) // 3).astype(str)

    d0 = date.fromisoformat(day)
    prior = []
    for k in range(1, HIST_DAYS + 1):
        fp = SRC / f"runs_{(d0 - timedelta(days=k)).isoformat()}.parquet"
        if fp.exists():
            r = pd.read_parquet(fp)
            r["key"] = r["key"] + "#" + (local_hour(r["t"].to_numpy()) // 3).astype(str)
            prior.append(r)
    kb = np.where(keys != "", keys + "#" + band, "")
    dw = dwell_features(prior, kb, s)
    f["dwell_h_rem_med"] = dw["dwell_rem_med"].to_numpy()
    f["dwell_h_p_more120"] = dw["dwell_p_more120"].to_numpy()
    f["dwell_h_n"] = dw["dwell_n"].to_numpy()

    # live: today's completed stops at the same location
    today = pd.read_parquet(SRC / f"runs_{day}.parquet")
    today = today[today["D"] >= 30].copy()
    today["te"] = today["t"] + today["D"]
    today = today.sort_values(["key", "te"])
    today["m3"] = (today.groupby("key")["D"].rolling(3, min_periods=1).mean()
                   .reset_index(level=0, drop=True).to_numpy())
    allk = pd.Index(pd.unique(np.concatenate([today["key"].to_numpy(dtype=object), keys])))
    today["k"] = allk.get_indexer(today["key"])
    q = pd.DataFrame({"rid": np.arange(len(f)), "k": allk.get_indexer(pd.Series(keys, dtype=object)),
                      "tq": t - DETECT_LAG_SEC}).sort_values("tq")
    m = pd.merge_asof(q, today[["k", "te", "D", "m3"]].sort_values("te"), left_on="tq",
                      right_on="te", by="k", direction="backward", tolerance=3600)
    m = m.set_index("rid").reindex(np.arange(len(f)))
    blank = keys == ""
    f["dwell_live_last"] = np.where(blank, np.nan, m["D"].to_numpy())
    f["dwell_live_age"] = np.where(blank, np.nan, (m["tq"] - m["te"]).to_numpy())
    f["dwell_live_m3"] = np.where(blank, np.nan, m["m3"].to_numpy())
    DST.mkdir(parents=True, exist_ok=True)
    f.to_parquet(out, index=False)
    print(f"  {day}: {len(f):,} rows, cov " +
          ", ".join(f"{c}={f[c].notna().mean():.2f}" for c in ("dwell_h_rem_med", "dwell_live_last")),
          flush=True)


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
