"""
Run 32 add-on: V32 duty / break-state cols appended to ms_features_v31 day files
(written to ms_features_v32, same rows).

The tail (routes 133 / 137 / 122) is driver breaks (runs 20-23). Trip-keyed hold
tables only gave +-0.5%. This tests the vehicle's own break history instead: a
driver who just had a long stop is unlikely to take another soon, and one who has
driven for hours is due one. All of it comes from the per-vehicle odometer ring
(V28/V29), so it is live-computable.

Stopped episodes: consecutive distinct GPS fixes (vehicle clock) of one vehicle
with odometer step <= 2 m and fix gap <= 120 s; length = last - first fix time.
Only episodes that already ended before the current fix count (the ongoing one is
stationary_sec).
  brk_since_s     feed t - end of the last episode >= 180 s (within 4 h, else NaN)
  brk_last_len    length of that episode
  brk_odo_since   odometer m since that episode ended
  brk10_since_s   feed t - end of the last episode >= 600 s (within 6 h)
  brk10_odo_since odometer m since then
  brk_n3h         episodes >= 180 s that ended in the last 3 h
  duty_s          feed t - first fix after the vehicle's last >= 30 min feed gap (shift start)
  duty_odo        odometer m since shift start
Serving: per-vehicle state of (last episode end, its length, its odometer, last
long-episode end, a 3 h deque of episode ends, shift start) updated from the feed
entity; persisted with the tracker state. Cheap.

Usage: python research/model-search/addon_v32.py --days 2026-09-15..2026-09-29
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

SRC = REPO / "data" / os.environ.get("MS_ADDON32_SRC", "ms_features_v31")
DST = REPO / "data" / os.environ.get("MS_ADDON32_DST", "ms_features_v32")
V32_COLS = ["brk_since_s", "brk_last_len", "brk_odo_since", "brk10_since_s", "brk10_odo_since",
            "brk_n3h", "duty_s", "duty_odo"]


def fixes(day: str) -> pd.DataFrame:
    a = pd.read_parquet(TRAIN_DIR / f"{day}.age.parquet")
    a = pd.DataFrame({"vehicle_id": a["vehicle_id"].astype(str).to_numpy(),
                      "t": _epoch(a["timestamp"]),
                      "vts": pd.to_numeric(a["vehicle_ts"]).to_numpy(dtype=float),
                      "odo": pd.to_numeric(a["odometer"]).to_numpy(dtype=float) * 1000.0})
    return a.drop_duplicates(["vehicle_id", "t"]).sort_values(["vehicle_id", "t"]).reset_index(drop=True)


def episodes(a: pd.DataFrame) -> pd.DataFrame:
    """Stopped episodes from distinct fixes: vehicle_id, start, end (vehicle clock), len, odo at end."""
    f = a.drop_duplicates(["vehicle_id", "vts"]).sort_values(["vehicle_id", "vts"]).reset_index(drop=True)
    v = f["vehicle_id"].to_numpy()
    same = np.r_[False, v[1:] == v[:-1]]
    dodo = np.r_[np.inf, np.diff(f["odo"].to_numpy())]
    dts = np.r_[np.inf, np.diff(f["vts"].to_numpy())]
    cont = same & (np.abs(dodo) <= 2.0) & (dts <= 120)
    run = np.cumsum(~cont)
    g = f.groupby(run, sort=False)
    e = pd.DataFrame({"vehicle_id": g["vehicle_id"].first(), "start": g["vts"].min(),
                      "end": g["vts"].max(), "odo_end": g["odo"].last()})
    e["len"] = e["end"] - e["start"]
    return e[e["len"] >= 180].reset_index(drop=True)


def _last_before(q: pd.DataFrame, e: pd.DataFrame, cols: list[str], tol: float) -> pd.DataFrame:
    """Per query row (vehicle_id, vts): the latest episode with end < vts (within tol)."""
    qq = q[["vehicle_id", "vts", "row"]].sort_values("vts")
    ee = e[["vehicle_id", "end"] + cols].rename(columns={"end": "vts"}).sort_values("vts")
    ee["e_end"] = ee["vts"]
    m = pd.merge_asof(qq, ee, on="vts", by="vehicle_id", direction="backward",
                      allow_exact_matches=False, tolerance=tol)
    return m.sort_values("row").reset_index(drop=True)


def state_table(day: str) -> pd.DataFrame:
    a = fixes(day)
    e = episodes(a)
    a["row"] = np.arange(len(a))
    m = _last_before(a, e, ["len", "odo_end"], 4 * 3600.0)
    a["brk_since_s"] = a["t"] - m["e_end"].to_numpy()
    a["brk_last_len"] = m["len"].to_numpy()
    a["brk_odo_since"] = a["odo"] - m["odo_end"].to_numpy()
    m10 = _last_before(a, e[e["len"] >= 600], ["odo_end"], 6 * 3600.0)
    a["brk10_since_s"] = a["t"] - m10["e_end"].to_numpy()
    a["brk10_odo_since"] = a["odo"] - m10["odo_end"].to_numpy()
    # episodes ended in (vts - 3h, vts): searchsorted on a vehicle-major composite key
    vc = {v: i for i, v in enumerate(pd.unique(a["vehicle_id"]))}
    base = float(np.nanmin(a["vts"])) - 86400.0
    span = 4 * 86400.0
    ek = np.sort(e["vehicle_id"].map(vc).to_numpy(dtype=float) * span + (e["end"].to_numpy() - base))
    qk = a["vehicle_id"].map(vc).to_numpy(dtype=float) * span + (a["vts"].to_numpy() - base)
    a["brk_n3h"] = (np.searchsorted(ek, qk, side="left")
                    - np.searchsorted(ek, qk - 3 * 3600.0, side="right")).astype(float)
    # shift start: first snapshot after the last >= 30 min feed gap of this vehicle
    v = a["vehicle_id"].to_numpy()
    t = a["t"].to_numpy(dtype=float)
    new = np.r_[True, (v[1:] != v[:-1]) | (np.diff(t) >= 1800)]
    sid = np.cumsum(new)
    st = pd.Series(t).groupby(sid).transform("first").to_numpy()
    so = pd.Series(a["odo"].to_numpy()).groupby(sid).transform("first").to_numpy()
    a["duty_s"] = t - st
    a["duty_odo"] = a["odo"].to_numpy() - so
    # stale fixes: feed t can run far ahead of the vehicle clock; keep the windows on the feed clock too
    a.loc[a["brk_since_s"] > 4 * 3600, ["brk_since_s", "brk_last_len", "brk_odo_since"]] = np.nan
    a.loc[a["brk10_since_s"] > 6 * 3600, ["brk10_since_s", "brk10_odo_since"]] = np.nan
    for c in ["brk_odo_since", "brk10_odo_since", "duty_odo"]:
        a.loc[a[c] < 0, c] = np.nan
    return a[["vehicle_id", "t"] + V32_COLS]


def add_day(day: str) -> None:
    out = DST / f"{day}_p3.parquet"
    if out.exists():
        return
    t0 = time.time()
    f = pd.read_parquet(SRC / f"{day}_p3.parquet")
    s = state_table(day)
    q = pd.DataFrame({"vehicle_id": f["vehicle_id"].astype(str).to_numpy(), "t": _epoch(f["snapshot_ts"]),
                      "row": np.arange(len(f))})
    q = q.merge(s, on=["vehicle_id", "t"], how="left").sort_values("row")
    for c in V32_COLS:
        f[c] = q[c].to_numpy()
    DST.mkdir(parents=True, exist_ok=True)
    f.to_parquet(out, index=False)
    print(f"  {day}: {len(f):,} rows, brk {f['brk_since_s'].notna().mean():.3f} p50 {f['brk_since_s'].median():.0f}s, "
          f"brk10 {f['brk10_since_s'].notna().mean():.3f}, n3h mean {f['brk_n3h'].mean():.2f}, "
          f"duty p50 {f['duty_s'].median() / 3600:.1f}h ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", required=True)
    a = ap.parse_args()
    for d in expand_days(a.days):
        add_day(d)


if __name__ == "__main__":
    main()
