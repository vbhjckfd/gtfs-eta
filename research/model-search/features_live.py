"""
Causal "live" features computed from the full day of labelled rows.

Everything here only uses information that exists at snapshot time t (stop
crossings completed at or before t, the vehicle's own past positions), so it
is computable by the serving daemon from state it already sees every 10s —
though it would need a new shared store of recent link crossings (see
JOURNAL.md, "serving work").

Features (per sampled row):
  own_speed_{60,180,300}   metres advanced by this vehicle on this trip over
                           the last W seconds / W  (NaN if no position ~W ago
                           or the run reset)
  live_path_sec            recent traversal time of the path vehicle->target:
                           for every stop-to-stop link on the path, the latest
                           traversal by any vehicle completed in [t-30min, t]
                           (links keyed by stop-id pair, so routes sharing a
                           street share traffic); the vehicle's current link is
                           weighted by the fraction still to go; missing links
                           are extrapolated from the covered weight
  live_path_cov            covered weight / total weight (0 when none found)
  live_path_age            mean age (s) of the link observations used
  live_ratio               live_path_sec / (remaining_dist_m / hist speed…)
                           -> left to the model; we expose live_speed instead:
  live_speed_mps           path length / live_path_sec
  since_last_at_target     seconds since any vehicle last arrived at the target
                           stop (headway proxy; NaN if none in the last hour)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import build_sched_profile

import os

LINK_WINDOW_SEC = int(os.environ.get("MS_LINK_WINDOW", 1800))
TARGET_WINDOW_SEC = 3600
# A crossing is only known to the daemon once a later snapshot shows the vehicle
# past the stop; interpolated arrival times can sit before that. Only use
# crossings at least this old so no feature peeks past t.
DETECT_LAG_SEC = int(os.environ.get("MS_DETECT_LAG", 30))

LIVE_COLS = ["own_speed_60", "own_speed_180", "own_speed_300",
             "live_path_sec", "live_path_cov", "live_path_age", "live_speed_mps",
             "since_last_at_target"]


def _epoch(s: pd.Series) -> np.ndarray:
    s = pd.to_datetime(s, utc=True)
    return (s - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()


def _profiles(gtfs, trip_ids):
    out = {}
    for tid in trip_ids:
        trip = gtfs.get_trip(str(tid))
        if trip is None:
            continue
        prof = build_sched_profile(gtfs, trip)
        if not prof:
            continue
        out[tid] = (np.array([p[2] for p in prof]), [p[0] for p in prof])
    return out


def _nearest_idx(dists: np.ndarray, d: np.ndarray) -> np.ndarray:
    i = np.clip(np.searchsorted(dists, d), 1, len(dists) - 1) if len(dists) > 1 else np.zeros(len(d), int)
    if len(dists) > 1:
        left_closer = np.abs(d - dists[i - 1]) <= np.abs(dists[i] - d)
        i = np.where(left_closer, i - 1, i)
    return i


def live_features(cross: pd.DataFrame, pos: pd.DataFrame, rows: pd.DataFrame,
                  gtfs) -> pd.DataFrame:
    """cross/pos: full-day side tables from pipeline_lite; rows: sampled rows."""
    arr_t = _epoch(cross["actual_arrival"])
    rows_t = _epoch(rows["snapshot_ts"])
    profs = _profiles(gtfs, pd.unique(cross["trip_id"]))

    # ---- stop crossings -> link traversals ---------------------------------
    cr = pd.DataFrame({
        "vehicle_id": cross["vehicle_id"].to_numpy(), "trip_id": cross["trip_id"].to_numpy(),
        "stop_id": cross["stop_id"].astype(str).to_numpy(),
        "sd": cross["stop_dist_along_m"].to_numpy(dtype=float), "t": arr_t,
    }).drop_duplicates(["vehicle_id", "trip_id", "sd", "t"])
    cr["pidx"] = -1
    for tid, g in cr.groupby("trip_id", sort=False):
        if tid in profs:
            cr.loc[g.index, "pidx"] = _nearest_idx(profs[tid][0], g["sd"].to_numpy())
    cr = cr[cr["pidx"] >= 0].sort_values(["vehicle_id", "trip_id", "t"])
    same = (cr["vehicle_id"].eq(cr["vehicle_id"].shift())
            & cr["trip_id"].eq(cr["trip_id"].shift()))
    prev_stop = cr["stop_id"].shift()
    dt = cr["t"] - cr["t"].shift()
    ok = same & (cr["pidx"] == cr["pidx"].shift() + 1) & (dt > 0) & (dt < 1800)
    links = pd.DataFrame({"key": (prev_stop + ">" + cr["stop_id"])[ok].to_numpy(),
                          "t": cr["t"][ok].to_numpy(), "lt": dt[ok].to_numpy()})

    # ---- explode sampled rows into the links on their path -----------------
    n = len(rows)
    rt = rows["trip_id"].to_numpy()
    dv = rows["dist_along_m"].to_numpy(dtype=float)
    dtg = rows["stop_dist_along_m"].to_numpy(dtype=float)
    rid, keys, wts, plen = [], [], [], np.full(n, np.nan)
    for tid, ix in pd.Series(np.arange(n)).groupby(rt, sort=False):
        if tid not in profs:
            continue
        dists, sids = profs[tid]
        ix = ix.to_numpy()
        i0 = np.searchsorted(dists, dv[ix], side="right")
        it = _nearest_idx(dists, dtg[ix])
        for p, a, b in zip(ix, i0, it):
            if a < 1 or b < a:
                continue
            span = dists[a] - dists[a - 1]
            w0 = (dists[a] - dv[p]) / span if span > 0 else 1.0
            for j in range(a, b + 1):
                rid.append(p)
                keys.append(sids[j - 1] + ">" + sids[j])
                wts.append(min(max(w0, 0.0), 1.0) if j == a else 1.0)
            plen[p] = dists[b] - dv[p]
    ex = pd.DataFrame({"rid": np.array(rid, dtype=np.int64), "key": keys,
                       "w": np.array(wts), "t": rows_t[np.array(rid, dtype=np.int64)] if rid else []})
    ex["tq"] = ex["t"] - DETECT_LAG_SEC

    # latest traversal of each link completed at or before t
    allkeys = pd.Index(pd.unique(pd.concat([links["key"], ex["key"]])))
    links["k"] = allkeys.get_indexer(links["key"])
    ex["k"] = allkeys.get_indexer(ex["key"])
    m = pd.merge_asof(ex.sort_values("tq"), links[["k", "t", "lt"]].sort_values("t")
                      .rename(columns={"t": "tl"}),
                      left_on="tq", right_on="tl", by="k", direction="backward",
                      tolerance=LINK_WINDOW_SEC)
    found = m["lt"].notna()
    m["wlt"] = np.where(found, m["w"] * m["lt"], 0.0)
    m["wf"] = np.where(found, m["w"], 0.0)
    m["age"] = np.where(found, m["t"] - m["tl"], np.nan)
    agg = m.groupby("rid").agg(wlt=("wlt", "sum"), wf=("wf", "sum"), wt=("w", "sum"),
                               age=("age", "mean"))
    out = pd.DataFrame(index=np.arange(n))
    cov = (agg["wf"] / agg["wt"].where(agg["wt"] > 0)).reindex(out.index)
    lps = (agg["wlt"] / agg["wf"].where(agg["wf"] > 0) * agg["wt"]).reindex(out.index)
    out["live_path_sec"] = lps.to_numpy()
    out["live_path_cov"] = cov.fillna(0.0).to_numpy()
    out["live_path_age"] = agg["age"].reindex(out.index).to_numpy()
    out["live_speed_mps"] = plen / np.where(lps > 0, lps, np.nan)

    # ---- time since any vehicle last arrived at the target stop ------------
    st = cr[["stop_id", "t"]].rename(columns={"t": "ta"}).sort_values("ta")
    q = pd.DataFrame({"rid": np.arange(n), "stop_id": rows["stop_id"].astype(str).to_numpy(),
                      "t": rows_t, "tq": rows_t - DETECT_LAG_SEC}).sort_values("tq")
    qa = pd.merge_asof(q, st, left_on="tq", right_on="ta", by="stop_id",
                       direction="backward", tolerance=TARGET_WINDOW_SEC)
    out.loc[qa["rid"].to_numpy(), "since_last_at_target"] = (qa["t"] - qa["ta"]).to_numpy()

    # ---- own windowed speed -----------------------------------------------
    pos_df = pd.DataFrame({"vehicle_id": pos["vehicle_id"].to_numpy(),
                           "trip_id": pos["trip_id"].to_numpy(), "t": _epoch(pos["snapshot_ts"]),
                           "d": pos["dist_along_m"].to_numpy(dtype=float)}
                          ).drop_duplicates(["vehicle_id", "trip_id", "t"])
    pos_df["vt"] = pos_df["vehicle_id"].astype(str) + "|" + pos_df["trip_id"].astype(str)
    pos_df = pos_df.sort_values("t")
    q = pd.DataFrame({"rid": np.arange(n),
                      "vt": rows["vehicle_id"].astype(str).to_numpy() + "|" + rt.astype(str),
                      "d": dv, "t": rows_t})
    for W in (60, 180, 300):
        qq = q.assign(tw=q["t"] - W).sort_values("tw")
        mm = pd.merge_asof(qq, pos_df[["vt", "t", "d"]].rename(columns={"t": "tp", "d": "dp"}),
                           left_on="tw", right_on="tp", by="vt", direction="backward",
                           tolerance=60)
        adv = mm["d"] - mm["dp"]
        el = mm["t"] - mm["tp"]
        spd = np.where((adv > -30) & (el > 0), np.maximum(adv, 0) / el, np.nan)
        out.loc[mm["rid"].to_numpy(), f"own_speed_{W}"] = spd
    return out[LIVE_COLS].reset_index(drop=True)
