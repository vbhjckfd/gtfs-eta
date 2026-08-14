"""
Head-to-head: the production direct-regression HistGBT vs a structurally
different *segment-additive* ETA model, on one identical train/test split.

Arm A (production)  — HistGradientBoostingRegressor predicts seconds_to_arrival
                      directly, one row per (snapshot, target stop), any horizon.
Arm B (challenger)  — no learned regressor at all. Decomposes the ETA into
                      stop-to-stop link times:
                          eta(k) = partial_current_segment
                                 + sum(link_time(stop_j) for j = 2..k)
                      link_time is the empirical median seconds between arrival
                      at stop j-1 and arrival at stop j, keyed by
                      (route_id, stop_id, hour) with hierarchical fallbacks.
                      Dwell is included for free (arrival-to-arrival).
Arm B1              — B plus a live congestion multiplier from the vehicle's
                      observed speed vs its route/hour historical speed.
Arm B2              — B plus a per-horizon additive calibration fitted on the
                      training split (summing per-link medians undershoots a
                      right-skewed sum, and the miss grows with link count).

Both arms are scored on exactly the same test rows (joined on uid), so the
numbers are directly comparable — a shared uid survives the arm-A feature build
and the arm-B group arithmetic.

Measured 2026-08-14 (30 days to 2026-08-04, 9.6M sampled rows, split at
2026-07-30, 1,493,431 test rows):

    arm                MAE    bias   medAE    p90     fit
    A  HistGBT       139.2   -55.0    57.4   287.9   1854s
    B  segment       172.7  -122.1    56.3   378.9     20s
    B2 + calibration 170.0   -91.5    58.5   358.6     20s
    B1 + congestion  246.4   -89.0   103.4   643.4     20s
    0.5A + 0.5B2     146.7   -73.3    55.2   300.8       —
    schedule         205.8   -64.2    98.8   438.1       —

A wins at every horizon (96 vs 140s at 1 stop ahead, 183 vs 219s at 10). B ties
A on the *median* row while losing badly in the tail (p99 2104 vs 1602): with no
live-vehicle state it keeps serving historical link times to a stuck vehicle.
B2 beats A on 48% of rows but on only 1 of 60 routes with n>2000, and every
blend weight > 0 is worse than A alone, so there is no cheap gate — an oracle
per-row pick would score 116.1s, so the complementarity is real but unexploited.

The result worth keeping: B2 is a 54k-key lookup table that fits in 20s and
needs no tree traversal, and at 170s it beats the schedule fallback by 17%.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.features import (  # noqa: E402
    BASE_FEATURE_COLS, FEATURE_COLS, TARGET_COL, _UA_HOLIDAYS,
    apply_priors, build_sched_profile,
)

RAW_COLS = [
    "vehicle_id", "trip_id", "route_id", "date", "snapshot_ts", "dist_along_m",
    "progress_speed_mps", "stop_id", "stop_sequence", "stop_dist_along_m",
    "stops_ahead", "stops_remaining", "seconds_to_arrival", "stationary_sec",
]
_BAD_ROUTE_IDS = frozenset({"2299", "138"})


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_raw(days: int, keep_pct: int) -> pd.DataFrame:
    """Recent training parquets, sampled at *whole snapshot group* granularity.

    Row-level sampling would break arm B (it needs every stop of a snapshot to
    difference consecutive arrivals into link times), so the filter keys on
    snapshot_ts — every row of a kept snapshot survives together.
    """
    files = sorted((REPO / "data/training").glob("*.parquet"))[-days:]
    pieces = []
    for i, f in enumerate(files):
        df = pd.read_parquet(f, columns=RAW_COLS)
        # Sample distinct snapshot timestamps, not seconds-modulo: the daemon's
        # poll times cluster on particular second residues, so a modulo filter
        # keeps wildly different shares from different days.
        snaps = df["snapshot_ts"].drop_duplicates()
        keep = snaps.sample(frac=keep_pct / 100.0, random_state=42 + i)
        df = df[df["snapshot_ts"].isin(set(keep))]
        pieces.append(df)
        print(f"  {f.name}: {len(df):,} rows kept", flush=True)
    raw = pd.concat(pieces, ignore_index=True)
    raw["uid"] = np.arange(len(raw), dtype=np.int64)
    print(f"  total {len(raw):,} rows, {raw['date'].nunique()} dates")
    return raw


# ---------------------------------------------------------------------------
# Arm A features (mirror of src.features.compute_features_for_training,
# carrying uid so both arms can be scored on identical rows)
# ---------------------------------------------------------------------------

def features_with_uid(rows: pd.DataFrame, gtfs) -> pd.DataFrame:
    rows = rows.copy()
    rows["snapshot_ts"] = pd.to_datetime(rows["snapshot_ts"], utc=True)
    pieces = []
    for trip_id, grp in rows.groupby("trip_id", sort=False):
        trip = gtfs.get_trip(str(trip_id))
        if trip is None:
            continue
        shape_len = max(gtfs.get_shape_length(trip.shape_id), 1.0)
        profile = build_sched_profile(gtfs, trip)
        if not profile:
            continue
        prof_dists = np.array([p[2] for p in profile])
        prof_cums = np.array([p[3] for p in profile])
        sched_by_seq = {seq: cum for (_, seq, _, cum) in profile}

        d_vehicle = grp["dist_along_m"].to_numpy(dtype=float)
        d_target = grp["stop_dist_along_m"].to_numpy(dtype=float)
        sched_at_pos = np.interp(d_vehicle, prof_dists, prof_cums)
        sched_target = (
            grp["stop_sequence"].astype(int).map(sched_by_seq)
            .fillna(pd.Series(sched_at_pos, index=grp.index))
            .to_numpy(dtype=float)
        )
        snap = grp["snapshot_ts"].dt
        dow = snap.weekday.to_numpy()
        month_day = snap.month * 100 + snap.day
        holiday = month_day.isin({m * 100 + d for m, d in _UA_HOLIDAYS}).astype(int)
        stops_ahead_arr = grp["stops_ahead"].astype(int).to_numpy()
        rem_dist = np.maximum(0.0, d_target - d_vehicle)

        pieces.append(pd.DataFrame({
            "uid": grp["uid"].to_numpy(),
            "route_id": trip.route_id,
            "stop_sequence": grp["stop_sequence"].astype(int).to_numpy(),
            "stops_ahead": stops_ahead_arr,
            "hour": snap.hour.to_numpy(),
            "day_of_week": dow,
            "month": snap.month.to_numpy(),
            "is_weekend": (dow >= 5).astype(int),
            "is_holiday": holiday.to_numpy(),
            "remaining_dist_m": rem_dist,
            "progress_speed_mps": grp["progress_speed_mps"].to_numpy(dtype=float),
            "stops_remaining": grp["stops_remaining"].astype(int).to_numpy(),
            "trip_progress_frac": d_target / shape_len,
            "dist_per_stop_m": rem_dist / np.maximum(1, stops_ahead_arr),
            "stationary_sec": grp["stationary_sec"].to_numpy(dtype=float),
            "sched_remaining_sec": np.maximum(0.0, sched_target - sched_at_pos),
            "date": grp["date"].to_numpy(),
            TARGET_COL: grp["seconds_to_arrival"].to_numpy(dtype=float),
        }))
    return pd.concat(pieces, ignore_index=True)


# ---------------------------------------------------------------------------
# Arm B: segment-additive link-time model
# ---------------------------------------------------------------------------

GROUP_KEYS = ["vehicle_id", "trip_id", "snapshot_ts"]


def _link_observations(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per observed stop-to-stop link: seconds and metres.

    Within a snapshot group the per-stop targets are y_1 < y_2 < ... < y_k, all
    anchored to the same instant, so y_j - y_{j-1} is the observed traversal
    time of the segment ending at stop j — arrival to arrival, dwell included.
    """
    df = raw.sort_values(GROUP_KEYS + ["stops_ahead"])
    g = df.groupby(GROUP_KEYS, sort=False)
    d_sec = g["seconds_to_arrival"].diff()
    d_m = g["stop_dist_along_m"].diff()
    d_ahead = g["stops_ahead"].diff()
    ok = (d_ahead == 1) & d_sec.notna() & (d_sec > 0) & (d_sec < 1800)
    out = df.loc[ok, ["route_id", "stop_id", "date"]].copy()
    out["hour"] = df.loc[ok, "snapshot_ts"].dt.hour.to_numpy()
    out["link_sec"] = d_sec[ok].to_numpy()
    out["link_m"] = d_m[ok].to_numpy()
    return out


def build_link_tables(raw_train: pd.DataFrame) -> dict:
    obs = _link_observations(raw_train)
    print(f"  link observations: {len(obs):,}")

    def med(keys, col="link_sec"):
        return obs.groupby(keys, observed=True)[col].median()

    tables = {
        "rsh": med(["route_id", "stop_id", "hour"]).to_dict(),
        "rs": med(["route_id", "stop_id"]).to_dict(),
        "s": med(["stop_id"]).to_dict(),
        "rh": med(["route_id", "hour"]).to_dict(),
        "global": float(obs["link_sec"].median()),
        # Segment length, for the partial first segment.
        "seglen_rs": med(["route_id", "stop_id"], "link_m").to_dict(),
        "seglen_global": float(obs["link_m"].median()),
    }
    # Live-speed reference for the congestion multiplier: historical median
    # link speed per (route, hour).
    obs = obs[obs["link_sec"] > 0]
    spd = (obs["link_m"] / obs["link_sec"]).groupby(
        [obs["route_id"], obs["hour"]]
    ).median()
    tables["hist_speed_rh"] = spd.to_dict()
    tables["hist_speed_global"] = float((obs["link_m"] / obs["link_sec"]).median())
    print(f"  keys: rsh={len(tables['rsh']):,} rs={len(tables['rs']):,} "
          f"s={len(tables['s']):,} global={tables['global']:.0f}s")
    return tables


def _lookup_link(raw: pd.DataFrame, t: dict) -> np.ndarray:
    """Per-row link time for the segment ending at that row's stop."""
    route = raw["route_id"].to_numpy()
    stop = raw["stop_id"].to_numpy()
    hour = raw["snapshot_ts"].dt.hour.to_numpy()
    rsh = pd.Series(list(zip(route, stop, hour))).map(t["rsh"]).to_numpy(dtype=float)
    rs = pd.Series(list(zip(route, stop))).map(t["rs"]).to_numpy(dtype=float)
    s = pd.Series(stop).map(t["s"]).to_numpy(dtype=float)
    rh = pd.Series(list(zip(route, hour))).map(t["rh"]).to_numpy(dtype=float)
    out = np.where(np.isfinite(rsh), rsh,
          np.where(np.isfinite(rs), rs,
          np.where(np.isfinite(s), s,
          np.where(np.isfinite(rh), rh, t["global"]))))
    return out


def predict_segment(raw: pd.DataFrame, t: dict, congestion: bool = False) -> np.ndarray:
    """eta(k) = partial(current segment) + sum of link times for stops 2..k."""
    df = raw.sort_values(GROUP_KEYS + ["stops_ahead"]).copy()
    df["_link"] = _lookup_link(df, t)

    if congestion:
        route = df["route_id"].to_numpy()
        hour = df["snapshot_ts"].dt.hour.to_numpy()
        hist = pd.Series(list(zip(route, hour))).map(t["hist_speed_rh"]).to_numpy(dtype=float)
        hist = np.where(np.isfinite(hist), hist, t["hist_speed_global"])
        obs_speed = df["progress_speed_mps"].to_numpy(dtype=float)
        ratio = np.where(obs_speed > 0.3, hist / np.maximum(obs_speed, 0.3), 1.0)
        # A vehicle slower than its historical norm needs proportionally longer;
        # clipped so one bad GPS sample can't triple the ETA.
        df["_link"] = df["_link"] * np.clip(ratio, 0.7, 2.0)

    g = df.groupby(GROUP_KEYS, sort=False)
    cum = g["_link"].cumsum()
    first_link = g["_link"].transform("first")
    # Stops 2..k only: drop the first segment's full link time, replace with the
    # partial remaining piece below.
    ahead_sum = cum - first_link

    seglen = pd.Series(
        list(zip(df["route_id"].to_numpy(), df["stop_id"].to_numpy()))
    ).map(t["seglen_rs"]).to_numpy(dtype=float)
    seglen = np.where(np.isfinite(seglen) & (seglen > 1), seglen, t["seglen_global"])
    first_seglen = pd.Series(seglen, index=df.index).groupby(
        [df[k] for k in GROUP_KEYS], sort=False
    ).transform("first").to_numpy()
    rem_first = (
        g["stop_dist_along_m"].transform("first").to_numpy()
        - df["dist_along_m"].to_numpy()
    )
    frac = np.clip(rem_first / np.maximum(first_seglen, 1.0), 0.0, 1.0)
    partial = frac * first_link.to_numpy()

    pred = partial + ahead_sum.to_numpy()
    return pd.Series(pred, index=df["uid"].to_numpy())


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    err = p - y
    ae = np.abs(err)
    return {
        "n": int(len(y)),
        "mae": round(float(ae.mean()), 1),
        "bias": round(float(err.mean()), 1),
        "median_ae": round(float(np.median(ae)), 1),
        "p90_ae": round(float(np.percentile(ae, 90)), 1),
    }


def by_group(df: pd.DataFrame, col: str, preds: dict) -> dict:
    out = {}
    for key, g in df.groupby(col, observed=True):
        out[str(key)] = {
            name: metrics(g[TARGET_COL].to_numpy(), g[name].to_numpy())
            for name in preds
        }
    return out


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--keep-pct", type=int, default=2)
    ap.add_argument("--test-fraction", type=float, default=0.2)
    ap.add_argument("--max-iter", type=int, default=1200)
    ap.add_argument("--out", default="segment_experiment.json")
    args = ap.parse_args()

    t0 = time.monotonic()
    print("Loading raw training rows…", flush=True)
    raw = load_raw(args.days, args.keep_pct)
    raw["snapshot_ts"] = pd.to_datetime(raw["snapshot_ts"], utc=True)

    dates = sorted(pd.to_datetime(raw["date"]).unique())
    cutoff = dates[int(len(dates) * (1 - args.test_fraction))]
    raw_date = pd.to_datetime(raw["date"])
    raw_train, raw_test = raw[raw_date < cutoff], raw[raw_date >= cutoff]
    print(f"  split at {pd.Timestamp(cutoff).date()}: "
          f"{len(raw_train):,} train / {len(raw_test):,} test raw rows")

    # ---- Arm B ----
    print("\nArm B: building link-time tables…", flush=True)
    tb = time.monotonic()
    tables = build_link_tables(raw_train)
    # Summing per-link medians underestimates a right-skewed sum, and the miss
    # grows with the number of links summed. Calibrate it out per horizon on the
    # training split only — still a lookup, no regressor.
    in_sample = predict_segment(raw_train, tables, congestion=False)
    tr = raw_train[["uid", "stops_ahead", "seconds_to_arrival"]].copy()
    tr["p"] = tr["uid"].map(in_sample)
    tr = tr.dropna(subset=["p"])
    corr = (tr["seconds_to_arrival"] - tr["p"]).groupby(tr["stops_ahead"]).median().to_dict()
    tables["horizon_corr"] = {int(k): float(v) for k, v in corr.items()}
    fit_b = time.monotonic() - tb
    print("  horizon correction (s): "
          + ", ".join(f"{h}:{v:+.0f}" for h, v in sorted(tables["horizon_corr"].items())))

    tb = time.monotonic()
    pred_b = predict_segment(raw_test, tables, congestion=False)
    pred_b1 = predict_segment(raw_test, tables, congestion=True)
    infer_b = time.monotonic() - tb
    print(f"  fit {fit_b:.1f}s, predict {infer_b:.1f}s")

    # ---- Arm A ----
    print("\nArm A: building features…", flush=True)
    from src.gtfs_static import get_gtfs
    gtfs = get_gtfs()
    feats = features_with_uid(raw, gtfs)
    feats = feats.dropna(subset=[TARGET_COL] + BASE_FEATURE_COLS)
    feats = feats[feats[TARGET_COL].between(0, 3600)]
    feats = feats[~feats["route_id"].astype(str).isin(_BAD_ROUTE_IDS)]
    fdate = pd.to_datetime(feats["date"])
    ftrain, ftest = feats[fdate < cutoff].copy(), feats[fdate >= cutoff].copy()
    print(f"  features: {len(ftrain):,} train / {len(ftest):,} test rows")

    from src.train import _build_pipeline, _build_sample_weights, _compute_route_hour_priors
    priors = _compute_route_hour_priors(ftrain)
    ftrain = apply_priors(ftrain, priors)
    ftest = apply_priors(ftest, priors)

    print("Fitting HistGBT…", flush=True)
    ta = time.monotonic()
    pipe = _build_pipeline()
    pipe.named_steps["model"].set_params(max_iter=args.max_iter)
    pipe.fit(ftrain[FEATURE_COLS], ftrain[TARGET_COL].astype(float),
             model__sample_weight=_build_sample_weights(ftrain))
    fit_a = time.monotonic() - ta
    ta = time.monotonic()
    ftest["pred_A"] = pipe.predict(ftest[FEATURE_COLS])
    infer_a = time.monotonic() - ta
    print(f"  stopped at iter {pipe.named_steps['model'].n_iter_}, "
          f"fit {fit_a:.1f}s, predict {infer_a:.1f}s")

    # ---- Score on identical rows ----
    ev = ftest.copy()
    ev["pred_B"] = ev["uid"].map(pred_b)
    ev["pred_B1"] = ev["uid"].map(pred_b1)
    ev = ev.dropna(subset=["pred_B", "pred_B1"])
    corr_vec = ev["stops_ahead"].map(tables["horizon_corr"]).fillna(0.0)
    ev["pred_B2"] = (ev["pred_B"] + corr_vec).clip(lower=0)
    ev["pred_blend"] = 0.5 * ev["pred_A"] + 0.5 * ev["pred_B2"]
    ev["pred_sched"] = ev["sched_remaining_sec"]
    ev["pred_speed"] = ev["speed_eta_warm"].clip(0, 3600)

    names = ["pred_A", "pred_B", "pred_B1", "pred_B2", "pred_blend",
             "pred_sched", "pred_speed"]
    y = ev[TARGET_COL].to_numpy()
    overall = {n: metrics(y, ev[n].to_numpy()) for n in names}

    report = {
        "config": vars(args),
        "n_eval_rows": int(len(ev)),
        "train_dates": [str(pd.Timestamp(d).date()) for d in (dates[0], cutoff)],
        "cost_sec": {"fit_A": round(fit_a, 1), "predict_A": round(infer_a, 2),
                     "fit_B": round(fit_b, 1), "predict_B": round(infer_b, 2)},
        "overall": overall,
        "by_stops_ahead": by_group(ev, "stops_ahead", names),
        "by_hour": by_group(ev, "hour", names),
    }
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    # Persist predictions + model so further arms can be scored on this exact
    # split without refitting.
    keep = ["uid", "route_id", "stops_ahead", "hour", TARGET_COL] + names
    ev[keep].to_parquet(Path(args.out).with_suffix(".preds.parquet"), index=False)
    import joblib
    joblib.dump({"pipeline": pipe, "priors": priors, "tables": tables},
                Path(args.out).with_suffix(".models.joblib"))

    print(f"\n{'model':<12} {'MAE':>8} {'bias':>8} {'medAE':>8} {'p90':>8}")
    for n in names:
        m = overall[n]
        print(f"{n:<12} {m['mae']:>8.1f} {m['bias']:>8.1f} "
              f"{m['median_ae']:>8.1f} {m['p90_ae']:>8.1f}")
    print(f"\nrows {len(ev):,}   total {time.monotonic() - t0:.0f}s   → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
