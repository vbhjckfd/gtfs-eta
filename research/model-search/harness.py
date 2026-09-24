"""
Model-search evaluation harness (research only — never touches R2 writes).

Reproduces the production recipe in src/train.py (same features, filters,
route+hour priors computed on the train split, sample weights, HistGBT
hyper-parameters) but with an explicit, fixed day split so every arm is
scored on the identical held-out rows.

Differences from `python -m src.train` that apply to baseline AND candidates:
  * features are built per day against get_gtfs_for_date(day) (era-pinned
    static) instead of the current static, so rows are reproducible;
  * split is by explicit day lists instead of the last 20% of dates;
  * rows are sampled per day at --keep-pct (seeded) to keep fits short.

Usage:
    python research/model-search/harness.py prep --days 2026-09-01..2026-09-21
    python research/model-search/harness.py run --arm baseline \
        --train 2026-09-01..2026-09-17 --test 2026-09-18..2026-09-21
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

from src import train as prod  # noqa: E402
from src.features import (  # noqa: E402
    BASE_FEATURE_COLS, FEATURE_COLS, TARGET_COL, apply_priors,
    build_sched_profile, compute_features_for_training,
)

TRAIN_DIR = REPO / "data" / "ms_lite"   # pipeline_lite output (10% snapshots)
FEAT_DIR = REPO / "data" / "ms_features"
RESULTS_DIR = Path(__file__).parent / "results"

PASSTHROUGH = ["vehicle_id", "trip_id", "stop_id", "snapshot_ts",
               "dist_along_m", "stop_dist_along_m", "actual_arrival"]


def expand_days(spec: str) -> list[str]:
    out = []
    for part in spec.split(","):
        if ".." in part:
            a, b = part.split("..")
            d, e = date.fromisoformat(a), date.fromisoformat(b)
            while d <= e:
                out.append(d.isoformat())
                d += timedelta(days=1)
        else:
            out.append(part)
    return out


# --------------------------------------------------------------------------
# prep: sampled, era-pinned feature rows per day (cached)
# --------------------------------------------------------------------------

def feat_path(day: str, keep_pct: float) -> Path:
    return FEAT_DIR / f"{day}_p{keep_pct:g}.parquet"


def prep_day(day: str, keep_pct: float, client=None) -> Path:
    out = feat_path(day, keep_pct)
    if out.exists():
        return out
    from src.gtfs_static import get_gtfs_for_date
    gtfs = get_gtfs_for_date(day, client=client)
    raw = pd.read_parquet(TRAIN_DIR / f"{day}.parquet")
    n_raw = len(raw)
    # Sample whole snapshots (vehicle, ts) so all horizons of one snapshot stay
    # together — needed by per-snapshot features and harmless otherwise.
    key = pd.util.hash_pandas_object(
        raw[["vehicle_id", "snapshot_ts"]].astype(str), index=False
    ).to_numpy()
    raw = raw[(key % 10000) < keep_pct * 100].reset_index(drop=True)

    # compute_features_for_training iterates groupby(trip_id, sort=False) and
    # skips trips missing from static / without a profile; replay that order
    # to carry passthrough columns alongside.
    order = []
    for trip_id, grp in raw.groupby("trip_id", sort=False):
        trip = gtfs.get_trip(str(trip_id))
        if trip is None or not build_sched_profile(gtfs, trip):
            continue
        order.append(grp.index.to_numpy())
    feats = compute_features_for_training(raw, gtfs)
    idx = np.concatenate(order) if order else np.array([], dtype=int)
    assert len(idx) == len(feats), (len(idx), len(feats))
    for c in PASSTHROUGH:
        if c in raw.columns:
            feats[c] = raw[c].to_numpy()[idx]
    feats["route_id"] = feats["route_id"].astype(str)
    from features_live import live_features
    cross = pd.read_parquet(TRAIN_DIR / f"{day}.cross.parquet")
    pos = pd.read_parquet(TRAIN_DIR / f"{day}.pos.parquet")
    live = live_features(cross, pos, raw.iloc[idx].reset_index(drop=True), gtfs)
    for c in live.columns:
        feats[c] = live[c].to_numpy()
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out, index=False)
    print(f"  {day}: {n_raw:,} raw -> {len(feats):,} feature rows @ {keep_pct}%")
    return out


def load_days(days: list[str], keep_pct: float) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(feat_path(d, keep_pct)) for d in days],
                     ignore_index=True)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def metrics(df: pd.DataFrame, pred: np.ndarray) -> dict:
    y = df[TARGET_COL].to_numpy(dtype=float)
    err = pred - y
    ae = np.abs(err)
    out = {
        "n": int(len(y)),
        "mae": float(ae.mean()),
        "median_ae": float(np.median(ae)),
        "p90_ae": float(np.percentile(ae, 90)),
        "bias": float(err.mean()),
    }
    sa = df["stops_ahead"].to_numpy()
    out["mae_by_stops_ahead"] = {
        int(k): float(ae[sa == k].mean()) for k in range(1, 11) if (sa == k).any()
    }
    by_route = (pd.DataFrame({"r": df["route_id"].astype(str).to_numpy(), "ae": ae})
                .groupby("r")["ae"].agg(["mean", "size"]))
    by_route = by_route[by_route["size"] >= 500].sort_values("mean", ascending=False)
    out["worst_routes"] = {r: round(float(m), 1) for r, m in by_route["mean"].head(10).items()}
    days = pd.to_datetime(df["date"]).dt.date.astype(str).to_numpy()
    out["mae_by_day"] = {d: float(ae[days == d].mean()) for d in sorted(set(days))}
    out["n_by_day"] = {d: int((days == d).sum()) for d in sorted(set(days))}
    return out


def judge(base: dict, cand: dict) -> dict:
    d_mae = (cand["mae"] - base["mae"]) / base["mae"]
    worst_bucket = max(
        (cand["mae_by_stops_ahead"][k] - base["mae_by_stops_ahead"][k])
        / base["mae_by_stops_ahead"][k]
        for k in base["mae_by_stops_ahead"] if k in cand["mae_by_stops_ahead"]
    )
    return {
        "mae_change_pct": 100 * d_mae,
        "p90_change_pct": 100 * (cand["p90_ae"] - base["p90_ae"]) / base["p90_ae"],
        "worst_bucket_change_pct": 100 * worst_bucket,
        "wins": bool(d_mae <= -0.03 and cand["p90_ae"] <= base["p90_ae"]
                     and worst_bucket <= 0.05),
    }


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------

def prepare_split(train_days, test_days, keep_pct, test_keep_pct):
    tr = load_days(train_days, keep_pct)
    te = load_days(test_days, test_keep_pct)
    out = []
    for df in (tr, te):
        df = df.dropna(subset=[TARGET_COL] + BASE_FEATURE_COLS)
        df = df[df[TARGET_COL].between(0, 3600)]
        df = df[~df["route_id"].astype(str).isin(prod._BAD_ROUTE_IDS)]
        out.append(df.reset_index(drop=True))
    tr, te = out
    priors = prod._compute_route_hour_priors(tr)
    tr = apply_priors(tr, priors)
    te = apply_priors(te, priors)
    return tr, te, priors


def fit_hgbt(X, y, w, cat_cols, **overrides):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder
    num = [c for c in X.columns if c not in cat_cols]
    pre = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
         ("num", "passthrough", num)], remainder="drop")
    params = dict(loss="absolute_error", max_iter=1200, learning_rate=0.05,
                  max_leaf_nodes=127, min_samples_leaf=20, early_stopping=True,
                  validation_fraction=0.1, n_iter_no_change=50, random_state=42)
    params.update(overrides)
    pipe = Pipeline([("prep", pre), ("model", HistGradientBoostingRegressor(**params))])
    pipe.fit(X, y, model__sample_weight=w)
    return pipe


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--days", required=True)
    p.add_argument("--keep-pct", type=float, default=10)
    r = sub.add_parser("run")
    r.add_argument("--arm", required=True)
    r.add_argument("--train", required=True)
    r.add_argument("--test", required=True)
    r.add_argument("--keep-pct", type=float, default=10)
    r.add_argument("--test-keep-pct", type=float, default=10)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--tag", default="")
    args = ap.parse_args()

    if args.cmd == "prep":
        from src.snapshots import _make_client
        c = _make_client()
        for d in expand_days(args.days):
            if (TRAIN_DIR / f"{d}.parquet").exists():
                prep_day(d, args.keep_pct, client=c)
            else:
                print(f"  {d}: no training parquet, skipped")
        return

    import arms  # noqa: E402  (research/model-search/arms.py)
    train_days, test_days = expand_days(args.train), expand_days(args.test)
    t0 = time.monotonic()
    tr, te, priors = prepare_split(train_days, test_days, args.keep_pct, args.test_keep_pct)
    print(f"train {len(tr):,} rows ({train_days[0]}..{train_days[-1]}), "
          f"test {len(te):,} rows ({test_days[0]}..{test_days[-1]})")
    arm = arms.ARMS[args.arm]
    pred, info = arm(tr, te, seed=args.seed)
    m = metrics(te, pred)
    m.update(arm=args.arm, info=info, train_days=[train_days[0], train_days[-1]],
             test_days=test_days, keep_pct=args.keep_pct, seed=args.seed,
             n_train=int(len(tr)), secs=round(time.monotonic() - t0))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{args.arm}{'_' + args.tag if args.tag else ''}_s{args.seed}"
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps(m, indent=1))
    base_path = RESULTS_DIR / f"baseline{'_' + args.tag if args.tag else ''}_s{args.seed}.json"
    print(json.dumps({k: m[k] for k in ("n", "mae", "median_ae", "p90_ae", "bias")}, indent=1))
    print("by stops_ahead:", {k: round(v, 1) for k, v in m["mae_by_stops_ahead"].items()})
    print("worst routes:", m["worst_routes"])
    if args.arm != "baseline" and base_path.exists():
        print("vs baseline:", judge(json.loads(base_path.read_text()),
                                    json.loads(json.dumps(m))))


if __name__ == "__main__":
    main()
