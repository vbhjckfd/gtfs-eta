"""
Train the ``live_v2`` ETA model: the production features plus the live
network / per-vehicle state of src/live_features.py.

Input is what scripts/run_pipeline.py writes per day under data/training/:
the snapshot-anchored rows (<day>.parquet) and two full-day side tables
(side/<day>.feed.parquet, side/<day>.pos.parquet). For each day this

  1. derives the day's link traversals and stationary runs by replaying its
     side tables through LiveState (cached as side/<day>.links|runs.parquet);
  2. builds that day's static tables (historical link medians, dwell) from
     the 14 days before it — strictly prior days, as at serving time where
     the export builds them from the days before it goes live;
  3. replays the day again, answering every sampled training snapshot with
     the state the daemon would have had at that moment (cached under
     data/features_live/).

The fit is the research recipe (``sc_big_fo_long``, branch
claude/model-search): production base features minus month / day_of_week /
stop_sequence, route and target stop (top 254) as native categoricals, the
36 live columns with -1 for missing, 255-leaf trees, max_features 0.3.

Usage:
    python -m src.train_live                       # latest 21 days, last 20% held out
    python -m src.train_live --days-back 28
    python -m src.train_live --train 2026-09-19..2026-09-30 --test 2026-10-01..2026-10-03 \\
        --compare-baseline                         # protocol check vs the legacy recipe
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import train as legacy
from src.features import BASE_FEATURE_COLS, TARGET_COL, apply_priors, compute_features_for_training
from src.live_features import (
    HIST_DAYS, LIVE_COLS, LiveTables, derived_paths, geom_from_gtfs, replay_day,
    side_paths, stationary_runs,
)

TRAIN_DIR = Path(os.environ.get("GTFS_ETA_TRAIN_DIR", "data/training"))
FEAT_DIR = Path(os.environ.get("GTFS_ETA_LIVE_FEAT_DIR", "data/features_live"))
MODEL_DIR = legacy.MODEL_DIR
MODEL_PATH = MODEL_DIR / "eta_live.joblib"
TABLES_PATH = MODEL_DIR / "live_tables.joblib"

FEATURE_SET = "live_v2"

# Model column order — the exported trees index features positionally, and
# src/inference.build_features_live assembles rows in exactly this order.
# Native categoricals first (HistGradientBoosting puts them first anyway).
CAT_COLS = ["route_code", "stop_cat"]
BASE_COLS = [
    "stops_ahead", "hour", "is_weekend", "is_holiday", "remaining_dist_m",
    "progress_speed_mps", "stops_remaining", "trip_progress_frac", "dist_per_stop_m",
    "speed_eta_warm", "hist_speed_mps", "hist_travel_time_est", "stationary_sec",
]
MODEL_COLS = CAT_COLS + BASE_COLS + LIVE_COLS
N_STOP_CODES = 254          # top target stops get their own category, the rest share 254

DEFAULT_DAYS_BACK = 21
# Share of the snapshots in each day's parquet that become training rows. The
# pipeline already keeps a sample of snapshots (run_pipeline --keep-pct), so
# the share of *all* snapshots is the product of the two.
DEFAULT_SNAP_FRAC = 0.3
TEST_SNAP_FRAC = 0.3

_PASSTHROUGH = ("vehicle_id", "trip_id", "stop_id", "dist_along_m", "stop_dist_along_m", "snapshot_ts")

HGBT_PARAMS = dict(
    loss="absolute_error",
    max_iter=1200,
    learning_rate=0.05,
    max_leaf_nodes=255,
    min_samples_leaf=50,
    max_features=0.3,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=50,
)


# ---------------------------------------------------------------------------
# Days
# ---------------------------------------------------------------------------

def expand_days(spec: str) -> list[str]:
    out = []
    for part in spec.split(","):
        if ".." in part:
            a, b = part.split("..")
            d, e = date.fromisoformat(a), date.fromisoformat(b)
            while d <= e:
                out.append(d.isoformat())
                d += timedelta(days=1)
        elif part:
            out.append(part)
    return out


def ready_days(train_dir: Path = TRAIN_DIR) -> list[str]:
    """Days with training rows and both side tables, sorted."""
    days = []
    for p in sorted(train_dir.glob("*.parquet")):
        day = p.stem
        if all(s.exists() for s in side_paths(train_dir, day)):
            days.append(day)
    return days


def _prior_days(day: str, n: int = HIST_DAYS) -> list[str]:
    d0 = date.fromisoformat(day)
    return [(d0 - timedelta(days=k)).isoformat() for k in range(n, 0, -1)]


# ---------------------------------------------------------------------------
# Per-day derived tables and features (cached)
# ---------------------------------------------------------------------------

_gtfs_cache: dict = {}


def _gtfs(day: str):
    if day not in _gtfs_cache:
        from src.gtfs_static import get_gtfs_for_date
        _gtfs_cache.clear()
        _gtfs_cache[day] = get_gtfs_for_date(day)
    return _gtfs_cache[day]


def ensure_derived(day: str, train_dir: Path = TRAIN_DIR) -> bool:
    """Link traversals + stationary runs for *day*; False when its side tables are missing."""
    links_p, runs_p = derived_paths(train_dir, day)
    if links_p.exists() and runs_p.exists():
        return True
    if not all(s.exists() for s in side_paths(train_dir, day)):
        return False
    gtfs = _gtfs(day)
    geom = geom_from_gtfs(gtfs)
    _, links = replay_day(train_dir, day, geom, None, queries=None, collect_links=True)
    pos = pd.read_parquet(side_paths(train_dir, day)[1])
    runs = stationary_runs(pos, geom)
    links.to_parquet(links_p, index=False)
    runs.to_parquet(runs_p, index=False)
    return True


def tables_for(day: str, train_dir: Path = TRAIN_DIR, inclusive: bool = False) -> LiveTables:
    """Static tables from the HIST_DAYS days before *day* (and *day* itself if inclusive)."""
    days = _prior_days(day) + ([day] if inclusive else [])
    links, runs = [], []
    for d in days:
        lp, rp = derived_paths(train_dir, d)
        if lp.exists() and rp.exists():
            links.append(pd.read_parquet(lp, columns=["key", "t", "lt"]))
            runs.append(pd.read_parquet(rp))
    return LiveTables.build(links, runs)


def _snap_mask(rows: pd.DataFrame, frac: float) -> np.ndarray:
    """Keep *frac* of the snapshots (whole snapshots), hash-based and stable.

    A different modulus from labeling.snapshot_sample_mask, so this sub-sample
    is independent of the pipeline's own sample.
    """
    if frac >= 1.0:
        return np.ones(len(rows), dtype=bool)
    key = pd.util.hash_pandas_object(
        rows[["vehicle_id", "snapshot_ts"]].astype(str), index=False).to_numpy()
    return (key % 7919) < 7919 * frac


def feature_path(day: str, frac: float) -> Path:
    return FEAT_DIR / f"{day}_f{frac:g}.parquet"


def day_features(day: str, frac: float, train_dir: Path = TRAIN_DIR) -> Path:
    """Base + live features for a sample of *day*'s snapshots (cached parquet)."""
    out = feature_path(day, frac)
    if out.exists():
        return out
    for d in _prior_days(day):
        ensure_derived(d, train_dir)
    ensure_derived(day, train_dir)
    rows = pd.read_parquet(train_dir / f"{day}.parquet")
    rows = rows[_snap_mask(rows, frac)].reset_index(drop=True)
    gtfs = _gtfs(day)
    feats = compute_features_for_training(rows, gtfs, passthrough=_PASSTHROUGH)
    feats["route_id"] = feats["route_id"].astype(str)
    feats["stop_id"] = feats["stop_id"].astype(str)
    snap = pd.to_datetime(feats["snapshot_ts"], utc=True)
    feats["t"] = (snap - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()
    tables = tables_for(day, train_dir)
    live, _ = replay_day(train_dir, day, geom_from_gtfs(gtfs), tables, queries=feats)
    for i, c in enumerate(LIVE_COLS):
        feats[c] = live[:, i]
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    feats.to_parquet(tmp, index=False)
    tmp.rename(out)
    return out


def _day_job(args) -> tuple[str, str]:
    day, frac = args
    t0 = time.monotonic()
    try:
        day_features(day, frac)
    except Exception as exc:  # noqa: BLE001 — report and keep the other days going
        return day, f"error {exc!r}"
    return day, f"ok {time.monotonic() - t0:.0f}s"


def build_features(days: list[str], frac: float, parallel: int = 1) -> None:
    todo = [d for d in days if not feature_path(d, frac).exists()]
    if not todo:
        return
    print(f"Building live features for {len(todo)} day(s) at {frac:g} of each day's snapshots…", flush=True)
    # Derived tables first, in order: every feature day reads the 14 before it.
    need = sorted({p for d in todo for p in _prior_days(d) + [d]})
    _run(need, lambda d: ensure_derived(d), parallel, "derived")
    if parallel > 1 and len(todo) > 1:
        with ProcessPoolExecutor(max_workers=parallel, max_tasks_per_child=1) as ex:
            for day, status in ex.map(_day_job, [(d, frac) for d in todo]):
                print(f"  {day}: {status}", flush=True)
    else:
        for d in todo:
            print(f"  {_day_job((d, frac))}", flush=True)


def _derived_job(day: str) -> tuple[str, bool]:
    return day, ensure_derived(day)


def _run(days, fn, parallel, label):
    todo = [d for d in days if not all(p.exists() for p in derived_paths(TRAIN_DIR, d))
            and all(p.exists() for p in side_paths(TRAIN_DIR, d))]
    if not todo:
        return
    print(f"  {label}: {len(todo)} day(s)", flush=True)
    if parallel > 1 and len(todo) > 1:
        with ProcessPoolExecutor(max_workers=parallel, max_tasks_per_child=1) as ex:
            for day, ok in ex.map(_derived_job, todo):
                print(f"    {day}: {'ok' if ok else 'missing side tables'}", flush=True)
    else:
        for d in todo:
            print(f"    {d}: {'ok' if fn(d) else 'missing side tables'}", flush=True)


# Cap on training rows. 17 days at the default sample is ~9M rows, and the
# fit plus the priors' frame copies then overran a 15 GB box; the research
# measured only ~2.5% from tripling 2.2M rows, so the cap costs little.
MAX_TRAIN_ROWS = int(os.environ.get("GTFS_ETA_LIVE_MAX_ROWS", 5_000_000))


def load_days(days: list[str], frac: float, max_rows: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Feature rows of *days*: only the columns the fit and the metrics use,
    numerics as float32 (a full frame of 9M rows is ~4x larger)."""
    num = sorted(set(BASE_FEATURE_COLS + BASE_COLS + LIVE_COLS + [TARGET_COL])
                 - {"route_id", "speed_eta_warm", "hist_speed_mps", "hist_travel_time_est"})
    cols = ["route_id", "stop_id", "date"] + num
    pieces = []
    for d in days:
        piece = pd.read_parquet(feature_path(d, frac), columns=cols)
        piece[num] = piece[num].astype(np.float32)
        pieces.append(piece)
    df = pd.concat(pieces, ignore_index=True)
    del pieces
    df = df.dropna(subset=[TARGET_COL] + [c for c in BASE_FEATURE_COLS if c in df.columns])
    df = df[df[TARGET_COL].between(0, 3600)]
    df = df[~df["route_id"].astype(str).isin(legacy._BAD_ROUTE_IDS)]
    if max_rows and len(df) > max_rows:
        print(f"  sampling {max_rows:,} of {len(df):,} rows (GTFS_ETA_LIVE_MAX_ROWS)", flush=True)
        df = df.sample(n=max_rows, random_state=seed)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def encode(df: pd.DataFrame, route_codes: dict, stop_codes: dict) -> np.ndarray:
    """Feature matrix in MODEL_COLS order (df must already carry the priors)."""
    X = np.empty((len(df), len(MODEL_COLS)), dtype=np.float64)
    X[:, 0] = df["route_id"].astype(str).map(route_codes).fillna(-1).to_numpy(dtype=float)
    X[:, 1] = df["stop_id"].astype(str).map(stop_codes).fillna(N_STOP_CODES).to_numpy(dtype=float)
    for i, c in enumerate(BASE_COLS + LIVE_COLS, start=2):
        X[:, i] = df[c].to_numpy(dtype=float)
    return X


def fit(train_df: pd.DataFrame, seed: int = 42) -> dict:
    """Fit on *train_df* (raw feature rows); returns the model bundle."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    priors = legacy._compute_route_hour_priors(train_df)
    tr = apply_priors(train_df, priors)
    route_codes = {r: i for i, r in enumerate(sorted(tr["route_id"].astype(str).unique()))}
    top = tr["stop_id"].astype(str).value_counts().index[:N_STOP_CODES]
    stop_codes = {s: i for i, s in enumerate(top)}
    X = encode(tr, route_codes, stop_codes)
    y = tr[TARGET_COL].to_numpy(dtype=float)
    w = legacy._build_sample_weights(tr)
    model = HistGradientBoostingRegressor(categorical_features=[0, 1], random_state=seed, **HGBT_PARAMS)
    t0 = time.monotonic()
    model.fit(X, y, sample_weight=w)
    print(f"  fit: {len(tr):,} rows x {X.shape[1]} features, {model.n_iter_} trees "
          f"({time.monotonic() - t0:.0f}s)", flush=True)
    return {
        "feature_set": FEATURE_SET,
        "model": model,
        "cols": list(MODEL_COLS),
        "route_codes": route_codes,
        "stop_codes": stop_codes,
        "priors": priors,
    }


def predict(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    d = apply_priors(df, bundle["priors"])
    return bundle["model"].predict(encode(d, bundle["route_codes"], bundle["stop_codes"]))


def metrics(df: pd.DataFrame, pred: np.ndarray) -> dict:
    y = df[TARGET_COL].to_numpy(dtype=float)
    err = pred - y
    ae = np.abs(err)
    sa = df["stops_ahead"].to_numpy()
    days = pd.to_datetime(df["date"]).dt.date.astype(str).to_numpy()
    by_route = (pd.DataFrame({"r": df["route_id"].astype(str).to_numpy(), "ae": ae})
                .groupby("r")["ae"].agg(["mean", "size"]))
    by_route = by_route[by_route["size"] >= 500].sort_values("mean", ascending=False)
    return {
        "n": int(len(y)),
        "mae": float(ae.mean()),
        "median_ae": float(np.median(ae)),
        "p90_ae": float(np.percentile(ae, 90)),
        "bias": float(err.mean()),
        "mae_by_stops_ahead": {int(k): float(ae[sa == k].mean()) for k in range(1, 11) if (sa == k).any()},
        "mae_by_day": {d: float(ae[days == d].mean()) for d in sorted(set(days))},
        "n_by_day": {d: int((days == d).sum()) for d in sorted(set(days))},
        "worst_routes": {r: round(float(m), 1) for r, m in by_route["mean"].head(10).items()},
    }


def _fmt(m: dict) -> str:
    sa = m["mae_by_stops_ahead"]
    return (f"MAE {m['mae']:.1f}  median {m['median_ae']:.1f}  p90 {m['p90_ae']:.1f}  "
            f"bias {m['bias']:+.1f}  n {m['n']:,}  sa1 {sa.get(1, float('nan')):.1f}  "
            f"sa5 {sa.get(5, float('nan')):.1f}  sa10 {sa.get(10, float('nan')):.1f}")


def _baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int) -> np.ndarray:
    """The legacy production recipe (src/train.py) on the same rows."""
    from src.features import FEATURE_COLS
    priors = legacy._compute_route_hour_priors(train_df)
    tr = apply_priors(train_df, priors)
    te = apply_priors(test_df, priors)
    pipe = legacy._build_pipeline()
    pipe.set_params(model__random_state=seed)
    t0 = time.monotonic()
    pipe.fit(tr[FEATURE_COLS], tr[TARGET_COL].astype(float),
             model__sample_weight=legacy._build_sample_weights(tr))
    print(f"  baseline fit: {pipe.named_steps['model'].n_iter_} trees ({time.monotonic() - t0:.0f}s)", flush=True)
    return pipe.predict(te[FEATURE_COLS])


def uncertainty_table(test_df: pd.DataFrame, pred: np.ndarray) -> dict:
    return legacy._compute_uncertainty(test_df, test_df[TARGET_COL].to_numpy(dtype=float), pred)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--days-back", type=int, default=DEFAULT_DAYS_BACK,
                    help="use the latest N ready days (default %(default)s)")
    ap.add_argument("--train", help="explicit train days, e.g. 2026-09-19..2026-09-30")
    ap.add_argument("--test", help="explicit held-out days (later than --train)")
    ap.add_argument("--snap-frac", type=float, default=DEFAULT_SNAP_FRAC,
                    help="share of each day's pipeline snapshots used for training rows")
    ap.add_argument("--test-snap-frac", type=float, default=TEST_SNAP_FRAC)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compare-baseline", action="store_true",
                    help="also fit the legacy recipe on the same rows and judge the protocol")
    ap.add_argument("--no-save", action="store_true", help="evaluate only, write no model files")
    ap.add_argument("--results", help="append a JSON line with the metrics to this file")
    args = ap.parse_args(argv)

    t_start = time.monotonic()
    if args.train:
        train_days = expand_days(args.train)
        test_days = expand_days(args.test) if args.test else []
    else:
        days = ready_days()[-args.days_back:]
        if len(days) < 5:
            raise SystemExit(f"only {len(days)} ready day(s) under {TRAIN_DIR} — need >= 5")
        n_test = max(1, int(round(len(days) * legacy.TEST_FRACTION)))
        train_days, test_days = days[:-n_test], days[-n_test:]
    print(f"Train days {train_days[0]}..{train_days[-1]} ({len(train_days)}), "
          f"test {test_days[0] + '..' + test_days[-1] if test_days else '-'} ({len(test_days)})", flush=True)

    build_features(train_days, args.snap_frac, args.parallel)
    if test_days:
        build_features(test_days, args.test_snap_frac, args.parallel)

    train_df = load_days(train_days, args.snap_frac, max_rows=MAX_TRAIN_ROWS, seed=args.seed)
    print(f"  train rows {len(train_df):,}", flush=True)
    bundle = fit(train_df, seed=args.seed)
    result: dict = {"train_days": [train_days[0], train_days[-1]], "test_days": test_days,
                    "n_train": int(len(train_df)), "seed": args.seed,
                    "n_iter": int(bundle["model"].n_iter_)}

    unc = None
    if test_days:
        test_df = load_days(test_days, args.test_snap_frac)
        pred = predict(bundle, test_df)
        m = metrics(test_df, pred)
        result["live_v2"] = m
        print(f"live_v2   {_fmt(m)}", flush=True)
        unc = uncertainty_table(test_df, pred)
        if args.compare_baseline:
            bpred = _baseline(train_df, test_df, args.seed)
            b = metrics(test_df, bpred)
            result["baseline"] = b
            print(f"baseline  {_fmt(b)}", flush=True)
            worst = max((m["mae_by_stops_ahead"][k] - b["mae_by_stops_ahead"][k]) / b["mae_by_stops_ahead"][k]
                        for k in b["mae_by_stops_ahead"] if k in m["mae_by_stops_ahead"])
            d_mae = (m["mae"] - b["mae"]) / b["mae"]
            result["judge"] = {
                "mae_change_pct": 100 * d_mae,
                "p90_change_pct": 100 * (m["p90_ae"] - b["p90_ae"]) / b["p90_ae"],
                "worst_bucket_change_pct": 100 * worst,
                "wins": bool(d_mae <= -0.03 and m["p90_ae"] <= b["p90_ae"] and worst <= 0.05),
            }
            print("judge:", json.dumps(result["judge"]), flush=True)
            for k in sorted(m["mae_by_stops_ahead"]):
                print(f"  sa{k:>2}: baseline {b['mae_by_stops_ahead'][k]:6.1f}  live_v2 {m['mae_by_stops_ahead'][k]:6.1f}")
            for d in m["mae_by_day"]:
                print(f"  {d}: baseline {b['mae_by_day'][d]:6.1f}  live_v2 {m['mae_by_day'][d]:6.1f}  n {m['n_by_day'][d]:,}")
            print("  worst routes baseline:", b["worst_routes"])
            print("  worst routes live_v2: ", m["worst_routes"])

    if not args.no_save:
        last = (test_days or train_days)[-1]
        tables = tables_for(last, inclusive=True)
        bundle["trained_through"] = train_days[-1]
        bundle["tables_through"] = last
        bundle["metrics"] = result.get("live_v2")
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, MODEL_PATH)
        joblib.dump(tables.to_dict(), TABLES_PATH)
        # The export reads these two for every model: the route×hour priors the
        # base features were trained with, and the held-out per-horizon bands.
        joblib.dump(bundle["priors"], legacy.PRIORS_PATH)
        if unc:
            joblib.dump(unc, legacy.UNCERTAINTY_PATH)
        print(f"Saved {MODEL_PATH}, {TABLES_PATH} ({len(tables.hist_k):,} links, "
              f"{len(tables.dwell):,} dwell locations), priors, uncertainty", flush=True)

    if args.results:
        with open(args.results, "a") as f:
            f.write(json.dumps(result) + "\n")
    print(f"Done in {(time.monotonic() - t_start) / 60:.1f} min", flush=True)
    return result


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Export format (scripts/export_worker_data.py, src/inference.predict_live)
# ---------------------------------------------------------------------------

def flatten_model(bundle: dict) -> dict:
    """The fitted trees as flat numpy arrays the daemon walks directly.

    Every tree's nodes are concatenated (child indices rebased), so a whole
    ensemble is one set of arrays. Categorical splits keep sklearn's
    semantics (_predictor.pyx): the raw code is first mapped to the encoded
    category the estimator saw (``cat_maps``; unknown -> NaN), NaN follows
    ``missing_left``, otherwise a category in the node's bitset goes left.
    Leaf values already include the learning rate.
    """
    model = bundle["model"]
    f_idx, thr, left, right, is_leaf, value, miss_left, is_cat, bitset, roots = ([] for _ in range(10))
    bits = []
    offset = 0
    for (pred,) in model._predictors:
        nodes = pred.nodes
        roots.append(offset)
        b0 = len(bits)
        if len(pred.raw_left_cat_bitsets):
            bits.extend(np.asarray(pred.raw_left_cat_bitsets, dtype=np.uint32).tolist())
        f_idx.append(np.where(nodes["is_leaf"], 0, nodes["feature_idx"]).astype(np.int32))
        thr.append(nodes["num_threshold"].astype(np.float64))
        left.append((nodes["left"] + offset).astype(np.int32))
        right.append((nodes["right"] + offset).astype(np.int32))
        is_leaf.append(nodes["is_leaf"].astype(bool))
        value.append(nodes["value"].astype(np.float64))
        miss_left.append(nodes["missing_go_to_left"].astype(bool))
        cat = nodes["is_categorical"].astype(bool)
        is_cat.append(cat)
        bitset.append(np.where(cat, nodes["bitset_idx"].astype(np.int64) + b0, 0).astype(np.int32))
        offset += len(nodes)

    enc = model._preprocessor.named_transformers_["encoder"] if model._preprocessor is not None else None
    cat_maps = []
    for cats in (enc.categories_ if enc is not None else []):
        m = np.full(256, np.nan)
        for i, raw in enumerate(cats):
            if raw == raw and 0 <= raw < 256:
                m[int(raw)] = float(i)
        cat_maps.append(m)

    cat = lambda xs, dt: np.concatenate(xs).astype(dt)  # noqa: E731
    return {
        "feature_set": FEATURE_SET,
        "cols": list(bundle["cols"]),
        "route_codes": dict(bundle["route_codes"]),
        "stop_codes": dict(bundle["stop_codes"]),
        "n_stop_codes": N_STOP_CODES,
        "cat_maps": cat_maps,                       # one per leading categorical column
        "baseline": float(model._baseline_prediction.flat[0]),
        "n_trees": len(roots),
        "flat": {
            "f_idx": cat(f_idx, np.int32), "thr": cat(thr, np.float64),
            "left": cat(left, np.int32), "right": cat(right, np.int32),
            "is_leaf": cat(is_leaf, bool), "value": cat(value, np.float64),
            "miss_left": cat(miss_left, bool), "is_cat": cat(is_cat, bool),
            "bitset": cat(bitset, np.int32),
            "bits": np.asarray(bits if bits else np.zeros((1, 8)), dtype=np.uint32).reshape(-1, 8),
            "roots": np.asarray(roots, dtype=np.int32),
        },
    }
