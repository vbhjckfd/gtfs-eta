"""
Train a sklearn HistGradientBoostingRegressor to predict seconds_to_arrival
(snapshot → target stop, direct multi-horizon).

Uses a Pipeline (OrdinalEncoder for route_id + HistGBT) so a single joblib file
contains everything needed for inference — no separate encoder step.

Baseline: sched_remaining_sec (the schedule's own prediction for the remaining
distance from the vehicle's position to the target stop).
"""

from __future__ import annotations

import os
from pathlib import Path

import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.features import (
    BASE_FEATURE_COLS, FEATURE_COLS, PRIOR_FEATURE_COLS, TARGET_COL,
    apply_priors, compute_features_for_training,
)

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODEL_DIR / "eta_pipeline.joblib"
PRIORS_PATH = MODEL_DIR / "route_hour_priors.joblib"
UNCERTAINTY_PATH = MODEL_DIR / "uncertainty.joblib"

# A horizon needs at least this many held-out rows before its error estimate is
# trustworthy enough to publish as an uncertainty band.
_MIN_HORIZON_SUPPORT = 200

TEST_FRACTION = 0.2

# Snapshot-anchored labeling yields ~3M rows/day — far more than HistGBT needs.
# Cap the loaded dataset with uniform per-file sampling to keep memory and fit
# time sane; override via GTFS_ETA_MAX_ROWS.
MAX_TRAINING_ROWS = int(os.environ.get("GTFS_ETA_MAX_ROWS", 8_000_000))

# Routes with confirmed trip-matching failures (issue #2): their training rows
# have bad anchors (vehicle matched to wrong trip/direction) that poison the
# model. Exclude until the pipeline-level trip-matching filter is tightened.
_BAD_ROUTE_IDS: frozenset[str] = frozenset({"2299", "138"})

_CAT_COLS = ["route_id"]
_NUM_COLS = [c for c in FEATURE_COLS if c not in _CAT_COLS]


def _build_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """Up-weight underrepresented hours to address time-of-day bias (issue #2).

    Hours are UTC.  Lviv is UTC+2/+3, so:
      UTC 14-16 ≈ local 17-19 (evening peak, congestion underestimated)
      UTC 18-20 ≈ local 21-23 (late night, very few training rows)
    """
    w = np.ones(len(df), dtype=float)
    hour = df["hour"].to_numpy()
    w[(hour >= 14) & (hour <= 16)] = 1.5   # evening peak
    w[(hour >= 18) | (hour <= 3)] = 2.0    # late night / early morning
    return w


def _compute_route_hour_priors(train_df: pd.DataFrame) -> dict:
    """Per-(route_id, hour) median observed speed and travel-time-per-stop.

    travel-time-per-stop = seconds_to_arrival / stops_ahead — includes dwell,
    signal delays, and speed variation; the HistGBT uses it to correct the
    long-horizon optimism that pure-speed extrapolation creates.

    Only rows with known speed are used for the speed median; all rows are used
    for the per-stop time since low/unknown speed still has a valid target.
    """
    known_speed = train_df[train_df["progress_speed_mps"] > 0].copy()
    all_rows = train_df.copy()
    all_rows["_tps"] = all_rows[TARGET_COL] / all_rows["stops_ahead"].clip(lower=1)

    speed_by_rh = (
        known_speed.groupby(["route_id", "hour"])["progress_speed_mps"]
        .median()
        .rename("speed")
    )
    # Median per-stop dwell.  The p65 bump tried in 7e3abdd paired with the
    # (stops_ahead-1) cut and together overshot into systematic optimism
    # (issue #5); reverted to the median that calibrated near-zero in f5ed3f2.
    tps_by_rh = (
        all_rows.groupby(["route_id", "hour"])["_tps"]
        .median()
        .rename("tps")
    )
    by_rh = pd.concat([speed_by_rh, tps_by_rh], axis=1).reset_index()

    g_speed = float(known_speed["progress_speed_mps"].median())
    g_tps   = float(all_rows["_tps"].median())

    lookup = {
        (str(row["route_id"]), int(row["hour"])): (
            float(row["speed"]) if pd.notna(row.get("speed")) else g_speed,
            float(row["tps"])   if pd.notna(row.get("tps"))   else g_tps,
        )
        for _, row in by_rh.iterrows()
    }
    print(f"  Priors: {len(lookup)} route×hour entries  "
          f"(global: speed={g_speed:.1f} m/s, tps={g_tps:.0f}s)")
    return {"lookup": lookup, "global_speed": g_speed, "global_tps": g_tps}


def _compute_uncertainty(
    test_df: pd.DataFrame, y_test: np.ndarray, y_pred: np.ndarray
) -> dict:
    """Per-horizon ± band (seconds) for the served predictions.

    Uses held-out absolute error grouped by ``stops_ahead`` — the model is
    trained with absolute_error loss, so its mean-absolute error is the natural,
    self-consistent confidence band to publish as GTFS-RT
    StopTimeEvent.uncertainty. Horizons with too few test rows are dropped;
    inference reuses the widest measured band beyond the largest key.
    """
    resid = np.abs(np.asarray(y_test, dtype=float) - np.asarray(y_pred, dtype=float))
    work = pd.DataFrame(
        {"stops_ahead": test_df["stops_ahead"].to_numpy(), "resid": resid}
    )
    table: dict[int, int] = {}
    for horizon, sub in work.groupby("stops_ahead"):
        if len(sub) >= _MIN_HORIZON_SUPPORT:
            table[int(horizon)] = int(round(float(sub["resid"].mean())))
    if table:
        bands = ", ".join(f"{h}:{s}s" for h, s in sorted(table.items()))
        print(f"  Uncertainty (per-horizon MAE): {bands}")
    else:
        print("  Uncertainty: insufficient test rows per horizon — none saved")
    return table


def _build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                _CAT_COLS,
            ),
            ("num", "passthrough", _NUM_COLS),
        ],
        remainder="drop",
    )
    hgbt = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=500,
        learning_rate=0.05,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
        random_state=42,
        verbose=0,
    )
    return Pipeline([("prep", pre), ("model", hgbt)])


def _load_features(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _load_training_dir(directory: Path, max_rows: int = MAX_TRAINING_ROWS) -> pd.DataFrame:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet files found in {directory}")

    import pyarrow.parquet as pq
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    frac = min(1.0, max_rows / max(total, 1))
    print(f"  Loading {len(files)} training parquets "
          f"({total:,} rows, sampling {frac:.0%})…")

    pieces = []
    for f in files:
        piece = pd.read_parquet(f)
        if frac < 1.0:
            piece = piece.sample(frac=frac, random_state=42)
        pieces.append(piece)
    df = pd.concat(pieces, ignore_index=True)
    print(f"  {len(df):,} training rows across {df['date'].nunique()} dates")
    return df


def _time_split(df: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df["date"] = pd.to_datetime(df["date"])
    dates = sorted(df["date"].unique())
    cutoff_idx = int(len(dates) * (1 - test_fraction))
    cutoff = dates[cutoff_idx]
    train_df = df[df["date"] < cutoff].copy()
    test_df  = df[df["date"] >= cutoff].copy()
    print(f"  Train: {len(train_df):,} rows  ({dates[0].date()} → {dates[cutoff_idx - 1].date()})")
    print(f"  Test:  {len(test_df):,} rows  ({cutoff.date()} → {dates[-1].date()})")
    return train_df, test_df


def train(
    input_path: str | Path,
    model_path: str | Path = MODEL_PATH,
    test_fraction: float = TEST_FRACTION,
) -> dict:
    t0 = time.monotonic()
    input_path = Path(input_path)

    if input_path.is_dir():
        from src.gtfs_static import get_gtfs
        print("Loading GTFS static…")
        gtfs = get_gtfs()
        print("Building feature matrix from training parquets…")
        t_feat = time.monotonic()
        training_rows = _load_training_dir(input_path)
        df = compute_features_for_training(training_rows, gtfs)
        print(f"  Feature matrix: {len(df):,} rows × {len(FEATURE_COLS)} features  ({time.monotonic() - t_feat:.1f}s)")
    else:
        print(f"Loading pre-built features from {input_path}…")
        t_feat = time.monotonic()
        df = _load_features(input_path)
        print(f"  Done  ({time.monotonic() - t_feat:.1f}s)")

    # Sanity filter on base features only — prior-derived features don't exist yet.
    df = df.dropna(subset=[TARGET_COL] + BASE_FEATURE_COLS).copy()
    df = df[df[TARGET_COL].between(0, 3600)].copy()
    print(f"  After sanity filter: {len(df):,} rows")

    n_before = len(df)
    df = df[~df["route_id"].astype(str).isin(_BAD_ROUTE_IDS)].copy()
    if len(df) < n_before:
        print(f"  Excluded {n_before - len(df):,} rows from known-bad routes {set(_BAD_ROUTE_IDS)}")

    train_df, test_df = _time_split(df, test_fraction)

    # Compute route+hour priors from training split only, then enrich both splits.
    print("Computing route+hour priors…")
    priors = _compute_route_hour_priors(train_df)
    train_df = apply_priors(train_df, priors)
    test_df  = apply_priors(test_df,  priors)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(priors, PRIORS_PATH)
    print(f"  Priors saved → {PRIORS_PATH}")

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL].astype(float)
    X_test  = test_df[FEATURE_COLS]
    y_test  = test_df[TARGET_COL].astype(float)

    print("\nFitting HistGradientBoostingRegressor…")
    t_fit = time.monotonic()
    pipeline = _build_pipeline()
    sample_weights = _build_sample_weights(train_df)
    pipeline.fit(X_train, y_train, model__sample_weight=sample_weights)
    fit_sec = time.monotonic() - t_fit

    n_iters = pipeline.named_steps["model"].n_iter_
    print(f"  Stopped at iteration {n_iters}  ({fit_sec:.1f}s)")
    print(f"  Stopped at iteration {n_iters}")

    y_pred_train = pipeline.predict(X_train)
    y_pred_test  = pipeline.predict(X_test)

    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae  = mean_absolute_error(y_test,  y_pred_test)

    # Per-horizon confidence bands, published as GTFS-RT StopTimeEvent.uncertainty.
    uncertainty = _compute_uncertainty(test_df, y_test.to_numpy(), y_pred_test)
    joblib.dump(uncertainty, UNCERTAINTY_PATH)
    print(f"  Uncertainty saved → {UNCERTAINTY_PATH}")

    # GPS warm-started baseline: remaining_dist / effective_speed (no NaN/-1 sentinel).
    if "speed_eta_warm" in test_df.columns:
        baseline_pred = test_df["speed_eta_warm"].values
    else:
        baseline_pred = np.full(len(y_test), y_train.mean())
    baseline_mae = mean_absolute_error(y_test, baseline_pred)

    if "sched_remaining_sec" in test_df.columns:
        sched_mae = mean_absolute_error(y_test, test_df["sched_remaining_sec"].values)
        print(f"Schedule MAE:  {sched_mae:6.1f}s  (GTFS remaining — reference only)")

    improvement_pct = 100.0 * (baseline_mae - test_mae) / baseline_mae if baseline_mae > 0 else 0.0

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    total_sec = time.monotonic() - t0
    total_str = f"{int(total_sec // 60)}m {int(total_sec % 60)}s" if total_sec >= 60 else f"{total_sec:.1f}s"

    print(f"\n{'─'*50}")
    print(f"Train MAE:    {train_mae:6.1f}s")
    print(f"Test  MAE:    {test_mae:6.1f}s")
    print(f"Baseline MAE: {baseline_mae:6.1f}s  (GPS speed-based ETA)")
    print(f"Improvement:  {improvement_pct:+.1f}% vs baseline")
    print(f"Model saved → {model_path}")
    print(f"Done in {total_str}")
    print(f"{'─'*50}")

    # HistGBT exposes permutation-style importances only after sklearn 1.6
    # Use them when available; silently skip otherwise.
    model_step = pipeline.named_steps["model"]
    raw_imp = getattr(model_step, "feature_importances_", None)
    if raw_imp is not None:
        feature_names = _CAT_COLS + _NUM_COLS
        importances = pd.Series(raw_imp, index=feature_names).sort_values(ascending=False)
        print("\nTop feature importances:")
        for name, imp in importances.head(10).items():
            print(f"  {name:<30} {imp:.4f}")

    return {
        "train_mae": train_mae,
        "test_mae": test_mae,
        "baseline_mae": baseline_mae,
        "improvement_pct": improvement_pct,
        "n_train": len(train_df),
        "n_test": len(test_df),
    }


def load_model(model_path: str | Path = MODEL_PATH):
    return joblib.load(model_path)


def predict(pipeline, X: pd.DataFrame) -> np.ndarray:
    return pipeline.predict(X[FEATURE_COLS])


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.train data/training/")
        print("       python -m src.train data/features.parquet")
        sys.exit(1)
    train(sys.argv[1])
