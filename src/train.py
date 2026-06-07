"""
Train a sklearn HistGradientBoostingRegressor to predict seconds_to_target_stop.

Uses a Pipeline (OrdinalEncoder for route_id + HistGBT) so a single joblib file
contains everything needed for inference — no separate encoder step.

Baseline: scheduled_segment_sec (schedule's own prediction for each segment).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.features import FEATURE_COLS, TARGET_COL, compute_features_for_training

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODEL_DIR / "eta_pipeline.joblib"

TEST_FRACTION = 0.2

_CAT_COLS = ["route_id"]
_NUM_COLS = [c for c in FEATURE_COLS if c not in _CAT_COLS]


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


def _load_labeled_dir(directory: Path) -> pd.DataFrame:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet files found in {directory}")
    print(f"  Loading {len(files)} labeled parquets…")
    pieces = [pd.read_parquet(f) for f in files]
    df = pd.concat(pieces, ignore_index=True)
    print(f"  {len(df):,} labeled rows across {df['date'].nunique()} dates")
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
    input_path = Path(input_path)

    if input_path.is_dir():
        from src.gtfs_static import get_gtfs
        print("Loading GTFS static…")
        gtfs = get_gtfs()
        print("Building feature matrix from labeled parquets…")
        labeled = _load_labeled_dir(input_path)
        df = compute_features_for_training(labeled, gtfs)
        print(f"  Feature matrix: {len(df):,} rows × {len(FEATURE_COLS)} features")
    else:
        print(f"Loading pre-built features from {input_path}…")
        df = _load_features(input_path)

    df = df.dropna(subset=[TARGET_COL] + FEATURE_COLS).copy()
    df = df[df[TARGET_COL].between(0, 3600)].copy()
    print(f"  After sanity filter: {len(df):,} rows")

    train_df, test_df = _time_split(df, test_fraction)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL].astype(float)
    X_test  = test_df[FEATURE_COLS]
    y_test  = test_df[TARGET_COL].astype(float)

    print("\nFitting HistGradientBoostingRegressor…")
    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)

    n_iters = pipeline.named_steps["model"].n_iter_
    print(f"  Stopped at iteration {n_iters}")

    y_pred_train = pipeline.predict(X_train)
    y_pred_test  = pipeline.predict(X_test)

    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae  = mean_absolute_error(y_test,  y_pred_test)

    if "scheduled_segment_sec" in test_df.columns:
        baseline_pred = test_df["scheduled_segment_sec"].values
    else:
        baseline_pred = np.full(len(y_test), y_train.mean())
    baseline_mae = mean_absolute_error(y_test, baseline_pred)

    improvement_pct = 100.0 * (baseline_mae - test_mae) / baseline_mae if baseline_mae > 0 else 0.0

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    print(f"\n{'─'*50}")
    print(f"Train MAE:    {train_mae:6.1f}s")
    print(f"Test  MAE:    {test_mae:6.1f}s")
    print(f"Baseline MAE: {baseline_mae:6.1f}s  (schedule + propagated delay)")
    print(f"Improvement:  {improvement_pct:+.1f}% vs baseline")
    print(f"Model saved → {model_path}")
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
        print("Usage: python -m src.train data/labeled/")
        print("       python -m src.train data/features.parquet")
        sys.exit(1)
    train(sys.argv[1])
