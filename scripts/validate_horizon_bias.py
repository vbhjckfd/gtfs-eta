"""Validate per-stops_ahead ETA bias on the held-out test split.

Reproduces train()'s exact data prep and time-split, loads the freshly trained
model + priors from models/, and reports bias = mean(pred - actual) and MAE per
stops_ahead bucket.  Negative bias = optimism (bus arrives later than predicted),
the failure mode from issue #5.  A healthy model stays near zero across the whole
horizon curve, not just in aggregate.

    python scripts/validate_horizon_bias.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import (
    BASE_FEATURE_COLS, FEATURE_COLS, TARGET_COL, apply_priors,
    compute_features_for_training,
)
from src.train import (
    MODEL_PATH, PRIORS_PATH, TEST_FRACTION, _BAD_ROUTE_IDS,
    _load_training_dir, _time_split,
)


def main() -> None:
    from src.gtfs_static import get_gtfs

    print("Loading GTFS static…")
    gtfs = get_gtfs()
    print("Building feature matrix from training parquets…")
    df = compute_features_for_training(_load_training_dir(Path("data/training/")), gtfs)

    df = df.dropna(subset=[TARGET_COL] + BASE_FEATURE_COLS).copy()
    df = df[df[TARGET_COL].between(0, 3600)].copy()
    df = df[~df["route_id"].astype(str).isin(_BAD_ROUTE_IDS)].copy()

    _, test_df = _time_split(df, TEST_FRACTION)

    priors = joblib.load(PRIORS_PATH)
    pipeline = joblib.load(MODEL_PATH)
    test_df = apply_priors(test_df, priors)

    y_true = test_df[TARGET_COL].astype(float).to_numpy()
    y_pred = pipeline.predict(test_df[FEATURE_COLS])
    err = y_pred - y_true  # negative = optimistic (arrives later than predicted)

    sa = test_df["stops_ahead"].astype(int).to_numpy()
    print(f"\nHeld-out test rows: {len(test_df):,}")
    print(f"Overall bias: {err.mean():+7.1f}s   MAE: {np.abs(err).mean():6.1f}s\n")
    print(f"{'stops_ahead':>11}  {'n':>9}  {'bias':>8}  {'MAE':>7}")
    print("-" * 42)
    worst = 0.0
    for k in range(1, int(sa.max()) + 1):
        m = sa == k
        if not m.any():
            continue
        bias = err[m].mean()
        worst = max(worst, abs(bias))
        print(f"{k:>11}  {m.sum():>9,}  {bias:>+8.1f}  {np.abs(err[m]).mean():>7.1f}")
    print("-" * 42)
    print(f"\nWorst per-horizon |bias|: {worst:.1f}s")


if __name__ == "__main__":
    main()
