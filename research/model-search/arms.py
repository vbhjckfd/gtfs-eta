"""Experiment arms. Each takes (train_df, test_df, seed) -> (pred, info)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from harness import fit_hgbt, prod
from src.features import FEATURE_COLS, TARGET_COL

ARMS = {}


def arm(fn):
    ARMS[fn.__name__] = fn
    return fn


def _fit_predict(tr, te, cols, seed, weights=None, target=None, inv=None, **hp):
    y = tr[TARGET_COL].astype(float) if target is None else target(tr)
    w = prod._build_sample_weights(tr) if weights is None else weights
    pipe = fit_hgbt(tr[cols], y, w, ["route_id"], random_state=seed, **hp)
    pred = pipe.predict(te[cols])
    if inv is not None:
        pred = inv(te, pred)
    return pred, {"n_iter": int(pipe.named_steps["model"].n_iter_), "cols": cols, "hp": hp}


@arm
def baseline(tr, te, seed=42):
    """Production recipe (src/train.py) on the harness split."""
    return _fit_predict(tr, te, FEATURE_COLS, seed)


from features_live import LIVE_COLS  # noqa: E402

_OWN = ["own_speed_60", "own_speed_180", "own_speed_300"]
_PATH = ["live_path_sec", "live_path_cov", "live_path_age", "live_speed_mps"]
_HEAD = ["since_last_at_target"]


@arm
def drop_calendar(tr, te, seed=42):
    """Drop month / day_of_week / stop_sequence (weakly-identified IDs)."""
    cols = [c for c in FEATURE_COLS if c not in ("month", "day_of_week", "stop_sequence")]
    return _fit_predict(tr, te, cols, seed)


@arm
def no_weights(tr, te, seed=42):
    return _fit_predict(tr, te, FEATURE_COLS, seed, weights=np.ones(len(tr)))


@arm
def own_speed(tr, te, seed=42):
    return _fit_predict(tr, te, FEATURE_COLS + _OWN, seed)


@arm
def live_path(tr, te, seed=42):
    return _fit_predict(tr, te, FEATURE_COLS + _PATH, seed)


@arm
def headway(tr, te, seed=42):
    return _fit_predict(tr, te, FEATURE_COLS + _HEAD, seed)


@arm
def live_all(tr, te, seed=42):
    return _fit_predict(tr, te, FEATURE_COLS + LIVE_COLS, seed)


@arm
def log_target(tr, te, seed=42):
    return _fit_predict(tr, te, FEATURE_COLS, seed,
                        target=lambda d: np.log1p(d[TARGET_COL].astype(float)),
                        inv=lambda d, p: np.expm1(p))


@arm
def big_leaves(tr, te, seed=42):
    return _fit_predict(tr, te, FEATURE_COLS, seed, max_leaf_nodes=255, min_samples_leaf=50)


@arm
def live_all_s(tr, te, seed=42):
    """live_all with NaN -> -1 sentinel: exported trees (src/inference.py)
    carry no missing-value direction, so NaN must never reach them."""
    tr, te = tr.copy(), te.copy()
    for d in (tr, te):
        d[LIVE_COLS] = d[LIVE_COLS].fillna(-1.0)
    return _fit_predict(tr, te, FEATURE_COLS + LIVE_COLS, seed)


def _sentinel(tr, te, cols):
    tr, te = tr.copy(), te.copy()
    for d in (tr, te):
        d[cols] = d[cols].fillna(-1.0)
    return tr, te


@arm
def path_s(tr, te, seed=42):
    """Ablation: only the live-path block (no own_speed, no headway)."""
    tr, te = _sentinel(tr, te, _PATH)
    return _fit_predict(tr, te, FEATURE_COLS + _PATH, seed)


@arm
def path_min_s(tr, te, seed=42):
    """Ablation: minimal serving set — live_path_sec + live_path_cov."""
    cols = ["live_path_sec", "live_path_cov"]
    tr, te = _sentinel(tr, te, cols)
    return _fit_predict(tr, te, FEATURE_COLS + cols, seed)


@arm
def path_own_s(tr, te, seed=42):
    """Ablation: live-path + own speed (drop headway store)."""
    tr, te = _sentinel(tr, te, _PATH + _OWN)
    return _fit_predict(tr, te, FEATURE_COLS + _PATH + _OWN, seed)


@arm
def live_all_s_lr10(tr, te, seed=42):
    """live_all_s with learning_rate 0.1 (n_iter cap of 1200 is binding at 0.05)."""
    tr, te = _sentinel(tr, te, LIVE_COLS)
    return _fit_predict(tr, te, FEATURE_COLS + LIVE_COLS, seed, learning_rate=0.1)


@arm
def live_all_s_cold(tr, te, seed=42):
    """Serving-safety check: train as live_all_s, but test with every live
    feature at its cold-start value (daemon just restarted: no link store,
    no position history) — must not be much worse than baseline."""
    tr, te = _sentinel(tr, te, LIVE_COLS)
    te = te.copy()
    te[LIVE_COLS] = -1.0
    te["live_path_cov"] = 0.0
    return _fit_predict(tr, te, FEATURE_COLS + LIVE_COLS, seed)
