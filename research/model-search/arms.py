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


_PO = _PATH + _OWN
DROP_FRAC = 0.15


def _cold(d, cols, mask=None):
    d = d.copy()
    m = slice(None) if mask is None else mask
    d.loc[m, cols] = -1.0
    if "live_path_cov" in cols:
        d.loc[m, "live_path_cov"] = 0.0
    return d


def _drop_mask(d, frac, seed):
    key = pd.util.hash_pandas_object(
        d[["vehicle_id", "snapshot_ts"]].astype(str), index=False).to_numpy()
    return ((key + seed) % 10007) < 10007 * frac


def _po_drop(tr, te, seed, cold_test):
    tr, te = _sentinel(tr, te, _PO)
    tr = _cold(tr, _PO, _drop_mask(tr, DROP_FRAC, seed))
    if cold_test:
        te = _cold(te, _PO)
    return _fit_predict(tr, te, FEATURE_COLS + _PO, seed)


@arm
def path_own_s_drop(tr, te, seed=42):
    """path_own_s trained with 15% of snapshots forced to cold-start values
    (live-feature dropout) so a restarted daemon degrades gracefully."""
    return _po_drop(tr, te, seed, cold_test=False)


@arm
def path_own_s_drop_cold(tr, te, seed=42):
    """path_own_s_drop evaluated with every live feature cold."""
    return _po_drop(tr, te, seed, cold_test=True)


@arm
def path_own_s_cold(tr, te, seed=42):
    """path_own_s (no dropout) evaluated cold."""
    tr, te = _sentinel(tr, te, _PO)
    return _fit_predict(tr, _cold(te, _PO), FEATURE_COLS + _PO, seed)


@arm
def path_own_s_linkcold(tr, te, seed=42):
    """path_own_s evaluated with only the link store cold (5 min after a
    restart: own-speed history is back, link store still empty)."""
    tr, te = _sentinel(tr, te, _PO)
    return _fit_predict(tr, _cold(te, _PATH), FEATURE_COLS + _PO, seed)


@arm
def path_own_s_big(tr, te, seed=42):
    """path_own_s with 255 leaves / min 50 per leaf."""
    tr, te = _sentinel(tr, te, _PO)
    return _fit_predict(tr, te, FEATURE_COLS + _PO, seed, max_leaf_nodes=255, min_samples_leaf=50)


# ---- run 3 ----------------------------------------------------------------
_M3 = ["live_path_sec_m3", "live_path_n"]


@arm
def path_own_m3(tr, te, seed=42):
    """path_own_s + median-of-last-3 link times and #links observed."""
    cols = _PO + _M3
    tr, te = _sentinel(tr, te, cols)
    return _fit_predict(tr, te, FEATURE_COLS + cols, seed)


@arm
def path_own_m3only(tr, te, seed=42):
    """path_own_s with live_path_sec replaced by its median-of-3 version."""
    cols = [c if c != "live_path_sec" else "live_path_sec_m3" for c in _PO]
    tr, te = _sentinel(tr, te, cols)
    return _fit_predict(tr, te, FEATURE_COLS + cols, seed)


@arm
def path_own_s_gate(tr, te, seed=42):
    """Served mix: path_own_s where the link store has any coverage, else the
    production (baseline) trees. info records MAE by coverage band."""
    tr2, te2 = _sentinel(tr, te, _PO)
    p_live, info = _fit_predict(tr2, te2, FEATURE_COLS + _PO, seed)
    p_base, _ = _fit_predict(tr, te, FEATURE_COLS, seed)
    y = te[TARGET_COL].to_numpy(dtype=float)
    cov = te2["live_path_cov"].to_numpy()
    bands = {}
    for lo, hi in [(0, 1e-9), (1e-9, 0.5), (0.5, 0.9), (0.9, 1.01)]:
        m = (cov >= lo) & (cov < hi)
        if m.any():
            bands[f"{lo:g}-{hi:g}"] = dict(n=int(m.sum()),
                                          live=float(np.abs(p_live[m] - y[m]).mean()),
                                          base=float(np.abs(p_base[m] - y[m]).mean()))
    info["cov_bands"] = bands
    return np.where(cov > 0, p_live, p_base), info


@arm
def path_own_s_nw(tr, te, seed=42):
    """path_own_s without the hour sample weights."""
    tr, te = _sentinel(tr, te, _PO)
    return _fit_predict(tr, te, FEATURE_COLS + _PO, seed, weights=np.ones(len(tr)))


@arm
def path_own_s_nocal(tr, te, seed=42):
    """path_own_s minus month / day_of_week / stop_sequence."""
    cols = [c for c in FEATURE_COLS if c not in ("month", "day_of_week", "stop_sequence")] + _PO
    tr, te = _sentinel(tr, te, _PO)
    return _fit_predict(tr, te, cols, seed)


@arm
def path_own_m35(tr, te, seed=42):
    """path_own_m3 + median-of-5 link times (needs features v2)."""
    cols = _PO + _M3 + ["live_path_sec_m5"]
    tr, te = _sentinel(tr, te, cols)
    return _fit_predict(tr, te, FEATURE_COLS + cols, seed)


@arm
def path_own_m3_big(tr, te, seed=42):
    """path_own_m3 with 255 leaves / min 50 per leaf."""
    cols = _PO + _M3
    tr, te = _sentinel(tr, te, cols)
    return _fit_predict(tr, te, FEATURE_COLS + cols, seed, max_leaf_nodes=255, min_samples_leaf=50)


_NOCAL = [c for c in FEATURE_COLS if c not in ("month", "day_of_week", "stop_sequence")]


@arm
def m3_nocal(tr, te, seed=42):
    """Run-3 stack: path_own with median-of-3 link times, minus calendar cols."""
    cols = [c if c != "live_path_sec" else "live_path_sec_m3" for c in _PO]
    tr, te = _sentinel(tr, te, cols)
    return _fit_predict(tr, te, _NOCAL + cols, seed)
