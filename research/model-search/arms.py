"""Experiment arms. Each takes (train_df, test_df, seed) -> (pred, info)."""

from __future__ import annotations

import os

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
    if os.environ.get("MS_SAVE_MODEL"):   # run 9: keep the fit for the serving timing test
        import joblib
        joblib.dump((pipe, cols, te[cols].sample(20000, random_state=0)), os.environ["MS_SAVE_MODEL"])
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
    # MS_INPLACE=1 (run 15, 3x fits): fill in place instead of copying the frames;
    # same numbers, ~half the peak memory (the harness does not reuse these cols)
    if not os.environ.get("MS_INPLACE"):
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
def _cold(d, cols, mask=None):
    d = d.copy()
    m = slice(None) if mask is None else mask
    d.loc[m, cols] = -1.0
    if "live_path_cov" in cols:
        d.loc[m, "live_path_cov"] = 0.0
    return d


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


_STACK = [c if c != "live_path_sec" else "live_path_sec_m3" for c in _PO] + ["live_path_sec_m5"]


@arm
def stack(tr, te, seed=42):
    """Run-3 stack: median-of-3 and -of-5 link times + own speed, no calendar cols."""
    tr, te = _sentinel(tr, te, _STACK)
    return _fit_predict(tr, te, _NOCAL + _STACK, seed)


@arm
def stack_big(tr, te, seed=42):
    """stack with 255 leaves / min 50 per leaf."""
    tr, te = _sentinel(tr, te, _STACK)
    return _fit_predict(tr, te, _NOCAL + _STACK, seed, max_leaf_nodes=255, min_samples_leaf=50)


# ---- run 4 (features v3: MS_FEAT_DIR=ms_features_v3) ----------------------
_HIST = ["hist_path_sec", "hist_path_cov"]


def _stack_plus(tr, te, seed, extra, **hp):
    cols = _STACK + extra
    tr, te = _sentinel(tr, te, cols)
    return _fit_predict(tr, te, _NOCAL + cols, seed, **hp)


@arm
def stack_m7(tr, te, seed=42):
    """stack + median-of-7 link times."""
    return _stack_plus(tr, te, seed, ["live_path_sec_m7"])


@arm
def stack_ewm(tr, te, seed=42):
    """stack + EWMA (alpha 0.4 per traversal) link times."""
    return _stack_plus(tr, te, seed, ["live_path_sec_ewm"])


@arm
def stack_hist(tr, te, seed=42):
    """stack + historical link times (prior days, same local hour)."""
    return _stack_plus(tr, te, seed, _HIST)


@arm
def stack_fill(tr, te, seed=42):
    """stack + historical path + live-or-historical filled path sum."""
    return _stack_plus(tr, te, seed, _HIST + ["fill_path_sec"])


@arm
def stack_v3(tr, te, seed=42):
    """stack + every run-4 column."""
    return _stack_plus(tr, te, seed, ["live_path_sec_m7", "live_path_sec_ewm"] + _HIST
                       + ["fill_path_sec"])


@arm
def stack_hist_cold(tr, te, seed=42):
    """stack_hist evaluated with the live store cold (historical table is a
    static artifact, so it stays)."""
    cols = _STACK + _HIST
    tr, te = _sentinel(tr, te, cols)
    return _fit_predict(tr, _cold(te, _STACK), _NOCAL + cols, seed)


@arm
def stack_cold(tr, te, seed=42):
    """stack evaluated with the live store cold (reference for stack_hist_cold)."""
    tr, te = _sentinel(tr, te, _STACK)
    return _fit_predict(tr, _cold(te, _STACK), _NOCAL + _STACK, seed)


# ---- run 5 (features v4: MS_FEAT_DIR=ms_features_v4, history from 08-31) ---

@arm
def stack_hist_dt(tr, te, seed=42):
    """stack_hist + weekday/weekend-specific historical path."""
    return _stack_plus(tr, te, seed, _HIST + ["hist_path_sec_dt"])


@arm
def stack_hist_r(tr, te, seed=42):
    """stack_hist + live/historical ratio on links observed by both."""
    return _stack_plus(tr, te, seed, _HIST + ["live_hist_ratio"])


@arm
def stack_v4(tr, te, seed=42):
    """stack_hist + every run-5 column."""
    return _stack_plus(tr, te, seed, _HIST + ["hist_path_sec_dt", "live_hist_ratio",
                                              "hist_path_p75"])


@arm
def stack_hist_l2(tr, te, seed=42):
    """stack_hist with l2_regularization 1.0."""
    return _stack_plus(tr, te, seed, _HIST, l2_regularization=1.0)


@arm
def stack_hist_leaf100(tr, te, seed=42):
    """stack_hist with min_samples_leaf 100."""
    return _stack_plus(tr, te, seed, _HIST, min_samples_leaf=100)


@arm
def stack_hist_catroute(tr, te, seed=42):
    """stack_hist with route_id as a native categorical (needs categorical-split
    support in the exported trees)."""
    return _stack_plus(tr, te, seed, _HIST, categorical_features=[0])


@arm
def stack_hist_dt_cat(tr, te, seed=42):
    """stack_hist_dt with route_id as a native categorical (run 6)."""
    return _stack_plus(tr, te, seed, _HIST + ["hist_path_sec_dt"], categorical_features=[0])


# ---- run 7 (features v4 + V5 own-vs-historical cols, same MS_FEAT_DIR) ------
_DT = _HIST + ["hist_path_sec_dt"]
_OWNH = ["own_hist_ratio3", "own_hist_ratio8", "own_excess8"]


@arm
def dtcat_own(tr, te, seed=42):
    """leader + own last-3/8 link times vs the historical table."""
    return _stack_plus(tr, te, seed, _DT + _OWNH, categorical_features=[0])


@arm
def dtcat_own_x(tr, te, seed=42):
    """dtcat_own + historical path scaled by the own ratio."""
    return _stack_plus(tr, te, seed, _DT + _OWNH + ["hist_x_own"], categorical_features=[0])


@arm
def dtcat_recency(tr, te, seed=42):
    """leader with training days down-weighted by age (half-life 7 days)."""
    days = pd.to_datetime(tr["date"])
    age = (days.max() - days).dt.days.to_numpy()
    w = prod._build_sample_weights(tr) * 0.5 ** (age / 7.0)
    return _stack_plus(tr, te, seed, _DT, weights=w, categorical_features=[0])


_PRUNE = ("is_holiday", "trip_progress_frac", "stops_remaining")


@arm
def dtcat_prune(tr, te, seed=42):
    """leader minus is_holiday / trip_progress_frac / stops_remaining
    (constant or block-schedule-derived)."""
    cols = _STACK + _DT
    tr, te = _sentinel(tr, te, cols)
    base = [c for c in _NOCAL if c not in _PRUNE]
    return _fit_predict(tr, te, base + cols, seed, categorical_features=[0])


@arm
def dtcat_stopcat(tr, te, seed=42):
    """leader + target stop as a native categorical (254 most frequent train
    stops, the rest pooled)."""
    cols = _STACK + _DT
    tr, te = _sentinel(tr, te, cols)
    top = tr["stop_id"].astype(str).value_counts().index[:254]
    code = {s: i for i, s in enumerate(top)}
    for d in (tr, te):
        d["stop_cat"] = d["stop_id"].astype(str).map(code).fillna(254).astype(float)
    allc = _NOCAL + cols + ["stop_cat"]
    return _fit_predict(tr, te, allc, seed, categorical_features=[0, len(allc) - 1])


def _base_eta(d):
    """Anchor for residual targets: historical day-type path, else live m5, else
    the warm speed ETA (all already features, so free at serving time)."""
    h = d["hist_path_sec_dt"].to_numpy(dtype=float)
    l5 = d["live_path_sec_m5"].to_numpy(dtype=float)
    s = d["speed_eta_warm"].to_numpy(dtype=float)
    b = np.where(h > 0, h, np.where(l5 > 0, l5, s))
    return np.clip(np.nan_to_num(b, nan=300.0), 5.0, 3600.0)


def _resid(tr, te, seed, cols, mode, **hp):
    tr, te = _sentinel(tr, te, _STACK + _DT)
    btr = _base_eta(tr)
    y = tr[TARGET_COL].to_numpy(dtype=float)
    if mode == "diff":
        target = lambda d: y - btr
        inv = lambda d, p: np.clip(p + _base_eta(d), 0, 3600)
    else:
        target = lambda d: np.log((y + 30.0) / (btr + 30.0))
        inv = lambda d, p: np.clip(np.exp(p) * (_base_eta(d) + 30.0) - 30.0, 0, 3600)
    return _fit_predict(tr, te, cols, seed, target=target, inv=inv, categorical_features=[0], **hp)


@arm
def dtcat_resid(tr, te, seed=42):
    """leader, target = y - base ETA (historical path)."""
    return _resid(tr, te, seed, _NOCAL + _STACK + _DT, "diff")


@arm
def dtcat_logratio(tr, te, seed=42):
    """leader, target = log((y+30)/(base+30)). Note: absolute_error on the log
    scale is a median of the ratio, which is still the median of y."""
    return _resid(tr, te, seed, _NOCAL + _STACK + _DT, "log")


@arm
def dtcat_stopcat_big(tr, te, seed=42):
    """dtcat_stopcat with 255 leaves / min 50 per leaf."""
    cols = _STACK + _DT
    tr, te = _sentinel(tr, te, cols)
    top = tr["stop_id"].astype(str).value_counts().index[:254]
    code = {s: i for i, s in enumerate(top)}
    for d in (tr, te):
        d["stop_cat"] = d["stop_id"].astype(str).map(code).fillna(254).astype(float)
    allc = _NOCAL + cols + ["stop_cat"]
    return _fit_predict(tr, te, allc, seed, categorical_features=[0, len(allc) - 1],
                        max_leaf_nodes=255, min_samples_leaf=50)


@arm
def dtcat_big(tr, te, seed=42):
    """Control for dtcat_stopcat_big: leader with 255 leaves / min 50 per leaf."""
    return _stack_plus(tr, te, seed, _DT, categorical_features=[0],
                       max_leaf_nodes=255, min_samples_leaf=50)


# ---- run 8 (V6 cols: the vehicle's current link) ---------------------------
_CUR = ["cur_link_live", "cur_link_hist", "cur_link_frac"]


def _topcode(tr, te, src, dst):
    top = tr[src].astype(str).value_counts().index[:254]
    code = {s: i for i, s in enumerate(top)}
    for d in (tr, te):
        d[dst] = d[src].astype(str).map(code).fillna(254).astype(float)


def _stopcat_big_plus(tr, te, seed, extra_num=(), next_cat=False, mono=(), **hp):
    cols = _STACK + _DT
    tr, te = _sentinel(tr, te, cols + list(extra_num))
    _topcode(tr, te, "stop_id", "stop_cat")
    allc = _NOCAL + cols + list(extra_num) + ["stop_cat"]
    cats = [0, len(allc) - 1]
    if next_cat:
        _topcode(tr, te, "next_stop_id", "next_cat")
        allc.append("next_cat")
        cats.append(len(allc) - 1)
    params = dict(max_leaf_nodes=255, min_samples_leaf=50)
    if mono:   # run 18: +1 constraints; the transformed order equals allc (route_id is first)
        params["monotonic_cst"] = [1 if c in mono else 0 for c in allc]
    params.update(hp)
    return _fit_predict(tr, te, allc, seed, categorical_features=cats, **params)


@arm
def sc_big_cur(tr, te, seed=42):
    """dtcat_stopcat_big + current-link live m5 / historical time / fraction left."""
    return _stopcat_big_plus(tr, te, seed, extra_num=_CUR)


@arm
def sc_big_lr08(tr, te, seed=42):
    """dtcat_stopcat_big with learning rate 0.08 (the 1200-iteration cap binds)."""
    return _stopcat_big_plus(tr, te, seed, learning_rate=0.08)


# ---- run 9 (V7 cols: own previous-lap link times) --------------------------
_LAP = ["lap_path_sec", "lap_path_cov", "lap_path_age", "lap_fill_path_sec"]


@arm
def sc_big_lap(tr, te, seed=42):
    """sc_big_cur + the vehicle's own previous traversal of the path links
    (fallback where the shared live store is empty; sparse routes)."""
    return _stopcat_big_plus(tr, te, seed, extra_num=_CUR + _LAP)


@arm
def sc_big_cur_reg(tr, te, seed=42):
    """sc_big_cur with min_samples_leaf 100 and l2_regularization 1."""
    return _stopcat_big_plus(tr, te, seed, extra_num=_CUR, min_samples_leaf=100,
                             l2_regularization=1.0)


# ---- run 11 -----------------------------------------------------------------
@arm
def sc_big_lap_reg(tr, te, seed=42):
    """sc_big_lap with min_samples_leaf 100 and l2_regularization 1."""
    return _stopcat_big_plus(tr, te, seed, extra_num=_CUR + _LAP, min_samples_leaf=100,
                             l2_regularization=1.0)


# ---- run 12 (V8 cols: 6 h previous lap, vehicle-level own/hist ratio) -------
_VEH = ["veh_hist_ratio60", "veh_hist_n60"]


@arm
def sc_big_veh(tr, te, seed=42):
    """sc_big_lap + vehicle-level (any trip) own/historical ratio over the last hour."""
    return _stopcat_big_plus(tr, te, seed, extra_num=_CUR + _LAP + _VEH)


# ---- run 14 (V10 cols: dwell at the vehicle's current location) -------------
_DWELL = ["dwell_rem_med", "dwell_rem_p75", "dwell_p_more120", "dwell_n", "dwell_long_share"]


@arm
def sc_big_dwell(tr, te, seed=42):
    """sc_big_veh + historical conditional remaining dwell at the current location."""
    return _stopcat_big_plus(tr, te, seed, extra_num=_CUR + _LAP + _VEH + _DWELL)


# ---- run 14 add-on (V11 cols, MS_FEAT_DIR=ms_features_v11) -------------------
_DWELL_H = ["dwell_h_rem_med", "dwell_h_p_more120", "dwell_h_n"]
_DWELL_LIVE = ["dwell_live_last", "dwell_live_age", "dwell_live_m3"]


@arm
def sc_big_dwell_h(tr, te, seed=42):
    """sc_big_dwell + remaining dwell keyed by location x 3-hour band."""
    return _stopcat_big_plus(tr, te, seed, extra_num=_CUR + _LAP + _VEH + _DWELL + _DWELL_H)


@arm
def sc_big_dwell_v11(tr, te, seed=42):
    """sc_big_dwell_h + today's last stop durations at the same location."""
    return _stopcat_big_plus(tr, te, seed,
                             extra_num=_CUR + _LAP + _VEH + _DWELL + _DWELL_H + _DWELL_LIVE)


# ---- run 15 add-on (V12 cols, MS_FEAT_DIR=ms_features_v12) -------------------
_HW = ["hw_ahead_cur", "hw_ahead_tgt", "hw_ahead_prev"]
_DWELL_C = ["dwell_c_rem_med", "dwell_c_p_more120", "dwell_c_n"]
_LEAD = _CUR + _LAP + _VEH + _DWELL


@arm
def sc_big_hw(tr, te, seed=42):
    """sc_big_dwell + same-route headway to the leader (at our last stop, at the target)."""
    return _stopcat_big_plus(tr, te, seed, extra_num=_LEAD + _HW)


@arm
def sc_big_dwc(tr, te, seed=42):
    """sc_big_dwell + conditional remaining dwell keyed by the last passed stop only."""
    return _stopcat_big_plus(tr, te, seed, extra_num=_LEAD + _DWELL_C)


@arm
def sc_big_v12(tr, te, seed=42):
    """sc_big_dwell + headway + coarse dwell."""
    return _stopcat_big_plus(tr, te, seed, extra_num=_LEAD + _HW + _DWELL_C)


# ---- run 18 (no new cols: constraints / capacity on the leader) -------------
# Only never-missing cols can be constrained: the live / hist path cols use the
# -1 sentinel for "missing", which a +1 constraint would force to the lowest ETA.
_MONO = ["remaining_dist_m", "stops_ahead", "speed_eta_warm", "hist_travel_time_est"]


@arm
def sc_big_mono(tr, te, seed=42):
    """sc_big_dwell with ETA monotone increasing in distance / stops ahead / prior ETAs."""
    return _stopcat_big_plus(tr, te, seed, extra_num=_LEAD, mono=_MONO)


@arm
def sc_big_511_it2k(tr, te, seed=42):
    """sc_big_511 with the 2400-iteration cap (both capacity levers together)."""
    return _stopcat_big_plus(tr, te, seed, extra_num=_LEAD, max_leaf_nodes=511,
                             min_samples_leaf=100, max_iter=2400)


# ---- run 20 (no new cols) ----------------------------------------------------
@arm
def sc_big_nw(tr, te, seed=42):
    """sc_big_dwell without the production hour sample weights (the protocol scores
    unweighted MAE; the weights were a bias fix for the old feature set)."""
    return _stopcat_big_plus(tr, te, seed, extra_num=_LEAD, weights=np.ones(len(tr)))


# ---- runs 23-24 (no new cols): per-split feature subsampling and a 2-model bag ----
@arm
def sc_big_bag2(tr, te, seed=42):
    """Mean of two max_features-0.7 fits (arm sc_big_mf7, removed in run 25) with different seeds (2x the trees at serving)."""
    p1, i1 = _stopcat_big_plus(tr.copy(), te.copy(), seed, extra_num=_LEAD, max_features=0.7)
    p2, i2 = _stopcat_big_plus(tr.copy(), te.copy(), seed + 1000, extra_num=_LEAD, max_features=0.7)
    return (p1 + p2) / 2, {**i1, "n_iter": i1["n_iter"] + i2["n_iter"]}


@arm
def sc_big_mf5(tr, te, seed=42):
    """sc_big_dwell with max_features 0.5."""
    return _stopcat_big_plus(tr, te, seed, extra_num=_LEAD, max_features=0.5)


@arm
def sc_big_mf3(tr, te, seed=42):
    """sc_big_dwell with max_features 0.3 (run 24)."""
    return _stopcat_big_plus(tr, te, seed, extra_num=_LEAD, max_features=0.3)


# ---- run 25 (no new cols): cheaper-to-serve variants of the hand-off recipe ----
@arm
def sc_big_mf3_fast(tr, te, seed=42):
    """sc_big_mf3 at learning rate 0.1 with a 600-tree cap: half the trees to walk at serving."""
    return _stopcat_big_plus(tr, te, seed, extra_num=_LEAD, max_features=0.3, learning_rate=0.1, max_iter=600)


@arm
def sc_big_mf3_127(tr, te, seed=42):
    """sc_big_mf3 with 127-leaf trees (the production export size per tree)."""
    return _stopcat_big_plus(tr, te, seed, extra_num=_LEAD, max_features=0.3, max_leaf_nodes=127)


# ---- run 26 add-on (V26 cols, MS_FEAT_DIR=ms_features_v26) -------------------
_AGE = ["pos_age_s", "pos_age_med600", "pos_age_dist"]


@arm
def sc_big_mf3_age(tr, te, seed=42):
    """sc_big_mf3 + age of the vehicle's GPS fix at the snapshot (feed ts - vehicle ts)."""
    return _stopcat_big_plus(tr, te, seed, extra_num=_LEAD + _AGE, max_features=0.3)


# ---- run 27: same arms on fix-clock features (MS_AGE_CORR prep -> ms_features_v27) ----
@arm
def sc_big_mf3_age1(tr, te, seed=42):
    """sc_big_mf3 + pos_age_s only (ablation of the V26 cols)."""
    return _stopcat_big_plus(tr, te, seed, extra_num=_LEAD + ["pos_age_s"], max_features=0.3)


# ---- run 28 add-on (V28 cols: feed-reported speed / odometer, MS_FEAT_DIR=ms_features_v28) ----
_FS = ["fs_now", "fs_mean60"]
_ODO = ["odo_spd_60", "odo_spd_180", "odo_m_300"]


@arm
def sc_big_mf3_age_spd(tr, te, seed=42):
    """sc_big_mf3_age + the feed's reported speed (now, 60 s mean)."""
    return _stopcat_big_plus(tr, te, seed, extra_num=_LEAD + _AGE + _FS, max_features=0.3)


@arm
def sc_big_mf3_age_odo(tr, te, seed=42):
    """sc_big_mf3_age + odometer speed over 60 / 180 s and metres in 300 s."""
    return _stopcat_big_plus(tr, te, seed, extra_num=_LEAD + _AGE + _ODO, max_features=0.3)


@arm
def sc_big_mf3_age_fo(tr, te, seed=42):
    """sc_big_mf3_age + all V28 cols (reported speed + odometer)."""
    return _stopcat_big_plus(tr, te, seed, extra_num=_LEAD + _AGE + _FS + _ODO, max_features=0.3)


# ---- run 28 part 2 (V28b cols, MS_FEAT_DIR=ms_features_v28b) ----
_FS2 = ["fs_lag1", "fs_acc30", "fs_max120", "fs_zero_sec", "odo_spd_30"]


@arm
def sc_big_mf3_age_fo2(tr, te, seed=42):
    """sc_big_mf3_age_fo + recent reported-speed trajectory (lag, 30 s accel, 120 s max, time since moving) and 30 s odometer speed."""
    return _stopcat_big_plus(tr, te, seed, extra_num=_LEAD + _AGE + _FS + _ODO + _FS2, max_features=0.3)


# ---- run 29 add-on (V29 cols: long-window odometer + odometer-vs-shape, MS_FEAT_DIR=ms_features_v29) ----
_ODOL = ["odo_spd_600", "odo_spd_1200", "fs_zero_frac600"]
_ODOG = ["odo_shape_gap300", "odo_trip_spd"]


@arm
def sc_big_fo_long(tr, te, seed=42):
    """sc_big_mf3_age_fo + odometer speed over 600 / 1200 s and the 600 s zero-speed share."""
    return _stopcat_big_plus(tr, te, seed, extra_num=_LEAD + _AGE + _FS + _ODO + _ODOL, max_features=0.3)


@arm
def sc_big_fo_gap(tr, te, seed=42):
    """sc_big_mf3_age_fo + odometer-vs-shape progress gap (300 s) and trip-average odometer speed."""
    return _stopcat_big_plus(tr, te, seed, extra_num=_LEAD + _AGE + _FS + _ODO + _ODOG, max_features=0.3)


@arm
def sc_big_fo_v29(tr, te, seed=42):
    """sc_big_mf3_age_fo + all V29 cols."""
    return _stopcat_big_plus(tr, te, seed, extra_num=_LEAD + _AGE + _FS + _ODO + _ODOL + _ODOG, max_features=0.3)


# ---- run 30 add-on (V30 cols: other vehicles on the path right now, MS_FEAT_DIR=ms_features_v30) ----
_PA = ["pa_n", "pa_spd_mean", "pa_spd_min", "pa_stop_frac", "pa_gap_m", "pa_gap_spd", "pa_cov"]
_FOL = _AGE + _FS + _ODO + _ODOL


@arm
def sc_big_fo_pa(tr, te, seed=42):
    """sc_big_fo_long + vehicles currently on the path links ahead (count, odometer speeds, stopped share, nearest gap)."""
    return _stopcat_big_plus(tr, te, seed, extra_num=_LEAD + _FOL + _PA, max_features=0.3)


# serving-simplification ablations: does the odometer make the per-vehicle crossing log (_LAP / _VEH) redundant?
@arm
def sc_big_fo_nolap(tr, te, seed=42):
    """sc_big_fo_long without the own previous-lap cols."""
    return _stopcat_big_plus(tr, te, seed, extra_num=_CUR + _VEH + _DWELL + _FOL, max_features=0.3)


@arm
def sc_big_fo_noveh(tr, te, seed=42):
    """sc_big_fo_long without the vehicle-level own/hist ratio cols."""
    return _stopcat_big_plus(tr, te, seed, extra_num=_CUR + _LAP + _DWELL + _FOL, max_features=0.3)


@arm
def sc_big_fo_novl(tr, te, seed=42):
    """sc_big_fo_long without _LAP and _VEH (no per-vehicle crossing log at serving)."""
    return _stopcat_big_plus(tr, te, seed, extra_num=_CUR + _DWELL + _FOL, max_features=0.3)


# ---- run 31 add-on (V31 cols: "now" path time from the vehicles on the path, 40/60 min odometer, MS_FEAT_DIR=ms_features_v31) ----
_PAN = ["pa_now_t", "pa_now_len", "pa_slow_gap"]
_ODOX = ["odo_spd_2400", "odo_spd_3600"]


@arm
def sc_big_fo_pan(tr, te, seed=42):
    """sc_big_fo_pa + occupied-link "now" path time, its length and the gap to the nearest slow vehicle ahead."""
    return _stopcat_big_plus(tr, te, seed, extra_num=_LEAD + _FOL + _PA + _PAN, max_features=0.3)


@arm
def sc_big_fo_pax(tr, te, seed=42):
    """sc_big_fo_pa + odometer speed over 40 / 60 min."""
    return _stopcat_big_plus(tr, te, seed, extra_num=_LEAD + _FOL + _PA + _ODOX, max_features=0.3)


@arm
def sc_big_fo_v31(tr, te, seed=42):
    """sc_big_fo_pa + all V31 cols."""
    return _stopcat_big_plus(tr, te, seed, extra_num=_LEAD + _FOL + _PA + _PAN + _ODOX, max_features=0.3)


# ---- run 32 add-on (V32 cols: own break / duty state from the odometer ring, MS_FEAT_DIR=ms_features_v32) ----
_BRK = ["brk_since_s", "brk_last_len", "brk_odo_since", "brk10_since_s", "brk10_odo_since", "brk_n3h"]
_DUTY = ["duty_s", "duty_odo"]


@arm
def sc_big_fo_brk(tr, te, seed=42):
    """sc_big_fo_pa + the vehicle's own past stopped episodes (since / length / odometer since, 3 h count)."""
    return _stopcat_big_plus(tr, te, seed, extra_num=_LEAD + _FOL + _PA + _BRK, max_features=0.3)


@arm
def sc_big_fo_duty(tr, te, seed=42):
    """sc_big_fo_pa + time / odometer since the vehicle's shift start."""
    return _stopcat_big_plus(tr, te, seed, extra_num=_LEAD + _FOL + _PA + _DUTY, max_features=0.3)


@arm
def sc_big_fo_v32(tr, te, seed=42):
    """sc_big_fo_pa + all V32 cols."""
    return _stopcat_big_plus(tr, te, seed, extra_num=_LEAD + _FOL + _PA + _BRK + _DUTY, max_features=0.3)
