"""Run 45: residual feedback — the model's own recent errors as live features.

At serving time the daemon knows, for every stop a vehicle has already reached, what it predicted for that stop
earlier. This arm feeds those matured errors (pred - actual of earlier snapshots whose target stop was reached
>= LAG s ago) back as features: per vehicle over the last 600 / 1800 s, and per route over the last 600 s.

Stage 1 = the production recipe (src/train_live.fit). Train-day errors are out-of-fold (3 folds by day), test-day
errors come from the baseline fit on all train days (= the protocol baseline, scored on the same rows).

    python research/model-search/rfb.py --train 2026-10-05..2026-10-14 --test 2026-10-15..2026-10-18 --seed 42 --split C

Appends lv2_<split>_s<seed> (baseline) and rfb_<split>_s<seed> (+ RFB_COLS) to results_tl.jsonl.
Stage-1 predictions are cached in data/rfb/<split>_s<seed>/ so a re-run only refits the candidate.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from src import train_live as tl  # noqa: E402
from src.features import BASE_FEATURE_COLS, TARGET_COL  # noqa: E402

HERE = Path(__file__).parent
LAG = 30.0            # a crossing is known this long after it happened (live link store uses the same lag)
SENT = -9999.0        # no matured prediction in the window
RFB_COLS = ["rfb_n600", "rfb_e600", "rfb_r600", "rfb_e1800", "rfb_r1800", "rfb_last_e", "rfb_last_age",
            "rfr_n600", "rfr_e600", "rfr_r600"]


def load(days: list[str], frac: float) -> pd.DataFrame:
    """tl.load_days + vehicle_id / t, before its row cap (same filters, same order)."""
    num = sorted(set(BASE_FEATURE_COLS + tl.BASE_COLS + tl.MODEL_LIVE_COLS + [TARGET_COL])
                 - {"route_id", "speed_eta_warm", "hist_speed_mps", "hist_travel_time_est"})
    cols = ["route_id", "stop_id", "date"] + num
    pieces = []
    for d in days:
        piece = pd.read_parquet(tl.feature_path(d, frac), columns=cols + ["vehicle_id", "t"])
        piece[num] = piece[num].astype(np.float32)
        pieces.append(piece)
    df = pd.concat(pieces, ignore_index=True)
    del pieces
    df = df.dropna(subset=[TARGET_COL] + [c for c in BASE_FEATURE_COLS if c in df.columns])
    df = df[df[TARGET_COL].between(0, 3600)]
    df = df[~df["route_id"].astype(str).isin(tl.BAD_ROUTE_IDS)]
    df = df.reset_index(drop=True)
    df["day"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    return df


def _window(kq, tq, ks, ms, vals, w):
    """For each query (key kq, time tq): count and sums of *vals* over sources of the same key with
    maturity ms in [tq - w, tq - LAG]; plus the index of the latest such source (-1 if none)."""
    span = 4e5
    t0 = min(tq.min(), ms.min()) - w - 1
    order = np.lexsort((ms, ks))
    s = ks[order] * span + (ms[order] - t0)
    hi = np.searchsorted(s, kq * span + (tq - LAG - t0), side="right")
    lo = np.searchsorted(s, kq * span + (tq - w - t0), side="left")
    n = hi - lo
    sums = []
    for v in vals:
        c = np.concatenate([[0.0], np.cumsum(v[order])])
        sums.append(c[hi] - c[lo])
    last = np.where(n > 0, order[np.maximum(hi - 1, 0)], -1)
    return n, sums, last


def rfb_features(df: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """RFB_COLS for every row of *df*, from the matured errors of *pred* on the same day's rows."""
    y = df[TARGET_COL].to_numpy(dtype=float)
    t = df["t"].to_numpy(dtype=float)
    e = pred - y
    r = e / np.maximum(y, 60.0)
    m = t + y
    out = {c: np.full(len(df), SENT) for c in RFB_COLS}
    veh = pd.factorize(df["vehicle_id"].astype(str))[0].astype(float)
    rte = pd.factorize(df["route_id"].astype(str))[0].astype(float)
    for day in sorted(df["day"].unique()):
        i = np.flatnonzero(df["day"].to_numpy() == day)
        for w in (600.0, 1800.0):
            n, (se, sr), last = _window(veh[i], t[i], veh[i], m[i], [e[i], r[i]], w)
            ok = n > 0
            tag = int(w)
            if w == 600.0:
                out["rfb_n600"][i] = n
                out["rfb_last_e"][i[ok]] = e[i][last[ok]]
                out["rfb_last_age"][i[ok]] = t[i][ok] - m[i][last[ok]]
            out[f"rfb_e{tag}"][i[ok]] = se[ok] / n[ok]
            out[f"rfb_r{tag}"][i[ok]] = sr[ok] / n[ok]
        n, (se, sr), _ = _window(rte[i], t[i], rte[i], m[i], [e[i], r[i]], 600.0)
        ok = n > 0
        out["rfr_n600"][i] = n
        out["rfr_e600"][i[ok]] = se[ok] / n[ok]
        out["rfr_r600"][i[ok]] = sr[ok] / n[ok]
    return pd.DataFrame(out, index=df.index)


def _fit_predict(tr: pd.DataFrame, te: pd.DataFrame, seed: int) -> tuple[np.ndarray, int]:
    b = tl.fit(tr, seed=seed)
    return tl.predict(b, te), int(b["model"].n_iter_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", required=True)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--drop", default="", help="RFB cols to leave out of the candidate (ablation)")
    ap.add_argument("--name", default="rfb")
    args = ap.parse_args()
    train_days, test_days = tl.expand_days(args.train), tl.expand_days(args.test)
    cache = REPO / "data" / "rfb" / f"{args.split}_s{args.seed}"
    cache.mkdir(parents=True, exist_ok=True)
    res_path = HERE / "results_tl.jsonl"

    tl.build_features(train_days, tl.DEFAULT_SNAP_FRAC, args.parallel)
    tl.build_features(test_days, tl.TEST_SNAP_FRAC, args.parallel)
    full = load(train_days, tl.DEFAULT_SNAP_FRAC)
    # The production row cap, drawn exactly as tl.load_days draws it. The RFB features are computed on all
    # rows (so their density matches the uncapped test days), the fits use the capped sample.
    if len(full) > tl.MAX_TRAIN_ROWS:
        idx = full.sample(n=tl.MAX_TRAIN_ROWS, random_state=args.seed).index
    else:
        idx = full.index
    tr = full.loc[idx].reset_index(drop=True)
    te = load(test_days, tl.TEST_SNAP_FRAC)
    print(f"train {len(tr):,} rows, test {len(te):,} rows", flush=True)
    base_res = {"train_days": [train_days[0], train_days[-1]], "test_days": test_days,
                "n_train": int(len(tr)), "seed": args.seed}

    # Stage 1: baseline on all train days -> test errors (and the protocol baseline row).
    p_te = cache / "pred_test.npy"
    if not p_te.exists():
        t0 = time.monotonic()
        pred, n_iter = _fit_predict(tr, te, args.seed)
        np.save(p_te, pred)
        m = tl.metrics(te, pred)
        print(f"lv2 {tl._fmt(m)} ({(time.monotonic() - t0) / 60:.0f} min)", flush=True)
        with open(res_path, "a") as f:
            f.write(json.dumps(dict(base_res, name=f"lv2_{args.split}_s{args.seed}", n_iter=n_iter,
                                    live_v2=m)) + "\n")
    pred_te = np.load(p_te)

    # Stage 1, out of fold: train-day errors from fits that never saw that day.
    p_oof = cache / f"pred_oof{args.folds}.npy"
    if not p_oof.exists():
        oof = np.full(len(full), np.nan)
        for k in range(args.folds):
            hold = [d for j, d in enumerate(train_days) if j % args.folds == k]
            mask = full["day"].isin(hold).to_numpy()
            print(f"fold {k}: hold {hold} ({mask.sum():,} rows)", flush=True)
            oof[mask], _ = _fit_predict(tr[~tr["day"].isin(hold)], full[mask], args.seed)
        np.save(p_oof, oof)
    oof = np.load(p_oof)
    print(f"OOF MAE on train days {np.abs(oof - full[TARGET_COL].to_numpy()).mean():.1f}", flush=True)

    # Candidate: production recipe + RFB cols.
    drop = {c for c in args.drop.split(",") if c}
    extra = [c for c in RFB_COLS if c not in drop]
    rf = rfb_features(full, oof).loc[idx].reset_index(drop=True).astype(np.float32)
    del full, oof
    gc.collect()
    keep = ["route_id", "stop_id", "date", "day", TARGET_COL] + tl.BASE_COLS + tl.MODEL_LIVE_COLS
    keep = [c for c in tr.columns if c in set(keep) or c in BASE_FEATURE_COLS]
    tr = pd.concat([tr[keep], rf], axis=1)
    del rf
    te = pd.concat([te, rfb_features(te, pred_te).astype(np.float32)], axis=1)
    gc.collect()
    for c in extra:
        cov = (te[c] != SENT).mean()
        print(f"  {c}: test coverage {cov:.1%}  median {te.loc[te[c] != SENT, c].median():.3g}", flush=True)
    tl.MODEL_LIVE_COLS = tl.MODEL_LIVE_COLS + extra
    tl.MODEL_COLS = tl.CAT_COLS + tl.BASE_COLS + tl.MODEL_LIVE_COLS
    t0 = time.monotonic()
    pred, n_iter = _fit_predict(tr, te, args.seed)
    m = tl.metrics(te, pred)
    print(f"{args.name} {tl._fmt(m)} ({(time.monotonic() - t0) / 60:.0f} min)", flush=True)
    with open(res_path, "a") as f:
        f.write(json.dumps(dict(base_res, name=f"{args.name}_{args.split}_s{args.seed}", n_iter=n_iter,
                                live_v2=m)) + "\n")


if __name__ == "__main__":
    main()
