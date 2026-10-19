"""
Serving timing test (run 9): how long does src/inference.py-style vectorised
tree walking take for a candidate model, including native categorical splits?

  MS_SAVE_MODEL=data/ms_models/<arm>.pkl python research/model-search/harness.py run --arm <arm> ...
  python research/model-search/timing.py data/ms_models/baseline.pkl data/ms_models/sc_big_cur.pkl

Exports trees the way scripts/export_worker_data.py does, plus the two extra
node fields categorical splits need (is_categorical, bitset index -> 256-bit
bitset), walks them with the predict_rows algorithm extended by one bitset
test, checks parity against sklearn, and times it on push-sized batches.
"""

from __future__ import annotations

import sys
import time

import joblib
import numpy as np


def export(pipe):
    model = pipe.named_steps["model"]
    base = float(model._baseline_prediction.flat[0])
    f, thr, left, right, leaf, val, iscat, bidx, mleft, roots = ([] for _ in range(10))
    bitsets = []
    off = 0
    for est in model._predictors:
        p = est[0]
        nodes = p.nodes
        roots.append(off)
        bs = p.raw_left_cat_bitsets
        boff = len(bitsets)
        bitsets.extend(bs.tolist() if len(bs) else [])
        for n in nodes:
            f.append(0 if n["is_leaf"] else int(n["feature_idx"]))
            thr.append(float(n["num_threshold"]))
            left.append(off + int(n["left"]))
            right.append(off + int(n["right"]))
            leaf.append(bool(n["is_leaf"]))
            val.append(float(n["value"]))
            iscat.append(bool(n["is_categorical"]))
            bidx.append(boff + int(n["bitset_idx"]) if n["is_categorical"] else 0)
            mleft.append(bool(n["missing_go_to_left"]))
        off += len(nodes)
    A = lambda x, t: np.asarray(x, dtype=t)
    return dict(base=base, f=A(f, np.int32), thr=A(thr, np.float64), left=A(left, np.int32),
                right=A(right, np.int32), leaf=A(leaf, bool), val=A(val, np.float64),
                iscat=A(iscat, bool), bidx=A(bidx, np.int32), mleft=A(mleft, bool),
                roots=A(roots, np.int32),
                bits=A(bitsets if bitsets else np.zeros((1, 8)), np.uint32).reshape(-1, 8),
                n_nodes=off, n_trees=len(roots), any_cat=bool(np.any(iscat)))


def predict(T, X, chunk=1024):
    roots = T["roots"]
    nt = len(roots)
    out = np.empty(len(X))
    for s in range(0, len(X), chunk):
        c = X[s:s + chunk]
        nc = len(c)
        state = np.tile(roots, nc)
        row_of = np.repeat(np.arange(nc), nt)
        active = np.arange(nc * nt)
        while active.size:
            at = state[active]
            landed = T["leaf"][at]
            if landed.any():
                active = active[~landed]
                if not active.size:
                    break
                at = state[active]
            x = c[row_of[active], T["f"][at]]
            miss = np.isnan(x)
            go_right = x > T["thr"][at]
            if T["any_cat"]:
                cat = T["iscat"][at]
                if cat.any():
                    xc = x[cat]
                    bad = np.isnan(xc) | (xc < 0)
                    xi = np.where(bad, 0, xc).astype(np.int64)
                    word = T["bits"][T["bidx"][at[cat]], xi >> 5]
                    inset = ((word >> (xi & 31).astype(np.uint32)) & 1).astype(bool)
                    go_right[cat] = ~inset
                    miss[cat] = bad
            go_right = np.where(miss, ~T["mleft"][at], go_right)
            state[active] = np.where(go_right, T["right"][at], T["left"][at])
        out[s:s + nc] = T["base"] + T["val"][state].reshape(nc, nt).sum(axis=1)
    return out


def main():
    for path in sys.argv[1:]:
        pipe, cols, sample = joblib.load(path)
        T = export(pipe)
        X = pipe.named_steps["prep"].transform(sample)
        # sklearn >= 1.4 with categorical_features moves categorical columns first
        # and re-encodes their values (sorted, unknown -> NaN) via an internal
        # ColumnTransformer; serving must replicate that (a column permutation +
        # one dict per categorical column).
        pre = getattr(pipe.named_steps["model"], "_preprocessor", None)
        if pre is not None:
            X = pre.transform(X)
        X = np.asarray(X, dtype=np.float64)
        ref = pipe.predict(sample)
        got = predict(T, X[:2000])
        err = np.max(np.abs(got - ref[:2000]))
        # Push-sized batches: ~3000 rows every 10 s (single thread, like the daemon).
        times = []
        for rep in range(5):
            t0 = time.perf_counter()
            predict(T, X[rep * 3000:(rep + 1) * 3000])
            times.append(time.perf_counter() - t0)
        print(f"{path}: trees={T['n_trees']} nodes={T['n_nodes']:,} "
              f"(export ~{T['n_nodes'] * 6 * 8 / 1e6:.0f} MB as float64 arrays) "
              f"cat_nodes={int(T['iscat'].sum()):,} parity max|d|={err:.2e} "
              f"3000 rows: median {np.median(times):.2f}s min {min(times):.2f}s")


if __name__ == "__main__":
    main()
