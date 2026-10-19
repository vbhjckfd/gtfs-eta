"""Per-route error diagnostic on saved test predictions (run 20).

  python research/model-search/diag_route.py data/pred_ds1v20.parquet 133 [122 ...]

Needs MS_SAVE_PRED output from harness.py run (test rows + `pred`). Prints, per route:
signed error (pred - actual) by stops_ahead, by target stop, by vehicle, by local hour and by
stationary bucket, and how much of the route's absolute error sits in rows whose actual time
exceeds the historical path by > 300 s (an unmodelled layover / hold inside the trip).
"""

import sys

import numpy as np
import pandas as pd

from src.features import TARGET_COL


def _agg(d, key, top=12):
    g = d.groupby(key, observed=True).agg(n=("err", "size"), mae=("ae", "mean"), bias=("err", "mean"),
                                          ae_share=("ae", "sum"))
    g["ae_share"] = (g["ae_share"] / d["ae"].sum()).round(3)
    return g.sort_values("ae_share", ascending=False).head(top).round(1)


def main(path, routes):
    df = pd.read_parquet(path)
    df["err"] = df["pred"] - df[TARGET_COL].astype(float)
    df["ae"] = df["err"].abs()
    df["route_id"] = df["route_id"].astype(str)
    print(f"all: n={len(df):,} MAE {df.ae.mean():.1f} bias {df.err.mean():.1f}")
    for r in routes:
        d = df[df.route_id == r].copy()
        if d.empty:
            print(f"route {r}: no rows")
            continue
        print(f"\n=== route {r}: n={len(d):,} MAE {d.ae.mean():.1f} bias {d.err.mean():.1f} "
              f"(share of total AE {d.ae.sum() / df.ae.sum():.3%})")
        d["sa"] = d["stops_ahead"].clip(upper=15)
        print("by stops_ahead:\n", _agg(d, "sa", 15).sort_index().to_string())
        if "stop_id" in d:
            print("by target stop:\n", _agg(d, "stop_id").to_string())
        for vc in ("vehicle_id", "trip_id"):
            if vc in d:
                print(f"by {vc}:\n", _agg(d, vc, 8).to_string())
                break
        d["lh"] = ((d["hour"] + 3) % 24).astype(int)   # UTC -> Lviv summer time
        print("by local hour:\n", _agg(d, "lh", 24).sort_index().to_string())
        d["st"] = pd.cut(d["stationary_sec"], [-1, 30, 120, 300, 900, 1e9],
                         labels=["<30", "30-120", "120-300", "300-900", ">900"])
        print("by stationary_sec:\n", _agg(d, "st").sort_index().to_string())
        if "date" in d:
            print("by date:\n", _agg(d, "date").sort_index().to_string())
        if "hist_path_sec_dt" in d:
            h = d["hist_path_sec_dt"].astype(float)
            ok = h > 0
            excess = d[TARGET_COL].astype(float) - h
            hold = ok & (excess > 300)
            print(f"rows with hist path: {ok.mean():.1%}; actual > hist + 300 s: {hold.mean():.1%} of rows, "
                  f"{d.loc[hold, 'ae'].sum() / d.ae.sum():.1%} of the route's AE, their bias "
                  f"{d.loc[hold, 'err'].mean():.0f} s")
            print(f"median actual/hist ratio {np.median(d.loc[ok, TARGET_COL] / h[ok]):.2f} "
                  f"(all routes {np.median(df.loc[df.hist_path_sec_dt > 0, TARGET_COL] / df.loc[df.hist_path_sec_dt > 0, 'hist_path_sec_dt']):.2f})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
