"""Regenerate LEADERBOARD.md from results/*.json (baseline row per tag+seed)."""

import json
from pathlib import Path

from harness import judge

HERE = Path(__file__).parent
SERVING = {
    "baseline": "prod",
    "live_all": "needs serving work (NaN routing + live link store)",
    "live_all_s": "needs serving work (live link store, ~1-2 days)",
    "live_path": "needs serving work (NaN routing + live link store)",
    "own_speed": "needs serving work (NaN routing + 5-min position buffer)",
    "headway": "needs serving work (NaN routing + stop-arrival store)",
    "path_s": "needs serving work (link store)",
    "path_min_s": "needs serving work (link store, 2 cols)",
    "path_own_s": "needs serving work (link store + 5-min position ring, ~1 day), state persisted across the 5-min push-feed restarts",
    "path_own_s_drop": "as path_own_s; degrades less when cold",
    "path_own_s_big": "as path_own_s; 255-leaf trees ~2x export size",
    "live_all_s_lr10": "as live_all_s",
    "live_all_s_cold": "diagnostic (cold start)",
    "path_own_s_cold": "diagnostic (cold start)",
    "path_own_s_drop_cold": "diagnostic (cold start)",
    "path_own_s_linkcold": "diagnostic (link store cold)",
    "path_own_m3": "as path_own_s; store keeps last 3 traversals per link",
    "path_own_m3only": "as path_own_s; store keeps last 3 traversals per link",
    "path_own_s_gate": "as path_own_s; baseline trees also shipped for cov=0 rows",
    "path_own_s_nw": "as path_own_s",
    "path_own_s_nocal": "as path_own_s",
    "path_own_m35": "as path_own_s; store keeps last 5 traversals per link",
    "path_own_m3_big": "as path_own_m3; 255-leaf trees ~2x export size",
    "stack": "link store (last 5 traversals/link) + 5-min position ring, persisted in tracker_state.json; ~1 day",
    "stack_big": "as stack; 255-leaf trees ~2x export size",
    "stack_m7": "as stack; store keeps last 7 traversals per link",
    "stack_ewm": "as stack; + per-link EWMA in the store",
    "stack_hist": "as stack + static link x hour median table built at export (like priors)",
    "stack_fill": "as stack_hist",
    "stack_v3": "as stack_hist + last 7 traversals + EWMA per link",
    "stack_cold": "diagnostic (cold start)",
    "stack_hist_dt": "as stack_hist + weekday/weekend link x hour table",
    "stack_hist_r": "as stack_hist",
    "stack_v4": "as stack_hist + day-type and p75 tables",
    "stack_hist_l2": "as stack_hist",
    "stack_hist_leaf100": "as stack_hist",
    "stack_hist_catroute": "as stack_hist + categorical-split (bitset) support in export/inference, ~0.5-1 day",
    "stack_hist_dt_cat": "as stack_hist_dt + categorical-split (bitset) support in export/inference; ~2 days total",
    "stack_hist_cold": "diagnostic (cold start, hist table kept)",
    "dtcat_own": "as stack_hist_dt_cat + own last-k link sums vs hist table (per-vehicle crossing log)",
    "dtcat_own_x": "as dtcat_own",
    "dtcat_recency": "as stack_hist_dt_cat",
    "dtcat_prune": "as stack_hist_dt_cat",
    "dtcat_stopcat": "as stack_hist_dt_cat + stop->code map (254 stops) shipped with the model",
    "dtcat_stopcat_big": "as dtcat_stopcat; 255-leaf trees ~2x export size; ~2-2.5 days total",
    "dtcat_big": "as stack_hist_dt_cat; 255-leaf trees ~2x export size",
    "dtcat_resid": "as stack_hist_dt_cat + add base ETA to tree output",
    "dtcat_logratio": "as dtcat_resid (exp transform)",
    "m3_nocal": "link store (last 3 traversals/link) + 5-min position ring, persisted in tracker_state.json; ~1 day",
}
HEAD = """# Leaderboard

Held-out raw-model metrics (seconds); Δ vs the baseline row with the same tag
(day set / split) and seed. Win = MAE ≤ −3%, p90 not worse, no stops_ahead bucket
worse by > 5%. Regenerate with `python research/model-search/report.py`.

Tags: `ds1v5` / `ds1shiftv5` = v4 features rebuilt in run 7 (+ own-vs-hist cols); baselines identical. `ds1v5k3` = ds1v5 with train rows at 3% of snapshots (3x) instead of 1%.
`ds1v4` / `ds1shiftv4` = same splits, features v4 (MS_FEAT_DIR=ms_features_v4: pipeline from 08-31 so the historical link table has 7-14 prior days for every train row, + day-type table, live/hist ratio, p75); baselines identical.
`ds1v3` / `ds1shiftv3` = same splits, features v3 (MS_FEAT_DIR=ms_features_v3: + m7, EWMA, historical link table from prior days); baselines identical.
`ds1v2` / `ds1shiftv2` = same splits, features prepped with the run-3 median-of-5 column (MS_FEAT_DIR=ms_features_v2); baselines identical.
`ds1lag90w15` = ds1 with live features rebuilt at 90 s detection lag, 15-min link window.
`ds1` = train 2026-09-07..09-18, test 09-19 (Sat), 09-20 (Sun), 09-21 (Mon).
`ds1shift` = train 09-07..09-17, test 09-18 (Fri), 09-19, 09-20. Train rows 1% of
snapshots/day, test 3% (pipeline_lite 10% → prep 3% → run).

| tag | arm | seed | n_train | n_test | MAE | median | p90 | bias | sa1 | sa10 | ΔMAE | Δp90 | worst bucket Δ | win | serving |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
"""


def main():
    res = {}
    for p in sorted((HERE / "results").glob("*.json")):
        m = json.loads(p.read_text())
        tag = p.stem.rsplit("_s", 1)[0][len(m["arm"]):].lstrip("_") or "-"
        res[(tag, m["seed"], m["arm"])] = m
    lines = [HEAD]
    for (tag, seed, arm_name), m in sorted(res.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] != "baseline", kv[0][2])):
        base = res.get((tag, seed, "baseline"))
        if base and arm_name != "baseline":
            j = judge(base, m)
            d = (f"{j['mae_change_pct']:+.1f}%", f"{j['p90_change_pct']:+.1f}%",
                 f"{j['worst_bucket_change_pct']:+.1f}%", "**yes**" if j["wins"] else "no")
        else:
            d = ("—",) * 4
        sa = m["mae_by_stops_ahead"]
        lines.append(
            f"| {tag} | {arm_name} | {seed} | {m['n_train']:,} | {m['n']:,} | {m['mae']:.1f} | "
            f"{m['median_ae']:.1f} | {m['p90_ae']:.1f} | {m['bias']:+.1f} | "
            f"{sa.get('1', 0):.1f} | {sa.get('10', 0):.1f} | {' | '.join(d)} | "
            f"{SERVING.get(arm_name, '?')} |")
    (HERE / "LEADERBOARD.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[1:]))


if __name__ == "__main__":
    main()
