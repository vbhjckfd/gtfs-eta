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
    "sc_big_cur": "as dtcat_stopcat_big + current-link live m5 / hist / fraction (already in the link store); ~2-2.5 days total",
    "sc_big_nxt": "as dtcat_stopcat_big + 3rd categorical (next stop code map)",
    "sc_big_v6": "as sc_big_cur + next-stop categorical",
    "sc_big_lap": "as sc_big_cur + per-vehicle last traversal of each link (3 h window; the link store already sees every traversal, keyed also by vehicle); ~2.5 days total",
    "sc_big_cur_reg": "as sc_big_cur",
    "sc_big_lap_reg": "as sc_big_lap",
    "sc_lap_127": "as sc_big_lap at 127 leaves (current export size)",
    "sc_big_lr08": "as dtcat_stopcat_big",
    "dtcat_big": "as stack_hist_dt_cat; 255-leaf trees ~2x export size",
    "dtcat_resid": "as stack_hist_dt_cat + add base ETA to tree output",
    "dtcat_logratio": "as dtcat_resid (exp transform)",
    "sc_big_lap6": "as sc_big_lap (6 h window)",
    "sc_big_veh": "as sc_big_lap + per-vehicle 1 h deque of (own link time, hist link time) from the same crossing events; ~2.7 days total",
    "sc_big_v8": "as sc_big_veh + 6 h lap window",
    "sc_big_net": "as sc_big_veh + one global 15-min ring of (link time, hist) sums",
    "sc_big_v9": "as sc_big_net + 20 min and clipped 60 min vehicle ratios",
    "sc_big_dwell": "as sc_big_veh + static location -> sorted stop-duration table (prior 14 days, built at export like hist_dt); lookup by (last passed stop, 50 m bin) and stationary_sec; ~+0.3 day, ~3 days total",
    "sc_big_dwell2": "as sc_big_dwell (2 of its 5 cols)",
    "sc_big_dwell_h": "as sc_big_dwell with the table also keyed by 3-hour band",
    "sc_big_dwell_v11": "as sc_big_dwell_h + per-location ring of the last 3 live stop durations",
    "m3_nocal": "link store (last 3 traversals/link) + 5-min position ring, persisted in tracker_state.json; ~1 day",
}
# Tags shown in LEADERBOARD.md (the current feature versions + the leader's scale check);
# every other tag goes to archive/LEADERBOARD-old.md.
CURRENT_TAGS = {"ds1v10", "ds1shiftv10", "ds1v11", "ds1shiftv11", "ds1v9k3",
                "ds1v12", "ds1shiftv12", "ds1v12k3"}
HEAD = """# Leaderboard

Held-out raw-model metrics (seconds); Δ vs the baseline row with the same tag
(day set / split) and seed. Win = MAE ≤ −3%, p90 not worse, no stops_ahead bucket
worse by > 5%. Regenerate with `python research/model-search/report.py`.

Superseded tags (ds1 .. ds1v9, ds1shift .. ds1shiftv9, ds1v5k3, ds1lag90w15) are in
`archive/LEADERBOARD-old.md`; their baselines are identical to the ones below.

Tags: `ds1v12` / `ds1shiftv12` = rebuilt in run 15 (v10 files + V12 add-on cols from `addon_v12.py`: same-route headway / bunching, coarse dwell fallback; MS_FEAT_DIR=ms_features_v12); baselines identical. `ds1v12k3` = 3x train rows.
`ds1v10` / `ds1shiftv10` = rebuilt in run 14 (+ V10 cols: conditional remaining dwell at the current location from prior days' stationary runs; MS_FEAT_DIR=ms_features_v10); `ds1v11` / `ds1shiftv11` = same rows + V11 add-on cols (hour-banded dwell, live same-location dwell; MS_FEAT_DIR=ms_features_v11); baselines identical.
`ds1v9` / `ds1shiftv9` = rebuilt in run 13 (+ V9 cols: network ratio net_ratio15 / net_n15, veh_hist_ratio20, clipped veh_hist_ratio60c; MS_FEAT_DIR=ms_features_v9); baselines identical. `ds1v9k3` = ds1v9 with train rows at 3% of snapshots (3x).
`ds1v8` / `ds1shiftv8` = rebuilt in run 12 (+ V8 cols: 6 h previous lap lap6_*, vehicle-level own/hist ratio veh_hist_ratio60 / veh_hist_n60; MS_FEAT_DIR=ms_features_v8); baselines identical.
`ds1v7` / `ds1shiftv7` = rebuilt in run 11 (+ V7 own previous-lap cols: lap_path_sec / cov / age, lap_fill_path_sec; MS_FEAT_DIR=ms_features_v7); baselines identical.
`ds1v6` / `ds1shiftv6` = v4/v5 features rebuilt in run 8 (+ V6 current-link cols: next stop id, current-link live m5 / hist / fraction left); baselines identical.
`ds1v5` / `ds1shiftv5` = v4 features rebuilt in run 7 (+ own-vs-hist cols); baselines identical. `ds1v5k3` = ds1v5 with train rows at 3% of snapshots (3x) instead of 1%.
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


def _rows(res, keep):
    lines = []
    for (tag, seed, arm_name), m in sorted(res.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] != "baseline", kv[0][2])):
        if keep(tag) is False:
            continue
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
    return lines


def main():
    res = {}
    for p in sorted((HERE / "results").glob("*.json")):
        m = json.loads(p.read_text())
        tag = p.stem.rsplit("_s", 1)[0][len(m["arm"]):].lstrip("_") or "-"
        res[(tag, m["seed"], m["arm"])] = m
    cur = _rows(res, lambda t: t in CURRENT_TAGS)
    old = _rows(res, lambda t: t not in CURRENT_TAGS)
    (HERE / "LEADERBOARD.md").write_text("\n".join([HEAD] + cur) + "\n")
    tbl = HEAD[HEAD.index("| tag |"):]
    (HERE / "archive").mkdir(exist_ok=True)
    (HERE / "archive" / "LEADERBOARD-old.md").write_text(
        "# Leaderboard — superseded tags (generated by report.py)\n\n"
        "Tag definitions: see the header of ../LEADERBOARD.md and archive/JOURNAL-runs-1-12.md.\n\n"
        + tbl + "\n".join(old) + "\n")
    print("\n".join(cur))


if __name__ == "__main__":
    main()
