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
    "path_own_s": "needs serving work (link store + 5-min position ring, ~1 day) + cold-start gate",
    "path_own_s_drop": "as path_own_s; degrades less when cold",
    "path_own_s_big": "as path_own_s; 255-leaf trees ~2x export size",
    "live_all_s_lr10": "as live_all_s",
    "live_all_s_cold": "diagnostic (cold start)",
    "path_own_s_cold": "diagnostic (cold start)",
    "path_own_s_drop_cold": "diagnostic (cold start)",
    "path_own_s_linkcold": "diagnostic (link store cold)",
}
HEAD = """# Leaderboard

Held-out raw-model metrics (seconds); Δ vs the baseline row with the same tag
(day set / split) and seed. Win = MAE ≤ −3%, p90 not worse, no stops_ahead bucket
worse by > 5%. Regenerate with `python research/model-search/report.py`.

Tags: `ds1lag90w15` = ds1 with live features rebuilt at 90 s detection lag, 15-min link window.
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
