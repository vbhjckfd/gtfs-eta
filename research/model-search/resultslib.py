"""Result-file helpers for report.py (split out of the archived harness in run 45: harness.py imported
src.train, which main removed with the legacy model)."""

import json
from pathlib import Path

RESULTS = Path(__file__).parent / "results.jsonl"   # harness-era fits (runs 1-43), one compact JSON line per fit


def load_results() -> dict:
    """name -> metrics dict (name = <arm>[_<tag>]_s<seed>)."""
    if not RESULTS.exists():
        return {}
    return {m["name"]: m for m in map(json.loads, RESULTS.read_text().splitlines()) if m}


def judge(base: dict, cand: dict) -> dict:
    d_mae = (cand["mae"] - base["mae"]) / base["mae"]
    worst_bucket = max(
        (cand["mae_by_stops_ahead"][k] - base["mae_by_stops_ahead"][k])
        / base["mae_by_stops_ahead"][k]
        for k in base["mae_by_stops_ahead"] if k in cand["mae_by_stops_ahead"]
    )
    return {
        "mae_change_pct": 100 * d_mae,
        "p90_change_pct": 100 * (cand["p90_ae"] - base["p90_ae"]) / base["p90_ae"],
        "worst_bucket_change_pct": 100 * worst_bucket,
        "wins": bool(d_mae <= -0.03 and cand["p90_ae"] <= base["p90_ae"]
                     and worst_bucket <= 0.05),
    }
