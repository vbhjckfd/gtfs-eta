"""
Auto-retrain gate (step 5 of the quality feedback loop).

Decides whether the live model has degraded enough to warrant an automatic
retrain. This is the *numeric*, authoritative trigger — distinct from the AI
diagnosis's advisory `recommend_retrain` flag (scripts/diagnose.py).

It evaluates a **rolling window** of recent daily quality reports, not a single
day, so one noisy day (a holiday, a feed outage) can't trigger a retrain. A
breach fires when the rolling mean MAE exceeds a ceiling, or rolling mean
coverage drops below a floor. Thresholds are env-tunable — the defaults are
deliberately conservative placeholders to set sensibly once a week of live data
exists (offline test MAE was ~123s; live MAE runs higher).

Pure and side-effect-free: dispatching the retrain + cooldown bookkeeping lives
in scripts/score_quality.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Defaults — override via env in the workflow once real live numbers are known.
DEFAULT_MAE_LIMIT_SEC = float(os.environ.get("GATE_MAE_LIMIT_SEC", "210"))
DEFAULT_COVERAGE_MIN = float(os.environ.get("GATE_COVERAGE_MIN", "0.55"))
DEFAULT_MIN_DAYS = int(os.environ.get("GATE_MIN_DAYS", "3"))
DEFAULT_WINDOW_DAYS = int(os.environ.get("GATE_WINDOW_DAYS", "7"))


@dataclass
class GateResult:
    breach: bool
    reason: str
    n_days: int
    rolling_mae_sec: float | None
    rolling_coverage: float | None


def evaluate_gate(
    reports: list[dict],
    *,
    mae_limit_sec: float = DEFAULT_MAE_LIMIT_SEC,
    coverage_min: float = DEFAULT_COVERAGE_MIN,
    min_days: int = DEFAULT_MIN_DAYS,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> GateResult:
    """Evaluate the retrain gate over recent daily reports.

    `reports` is any iterable of per-day report dicts (as produced by
    src.scoring.score_date). Order doesn't matter — we sort by date and take the
    most recent `window_days` that scored OK.
    """
    ok = [r for r in reports if r.get("status") == "ok" and r.get("overall")]
    ok.sort(key=lambda r: str(r.get("date")))
    window = ok[-window_days:]

    if len(window) < min_days:
        return GateResult(
            breach=False,
            reason=f"insufficient data: {len(window)} scored day(s) < {min_days} required",
            n_days=len(window),
            rolling_mae_sec=None,
            rolling_coverage=None,
        )

    maes = [r["overall"]["mae_sec"] for r in window]
    covs = [r.get("coverage_frac", 0.0) for r in window]
    mean_mae = sum(maes) / len(maes)
    mean_cov = sum(covs) / len(covs)

    reasons = []
    if mean_mae > mae_limit_sec:
        reasons.append(f"rolling MAE {mean_mae:.0f}s > limit {mae_limit_sec:.0f}s")
    if mean_cov < coverage_min:
        reasons.append(f"rolling coverage {mean_cov:.0%} < floor {coverage_min:.0%}")

    breach = bool(reasons)
    reason = (
        "; ".join(reasons)
        if breach
        else f"healthy: MAE {mean_mae:.0f}s ≤ {mae_limit_sec:.0f}s, "
        f"coverage {mean_cov:.0%} ≥ {coverage_min:.0%} over {len(window)} day(s)"
    )
    return GateResult(
        breach=breach,
        reason=reason,
        n_days=len(window),
        rolling_mae_sec=round(mean_mae, 1),
        rolling_coverage=round(mean_cov, 3),
    )
