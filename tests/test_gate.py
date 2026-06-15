"""Unit tests for the auto-retrain gate (src/gate.py)."""

from __future__ import annotations

from src.gate import evaluate_gate


def _day(date: str, mae: float, coverage: float) -> dict:
    return {
        "date": date,
        "status": "ok",
        "coverage_frac": coverage,
        "overall": {"mae_sec": mae},
    }


def test_insufficient_data_never_breaches():
    reports = [_day("2026-06-14", 999, 0.1), _day("2026-06-15", 999, 0.1)]
    r = evaluate_gate(reports, min_days=3)
    assert r.breach is False
    assert "insufficient data" in r.reason


def test_healthy_window_does_not_breach():
    reports = [_day(f"2026-06-1{d}", 120, 0.8) for d in range(1, 6)]
    r = evaluate_gate(reports, mae_limit_sec=210, coverage_min=0.55)
    assert r.breach is False
    assert r.rolling_mae_sec == 120


def test_high_mae_breaches():
    reports = [_day(f"2026-06-1{d}", 260, 0.8) for d in range(1, 6)]
    r = evaluate_gate(reports, mae_limit_sec=210)
    assert r.breach is True
    assert "MAE" in r.reason


def test_low_coverage_breaches():
    reports = [_day(f"2026-06-1{d}", 120, 0.40) for d in range(1, 6)]
    r = evaluate_gate(reports, coverage_min=0.55)
    assert r.breach is True
    assert "coverage" in r.reason


def test_window_takes_most_recent_days():
    # Old bad days fall outside a 3-day window; recent days are healthy.
    reports = [
        _day("2026-06-01", 500, 0.1),
        _day("2026-06-02", 500, 0.1),
        _day("2026-06-13", 120, 0.8),
        _day("2026-06-14", 120, 0.8),
        _day("2026-06-15", 120, 0.8),
    ]
    r = evaluate_gate(reports, window_days=3, min_days=3, mae_limit_sec=210)
    assert r.n_days == 3
    assert r.breach is False


def test_non_ok_reports_ignored():
    reports = [
        {"date": "2026-06-13", "status": "no_predictions", "overall": None},
        _day("2026-06-14", 260, 0.8),
        _day("2026-06-15", 260, 0.8),
        _day("2026-06-16", 260, 0.8),
    ]
    r = evaluate_gate(reports, min_days=3, mae_limit_sec=210)
    assert r.n_days == 3
    assert r.breach is True
