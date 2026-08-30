"""Unit tests for the day-completeness guard in scripts/route_mae.py.

route-mae.yml's `schedule:` cron is meant to fire at 22:00 UTC, once Lviv's
service day has closed, and --date defaults to date.today() on that
assumption. GitHub's own schedule trigger is unreliable on low-activity
repos and has drifted past midnight UTC -- when that happens "today" quietly
becomes the day that just *started*, and publishing it under
quality/<date>.json would lock a few hours of data in as the whole day (see
score-quality.yml's shared idempotency guard). _day_looks_complete is the
predicate that catches this before anything gets published.

Hermetic: no R2/GitHub credentials or network involved -- the guard is pure.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

for _k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
    os.environ.setdefault(_k, "test")

# scripts/ isn't an importable package, so load route_mae.py by path.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
_spec = importlib.util.spec_from_file_location(
    "route_mae", _ROOT / "scripts" / "route_mae.py"
)
route_mae = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(route_mae)


def _report(hours: list[str]) -> dict:
    return {"status": "ok", "by_hour": {h: {} for h in hours}}


def test_a_full_service_day_looks_complete():
    hours = [str(h) for h in range(5, 24)]  # 05:00-23:00, 19 hours
    assert route_mae._day_looks_complete(_report(hours))


def test_the_first_hour_after_a_midnight_drift_does_not():
    # This is exactly what quality/2026-08-27.json looked like: the cron fired
    # at 02:59 UTC, "today" became 2026-08-27, and only its first hour existed.
    assert not route_mae._day_looks_complete(_report(["5"]))


def test_a_partial_morning_still_does_not():
    # quality/2026-08-28.json: fired at 05:59 UTC, a few hours in.
    assert not route_mae._day_looks_complete(_report(["5", "6"]))


def test_right_at_the_floor_counts_as_complete():
    hours = [str(h) for h in range(5, 5 + route_mae.MIN_HOURS_COVERED)]
    assert len(hours) == route_mae.MIN_HOURS_COVERED
    assert route_mae._day_looks_complete(_report(hours))


def test_one_hour_short_of_the_floor_does_not():
    hours = [str(h) for h in range(5, 5 + route_mae.MIN_HOURS_COVERED - 1)]
    assert not route_mae._day_looks_complete(_report(hours))


def test_a_missing_by_hour_key_does_not_crash():
    assert not route_mae._day_looks_complete({"status": "ok"})
