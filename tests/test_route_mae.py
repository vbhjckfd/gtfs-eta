"""Unit tests for the two wrong-day guards in scripts/route_mae.py.

route-mae.yml's `schedule:` cron is meant to fire at 22:00 UTC, once Lviv's
service day has closed, and --date defaults to today's UTC date on that
assumption. GitHub's own schedule trigger is unreliable on low-activity
repos and routinely drifts past midnight UTC -- when that happens "today"
is the day that just *started*, and publishing it under quality/<date>.json
would lock a few hours of data in as the whole day (see score-quality.yml's
shared idempotency guard).

_default_date is the primary guard: a wall-clock rule applied before
score_date is even called (before 21:00 UTC, the closed day is yesterday), since a too-early run can come back
status="no_predictions"/"no_matches"/"no_actuals" just as easily as a thin
"ok" report, and every one of those still gets published deliberately.
_day_looks_complete is the second-line defense against an explicit --date
that still comes back thin for some other reason.

Hermetic: no R2/GitHub credentials or network involved -- both guards are
pure.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from datetime import datetime, timezone
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


def _at(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 8, 31, hour, minute, tzinfo=timezone.utc)


def test_the_intended_fire_time_scores_today():
    assert route_mae._default_date(_at(22, 0)) == "2026-08-31"


def test_a_bit_early_still_scores_today():
    assert route_mae._default_date(_at(21, 0)) == "2026-08-31"


def test_just_before_the_floor_scores_yesterday():
    assert route_mae._default_date(_at(20, 59)) == "2026-08-30"


def test_a_run_past_midnight_scores_the_day_that_closed():
    # The cron has fired at 00:01, 00:11, 00:15, 00:22, 02:59 and 05:59 UTC;
    # each time "today" was the wrong day and yesterday was the right one.
    for hour, minute in [(0, 1), (0, 15), (2, 59), (5, 59)]:
        assert route_mae._default_date(_at(hour, minute)) == "2026-08-30"


def test_crosses_a_month_boundary():
    first = datetime(2026, 9, 1, 0, 11, tzinfo=timezone.utc)
    assert route_mae._default_date(first) == "2026-08-31"


def test_dropped_joins_section_renders_counts():
    report = {
        "past_crossing_dropped": {"n": 47713, "frac": 0.0733, "tolerance_sec": 60},
        "implausible_dropped": {"n": 13835, "frac": 0.0213, "threshold_sec": 3600},
    }
    text = "\n".join(route_mae._dropped_joins_section(report))
    assert "## Dropped joins" in text
    assert "| 47,713 | 7.3% |" in text
    assert "> 3600s" in text
    assert "trip not running" not in text  # absent key → no row


def test_dropped_joins_section_absent_on_old_reports():
    assert route_mae._dropped_joins_section({"overall": {}}) == []
