"""Unit tests for src/snapshots.py's repeated-feed dedup.

The collector keys raw/ objects by fetch wall-clock, so the same upstream feed
(same header timestamp) can be stored under two keys — on 2026-10-12..14 every
snapshot was. drop_repeated_snapshots must keep one copy per feed timestamp.
"""

from __future__ import annotations

import os

for _k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
    os.environ.setdefault(_k, "test")

from datetime import datetime, timedelta, timezone  # noqa: E402

from src.snapshots import drop_repeated_snapshots, thin_dense_hours  # noqa: E402


def _snap(sec: int, vehicles=("v1", "v2")) -> list[dict]:
    ts = datetime(2026, 10, 13, 10, 0, sec, tzinfo=timezone.utc)
    return [{"timestamp": ts, "vehicle_id": v} for v in vehicles]


def test_repeated_feed_timestamp_is_dropped():
    a, b = _snap(2), _snap(8)
    out = list(drop_repeated_snapshots([a, _snap(2), b, _snap(8), _snap(8)]))
    assert out == [a, b]


def test_distinct_snapshots_are_kept_in_order():
    snaps = [_snap(s) for s in (2, 8, 13, 18)]
    assert list(drop_repeated_snapshots(snaps)) == snaps


def test_a_repeat_out_of_order_is_still_dropped():
    # Two overlapping writers: keys sort by wall-clock, so a repeat need not
    # sit right after its original.
    out = list(drop_repeated_snapshots([_snap(2), _snap(8), _snap(2)]))
    assert [r[0]["timestamp"].second for r in out] == [2, 8]


def test_empty_snapshots_pass_through():
    # _fetch_and_parse returns [] on a parse error; that must not poison dedup.
    assert list(drop_repeated_snapshots([[], _snap(2), []])) == [[], _snap(2), []]


# ── thin_dense_hours ─────────────────────────────────────────────────────────

def _hour(fetch_secs: list[float], feed_secs: list[int], hour: int = 10):
    """Keys (fetch wall-clock) and their parsed snapshots (feed header ts)."""
    base = datetime(2026, 10, 13, hour, tzinfo=timezone.utc)
    keys = [f"raw/2026-10-13/{(base + timedelta(seconds=s)).isoformat()[:23]}Z.pb"
            for s in fetch_secs]
    snaps = [[{"timestamp": base + timedelta(seconds=s), "vehicle_id": "v1"}]
             for s in feed_secs]
    return keys, snaps


def _secs(snaps) -> list[int]:
    return [int(r[0]["timestamp"].second + 60 * r[0]["timestamp"].minute) for r in snaps]


def test_dense_hour_is_thinned_to_the_min_gap():
    # Two pollers 5 s apart, each catching a fresh ~5 s feed.
    keys, snaps = _hour([0, 5, 10, 15, 20, 25], [0, 5, 10, 15, 21, 26])
    out = list(thin_dense_hours(keys, snaps, max_per_hour=3, min_gap_sec=7))
    assert _secs(out) == [0, 10, 21]


def test_normal_hour_is_left_untouched():
    # Cron-boundary 5 s gaps happen on normal days too; keep them there.
    keys, snaps = _hour([0, 5, 15, 25], [0, 5, 15, 25])
    assert list(thin_dense_hours(keys, snaps, max_per_hour=4, min_gap_sec=7)) == snaps


def test_only_the_dense_hour_is_thinned():
    k9, s9 = _hour([0, 5, 10], [0, 5, 10], hour=9)
    k10, s10 = _hour([0, 5, 10, 15, 20], [0, 5, 10, 15, 20], hour=10)
    out = list(thin_dense_hours(k9 + k10, s9 + s10, max_per_hour=4, min_gap_sec=7))
    assert len(out) == len(s9) + 3          # hour 9 whole; hour 10 keeps 0, 10, 20


def test_thinning_then_dedup_leaves_one_copy_per_kept_feed():
    # Doubled hour: each feed stored twice, 0.3 s apart.
    keys, snaps = _hour([0, 0.3, 5, 5.3, 10, 10.3], [0, 0, 5, 5, 10, 10])
    out = list(drop_repeated_snapshots(thin_dense_hours(keys, snaps, max_per_hour=3,
                                                        min_gap_sec=7)))
    assert _secs(out) == [0, 10]


def test_empty_snapshots_do_not_reset_the_gap():
    keys, snaps = _hour([0, 3, 5, 10], [0, 0, 5, 10])
    snaps[1] = []                           # parse error
    out = [s for s in thin_dense_hours(keys, snaps, max_per_hour=3, min_gap_sec=7) if s]
    assert _secs(out) == [0, 10]
