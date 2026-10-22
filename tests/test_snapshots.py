"""Unit tests for src/snapshots.py's repeated-feed dedup.

The collector keys raw/ objects by fetch wall-clock, so the same upstream feed
(same header timestamp) can be stored under two keys — on 2026-10-12..14 every
snapshot was. drop_repeated_snapshots must keep one copy per feed timestamp.
"""

from __future__ import annotations

import os

for _k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
    os.environ.setdefault(_k, "test")

from datetime import datetime, timezone  # noqa: E402

from src.snapshots import drop_repeated_snapshots  # noqa: E402


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
