"""Unit tests for the cross-run stationary-anchor carryover in scripts/push_feed.py.

The daemon is restarted by every cron tick (`--loop 10 --count 33`, ~5.5 min),
so without this its in-memory trackers would cap stationary_sec at ~330 s while
the model was trained on a whole day's trajectories (values reach hours). That
mismatch would land precisely on the stopped vehicles the feature exists to
catch, so the anchors are round-tripped through R2 between runs.

Hermetic: the R2 client is a fake, no network or credentials involved.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import time

import pytest
from pathlib import Path

for _k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
    os.environ.setdefault(_k, "test")

# scripts/ isn't an importable package, so load push_feed.py by path.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
_spec = importlib.util.spec_from_file_location(
    "push_feed", _ROOT / "scripts" / "push_feed.py"
)
push_feed = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(push_feed)


class FakeR2:
    def __init__(self, payload: dict | None = None):
        self.payload = payload
        self.put: dict | None = None

    def get_object(self, **kw):
        if self.payload is None:
            raise RuntimeError("NoSuchKey")
        return {"Body": io.BytesIO(json.dumps(self.payload).encode())}

    def put_object(self, **kw):
        self.put = kw


class BrokenR2:
    def get_object(self, **kw):
        raise RuntimeError("r2 down")

    def put_object(self, **kw):
        raise RuntimeError("r2 down")


def test_fresh_anchor_is_carried_over():
    now = time.time()
    client = FakeR2({"still": {"v1": [now - 120.0, 500.0, "trip-a"]}})
    trackers = push_feed._load_tracker_state(client)

    assert trackers["v1"]["still"] == (now - 120.0, 500.0, "trip-a")
    # Off-route debounce counters are deliberately NOT carried — they are a
    # consecutive-push debounce and must restart cold.
    assert trackers["v1"]["off"] == 0
    assert trackers["v1"]["on"] == 0
    assert trackers["v1"]["status"] == "on_route"


def test_stale_and_malformed_anchors_are_dropped():
    now = time.time()
    client = FakeR2({"still": {
        "fresh": [now - 60.0, 10.0, "t"],
        "stale": [now - push_feed.TRACKER_STATE_MAX_AGE_SEC - 1, 10.0, "t"],
        "short": ["only-one-field"],
        "junk": ["not-a-number", 10.0, "t"],
    }})
    assert sorted(push_feed._load_tracker_state(client)) == ["fresh"]


def test_missing_or_broken_state_starts_cold_without_raising():
    assert push_feed._load_tracker_state(FakeR2(None)) == {}
    assert push_feed._load_tracker_state(BrokenR2()) == {}


def test_save_round_trips_through_load():
    now = time.time()
    saved = FakeR2()
    trackers = {
        "v1": {"status": "on_route", "off": 0, "on": 0,
               "still": (now, 42.0, "trip-a"), "pos": (now, 42.0, "trip-a")},
        # No anchor yet (never seen twice) — nothing to persist for this one.
        "v2": {"status": "on_route", "off": 0, "on": 0},
    }
    push_feed._save_tracker_state(saved, trackers)
    assert saved.put["Key"] == push_feed.TRACKER_STATE_KEY

    reloaded = push_feed._load_tracker_state(FakeR2(json.loads(saved.put["Body"])))
    assert sorted(reloaded) == ["v1"]
    assert reloaded["v1"]["still"] == (now, 42.0, "trip-a")
    assert reloaded["v1"]["pos"] == (now, 42.0, "trip-a")


def test_carried_position_lets_speed_measure_on_the_first_push():
    """Without pos, every restart served one feed with SPEED_UNKNOWN throughout."""
    from src.inference import SPEED_UNKNOWN, progress_speed

    now = time.time()
    client = FakeR2({"pos": {"v1": [now, 100.0, "trip-a"]}})
    trackers = push_feed._load_tracker_state(client)

    speed = progress_speed(trackers, "v1", "trip-a", 250.0, now + 15.0)
    assert speed == pytest.approx(10.0)

    # A pos left by a long-dead run is still rejected by the existing gap guard.
    stale = push_feed._load_tracker_state(FakeR2({"pos": {"v1": [now - 600, 100.0, "trip-a"]}}))
    assert progress_speed(stale, "v1", "trip-a", 250.0, now) == SPEED_UNKNOWN


def test_save_failure_is_tolerated():
    now = time.time()
    push_feed._save_tracker_state(
        BrokenR2(),
        {"v1": {"status": "on_route", "off": 0, "on": 0, "still": (now, 1.0, "t")}},
    )  # must not raise — losing anchors costs accuracy, not the feed
