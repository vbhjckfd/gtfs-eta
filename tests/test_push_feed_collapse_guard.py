"""Unit tests for the collapse guard and status sidecar in scripts/push_feed.py.

A pass that produces a small fraction of the previous pass's trips is almost
always an upstream hiccup, not reality. Publishing it empties every consumer's
timetable at once (they filter arrivals already in the past), while keeping the
previous blob costs only freshness — so the daemon skips the upload, reports,
and records the skip in feed/status.json.

Hermetic: the R2 client is a fake, inference is monkeypatched, and no network
or credentials are involved.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from google.transit import gtfs_realtime_pb2

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


def _make_tu_bytes(n_entities: int, timestamp: int = 1_786_000_000) -> bytes:
    """A minimal but valid TripUpdates protobuf with *n_entities* trips."""
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.header.gtfs_realtime_version = "2.0"
    feed.header.timestamp = timestamp
    for i in range(n_entities):
        ent = feed.entity.add()
        ent.id = f"t{i}"
        ent.trip_update.trip.trip_id = f"{1000 + i}_1_1"
    return feed.SerializeToString()


class _FakeClient:
    """Records put_object calls by key."""

    def __init__(self):
        self.puts: list[dict] = []

    def put_object(self, **kwargs):
        self.puts.append(kwargs)

    def keys(self) -> list[str]:
        return [p["Key"] for p in self.puts]

    def body(self, key: str):
        for put in reversed(self.puts):
            if put["Key"] == key:
                return put["Body"]
        raise KeyError(key)


@pytest.fixture(autouse=True)
def _isolate():
    push_feed.__reset_cache()
    yield
    push_feed.__reset_cache()


def _run_push(monkeypatch, client, n_entities: int, stats: dict | None = None):
    """Drive _push_once with inference stubbed to emit *n_entities* trips."""
    monkeypatch.setattr(push_feed, "_get_vp_bytes", lambda: b"")

    def fake_inference(gtfs_data, model_data, trackers, vp_bytes, **kwargs):
        if kwargs.get("stats") is not None and stats:
            kwargs["stats"].update(stats)
        return _make_tu_bytes(n_entities), b"vp"

    monkeypatch.setattr(push_feed, "run_inference", fake_inference)
    push_feed._push_once(client, {}, {}, {})


# ── _should_publish ──────────────────────────────────────────────────────────

def test_first_push_of_a_run_always_publishes():
    # Nothing to compare against yet — a fresh process must not sit silent.
    assert push_feed._should_publish(None, 0) is True


def test_small_previous_count_disables_the_ratio():
    # Night service: 3 trips one pass, 1 the next is a normal fluctuation, not
    # a collapse, and the ratio would block it.
    assert push_feed._should_publish(3, 1) is True


def test_collapse_is_blocked():
    assert push_feed._should_publish(500, 20) is False


def test_growth_and_mild_shrink_publish():
    assert push_feed._should_publish(500, 600) is True
    assert push_feed._should_publish(500, 400) is True
    # Exactly at the ratio boundary still publishes.
    assert push_feed._should_publish(500, 150) is True


# ── _push_once ───────────────────────────────────────────────────────────────

def test_publishes_feed_and_status(monkeypatch):
    client = _FakeClient()
    _run_push(monkeypatch, client, 500, {"vehicles_in": 300, "vehicles_stale": 2})

    assert push_feed.FEED_KEY in client.keys()
    assert push_feed.VP_FEED_KEY in client.keys()

    status = json.loads(client.body(push_feed.STATUS_KEY))
    assert status["published"] is True
    assert status["entities"] == 500
    assert status["feed_timestamp"] == 1_786_000_000
    assert status["vehicles_in"] == 300
    assert status["vehicles_stale"] == 2


def test_collapsed_pass_keeps_the_previous_blob(monkeypatch):
    client = _FakeClient()
    _run_push(monkeypatch, client, 500)

    collapsed = _FakeClient()
    _run_push(monkeypatch, collapsed, 5, {"vehicles_in": 300, "vehicles_stale": 300})

    # Neither feed was overwritten — consumers keep the last good blob.
    assert push_feed.FEED_KEY not in collapsed.keys()
    assert push_feed.VP_FEED_KEY not in collapsed.keys()

    # …but the skip is visible rather than silent.
    status = json.loads(collapsed.body(push_feed.STATUS_KEY))
    assert status["published"] is False
    assert status["entities"] == 5
    assert "collapsed" in status["detail"]


def test_recovery_publishes_again(monkeypatch):
    client = _FakeClient()
    _run_push(monkeypatch, client, 500)
    _run_push(monkeypatch, client, 5)  # blocked, so the baseline stays at 500

    recovered = _FakeClient()
    _run_push(monkeypatch, recovered, 480)
    assert push_feed.FEED_KEY in recovered.keys()


def test_status_write_failure_does_not_break_the_push(monkeypatch):
    class _StatusFails(_FakeClient):
        def put_object(self, **kwargs):
            if kwargs["Key"] == push_feed.STATUS_KEY:
                raise RuntimeError("R2 down")
            super().put_object(**kwargs)

    client = _StatusFails()
    _run_push(monkeypatch, client, 500)
    assert push_feed.FEED_KEY in client.keys()


# ── _load_prev_entities ──────────────────────────────────────────────────────
#
# The guard's baseline used to die with the process, and this process is
# replaced every ~5.5 min. An upstream outage lasting longer than one cycle
# therefore beat the guard outright — the fresh process had no baseline and
# published the empty feed over the good one. These cover the seed that carries
# the baseline across the restart.

class _FakeHead(_FakeClient):
    """A fake R2 that answers HEAD on FEED_KEY with a stamped, aged object."""

    def __init__(self, metadata: dict | None = None, age_sec: float = 5.0, exc=None):
        super().__init__()
        self._metadata = metadata
        self._age_sec = age_sec
        self._exc = exc

    def head_object(self, **kwargs):
        assert kwargs["Key"] == push_feed.FEED_KEY
        if self._exc is not None:
            raise self._exc
        return {
            "LastModified": datetime.now(timezone.utc) - timedelta(seconds=self._age_sec),
            "Metadata": self._metadata,
        }


def test_seeds_the_baseline_from_the_served_feed():
    client = _FakeHead({"entities": "530", "commit": "abc1234"}, age_sec=20)
    assert push_feed._load_prev_entities(client) == 530


def test_stale_served_feed_does_not_seed():
    # Past the window the daemon itself was likely down (a runner gap) and real
    # service may have wound down since — a baseline from another part of the
    # day would block every publish instead of protecting one.
    client = _FakeHead(
        {"entities": "530"}, age_sec=push_feed.COLLAPSE_SEED_MAX_AGE_SEC + 1
    )
    assert push_feed._load_prev_entities(client) is None


def test_missing_or_unreadable_feed_starts_the_guard_cold():
    assert push_feed._load_prev_entities(_FakeHead(exc=RuntimeError("no such key"))) is None
    # Written by a daemon predating the metadata stamp.
    assert push_feed._load_prev_entities(_FakeHead({"commit": "abc1234"})) is None
    assert push_feed._load_prev_entities(_FakeHead(None)) is None
    assert push_feed._load_prev_entities(_FakeHead({"entities": "not-a-number"})) is None


def test_published_feed_carries_its_entity_count(monkeypatch):
    # The seed reads this back with a HEAD, so the stamp must be on the object.
    client = _FakeClient()
    _run_push(monkeypatch, client, 500)

    feed_put = next(p for p in client.puts if p["Key"] == push_feed.FEED_KEY)
    assert feed_put["Metadata"]["entities"] == "500"


def test_seeded_baseline_blocks_an_empty_first_push(monkeypatch):
    # The 2026-08-26 regression: upstream returned no vehicles for ~12 min, the
    # guard held for the first process's whole lifetime, then the restart
    # published the empty feed and /health went 503.
    push_feed._last_published_entities = push_feed._load_prev_entities(
        _FakeHead({"entities": "530"}, age_sec=230)
    )

    restarted = _FakeClient()
    _run_push(monkeypatch, restarted, 0, {"vehicles_in": 0, "vehicles_stale": 0})

    assert push_feed.FEED_KEY not in restarted.keys()
    status = json.loads(restarted.body(push_feed.STATUS_KEY))
    assert status["published"] is False
    assert "collapsed to 0 trips from 530" in status["detail"]
