"""Unit tests for push-on-change polling in scripts/push_feed.py.

With --poll the daemon re-reads the upstream vehicle-position feed between
pushes and publishes as soon as its vehicles change, instead of on a fixed
10 s clock that ran out of step with upstream's own refresh. A push still
happens every --loop seconds without a change, so the header keeps moving.

Hermetic: clock, sleep, fetch and push are fakes; no network or credentials.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from google.transit import gtfs_realtime_pb2

for _k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
    os.environ.setdefault(_k, "test")

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
_spec = importlib.util.spec_from_file_location(
    "push_feed_poll", _ROOT / "scripts" / "push_feed.py"
)
push_feed = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(push_feed)


def _vp_bytes(lat: float, header_ts: int = 1_786_000_000) -> bytes:
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.header.gtfs_realtime_version = "2.0"
    feed.header.timestamp = header_ts
    ent = feed.entity.add()
    ent.id = "v1"
    ent.vehicle.position.latitude = lat
    ent.vehicle.position.longitude = 24.0
    return feed.SerializeToString()


class _Clock:
    """Fake monotonic clock; sleep() and push() advance it."""

    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _run(upstream, *, clock, push_cost=5.0, **kwargs):
    """Run the loop against a scripted upstream: upstream(t) -> bytes."""
    pushes: list[tuple[float, bytes | None]] = []

    def push(vp):
        pushes.append((clock.now, vp))
        clock.now += push_cost

    n, failed = push_feed._run_loop(
        push, lambda n: None, fetch=lambda: upstream(clock.now),
        clock=clock, sleep=clock.sleep, **kwargs,
    )
    return pushes, n, failed


def test_fingerprint_ignores_the_header_but_not_the_vehicles():
    a = push_feed._vp_fingerprint(_vp_bytes(49.80, header_ts=1))
    assert a == push_feed._vp_fingerprint(_vp_bytes(49.80, header_ts=2))
    assert a != push_feed._vp_fingerprint(_vp_bytes(49.81, header_ts=1))


def test_pushes_soon_after_upstream_changes():
    clock = _Clock()
    # Upstream publishes a new snapshot at t=0, 13, 26, ...
    upstream = lambda t: _vp_bytes(49.0 + int(t // 13) / 100)
    pushes, n, _ = _run(upstream, clock=clock, loop=20, poll=2, duration=60)

    times = [t for t, _ in pushes]
    # Each snapshot is picked up within one poll of appearing, never a whole
    # --loop period later, and no heartbeat push gets in its way.
    for change in (13, 26, 39, 52):
        assert any(change <= t <= change + 2 for t in times), (change, times)
    # Every push carries the bytes that were polled, so it is not re-fetched.
    assert all(vp is not None for _, vp in pushes)


def test_unchanged_upstream_still_pushes_every_loop_seconds():
    clock = _Clock()
    frozen = _vp_bytes(49.8)
    pushes, _, _ = _run(lambda t: frozen, clock=clock, loop=20, poll=2, duration=70)

    times = [t for t, _ in pushes]
    assert times[0] == 0
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert gaps and all(20 <= g <= 22 for g in gaps), gaps


def test_failed_poll_falls_back_to_a_plain_push_on_the_heartbeat():
    clock = _Clock()

    def upstream(t):
        raise ConnectionError("upstream down")

    pushes, _, _ = _run(upstream, clock=clock, loop=20, poll=2, duration=45)

    # Only heartbeat pushes, each told to fetch for itself (vp=None) so the
    # stale fallback and error reporting in _push_once still apply.
    assert [vp for _, vp in pushes] == [None] * len(pushes)
    gaps = [b - a for a, b in zip([t for t, _ in pushes], [t for t, _ in pushes][1:])]
    assert gaps and all(20 <= g <= 22 for g in gaps), gaps


def test_a_failed_push_is_counted_and_the_loop_goes_on(monkeypatch):
    monkeypatch.setattr(push_feed, "_report", lambda *a, **k: None)
    clock = _Clock()
    calls = []

    def push(vp):
        calls.append(vp)
        clock.now += 5
        if len(calls) == 1:
            raise RuntimeError("R2 hiccup")

    n, failed = push_feed._run_loop(
        push, lambda n: None, loop=10, poll=2, duration=30,
        fetch=lambda: _vp_bytes(49.0 + clock.now / 100),
        clock=clock, sleep=clock.sleep,
    )
    assert failed == 1
    assert n == len(calls) > 1


def test_without_poll_keeps_the_fixed_cadence_and_count():
    clock = _Clock()
    fetched = []
    pushes, n, _ = _run(
        lambda t: fetched.append(t), clock=clock, loop=10, count=3,
    )
    assert n == 3
    assert [t for t, _ in pushes] == [0, 10, 20]
    # The old mode never polls; each push fetches for itself.
    assert fetched == [] and [vp for _, vp in pushes] == [None] * 3


def test_checkpoints_follow_the_push_count():
    clock = _Clock()
    seen = []
    push_feed._run_loop(
        lambda vp: setattr(clock, "now", clock.now + 5), seen.append,
        loop=10, poll=2, duration=40,
        fetch=lambda: _vp_bytes(49.0 + clock.now / 100),
        clock=clock, sleep=clock.sleep,
    )
    assert seen == list(range(1, len(seen) + 1))
