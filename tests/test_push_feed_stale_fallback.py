"""Unit tests for the stale-on-error VP fallback in scripts/push_feed.py.

When the upstream vehicle-position fetch fails after all retries, the daemon
should reuse the last feed it successfully decoded — but only while it's within
STALE_MAX_AGE_MS. Older than that (or never cached) → re-raise the error.

These are hermetic: requests.get and time.sleep are monkeypatched, so nothing
touches the network and backoff doesn't actually sleep.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import requests
from google.protobuf.message import DecodeError
from google.transit import gtfs_realtime_pb2

# push_feed reads R2 credentials from the environment at import time; provide
# dummies so the import succeeds in CI without a populated .env. (load_dotenv
# does not override already-set vars, so these win.)
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


def _make_vp_bytes(n_entities: int = 1) -> bytes:
    """A minimal but valid VehiclePositions protobuf payload."""
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.header.gtfs_realtime_version = "2.0"
    for i in range(n_entities):
        ent = feed.entity.add()
        ent.id = f"v{i}"
        ent.vehicle.trip.trip_id = f"{1000 + i}_1_1"
    return feed.SerializeToString()


class _FakeResp:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self) -> None:  # 2xx — nothing to raise
        pass


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Reset the module-level cache and neutralize backoff sleeps per test."""
    push_feed.__reset_cache()
    monkeypatch.setattr(push_feed.time, "sleep", lambda *_: None)
    yield
    push_feed.__reset_cache()


def _patch_get(monkeypatch, fn):
    monkeypatch.setattr(push_feed.requests, "get", fn)


# ── (a) serves cached feed when a later fetch fails within the window ─────────

def test_serves_cached_feed_within_stale_window(monkeypatch):
    good = _make_vp_bytes(3)

    # First call succeeds and populates the cache.
    _patch_get(monkeypatch, lambda *a, **k: _FakeResp(good))
    assert push_feed._get_vp_bytes() == good
    assert push_feed._vp_cache is not None

    # Next fetch fails on every retry, but the cache is fresh → serve it.
    def boom(*a, **k):
        raise requests.exceptions.ConnectionError("upstream down")

    _patch_get(monkeypatch, boom)
    assert push_feed._get_vp_bytes() == good


def test_decode_failure_also_falls_back_to_cache(monkeypatch):
    good = _make_vp_bytes(2)
    _patch_get(monkeypatch, lambda *a, **k: _FakeResp(good))
    assert push_feed._get_vp_bytes() == good

    # HTTP succeeds but the body is not valid protobuf → DecodeError path.
    _patch_get(monkeypatch, lambda *a, **k: _FakeResp(b"\xff\xff not protobuf"))
    assert push_feed._get_vp_bytes() == good


# ── (b) re-raises when the cache is too old or empty ──────────────────────────

def test_rethrows_when_cache_older_than_window(monkeypatch):
    good = _make_vp_bytes(1)
    _patch_get(monkeypatch, lambda *a, **k: _FakeResp(good))
    push_feed._get_vp_bytes()

    # Age the cached entry just past the stale window.
    import time

    push_feed._vp_cache["at"] = int(time.time() * 1000) - (
        push_feed.STALE_MAX_AGE_MS + 1000
    )

    def boom(*a, **k):
        raise requests.exceptions.ConnectionError("upstream down")

    _patch_get(monkeypatch, boom)
    with pytest.raises(requests.exceptions.ConnectionError):
        push_feed._get_vp_bytes()


def test_rethrows_when_cache_empty(monkeypatch):
    # No successful fetch has ever happened (cache reset by the autouse fixture).
    assert push_feed._vp_cache is None

    def boom(*a, **k):
        raise requests.exceptions.ConnectionError("upstream down")

    _patch_get(monkeypatch, boom)
    with pytest.raises(requests.exceptions.ConnectionError):
        push_feed._get_vp_bytes()
