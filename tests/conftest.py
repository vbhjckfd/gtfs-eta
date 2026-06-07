"""Shared fixtures for the worker smoke suite.

The suite hits a *live* GTFS-RT TripUpdates endpoint and validates that it
behaves as a drop-in replacement for the upstream Lviv feed. Both URLs are
configurable via environment variables so the same tests can run against
production, a preview deployment, or `pywrangler dev` running locally.

    SMOKE_URL      endpoint under test   (default: the deployed worker)
    SMOKE_REF_URL  upstream reference    (default: track.ua-gis.com)
    SMOKE_TIMEOUT  per-request seconds   (default: 30)
"""

from __future__ import annotations

import os

import pytest
import requests
from google.transit import gtfs_realtime_pb2

WORKER_URL = os.environ.get(
    "SMOKE_URL", "https://gtfs-eta.vbhjckfd.workers.dev"
)
REF_URL = os.environ.get(
    "SMOKE_REF_URL", "https://track.ua-gis.com/gtfs/lviv/trip_updates"
)
TIMEOUT = float(os.environ.get("SMOKE_TIMEOUT", "30"))


def _fetch(url: str) -> requests.Response:
    return requests.get(url, timeout=TIMEOUT)


def _parse(content: bytes) -> gtfs_realtime_pb2.FeedMessage:
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(content)
    return feed


@pytest.fixture(scope="session")
def worker_response() -> requests.Response:
    """Raw HTTP response from the endpoint under test (fetched once)."""
    try:
        return _fetch(WORKER_URL)
    except requests.RequestException as exc:
        pytest.fail(f"Could not reach worker at {WORKER_URL}: {exc}")


@pytest.fixture(scope="session")
def worker_feed(worker_response) -> gtfs_realtime_pb2.FeedMessage:
    """Parsed FeedMessage from the endpoint under test.

    Skips downstream tests (rather than erroring) when the endpoint is not
    serving a parseable feed — the reachability/validity tests own that
    failure so we don't report the same outage a dozen times.
    """
    if worker_response.status_code != 200:
        body = worker_response.text[:200]
        pytest.skip(
            f"endpoint returned HTTP {worker_response.status_code}: {body!r}"
        )
    try:
        return _parse(worker_response.content)
    except Exception as exc:  # noqa: BLE001 - any decode failure means skip
        pytest.skip(f"endpoint body is not a parseable FeedMessage: {exc}")


@pytest.fixture(scope="session")
def reference_feed() -> gtfs_realtime_pb2.FeedMessage:
    """Parsed FeedMessage from the upstream reference (skips if unavailable)."""
    try:
        resp = _fetch(REF_URL)
    except requests.RequestException as exc:
        pytest.skip(f"reference endpoint unreachable: {exc}")
    if resp.status_code != 200:
        pytest.skip(f"reference endpoint returned HTTP {resp.status_code}")
    try:
        return _parse(resp.content)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"reference body not parseable: {exc}")
