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
import time

import pytest
import requests
from google.protobuf.message import DecodeError
from google.transit import gtfs_realtime_pb2

WORKER_URL = os.environ.get(
    "SMOKE_URL", "https://gtfs-eta.vbhjckfd.workers.dev"
)
REF_URL = os.environ.get(
    "SMOKE_REF_URL", "https://track.ua-gis.com/gtfs/lviv/trip_updates"
)
VP_URL = os.environ.get(
    "SMOKE_VP_URL", "https://track.ua-gis.com/gtfs/lviv/vehicle_position"
)
OUR_VP_URL = os.environ.get(
    "SMOKE_OUR_VP_URL", "https://eta.lad.lviv.ua/feed/vehicle_positions.pb"
)
TIMEOUT = float(os.environ.get("SMOKE_TIMEOUT", "30"))


def _fetch(url: str, _retries: int = 5) -> requests.Response:
    for attempt in range(_retries):
        try:
            return requests.get(url, timeout=TIMEOUT)
        except requests.exceptions.RequestException:
            if attempt == _retries - 1:
                raise
            time.sleep(0.2 * (2 ** attempt))
    raise RuntimeError("unreachable")


def _fetch_proto(url: str, _retries: int = 5) -> gtfs_realtime_pb2.FeedMessage:
    for attempt in range(_retries):
        try:
            resp = requests.get(url, timeout=TIMEOUT)
            resp.raise_for_status()
            return _parse(resp.content)
        except (requests.exceptions.RequestException, DecodeError):
            if attempt == _retries - 1:
                raise
            time.sleep(0.2 * (2 ** attempt))
    raise RuntimeError("unreachable")


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
def vehicle_positions_feed() -> gtfs_realtime_pb2.FeedMessage:
    """Parsed VehiclePositions feed from the upstream source (skips if unavailable)."""
    try:
        return _fetch_proto(VP_URL)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"vehicle positions feed unavailable: {exc}")


@pytest.fixture(scope="session")
def our_vp_feed() -> gtfs_realtime_pb2.FeedMessage:
    """Parsed VehiclePositions feed from our own R2 publication (skips if unavailable)."""
    try:
        return _fetch_proto(OUR_VP_URL)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"our vehicle positions feed unavailable: {exc}")


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
        ctype = resp.headers.get("content-type", "<no content-type>")
        preview = resp.content[:120]
        pytest.skip(
            f"reference body not parseable ({exc}); "
            f"content-type={ctype!r}, first bytes={preview!r}"
        )
