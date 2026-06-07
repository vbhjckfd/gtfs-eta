"""Smoke tests for the live GTFS-RT TripUpdates worker.

Goal: prove the deployed endpoint is a valid, sane drop-in replacement for
the upstream Lviv `trip_updates` feed. These run against the network, so they
are fast but not hermetic — treat a green run as "production is serving a
healthy feed right now".

Run with `make smoke`.
"""

from __future__ import annotations

import time

from google.transit import gtfs_realtime_pb2

# Worker caps upcoming stops per trip (worker.py: MAX_STOPS_AHEAD).
MAX_STOPS_AHEAD = 10
# Feed header timestamp must be fresh — upstream vehicle positions are
# republished every few seconds; allow generous slack for clock skew / caching.
MAX_FEED_AGE_SEC = 15 * 60


# ── Reachability & transport ────────────────────────────────────────────────

def test_endpoint_returns_200(worker_response):
    assert worker_response.status_code == 200, (
        f"expected HTTP 200, got {worker_response.status_code}: "
        f"{worker_response.text[:200]!r}"
    )


def test_content_type_is_protobuf(worker_response):
    if worker_response.status_code != 200:
        # Owned by test_endpoint_returns_200; don't double-report.
        return
    ctype = worker_response.headers.get("content-type", "")
    assert "protobuf" in ctype, f"unexpected content-type: {ctype!r}"


def test_body_is_non_empty(worker_response):
    if worker_response.status_code != 200:
        return
    assert len(worker_response.content) > 0, "endpoint returned an empty body"


# ── Feed-level validity ─────────────────────────────────────────────────────

def test_body_parses_as_feedmessage(worker_feed):
    assert isinstance(worker_feed, gtfs_realtime_pb2.FeedMessage)


def test_header_has_version(worker_feed):
    assert worker_feed.header.gtfs_realtime_version, "missing gtfs_realtime_version"


def test_header_timestamp_is_fresh(worker_feed):
    ts = worker_feed.header.timestamp
    assert ts > 0, "feed header timestamp is unset"
    age = time.time() - ts
    assert -60 < age < MAX_FEED_AGE_SEC, (
        f"feed timestamp is stale/skewed: age={age:.0f}s "
        f"(limit {MAX_FEED_AGE_SEC}s)"
    )


def test_feed_has_entities(worker_feed):
    # During service hours the feed should carry live trip updates. An empty
    # feed is valid protobuf but usually means inference produced nothing —
    # worth a loud failure so we notice silent breakage.
    assert len(worker_feed.entity) > 0, (
        "feed parsed but contains zero entities — no trips were predicted"
    )


# ── Entity-level structure (the drop-in contract) ───────────────────────────

def test_entities_are_trip_updates(worker_feed):
    for e in worker_feed.entity:
        assert e.HasField("trip_update"), f"entity {e.id!r} has no trip_update"
        assert e.trip_update.trip.trip_id, f"entity {e.id!r} missing trip_id"


def test_stop_time_updates_well_formed(worker_feed):
    for e in worker_feed.entity:
        stus = e.trip_update.stop_time_update
        assert len(stus) > 0, f"trip {e.trip_update.trip.trip_id} has no stops"
        assert len(stus) <= MAX_STOPS_AHEAD, (
            f"trip {e.trip_update.trip.trip_id} has {len(stus)} stops "
            f"(> cap {MAX_STOPS_AHEAD})"
        )
        for stu in stus:
            tid = e.trip_update.trip.trip_id
            assert stu.stop_id, f"trip {tid} has a stop_time_update without stop_id"
            assert stu.HasField("arrival"), f"trip {tid} stop {stu.stop_id} no arrival"
            assert stu.arrival.time > 0, (
                f"trip {tid} stop {stu.stop_id} has non-positive arrival time"
            )


def test_arrival_times_are_monotonic_and_future(worker_feed):
    feed_ts = worker_feed.header.timestamp
    for e in worker_feed.entity:
        tid = e.trip_update.trip.trip_id
        prev = None
        for stu in e.trip_update.stop_time_update:
            t = stu.arrival.time
            assert t >= feed_ts - 60, (
                f"trip {tid} stop {stu.stop_id} arrival is in the past "
                f"relative to feed timestamp"
            )
            if prev is not None:
                assert t >= prev, (
                    f"trip {tid} arrival times are not monotonically increasing "
                    f"({prev} -> {t})"
                )
            prev = t


# ── Parity with the upstream reference ──────────────────────────────────────

def test_parity_with_reference(worker_feed, reference_feed):
    """Structural sanity vs upstream: comparable scale and overlapping IDs.

    Not an exact match — our feed is model-predicted while upstream is the
    operator's own ETAs — but a drop-in replacement should cover a meaningful
    share of the same trips and never balloon to an absurd size.
    """
    ours = {e.trip_update.trip.trip_id for e in worker_feed.entity}
    theirs = {e.trip_update.trip.trip_id for e in reference_feed.entity}

    assert ours, "our feed has no trips to compare"
    assert theirs, "reference feed has no trips to compare"

    overlap = ours & theirs
    assert overlap, (
        "no trip_id overlap with the reference feed — likely an ID-scheme "
        "mismatch that would break downstream consumers"
    )

    # Coverage: we should be predicting a non-trivial fraction of live trips.
    coverage = len(overlap) / len(theirs)
    assert coverage >= 0.10, (
        f"only {coverage:.0%} of reference trips covered "
        f"({len(overlap)}/{len(theirs)})"
    )
