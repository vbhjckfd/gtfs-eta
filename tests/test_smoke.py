"""Smoke tests for the live GTFS-RT TripUpdates worker.

Goal: prove the deployed endpoint is a valid, sane drop-in replacement for
the upstream Lviv `trip_updates` feed. These run against the network, so they
are fast but not hermetic — treat a green run as "production is serving a
healthy feed right now".

Run with `make smoke`.
"""

from __future__ import annotations

import re
import time

import pytest
from google.transit import gtfs_realtime_pb2

# Lviv trip IDs are always three underscore-separated groups of digits,
# e.g. "29734_3_1".  Any other format means our inference produced garbage.
_LVIV_TRIP_ID_RE = re.compile(r"^\d+_\d+_\d+$")

# Worker caps upcoming stops per trip (worker.py: MAX_STOPS_AHEAD).
MAX_STOPS_AHEAD = 10
# Feed header timestamp must be fresh — upstream vehicle positions are
# republished every few seconds; allow generous slack for clock skew / caching.
MAX_FEED_AGE_SEC = 15 * 60

# Stop sign-code 60 (Захисників України) → internal GTFS stop_id 4577.
HEALTH_CHECK_STOP_ID = "4577"
# Working-hours window for the arrivals check (mirrors worker.py WORKING_HOURS_UTC).
# Lviv transit doesn't run overnight, so 0 arrivals off-hours is healthy, not a
# regression. Expressed in UTC: 05:00–18:00 UTC ≈ 07:00–21:00 Lviv in both DST
# states — unambiguously inside the service day. Outside it, the arrivals
# assertion is skipped rather than failing a perfectly healthy feed.
WORKING_HOURS_UTC = range(5, 18)


def _within_working_hours() -> bool:
    return time.gmtime().tm_hour in WORKING_HOURS_UTC


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


# ── Real Lviv data checks ────────────────────────────────────────────────────

def test_trip_id_format(worker_feed):
    """Trip IDs must follow Lviv's DIGITS_DIGIT_DIGIT scheme (e.g. 29734_3_1)."""
    bad = [
        e.trip_update.trip.trip_id
        for e in worker_feed.entity
        if not _LVIV_TRIP_ID_RE.match(e.trip_update.trip.trip_id)
    ]
    assert not bad, f"malformed trip_ids: {bad[:5]}"


def test_stop_codes_are_numeric(worker_feed):
    """Stop codes printed on Lviv street signs are always numeric strings."""
    bad = {
        stu.stop_id
        for e in worker_feed.entity
        for stu in e.trip_update.stop_time_update
        if not stu.stop_id.isdigit()
    }
    assert not bad, f"non-numeric stop codes: {sorted(bad)[:10]}"


def test_predicted_routes_are_currently_active(worker_feed, vehicle_positions_feed):
    """≥95% of our route_ids must have at least one vehicle currently on road.

    The 5% slack covers timing: our feed is pre-computed (up to 5 min old) while
    the VP feed is live, so a vehicle can disappear between the two fetches.
    """
    active_routes = {
        str(e.vehicle.trip.route_id)
        for e in vehicle_positions_feed.entity
        if e.HasField("vehicle")
    }
    our_routes = {e.trip_update.trip.route_id for e in worker_feed.entity}
    unknown = our_routes - active_routes
    stale_frac = len(unknown) / max(len(our_routes), 1)
    assert stale_frac <= 0.05, (
        f"{stale_frac:.0%} of predicted routes have no active vehicles: "
        f"{sorted(unknown)} — inference may be matching wrong trips"
    )


def test_vehicle_coverage(worker_feed, vehicle_positions_feed):
    """We should predict ETAs for the majority of vehicles that report a trip_id."""
    # Vehicles with a blank trip_id cannot be matched; exclude them.
    vp_trips = {
        str(e.vehicle.trip.trip_id)
        for e in vehicle_positions_feed.entity
        if e.HasField("vehicle") and e.vehicle.trip.trip_id
    }
    our_trips = {e.trip_update.trip.trip_id for e in worker_feed.entity}
    if not vp_trips:
        pytest.skip("vehicle positions feed has no vehicles with trip_ids")
    covered = len(vp_trips & our_trips) / len(vp_trips)
    assert covered >= 0.50, (
        f"only {covered:.0%} of active vehicles have ETA predictions "
        f"({len(vp_trips & our_trips)}/{len(vp_trips)})"
    )


def test_stop_codes_overlap_with_reference(worker_feed, reference_feed):
    """Our stop codes should come from the same numbering system as the reference feed."""
    our_stops = {
        stu.stop_id
        for e in worker_feed.entity
        for stu in e.trip_update.stop_time_update
    }
    ref_stops = {
        stu.stop_id
        for e in reference_feed.entity
        for stu in e.trip_update.stop_time_update
    }
    if not ref_stops:
        pytest.skip("reference feed has no stop_time_updates to compare")
    overlap = len(our_stops & ref_stops) / len(our_stops)
    assert overlap >= 0.30, (
        f"only {overlap:.0%} of our stop codes appear in the reference feed — "
        f"stop ID scheme mismatch likely"
    )


# ── Stop-level coverage ─────────────────────────────────────────────────────

def test_stop_60_has_arrivals(worker_feed):
    """Stop sign-code 60 (Захисників України, highest-traffic in Lviv) must have arrivals.

    The feed uses internal GTFS stop_id values, not sign codes. Sign code 60
    maps to internal stop_id 4577 (stops.txt: stop_code=60, stop_id=4577).

    Only asserted during working hours: overnight Lviv transit isn't running,
    so 0 arrivals is healthy, not a regression (mirrors worker.py /health).
    """
    if not _within_working_hours():
        pytest.skip(
            f"outside working hours (UTC hour {time.gmtime().tm_hour} not in "
            f"{WORKING_HOURS_UTC}) — 0 arrivals is expected, not a failure"
        )
    arrivals = [
        (e.trip_update.trip.trip_id, stu)
        for e in worker_feed.entity
        for stu in e.trip_update.stop_time_update
        if stu.stop_id == HEALTH_CHECK_STOP_ID
    ]
    assert arrivals, (
        "no arrival predictions found for stop 60 / internal id 4577 — "
        "the busiest stop in Lviv should always have active trips"
    )


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
