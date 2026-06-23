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

# Worker caps upcoming stops per trip (src/predict.py: MAX_STOPS_AHEAD).
MAX_STOPS_AHEAD = 10
# Feed header timestamp must be fresh — upstream vehicle positions are
# republished every few seconds; allow generous slack for clock skew / caching.
MAX_FEED_AGE_SEC = 15 * 60

# Stop sign-code 60 (Захисників України) → internal GTFS stop_id 4577.
HEALTH_CHECK_STOP_ID = "4577"
# Working-hours window for the arrivals check (mirrors worker/worker.js WORKING_HOURS_UTC).
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
    # worth a loud failure so we notice silent breakage. Overnight, though,
    # transit isn't running and an empty feed is expected, so only assert
    # during working hours (mirrors worker/worker.js /health).
    if not _within_working_hours():
        pytest.skip(
            f"outside working hours (UTC hour {time.gmtime().tm_hour} not in "
            f"{WORKING_HOURS_UTC}) — an empty feed is expected, not a failure"
        )
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
    """We should predict ETAs for the majority of vehicles that report a trip_id.

    Only asserted during working hours: as service winds down overnight, few
    vehicles run and matching is sparse, so coverage legitimately drops below
    50% on a perfectly healthy feed (mirrors worker/worker.js /health and the other
    coverage tests in this file).
    """
    if not _within_working_hours():
        pytest.skip(
            f"outside working hours (UTC hour {time.gmtime().tm_hour} not in "
            f"{WORKING_HOURS_UTC}) — sparse coverage is expected, not a failure"
        )
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
    so 0 arrivals is healthy, not a regression (mirrors worker/worker.js /health).
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


# ── Our VehiclePositions feed ───────────────────────────────────────────────

# Lviv city bounding box (with ~5 km margin for outskirt depots/terminals).
# Tighter than an oblast box — catches swapped lat/lon and zero-initialised coords.
_VP_LAT_MIN, _VP_LAT_MAX = 49.72, 49.97
_VP_LON_MIN, _VP_LON_MAX = 23.82, 24.22

# Urban transit hard cap: ~130 km/h covers articulated trams on downhill tracks;
# anything above is a GPS artefact or a unit error (m/s vs km/h).
_SPEED_MAX_MPS = 36.0

# GTFS-RT VehiclePosition.VehicleStopStatus values we actively assign.
_VALID_STATUSES = frozenset({
    gtfs_realtime_pb2.VehiclePosition.STOPPED_AT,
    gtfs_realtime_pb2.VehiclePosition.INCOMING_AT,
    gtfs_realtime_pb2.VehiclePosition.IN_TRANSIT_TO,
})


def _vp_vehicles(our_vp_feed):
    """Yield VehiclePosition messages from the feed."""
    for e in our_vp_feed.entity:
        if e.HasField("vehicle"):
            yield e.id, e.vehicle


def test_our_vp_feed_has_entities(our_vp_feed):
    if not _within_working_hours():
        pytest.skip(
            f"outside working hours (UTC hour {time.gmtime().tm_hour} not in "
            f"{WORKING_HOURS_UTC}) — an empty VP feed is expected, not a failure"
        )
    assert len(our_vp_feed.entity) > 0, (
        "our vehicle_positions feed has zero entities — inference may have stopped"
    )


def test_our_vp_feed_vehicle_ids_are_unique(our_vp_feed):
    """Each vehicle must appear exactly once; duplicates mean a tracker bug."""
    ids = [eid for eid, _ in _vp_vehicles(our_vp_feed)]
    dupes = {v for v in ids if ids.count(v) > 1}
    assert not dupes, f"duplicate vehicle entity IDs: {sorted(dupes)[:10]}"


def test_our_vp_feed_coords_within_lviv(our_vp_feed):
    """All positions must fall inside the Lviv city bounding box.

    Catches swapped lat/lon, zero-initialised fields, and vehicles projected
    into a neighbouring city due to a shape-matching bug.
    """
    bad = []
    for eid, v in _vp_vehicles(our_vp_feed):
        pos = v.position
        if not (_VP_LAT_MIN <= pos.latitude  <= _VP_LAT_MAX and
                _VP_LON_MIN <= pos.longitude <= _VP_LON_MAX):
            bad.append((eid, f"lat={pos.latitude:.5f} lon={pos.longitude:.5f}"))
    assert not bad, f"out-of-bounds positions: {bad[:5]}"


def test_our_vp_feed_has_trip_and_route(our_vp_feed):
    """Every vehicle must carry a matched trip_id and route_id."""
    bad = []
    for eid, v in _vp_vehicles(our_vp_feed):
        if not v.trip.trip_id or not v.trip.route_id:
            bad.append((eid, f"trip_id={v.trip.trip_id!r} route_id={v.trip.route_id!r}"))
    assert not bad, f"entities missing trip/route: {bad[:5]}"


def test_our_vp_feed_bearing_present_on_all(our_vp_feed):
    """Every on-route vehicle must report a bearing (1–359°).

    Inference derives bearing from the shape tangent at the matched point, so
    bearing=0 would mean the tangent calculation returned north or was skipped.
    The upstream feed also supplies bearing, so a complete zero run signals
    a field-copy regression.
    """
    bad = []
    for eid, v in _vp_vehicles(our_vp_feed):
        b = v.position.bearing
        if not (1.0 <= b <= 359.0):
            bad.append((eid, f"bearing={b}"))
    assert not bad, f"vehicles with missing/invalid bearing: {bad[:10]}"


def test_our_vp_feed_speed_in_realistic_range(our_vp_feed):
    """Non-zero speeds must be physically plausible for Lviv urban transit.

    Lower bound: 0.5 m/s filters GPS noise from truly stationary vehicles.
    Upper bound: _SPEED_MAX_MPS catches unit errors (e.g. km/h reported as m/s
    would turn a 60 km/h tram into a 60 m/s ≈ 216 km/h rocket).
    """
    bad = []
    for eid, v in _vp_vehicles(our_vp_feed):
        s = v.position.speed
        if 0 < s < 0.5 or s > _SPEED_MAX_MPS:
            bad.append((eid, f"speed={s:.2f} m/s ({s * 3.6:.1f} km/h)"))
    assert not bad, f"implausible speed values: {bad[:10]}"


def test_our_vp_feed_some_vehicles_have_speed(our_vp_feed):
    """During service hours, ≥30% of vehicles must report a non-zero GPS speed.

    The upstream tracker supplies speed for ~50–70% of active vehicles. Dropping
    below 30% indicates the speed field is being silently zeroed somewhere in
    the inference pipeline.
    """
    if not _within_working_hours():
        pytest.skip(
            f"outside working hours (UTC hour {time.gmtime().tm_hour} not in "
            f"{WORKING_HOURS_UTC}) — stationary/absent vehicles expected"
        )
    vehicles = list(_vp_vehicles(our_vp_feed))
    if not vehicles:
        pytest.skip("no vehicles in feed")
    nonzero = sum(1 for _, v in vehicles if v.position.speed > 0)
    frac = nonzero / len(vehicles)
    assert frac >= 0.30, (
        f"only {frac:.0%} of vehicles report speed > 0 ({nonzero}/{len(vehicles)}) — "
        "GPS speed may be dropped in encode_vehicle_positions"
    )


def test_our_vp_feed_status_is_valid(our_vp_feed):
    """current_status must be STOPPED_AT, INCOMING_AT, or IN_TRANSIT_TO."""
    bad = []
    for eid, v in _vp_vehicles(our_vp_feed):
        if v.current_status not in _VALID_STATUSES:
            bad.append((eid, f"current_status={v.current_status}"))
    assert not bad, f"unexpected current_status values: {bad[:10]}"


def test_our_vp_feed_next_stop_present_on_all(our_vp_feed):
    """Every vehicle must have a next stop_id (numeric GTFS stop code).

    Inference populates stop_id from the first remaining stop on the matched
    trip shape. A blank or non-numeric stop_id means the trip had no upcoming
    stops — likely an end-of-line vehicle that slipped through the pre-departure
    filter.
    """
    bad = []
    for eid, v in _vp_vehicles(our_vp_feed):
        if not v.stop_id or not v.stop_id.isdigit():
            bad.append((eid, f"stop_id={v.stop_id!r}"))
    assert not bad, f"vehicles with missing/non-numeric stop_id: {bad[:10]}"


def test_our_vp_feed_congestion_set_for_some_vehicles(our_vp_feed):
    """At least some vehicles must carry a non-UNKNOWN congestion level.

    Congestion is computed from progress_speed (position deltas between pushes),
    not from the raw GPS speed field, so newly-seen vehicles legitimately stay
    UNKNOWN until two consecutive snapshots are available. But during service
    hours the vast majority of vehicles have history, so a complete absence of
    congestion data is a clear pipeline regression.
    """
    if not _within_working_hours():
        pytest.skip(
            f"outside working hours (UTC hour {time.gmtime().tm_hour} not in "
            f"{WORKING_HOURS_UTC}) — no active vehicles expected"
        )
    UNKNOWN = gtfs_realtime_pb2.VehiclePosition.UNKNOWN_CONGESTION_LEVEL
    vehicles = list(_vp_vehicles(our_vp_feed))
    if not vehicles:
        pytest.skip("no vehicles in feed")
    with_congestion = sum(1 for _, v in vehicles if v.congestion_level != UNKNOWN)
    assert with_congestion > 0, (
        "every vehicle has UNKNOWN congestion level — "
        "congestion_level is not being written (check _congestion_level / encode_vehicle_positions)"
    )


# ── Parity with the upstream reference ──────────────────────────────────────

def test_parity_with_reference(worker_feed, reference_feed):
    """Structural sanity vs upstream: comparable scale and overlapping IDs.

    Not an exact match — our feed is model-predicted while upstream is the
    operator's own ETAs — but a drop-in replacement should cover a meaningful
    share of the same trips and never balloon to an absurd size.

    Only meaningful during working hours: overnight both feeds are empty
    (transit isn't running), so there's nothing to compare (mirrors worker/worker.js).
    """
    if not _within_working_hours():
        pytest.skip(
            f"outside working hours (UTC hour {time.gmtime().tm_hour} not in "
            f"{WORKING_HOURS_UTC}) — no live trips to compare"
        )
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
