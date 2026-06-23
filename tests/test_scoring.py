"""Hermetic unit tests for src/scoring.py.

No R2, no GTFS: we build a TripUpdates protobuf in memory, parse it back, and
drive the join/aggregation with synthetic actuals so the residual maths is
pinned down exactly.
"""

from __future__ import annotations

import os

# scoring imports src.snapshots, which reads R2 creds at import time.
for _k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
    os.environ.setdefault(_k, "test")

import pandas as pd  # noqa: E402
from google.transit import gtfs_realtime_pb2  # noqa: E402

from src import scoring  # noqa: E402


def _make_feed(feed_ts: int, entities: list[dict]) -> bytes:
    """entities: [{vehicle_id, trip_id, route_id, stops: [(stop_id, seq, arr_ts)]}]"""
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.header.gtfs_realtime_version = "2.0"
    feed.header.timestamp = feed_ts
    for e in entities:
        ent = feed.entity.add()
        ent.id = e["vehicle_id"]
        tu = ent.trip_update
        tu.trip.trip_id = e["trip_id"]
        if e.get("route_id"):
            tu.trip.route_id = e["route_id"]
        tu.vehicle.id = e["vehicle_id"]
        for stop_id, seq, arr in e["stops"]:
            stu = tu.stop_time_update.add()
            stu.stop_id = stop_id
            stu.stop_sequence = seq
            stu.arrival.time = arr
            stu.departure.time = arr
    return feed.SerializeToString()


def test_parse_prediction_feed_extracts_rows_and_stops_ahead():
    data = _make_feed(
        1000,
        [{
            "vehicle_id": "v1", "trip_id": "t1", "route_id": "32",
            "stops": [("100", 5, 1120), ("101", 6, 1300)],
        }],
    )
    rows = scoring._parse_prediction_feed(data)
    assert len(rows) == 2
    assert rows[0] == {
        "feed_ts": 1000, "vehicle_id": "v1", "trip_id": "t1", "route_id": "32",
        "stop_id": "100", "stop_sequence": 5, "stops_ahead": 1,
        "predicted_arrival": 1120,
    }
    # stops_ahead is the position within the vehicle's update list.
    assert rows[1]["stops_ahead"] == 2


def test_parse_prediction_feed_ignores_garbage():
    assert scoring._parse_prediction_feed(b"\xff\xff not a feed") == []


def _predictions_df(rows):
    return pd.DataFrame(rows)


def test_join_computes_signed_error_and_lead():
    # Predicted arrival 1120, actual 1100 → 20 s late; lead = 1120-1010 = 110.
    preds = _predictions_df([{
        "feed_ts": 1010, "vehicle_id": "v1", "trip_id": "t1", "route_id": "32",
        "stop_id": "100", "stop_sequence": 5, "stops_ahead": 1,
        "predicted_arrival": 1120,
    }])
    actuals = pd.DataFrame([{
        "vehicle_id": "v1", "trip_id": "t1", "route_id": "32",
        "stop_id": "100", "stop_sequence": 5, "actual_arrival_ts": 1100,
    }])
    joined = scoring.join_predictions_actuals(preds, actuals)
    assert len(joined) == 1
    r = joined.iloc[0]
    assert r["error_sec"] == 20
    assert r["abs_error_sec"] == 20
    assert r["lead_sec"] == 110
    assert r["lead_bucket"] == "0-2m"  # 110 s lands in [0,120)


def test_lead_bucket_boundaries_are_left_closed():
    # lead_sec exactly 120 should fall into the 2-5m bucket (bins right=False).
    preds = _predictions_df([{
        "feed_ts": 1000, "vehicle_id": "v1", "trip_id": "t1", "route_id": "32",
        "stop_id": "100", "stop_sequence": 5, "stops_ahead": 1,
        "predicted_arrival": 1121,  # lead 121
    }])
    actuals = pd.DataFrame([{
        "vehicle_id": "v1", "trip_id": "t1", "route_id": "32",
        "stop_id": "100", "stop_sequence": 5, "actual_arrival_ts": 1121,
    }])
    joined = scoring.join_predictions_actuals(preds, actuals)
    assert joined.iloc[0]["lead_bucket"] == "2-5m"


def test_join_drops_past_and_implausible():
    preds = _predictions_df([
        # lead <= 0 → dropped (rider never saw a future ETA)
        {"feed_ts": 2000, "vehicle_id": "v1", "trip_id": "t1", "route_id": "1",
         "stop_id": "1", "stop_sequence": 1, "stops_ahead": 1, "predicted_arrival": 1900},
        # implausible residual → dropped (bad join, not bad model)
        {"feed_ts": 1000, "vehicle_id": "v2", "trip_id": "t2", "route_id": "1",
         "stop_id": "2", "stop_sequence": 1, "stops_ahead": 1, "predicted_arrival": 1500},
    ])
    actuals = pd.DataFrame([
        {"vehicle_id": "v1", "trip_id": "t1", "route_id": "1", "stop_id": "1",
         "stop_sequence": 1, "actual_arrival_ts": 1850},
        {"vehicle_id": "v2", "trip_id": "t2", "route_id": "1", "stop_id": "2",
         "stop_sequence": 1, "actual_arrival_ts": 1500 + scoring.MAX_PLAUSIBLE_ERROR_SEC + 10},
    ])
    joined = scoring.join_predictions_actuals(preds, actuals)
    assert joined.empty


def test_score_report_metrics_and_coverage():
    # Two stops actually arrived; only one was predicted → coverage 0.5.
    preds = _predictions_df([
        {"feed_ts": 1000, "vehicle_id": "v1", "trip_id": "t1", "route_id": "32",
         "stop_id": "100", "stop_sequence": 5, "stops_ahead": 1, "predicted_arrival": 1130},
    ])
    actuals = pd.DataFrame([
        {"vehicle_id": "v1", "trip_id": "t1", "route_id": "32", "stop_id": "100",
         "stop_sequence": 5, "actual_arrival_ts": 1100},  # predicted 30s late
        {"vehicle_id": "v1", "trip_id": "t1", "route_id": "32", "stop_id": "101",
         "stop_sequence": 6, "actual_arrival_ts": 1300},  # never predicted
    ])
    joined = scoring.join_predictions_actuals(preds, actuals)
    report = scoring.score_report(joined, actuals, "2026-06-15")
    assert report["status"] == "ok"
    assert report["coverage_frac"] == 0.5
    assert report["overall"]["bias_sec"] == 30.0
    assert report["overall"]["mae_sec"] == 30.0
    assert report["n_actual_arrivals"] == 2


def test_score_report_empty_join():
    report = scoring.score_report(pd.DataFrame(), pd.DataFrame(), "2026-06-15")
    assert report["status"] == "no_matches"


def test_coverage_gap_cause_tagging():
    # Served feed: v1 predicts its trip t1; v2 is served but only under t2-alt
    # (a trip-inference disagreement); v3 never appears in the feed at all.
    preds = _predictions_df([
        {"feed_ts": 1000, "vehicle_id": "v1", "trip_id": "t1", "route_id": "32",
         "stop_id": "100", "stop_sequence": 5, "stops_ahead": 1, "predicted_arrival": 1130},
        {"feed_ts": 1000, "vehicle_id": "v2", "trip_id": "t2-alt", "route_id": "32",
         "stop_id": "200", "stop_sequence": 5, "stops_ahead": 1, "predicted_arrival": 1130},
    ])
    actuals = pd.DataFrame([
        # covered
        {"vehicle_id": "v1", "trip_id": "t1", "route_id": "32", "stop_id": "100",
         "stop_sequence": 5, "actual_arrival_ts": 1100},
        # served vehicle+trip but this stop was never in the served window → stop_missing
        {"vehicle_id": "v1", "trip_id": "t1", "route_id": "32", "stop_id": "101",
         "stop_sequence": 6, "actual_arrival_ts": 1300},
        # vehicle served, but actuals matched it to t2 (live said t2-alt) → trip_mismatch
        {"vehicle_id": "v2", "trip_id": "t2", "route_id": "32", "stop_id": "200",
         "stop_sequence": 5, "actual_arrival_ts": 1300},
        # vehicle never appeared in the served feed → vehicle_absent
        {"vehicle_id": "v3", "trip_id": "t3", "route_id": "88", "stop_id": "300",
         "stop_sequence": 5, "actual_arrival_ts": 1300},
    ])
    joined = scoring.join_predictions_actuals(preds, actuals)
    report = scoring.score_report(joined, actuals, "2026-06-15", predictions=preds)
    assert report["coverage_gap"]["by_cause"] == {
        "stop_missing": 1, "trip_mismatch": 1, "vehicle_absent": 1
    }
    by_route = report["coverage_gap"]["by_route"]
    assert by_route["88"]["uncovered_by_cause"] == {"vehicle_absent": 1}
    assert by_route["32"]["uncovered_by_cause"] == {
        "stop_missing": 1, "trip_mismatch": 1
    }


def test_coverage_gap_no_causes_without_predictions():
    # Backward-compatible: omit the served feed → no cause keys, as before.
    preds = _predictions_df([
        {"feed_ts": 1000, "vehicle_id": "v1", "trip_id": "t1", "route_id": "32",
         "stop_id": "100", "stop_sequence": 5, "stops_ahead": 1, "predicted_arrival": 1130},
    ])
    actuals = pd.DataFrame([
        {"vehicle_id": "v1", "trip_id": "t1", "route_id": "32", "stop_id": "100",
         "stop_sequence": 5, "actual_arrival_ts": 1100},
    ])
    joined = scoring.join_predictions_actuals(preds, actuals)
    report = scoring.score_report(joined, actuals, "2026-06-15")
    assert "by_cause" not in report["coverage_gap"]
    assert all("uncovered_by_cause" not in g
               for g in report["coverage_gap"]["by_route"].values())


def test_relaxed_join_recovers_trip_instance_mismatch():
    # Live served the physical stop under trip "t1-run2" at its sequence 5; batch
    # labeled the same arrival as trip "t1" sequence 12 (trip-relative numbering
    # differs). Strict join misses it (trip_id AND stop_sequence differ); the
    # rider-centric (vehicle, stop_id) + nearest-time join recovers it.
    preds = _predictions_df([
        {"feed_ts": 1000, "vehicle_id": "v1", "trip_id": "t1-run2", "route_id": "32",
         "stop_id": "100", "stop_sequence": 5, "stops_ahead": 1, "predicted_arrival": 1130},
    ])
    actuals = pd.DataFrame([
        {"vehicle_id": "v1", "trip_id": "t1", "route_id": "32", "stop_id": "100",
         "stop_sequence": 12, "actual_arrival_ts": 1100},
    ])
    assert scoring.join_predictions_actuals(preds, actuals).empty  # strict misses

    rj = scoring.join_predictions_actuals_relaxed(preds, actuals)
    assert len(rj) == 1
    assert rj.iloc[0]["error_sec"] == 30  # predicted 1130 vs actual 1100
    assert rj.iloc[0]["actual_trip_id"] == "t1"
    assert rj.iloc[0]["actual_stop_sequence"] == 12

    report = scoring.score_report(
        scoring.join_predictions_actuals(preds, actuals), actuals,
        "2026-06-15", predictions=preds,
    )
    assert report["status"] == "no_matches"        # strict found nothing
    assert report["relaxed_join"]["coverage_frac"] == 1.0  # recovered
    assert report["relaxed_join"]["overall"]["mae_sec"] == 30.0


def test_relaxed_join_separates_repeat_stop_visits_by_time():
    # Same vehicle/stop/sequence visited twice (two runs hours apart). Each
    # prediction must bind to its own visit, not collapse to one.
    preds = _predictions_df([
        {"feed_ts": 1000, "vehicle_id": "v1", "trip_id": "A", "route_id": "9",
         "stop_id": "1", "stop_sequence": 3, "stops_ahead": 1, "predicted_arrival": 1100},
        {"feed_ts": 9000, "vehicle_id": "v1", "trip_id": "B", "route_id": "9",
         "stop_id": "1", "stop_sequence": 3, "stops_ahead": 1, "predicted_arrival": 9100},
    ])
    actuals = pd.DataFrame([
        {"vehicle_id": "v1", "trip_id": "A", "route_id": "9", "stop_id": "1",
         "stop_sequence": 3, "actual_arrival_ts": 1090},
        {"vehicle_id": "v1", "trip_id": "B", "route_id": "9", "stop_id": "1",
         "stop_sequence": 3, "actual_arrival_ts": 9080},
    ])
    rj = scoring.join_predictions_actuals_relaxed(preds, actuals).sort_values("feed_ts")
    assert list(rj["error_sec"]) == [10, 20]  # 1100-1090, 9100-9080 — bound separately


def test_aggregate_stops_ahead_mae_n_weighted_pool():
    # mae_sec is a per-day mean of |error|, so pooling across days must weight by
    # n: day1 (100s, n=100) + day2 (200s, n=300) → (100*100 + 200*300)/400 = 175.
    reports = [
        {"by_stops_ahead": {"1": {"n": 100, "mae_sec": 100.0}}},
        {"by_stops_ahead": {"1": {"n": 300, "mae_sec": 200.0}}},
    ]
    table = scoring._aggregate_stops_ahead_mae(reports, min_n=200)
    assert table == {1: 175}


def test_aggregate_stops_ahead_mae_drops_low_support_and_enforces_monotone():
    reports = [{
        "by_stops_ahead": {
            "1": {"n": 500, "mae_sec": 180.0},
            "2": {"n": 500, "mae_sec": 160.0},   # dips below h1 → lifted to 180
            "3": {"n": 50,  "mae_sec": 999.0},   # under min_n → dropped
            "4": {"n": 500, "mae_sec": 300.0},
        }
    }]
    table = scoring._aggregate_stops_ahead_mae(reports, min_n=200)
    assert table == {1: 180, 2: 180, 4: 300}   # non-decreasing, low-n horizon gone


def test_aggregate_stops_ahead_mae_empty_when_no_support():
    assert scoring._aggregate_stops_ahead_mae([], min_n=200) == {}
    assert scoring._aggregate_stops_ahead_mae(
        [{"by_stops_ahead": {"1": {"n": 10, "mae_sec": 50.0}}}], min_n=200
    ) == {}


def test_to_epoch_seconds_is_resolution_independent():
    # 2026-06-15T12:25:46Z = 1781526346 epoch seconds. pandas 3.0 builds
    # datetime64 at microsecond resolution; the conversion must still land in
    # seconds (the //1e9 bug returned this 1000x low and zeroed every join).
    expected = 1781526346
    for unit in ("s", "ms", "us", "ns"):
        ts = pd.Series(pd.to_datetime([expected], unit="s", utc=True)).astype(f"datetime64[{unit}, UTC]")
        out = scoring._to_epoch_seconds(ts)
        assert int(out.iloc[0]) == expected, f"unit={unit} → {out.iloc[0]}"
