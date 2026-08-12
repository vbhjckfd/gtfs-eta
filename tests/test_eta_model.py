"""
Unit tests for the snapshot-anchored training data and the compact
production inference path.

The regression these tests guard: a bus sitting *at* a stop must get a
near-zero remaining distance (and therefore a near-zero ETA from a model
trained on snapshot-anchored rows), never the full previous-segment time.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from src.features import (
    BASE_FEATURE_COLS,
    FEATURE_COLS,
    SPEED_UNKNOWN,
    TARGET_COL,
    apply_priors,
    compute_features_for_inference,
    compute_features_for_training,
    sched_sec_at_dist,
)
from src.gtfs_static import StopTime, TripInfo
from src.inference import (
    MAX_FEED_CLOCK_SKEW_SEC,
    MAX_STOPS_AHEAD,
    STALE_VEHICLE_MAX_AGE_SEC,
    _TERMINUS_MODEL_UNCERTAINTY_SEC,
    _TERMINUS_UNCERTAINTY_SEC,
    _isotonic,
    _uncertainty_for,
    build_features,
    staleness_reference,
    terminus_seconds_until_departure,
    encode_trip_updates,
    encode_vehicle_positions,
    infer_trip,
    progress_speed,
    run_inference,
    vehicle_dist_along,
)
from src.labeling import build_training_rows
from src.utm import project_xy

# ---------------------------------------------------------------------------
# Synthetic straight-line route near Lviv: stops every 500 m, 60 s apart.
# ---------------------------------------------------------------------------

LAT = 49.84
LON0 = 24.00
TRIP_ID = "t1"
SHAPE_ID = "s1"
ROUTE_ID = "A25"
STOP_SPACING_M = 500.0
N_STOPS = 5  # stops at 0, 500, 1000, 1500, 2000 m


def _lon_at(meters: float) -> float:
    """Longitude offset that is ~meters east of LON0 at LAT (flat-earth ok here)."""
    x0, y0 = project_xy(LON0, LAT)
    # one degree of longitude in metres at this latitude
    x1, _ = project_xy(LON0 + 0.01, LAT)
    return LON0 + 0.01 * meters / (x1 - x0)


class FakeGTFS:
    """Minimal stand-in for GTFSStatic over a straight 2 km shape."""

    def __init__(self):
        xs = [project_xy(_lon_at(m), LAT) for m in (0.0, N_STOPS * STOP_SPACING_M)]
        self._shape = LineString(xs)
        self._stop_times = [
            StopTime(
                stop_id=f"stop{i}",
                stop_sequence=i + 1,
                arrival_time=f"10:{i:02d}:00",  # 60 s per stop
                departure_time=f"10:{i:02d}:00",
                shape_dist_traveled=None,
            )
            for i in range(N_STOPS)
        ]
        self._trip = TripInfo(
            trip_id=TRIP_ID,
            route_id=ROUTE_ID,
            service_id="svc",
            shape_id=SHAPE_ID,
            direction_id=0,
            stop_times=self._stop_times,
        )
        self._stop_dists = {
            (SHAPE_ID, f"stop{i}"): self._shape.project(
                Point(project_xy(_lon_at(i * STOP_SPACING_M), LAT))
            )
            for i in range(N_STOPS)
        }

    def get_trip(self, trip_id):
        return self._trip if trip_id == TRIP_ID else None

    def get_shape_linestring(self, shape_id):
        return self._shape if shape_id == SHAPE_ID else None

    def get_shape_length(self, shape_id):
        return self._shape.length

    def get_stop_distance_along_shape(self, shape_id, stop_id):
        return self._stop_dists.get((shape_id, str(stop_id)))

    def get_stop(self, stop_id):
        return None

    def is_ambiguous_shape(self, shape_id):
        return False


@pytest.fixture(scope="module")
def gtfs():
    return FakeGTFS()


def _trajectory(speed_mps: float = 10.0, interval_s: float = 30.0) -> pd.DataFrame:
    """Vehicle driving the whole route at constant speed, snapshot every 30 s."""
    t0 = datetime(2026, 6, 3, 7, 0, 0, tzinfo=timezone.utc)  # Wednesday
    total_m = N_STOPS * STOP_SPACING_M
    rows = []
    m = 0.0
    i = 0
    while m <= total_m:
        rows.append({
            "timestamp": pd.Timestamp(t0 + timedelta(seconds=i * interval_s)),
            "vehicle_id": "v1",
            "inferred_trip_id": TRIP_ID,
            "lat": LAT,
            "lon": _lon_at(m),
        })
        i += 1
        m = i * interval_s * speed_mps
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Snapshot-anchored training rows
# ---------------------------------------------------------------------------

class TestTrainingRows:
    def test_targets_match_constant_speed(self, gtfs):
        rows = build_training_rows(_trajectory(), gtfs)
        assert not rows.empty
        # At 10 m/s the target must be ≈ remaining_dist / 10 for every row.
        remaining = rows["stop_dist_along_m"] - rows["dist_along_m"]
        expected = remaining / 10.0
        np.testing.assert_allclose(
            rows["seconds_to_arrival"], expected, atol=3.0
        )

    def test_bus_dwelling_at_stop_gets_near_zero_target(self, gtfs):
        # Vehicle dwells three snapshots ~3 m short of stop2 (≈1000 m), then
        # moves on. Training rows must include near-zero remaining distances,
        # and the last dwell snapshot's target must be small — these are the
        # examples that teach the model "bus at stop → ETA ≈ 0".
        t0 = datetime(2026, 6, 3, 7, 0, 0, tzinfo=timezone.utc)
        meters = [0, 300, 600, 900, 998, 998, 998, 1100, 1400, 1700, 2000]
        traj = pd.DataFrame([
            {
                "timestamp": pd.Timestamp(t0 + timedelta(seconds=30 * i)),
                "vehicle_id": "v1",
                "inferred_trip_id": TRIP_ID,
                "lat": LAT,
                "lon": _lon_at(m),
            }
            for i, m in enumerate(meters)
        ])
        rows = build_training_rows(traj, gtfs)
        near = rows[
            (rows["stop_dist_along_m"] - rows["dist_along_m"]).between(0, 5.0)
        ]
        assert not near.empty
        # Last dwell snapshot crosses the stop within seconds
        assert near["seconds_to_arrival"].min() < 5.0
        # Earlier dwell snapshots have the same tiny distance but larger
        # targets (dwell time) — both kinds of examples must exist.
        assert near["seconds_to_arrival"].max() > 30.0

    def test_progress_speed_measured(self, gtfs):
        rows = build_training_rows(_trajectory(), gtfs)
        known = rows[rows["progress_speed_mps"] != SPEED_UNKNOWN]
        assert not known.empty
        np.testing.assert_allclose(known["progress_speed_mps"], 10.0, atol=0.5)
        # First snapshot has no history → sentinel
        first_ts = rows["snapshot_ts"].min()
        assert (
            rows.loc[rows["snapshot_ts"] == first_ts, "progress_speed_mps"]
            == SPEED_UNKNOWN
        ).all()

    def test_stops_ahead_starts_at_one(self, gtfs):
        rows = build_training_rows(_trajectory(), gtfs)
        assert rows["stops_ahead"].min() == 1
        assert rows["stops_ahead"].max() <= 10


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------

class TestTrainingFeatures:
    def test_feature_columns_complete(self, gtfs):
        rows = build_training_rows(_trajectory(), gtfs)
        feats = compute_features_for_training(rows, gtfs)
        # Base features are present directly; prior-derived features need apply_priors.
        for col in BASE_FEATURE_COLS + [TARGET_COL, "date"]:
            assert col in feats.columns, col
        assert len(feats) == len(rows)
        feats = apply_priors(feats, None)  # use global fallback
        for col in FEATURE_COLS:
            assert col in feats.columns, col

    def test_sched_remaining_interpolated_at_position(self, gtfs):
        # Vehicle halfway between stop0 (cum 0 s) and stop1 (cum 60 s):
        # sched_at_pos = 30 s → remaining to stop1 = 30 s, to stop2 = 90 s.
        trip = gtfs.get_trip(TRIP_ID)
        feats = compute_features_for_inference(
            trip_id=TRIP_ID,
            vehicle_dist_m=STOP_SPACING_M / 2,
            snapshot_time=datetime(2026, 6, 3, 7, 0, tzinfo=timezone.utc),
            progress_speed_mps=10.0,
            gtfs=gtfs,
        )
        first, second = feats.iloc[0], feats.iloc[1]
        assert first["sched_remaining_sec"] == pytest.approx(30.0, abs=2.0)
        assert second["sched_remaining_sec"] == pytest.approx(90.0, abs=2.0)

    def test_inference_features_at_stop_have_zero_remaining(self, gtfs):
        # Vehicle 3 m short of stop2 → remaining_dist ≈ 3 m, not a full segment.
        feats = compute_features_for_inference(
            trip_id=TRIP_ID,
            vehicle_dist_m=2 * STOP_SPACING_M - 3.0,
            snapshot_time=datetime(2026, 6, 3, 7, 0, tzinfo=timezone.utc),
            progress_speed_mps=0.0,
            gtfs=gtfs,
        )
        first = feats.iloc[0]
        assert first["stop_sequence"] == 3
        assert first["remaining_dist_m"] == pytest.approx(3.0, abs=1.0)
        assert first["stops_ahead"] == 1


# ---------------------------------------------------------------------------
# Compact production path (src/inference.py)
# ---------------------------------------------------------------------------

def _compact_data(gtfs: FakeGTFS, route_type: int | None = None,
                  start_sec: float | None = None) -> dict:
    """Worker-format data for the fake route.

    ``route_type``/``start_sec`` are what the terminus schedule path needs;
    leaving them unset reproduces an export made before that path existed.
    """
    import struct
    coords = list(gtfs._shape.coords)
    shape_bytes = struct.pack(f"{2 * len(coords)}d", *(v for xy in coords for v in xy))
    # Cumulative scheduled seconds: 60 s per stop (matches FakeGTFS times)
    stop_times = [(f"stop{i}", i + 1, 60.0 * i) for i in range(N_STOPS)]
    data = {
        "shapes": {SHAPE_ID: shape_bytes},
        "shape_lengths": {SHAPE_ID: gtfs._shape.length},
        "stop_distances": dict(gtfs._stop_dists),
        "trip_index": {
            TRIP_ID: {"route_id": ROUTE_ID, "shape_id": SHAPE_ID,
                      "stop_times": stop_times, "start_sec": start_sec}
        },
        "route_trips": {ROUTE_ID: [TRIP_ID]},
        "route_hour_priors": {
            f"{ROUTE_ID}:7": (8.0, 45.0),  # (hist_speed_mps, hist_time_per_stop_sec)
            "_global": (5.0, 40.0),
        },
        "feed_timezone": "Europe/Kiev",
    }
    if route_type is not None:
        data["route_types"] = {ROUTE_ID: route_type}
    return data


class TestCompactInference:
    def test_vehicle_dist_along(self, gtfs):
        data = _compact_data(gtfs)
        x, y = project_xy(_lon_at(700.0), LAT)
        assert vehicle_dist_along(TRIP_ID, x, y, data) == pytest.approx(700.0, abs=2.0)

    def test_build_features_uses_vehicle_position(self, gtfs):
        data = _compact_data(gtfs)
        snap = datetime(2026, 6, 3, 7, 0, tzinfo=timezone.utc)
        rows = build_features(TRIP_ID, 2 * STOP_SPACING_M - 5.0, 8.0, snap, data)
        feat, stop_id, stop_seq = rows[0]
        assert stop_id == "stop2"
        assert stop_seq == 3
        assert feat[2] == 1                                   # stops_ahead
        assert feat[8] == pytest.approx(5.0, abs=1.0)         # remaining_dist_m
        assert feat[9]  == 8.0                                  # progress_speed_mps
        assert feat[13] == pytest.approx(5.0 / 8.0, abs=0.1)  # speed_eta_warm (speed known)
        assert feat[14] == pytest.approx(8.0, abs=0.1)         # hist_speed_mps from priors
        assert feat[15] == pytest.approx(1 * 45.0, abs=1.0)    # hist_travel_time_est (1 stop)
        assert len(feat) == len(FEATURE_COLS)

    def test_build_features_caps_horizon(self, gtfs):
        data = _compact_data(gtfs)
        snap = datetime(2026, 6, 3, 7, 0, tzinfo=timezone.utc)
        rows = build_features(TRIP_ID, 0.0, SPEED_UNKNOWN, snap, data)
        # Vehicle sits exactly at stop0 → that stop is already reached and
        # excluded; the remaining 4 are all within the 10-stop horizon.
        assert len(rows) == min(N_STOPS - 1, MAX_STOPS_AHEAD)
        # stops_ahead increments along the horizon
        assert [r[0][2] for r in rows] == list(range(1, len(rows) + 1))

    def test_progress_speed_state_machine(self):
        trackers: dict = {}
        assert progress_speed(trackers, "v1", TRIP_ID, 100.0, 1000.0) == SPEED_UNKNOWN
        speed = progress_speed(trackers, "v1", TRIP_ID, 250.0, 1015.0)
        assert speed == pytest.approx(10.0)
        # Same snapshot re-served (dt≈0, e.g. upstream slower than push
        # cadence): the last measurement is kept, not flipped to UNKNOWN
        assert progress_speed(trackers, "v1", TRIP_ID, 250.0, 1015.0) == pytest.approx(10.0)
        # Trip change resets the measurement
        assert progress_speed(trackers, "v1", "other", 50.0, 1030.0) == SPEED_UNKNOWN
        # Excessive gap → unknown
        assert progress_speed(trackers, "v1", "other", 500.0, 2000.0) == SPEED_UNKNOWN

    def test_stationary_seconds_accumulates_and_resets(self):
        from src.inference import stationary_seconds

        trackers: dict = {}
        # First sighting has no history — "moving" is the neutral answer.
        assert stationary_seconds(trackers, "v1", TRIP_ID, 100.0, 1000.0) == 0.0
        # Parked: jitter below _MOVE_EPS_M must not reset the anchor, so the
        # count keeps growing across pushes instead of restarting each time.
        assert stationary_seconds(trackers, "v1", TRIP_ID, 105.0, 1060.0) == 60.0
        assert stationary_seconds(trackers, "v1", TRIP_ID, 118.0, 1300.0) == 300.0
        # A real advance re-anchors.
        assert stationary_seconds(trackers, "v1", TRIP_ID, 400.0, 1330.0) == 0.0
        assert stationary_seconds(trackers, "v1", TRIP_ID, 402.0, 1390.0) == 60.0
        # Trip change re-anchors too — layover time is per trip, not per vehicle.
        assert stationary_seconds(trackers, "v1", "other", 402.0, 1400.0) == 0.0

    def test_stationary_seconds_matches_batch_labeling(self):
        """Live and batch must agree — they are hand-synced (see #2)."""
        from src.inference import stationary_seconds
        from src.labeling import _stationary_seconds

        dists = np.array([0.0, 5.0, 12.0, 300.0, 305.0, 310.0, 900.0])
        times = np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0])

        trackers: dict = {}
        live = [
            stationary_seconds(trackers, "v1", TRIP_ID, float(d), float(t))
            for d, t in zip(dists, times)
        ]
        np.testing.assert_allclose(live, _stationary_seconds(dists, times))

    def test_encode_trip_updates_direct_and_monotonic(self):
        now = datetime.now(tz=timezone.utc)
        updates = [{
            "vehicle_id": "v1",
            "trip_id": TRIP_ID,
            "route_id": ROUTE_ID,
            "snap_ts": now,
            "predictions": [
                {"stop_id": "stop1", "stop_sequence": 2, "seconds": 120.0},
                # Out-of-order prediction must be clamped, not summed
                {"stop_id": "stop2", "stop_sequence": 3, "seconds": 100.0},
                {"stop_id": "stop3", "stop_sequence": 4, "seconds": 300.0},
            ],
        }]
        from google.transit import gtfs_realtime_pb2
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(encode_trip_updates(updates, int(now.timestamp())))
        stus = feed.entity[0].trip_update.stop_time_update
        assert len(stus) == 3
        times = [s.arrival.time for s in stus]
        assert times == sorted(times)
        # Direct seconds, not cumulative: last stop is ~300 s out, not 520 s
        assert times[-1] - int(now.timestamp()) == pytest.approx(300, abs=2)


# ---------------------------------------------------------------------------
# Exported-tree parity with sklearn
# ---------------------------------------------------------------------------

class TestTreeExportParity:
    def test_predict_rows_matches_sklearn_predict(self):
        """The compact tree traversal must reproduce sklearn's predict exactly.

        Regression: HistGradientBoosting leaf values already include the
        learning rate (shrinkage is applied at fit time); multiplying by it
        again during traversal collapsed every prediction to ≈ baseline.
        """
        import os
        # export_worker_data reads R2 credentials at import time
        for var in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
            os.environ.setdefault(var, "test-dummy")
        from scripts.export_worker_data import _extract_trees
        from src.inference import predict_rows
        from src.train import _build_pipeline

        rng = np.random.default_rng(42)
        n = 2000
        stops_ahead    = rng.integers(1, 11, n)
        remaining_dist = rng.uniform(0, 5000, n)
        speed_mps      = rng.uniform(-1, 15, n)
        hist_speed     = rng.uniform(3, 10, n)
        hist_tps       = rng.uniform(20, 80, n)
        eff_speed      = np.where(speed_mps > 0, speed_mps, hist_speed)
        df = pd.DataFrame({
            "route_id": rng.choice(["10", "25", "117"], n),
            "stop_sequence": rng.integers(1, 60, n),
            "stops_ahead": stops_ahead,
            "hour": rng.integers(5, 23, n),
            "day_of_week": rng.integers(0, 7, n),
            "month": rng.integers(1, 13, n),
            "is_weekend": rng.integers(0, 2, n),
            "is_holiday": np.zeros(n, dtype=int),
            "remaining_dist_m": remaining_dist,
            "progress_speed_mps": speed_mps,
            "stops_remaining": rng.integers(1, 40, n),
            "trip_progress_frac": rng.uniform(0, 1, n),
            "dist_per_stop_m": remaining_dist / np.maximum(1, stops_ahead),
            "speed_eta_warm": remaining_dist / np.maximum(eff_speed, 0.1),
            "hist_speed_mps": hist_speed,
            "hist_travel_time_est": stops_ahead * hist_tps,
            # Mostly moving, with a stopped minority — the real shape of this
            # feature, so the traversal is exercised on both sides of its splits.
            "stationary_sec": np.where(rng.random(n) < 0.1, rng.uniform(0, 2400, n), 0.0),
        })
        y = (df["remaining_dist_m"] / 6.0 + rng.normal(0, 10, n)).clip(0)

        pipeline = _build_pipeline()
        pipeline.set_params(model__max_iter=25, model__early_stopping=False)
        pipeline.fit(df[FEATURE_COLS], y)

        tree_data = _extract_trees(pipeline)
        sample = df[FEATURE_COLS].head(100)
        compact = predict_rows(tree_data, sample.values.tolist())
        expected = pipeline.predict(sample)
        np.testing.assert_allclose(compact, expected, rtol=1e-9, atol=1e-6)


# ---------------------------------------------------------------------------
# Schedule interpolation helper
# ---------------------------------------------------------------------------

def test_sched_sec_at_dist_clamps_and_interpolates():
    profile = [("a", 1, 0.0, 0.0), ("b", 2, 100.0, 60.0), ("c", 3, 300.0, 120.0)]
    assert sched_sec_at_dist(profile, -5.0) == 0.0
    assert sched_sec_at_dist(profile, 50.0) == pytest.approx(30.0)
    assert sched_sec_at_dist(profile, 200.0) == pytest.approx(90.0)
    assert sched_sec_at_dist(profile, 999.0) == 120.0


# ---------------------------------------------------------------------------
# Code-review fixes (#1–#8)
# ---------------------------------------------------------------------------

import struct


def _pack_shape(pts):
    return struct.pack(f"{2 * len(pts)}d", *(v for xy in pts for v in xy))


class TestFeatureParity:
    """The training feature builder and the compact serving builder must emit
    byte-for-byte identical feature vectors — the positional tree export depends
    on it, and the order is otherwise only kept in sync by hand (#2)."""

    def test_inference_paths_agree(self, gtfs):
        v_dist, speed = 700.0, 6.0
        snap = datetime(2026, 6, 3, 7, 0, tzinfo=timezone.utc)

        # Same route+hour prior expressed in each path's own format.
        priors_features = {
            "lookup": {(ROUTE_ID, 7): (8.0, 45.0)},
            "global_speed": 5.0,
            "global_tps": 40.0,
        }
        feats_df = compute_features_for_inference(
            trip_id=TRIP_ID, vehicle_dist_m=v_dist, snapshot_time=snap,
            progress_speed_mps=speed, gtfs=gtfs, priors=priors_features,
        )
        compact_rows = build_features(TRIP_ID, v_dist, speed, snap, _compact_data(gtfs))

        assert len(feats_df) == len(compact_rows)
        for (_, frow), (cvec, _, _) in zip(feats_df.iterrows(), compact_rows):
            fvec = frow[FEATURE_COLS].tolist()
            assert str(fvec[0]) == str(cvec[0])          # route_id (string)
            np.testing.assert_allclose(
                np.array(fvec[1:], dtype=float),
                np.array(cvec[1:], dtype=float),
                rtol=1e-6, atol=1e-6,
            )


class TestBearingMatcher:
    """Overlapping opposite-direction shapes are told apart by heading (#1)."""

    def _data(self):
        fwd = _pack_shape([(0.0, 0.0), (1000.0, 0.0)])   # tangent → East (90°)
        rev = _pack_shape([(1000.0, 0.0), (0.0, 0.0)])   # tangent → West (270°)
        return {
            "shapes": {"f": fwd, "r": rev},
            "trip_index": {
                "tf": {"route_id": "R", "shape_id": "f", "stop_times": []},
                "tr": {"route_id": "R", "shape_id": "r", "stop_times": []},
            },
            "route_trips": {"R": ["tf", "tr"]},
        }

    def test_heading_picks_direction(self):
        data = self._data()
        tid_e, _, _ = infer_trip("R", None, 500.0, 0.0, 90.0, data)
        tid_w, _, _ = infer_trip("R", None, 500.0, 0.0, 270.0, data)
        assert tid_e == "tf"
        assert tid_w == "tr"

    def test_wrong_direction_reported_trip_rejected(self):
        data = self._data()
        # Reported trip says reverse, but the vehicle heads East → must override.
        tid, _, _ = infer_trip("R", "tr", 500.0, 0.0, 90.0, data)
        assert tid == "tf"

    def test_no_bearing_falls_back_to_distance(self):
        data = self._data()
        tid, dist, _ = infer_trip("R", None, 500.0, 0.0, None, data)
        assert tid in {"tf", "tr"}
        assert dist == pytest.approx(0.0, abs=1e-6)


class TestScheduleProgressMatcher:
    """Same-shape trips running at different times of day are indistinguishable
    by position/bearing alone — e.g. route 105's back-to-back weekday runs on
    the one shape it doesn't stop sharing with the weekend calendar. Only the
    schedule-progress term (now_sec) can tell them apart (#3)."""

    def _data(self):
        shape = _pack_shape([(0.0, 0.0), (1000.0, 0.0)])
        return {
            "shapes": {"s": shape},
            "shape_lengths": {"s": 1000.0},
            "stop_distances": {("s", "A"): 0.0, ("s", "B"): 1000.0},
            "trip_index": {
                "early": {
                    "route_id": "R", "shape_id": "s", "start_sec": 0.0,
                    "stop_times": [("A", 1, 0.0), ("B", 2, 600.0)],
                },
                "late": {
                    "route_id": "R", "shape_id": "s", "start_sec": 900.0,
                    "stop_times": [("A", 1, 0.0), ("B", 2, 600.0)],
                },
            },
            "route_trips": {"R": ["early", "late"]},
        }

    def test_ties_without_schedule_signal(self):
        data = self._data()
        # Halfway along the shared shape — spatially identical for both trips.
        tid, _, _ = infer_trip("R", None, 500.0, 0.0, None, data)
        assert tid in {"early", "late"}

    def test_schedule_progress_picks_the_running_trip(self):
        data = self._data()
        # now_sec=300: "early" (start 0) is halfway at x=500 right now;
        # "late" (start 900) hasn't started — expected at its first stop, x=0.
        tid, _, _ = infer_trip("R", None, 500.0, 0.0, None, data, now_sec=300.0)
        assert tid == "early"

        # now_sec=1200: "late" (start 900) is halfway at x=500 right now;
        # "early" (ended at 600) is long finished — expected at its last stop, x=1000.
        tid, _, _ = infer_trip("R", None, 500.0, 0.0, None, data, now_sec=1200.0)
        assert tid == "late"


class TestAmbiguousShapeProjection:
    """Self-intersecting shapes (out-and-back routes, tram turnarounds — e.g.
    route 122) revisit the same physical location at two far-apart
    distances-along-shape, so raw nearest-point projection can tie between
    them and pick either arbitrarily. vehicle_dist_along() resolves the tie
    using the schedule-implied position instead (#4).

    An earlier version clamped/confirmed jumps across pushes instead — that
    breaks for a real out-and-back trip, where re-approaching the
    intersection mid-route also looks like a "jump that gets confirmed".
    Measured live: made route 122 worse. This is the source-level fix, no
    per-vehicle state needed.
    """

    def _data(self, ambiguous):
        # Out-and-back: (0,0)->(1000,0) is the outbound leg (dist_along
        # 0-1000), a short hop up to (1000,1) (1000-1001), then the return
        # leg (1000,1)->(0,1) (1001-2001). A vehicle near x=500 sits ~1m
        # from BOTH legs — a real tie, not a rounding fluke. Two stops
        # spanning the whole shape 1:1 with scheduled seconds, so
        # _expected_dist_along(now_sec) == now_sec exactly — whichever leg
        # now_sec numerically lands near is the schedule-implied occurrence.
        shape = _pack_shape([(0.0, 0.0), (1000.0, 0.0), (1000.0, 1.0), (0.0, 1.0)])
        data = {
            "shapes": {"s": shape},
            "stop_distances": {("s", "origin"): 0.0, ("s", "terminus"): 2001.0},
            "trip_index": {
                "t": {
                    "shape_id": "s",
                    "start_sec": 0.0,
                    "stop_times": [("origin", 1, 0.0), ("terminus", 2, 2001.0)],
                },
            },
        }
        if ambiguous:
            data["ambiguous_shapes"] = {"s"}
        return data

    def test_ordinary_shape_ignores_schedule_hint(self):
        data = self._data(ambiguous=False)
        # now_sec=1501 would favor the return leg if this shape were
        # ambiguous — an ordinary shape ignores it: plain nearest point,
        # ties broken by walk order, wins (the outbound leg, walked first).
        d = vehicle_dist_along("t", 500.0, 0.5, data, now_sec=1501.0)
        assert d == pytest.approx(500.0, abs=1.0)

    def test_no_schedule_hint_falls_back_to_plain_nearest(self):
        data = self._data(ambiguous=True)
        d = vehicle_dist_along("t", 500.0, 0.5, data, now_sec=None)
        assert d == pytest.approx(500.0, abs=1.0)

    def test_tie_breaks_toward_outbound_when_schedule_says_early(self):
        data = self._data(ambiguous=True)
        d = vehicle_dist_along("t", 500.0, 0.5, data, now_sec=500.0)
        assert d == pytest.approx(500.0, abs=1.0)

    def test_tie_breaks_toward_return_leg_when_schedule_says_late(self):
        data = self._data(ambiguous=True)
        # Same geometry, same vehicle position — only the schedule-implied
        # time changes, and it alone flips which occurrence wins.
        d = vehicle_dist_along("t", 500.0, 0.5, data, now_sec=1501.0)
        assert d == pytest.approx(1501.0, abs=1.0)


class TestIsotonicMonotonicity:
    """PAVA distributes the correction instead of only pushing stops late (#4)."""

    def test_violator_pulled_down_not_up(self):
        # [120, 100] violate; a running max would report 120/120 (late bias),
        # the isotonic fit averages them to 110/110.
        assert _isotonic([120.0, 100.0, 300.0]) == [110.0, 110.0, 300.0]

    def test_already_monotone_unchanged(self):
        assert _isotonic([10.0, 20.0, 30.0]) == [10.0, 20.0, 30.0]

    def test_encode_uses_isotonic_value(self):
        now = datetime.now(tz=timezone.utc)
        updates = [{
            "vehicle_id": "v1", "trip_id": TRIP_ID, "route_id": ROUTE_ID,
            "snap_ts": now,
            "predictions": [
                {"stop_id": "stop1", "stop_sequence": 2, "seconds": 120.0},
                {"stop_id": "stop2", "stop_sequence": 3, "seconds": 100.0},
                {"stop_id": "stop3", "stop_sequence": 4, "seconds": 300.0},
            ],
        }]
        from google.transit import gtfs_realtime_pb2
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(encode_trip_updates(updates, int(now.timestamp())))
        stus = feed.entity[0].trip_update.stop_time_update
        # Middle stop reported ~110 s out (averaged), not bumped to 120.
        assert stus[1].arrival.time - int(now.timestamp()) == pytest.approx(110, abs=2)


def _vp_bytes(lat, lon, *, route_id=ROUTE_ID, trip_id=TRIP_ID, ts=None, bearing=None):
    from google.transit import gtfs_realtime_pb2
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.header.gtfs_realtime_version = "2.0"
    feed.header.timestamp = int(ts.timestamp()) if ts else int(datetime.now(timezone.utc).timestamp())
    ent = feed.entity.add()
    ent.id = "v1"
    ent.vehicle.vehicle.id = "v1"
    ent.vehicle.trip.trip_id = trip_id
    ent.vehicle.trip.route_id = route_id
    ent.vehicle.position.latitude = lat
    ent.vehicle.position.longitude = lon
    if bearing is not None:
        ent.vehicle.position.bearing = bearing
    return feed.SerializeToString()


class TestNotDepartedGate:
    """An idling vehicle is served only if it is standing at the terminus (#6).

    It used to be served nothing at all there; it is now served either the
    timetable or the model, depending on mode (see TestTerminusSchedule).  What
    survives from the original gate is that idling *short of* the first stop —
    on the lead-in, in a yard — still gets nothing.
    """

    def _model(self):
        # Constant +300 s predictor (no trees), so any emitted prediction lands
        # in the future and survives encoding — lets us count gated vehicles.
        return {"route_to_int": {ROUTE_ID: 0}, "baseline": 300.0, "trees": []}

    def _entities(self, vp):
        from google.transit import gtfs_realtime_pb2
        out = run_inference(self._gtfs_data, self._model(), {}, vp)
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(out)
        return len(feed.entity)

    def test_idle_short_of_the_first_stop_withheld(self, gtfs):
        self._gtfs_data = _compact_data(gtfs)
        self._gtfs_data["stop_distances"][(SHAPE_ID, "stop0")] = 400.0
        now = datetime.now(timezone.utc)
        # At the shape start, first sighting → speed unknown, v_dist ≈ 0, and
        # the terminus stop is 400 m further on → still gated.
        at_origin = _vp_bytes(LAT, _lon_at(2.0), ts=now)
        assert self._entities(at_origin) == 0
        # Mid-route, first sighting (speed still unknown) but v_dist large → served.
        mid = _vp_bytes(LAT, _lon_at(1200.0), ts=now)
        assert self._entities(mid) == 1

    def test_idle_at_the_first_stop_served(self, gtfs):
        """Standing at the terminus is served — that is the whole change."""
        self._gtfs_data = _compact_data(gtfs)
        now = datetime.now(timezone.utc)
        assert self._entities(_vp_bytes(LAT, _lon_at(2.0), ts=now)) == 1


# ---------------------------------------------------------------------------
# Schedule-anchored terminus ETAs (tram / trolleybus only)
# ---------------------------------------------------------------------------

class TestTerminusSchedule:
    """A tram or trolleybus idling at its terminus is served from the
    timetable instead of being withheld — its schedule holds there, unlike a
    bus's, and unlike either mode's mid-route schedule."""

    TRAM, TROLLEYBUS, BUS = 0, 11, 3

    def _model(self):
        return {"route_to_int": {ROUTE_ID: 0}, "baseline": 300.0, "trees": []}

    def _data(self, gtfs, route_type, depart_in_sec, now):
        """Compact data whose trip departs *depart_in_sec* from *now*."""
        local = now.astimezone(ZoneInfo("Europe/Kiev"))
        sod = local.hour * 3600 + local.minute * 60 + local.second
        return _compact_data(gtfs, route_type=route_type,
                             start_sec=sod + depart_in_sec)

    def _feed(self, gtfs, route_type, depart_in_sec, now, lat=LAT, meters=2.0,
              data=None):
        from google.transit import gtfs_realtime_pb2
        vp = _vp_bytes(lat, _lon_at(meters), ts=now)   # parked at the terminus
        out = run_inference(
            data if data is not None
            else self._data(gtfs, route_type, depart_in_sec, now),
            self._model(), {}, vp,
        )
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(out)
        return feed

    def test_tram_at_terminus_served_from_schedule(self, gtfs):
        now = datetime.now(timezone.utc)
        feed = self._feed(gtfs, self.TRAM, 300.0, now)
        assert len(feed.entity) == 1
        stus = feed.entity[0].trip_update.stop_time_update
        # stop0 sits at the vehicle's own position, so stop1 (500 m, +60 s of
        # scheduled running time) leads: 300 s of waiting + 60 s of driving.
        assert stus[0].stop_id == "stop1"
        assert stus[0].arrival.time == pytest.approx(now.timestamp() + 360, abs=2)
        # …and the stop after it another scheduled minute later.
        assert stus[1].arrival.time == pytest.approx(now.timestamp() + 420, abs=2)
        # Not model output — carries its own band, not the model's.
        assert stus[0].arrival.uncertainty == _TERMINUS_UNCERTAINTY_SEC

    def test_trolleybus_at_terminus_served(self, gtfs):
        now = datetime.now(timezone.utc)
        assert len(self._feed(gtfs, self.TROLLEYBUS, 300.0, now).entity) == 1

    def test_bus_at_terminus_served_from_the_model(self, gtfs):
        """A bus is served too, but never from the timetable — its schedule says
        nothing (MAE 222 s at best), so the model answers and the wide band
        says how little either of them knows."""
        now = datetime.now(timezone.utc)
        feed = self._feed(gtfs, self.BUS, 300.0, now)
        assert len(feed.entity) == 1
        stus = feed.entity[0].trip_update.stop_time_update
        # Constant +300 s model, untouched by the 300 s of scheduled waiting
        # (which would have put stop1 at +360 s, as it does for a tram).
        assert stus[0].arrival.time == pytest.approx(now.timestamp() + 300, abs=2)
        assert stus[0].arrival.uncertainty == _TERMINUS_MODEL_UNCERTAINTY_SEC

    def test_bus_terminus_ignores_the_timetable_entirely(self, gtfs):
        """Same bus, wildly different scheduled departure → same ETAs."""
        now = datetime.now(timezone.utc)
        early = self._feed(gtfs, self.BUS, 60.0, now).entity[0].trip_update
        late = self._feed(gtfs, self.BUS, 1500.0, now).entity[0].trip_update
        assert (early.stop_time_update[0].arrival.time
                == late.stop_time_update[0].arrival.time)

    def test_bus_short_of_the_terminus_stop_withheld(self, gtfs):
        """Serving buses at the terminus does not mean serving them anywhere
        near the shape start."""
        now = datetime.now(timezone.utc)
        data = self._data(gtfs, self.BUS, 300.0, now)
        data["stop_distances"][(SHAPE_ID, "stop0")] = 400.0
        assert len(self._feed(gtfs, self.BUS, 300.0, now, data=data).entity) == 0

    def test_moving_vehicle_keeps_the_model_bands(self, gtfs):
        """The wide terminus band must not leak onto ordinary predictions."""
        from google.transit import gtfs_realtime_pb2
        now = datetime.now(timezone.utc)
        data = self._data(gtfs, self.BUS, 300.0, now)
        vp = _vp_bytes(LAT, _lon_at(1200.0), ts=now)
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(run_inference(data, self._model(), {}, vp, ))
        arr = feed.entity[0].trip_update.stop_time_update[0].arrival
        assert not arr.HasField("uncertainty")   # no table passed → no field

    def test_departure_too_far_away_falls_back_to_the_model(self, gtfs):
        """An hour of parking means the matched trip is the wrong instance, so
        the timetable is dropped — but the tram is still served, like a bus."""
        now = datetime.now(timezone.utc)
        feed = self._feed(gtfs, self.TRAM, 3600.0, now)
        assert len(feed.entity) == 1
        stus = feed.entity[0].trip_update.stop_time_update
        assert stus[0].arrival.time == pytest.approx(now.timestamp() + 300, abs=2)
        assert stus[0].arrival.uncertainty == _TERMINUS_MODEL_UNCERTAINTY_SEC

    def test_slightly_overdue_departure_leaves_now(self, gtfs):
        now = datetime.now(timezone.utc)
        feed = self._feed(gtfs, self.TRAM, -60.0, now)
        assert len(feed.entity) == 1
        stus = feed.entity[0].trip_update.stop_time_update
        # Overdue wait clamps to zero rather than pulling arrivals into the past.
        assert stus[0].arrival.time == pytest.approx(now.timestamp() + 60, abs=2)

    def test_long_overdue_departure_falls_back_to_the_model(self, gtfs):
        """Parked well past its departure — a layover, not a pending run — so
        the timetable is dropped, but the tram is still served like a bus."""
        now = datetime.now(timezone.utc)
        feed = self._feed(gtfs, self.TRAM, -600.0, now)
        assert len(feed.entity) == 1
        arr = feed.entity[0].trip_update.stop_time_update[0].arrival
        assert arr.time == pytest.approx(now.timestamp() + 300, abs=2)
        assert arr.uncertainty == _TERMINUS_MODEL_UNCERTAINTY_SEC

    def test_parked_off_the_shape_but_on_route_still_served(self, gtfs):
        """A rail- or wire-bound vehicle 100 m off the drawn shape is still at
        the terminus — that offset is loop geometry, not a vehicle elsewhere."""
        now = datetime.now(timezone.utc)
        beside = LAT + 100.0 / 111_320.0    # ≈100 m north of the shape
        assert len(self._feed(gtfs, self.TRAM, 300.0, now, lat=beside).entity) == 1

    def test_parked_at_shape_start_but_short_of_the_stop_withheld(self, gtfs):
        """Some trips' first stop sits well into the shape; waiting on the
        lead-in to it is not standing at the terminus."""
        now = datetime.now(timezone.utc)
        data = self._data(gtfs, self.TRAM, 300.0, now)
        data["stop_distances"][(SHAPE_ID, "stop0")] = 400.0
        assert len(self._feed(gtfs, self.TRAM, 300.0, now, data=data).entity) == 0

    def test_legacy_export_without_route_types_uses_the_model(self, gtfs):
        """Data exported before this feature knows no modes, so nothing can be
        schedule-anchored and every terminus vehicle falls to the model."""
        from google.transit import gtfs_realtime_pb2
        now = datetime.now(timezone.utc)
        vp = _vp_bytes(LAT, _lon_at(2.0), ts=now)
        out = run_inference(_compact_data(gtfs), self._model(), {}, vp)
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(out)
        assert len(feed.entity) == 1
        arr = feed.entity[0].trip_update.stop_time_update[0].arrival
        assert arr.time == pytest.approx(now.timestamp() + 300, abs=2)
        assert arr.uncertainty == _TERMINUS_MODEL_UNCERTAINTY_SEC

    def test_moving_tram_uses_the_model_not_the_schedule(self, gtfs):
        """The schedule is trusted at the terminus only; once rolling, the
        vehicle goes back through the model."""
        from google.transit import gtfs_realtime_pb2
        now = datetime.now(timezone.utc)
        data = self._data(gtfs, self.TRAM, 300.0, now)
        vp = _vp_bytes(LAT, _lon_at(1200.0), ts=now)
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(run_inference(data, self._model(), {}, vp))
        stus = feed.entity[0].trip_update.stop_time_update
        # Constant +300 s model, not 300 s of waiting plus scheduled running.
        assert stus[0].arrival.time == pytest.approx(now.timestamp() + 300, abs=2)

    def test_service_day_wrap(self, gtfs):
        """A 23:50 departure seen at 00:05 is 15 min overdue, not 23 h early."""
        data = _compact_data(gtfs, route_type=self.TRAM,
                             start_sec=23 * 3600 + 50 * 60)
        at_five_past = datetime(2026, 6, 3, 0, 5, tzinfo=ZoneInfo("Europe/Kiev"))
        assert terminus_seconds_until_departure(TRIP_ID, at_five_past, data) is None
        # Mirror case: an after-midnight trip (start_sec > 86400) seen just
        # before it is due, from the previous calendar day's clock.
        late = _compact_data(gtfs, route_type=self.TRAM,
                             start_sec=24 * 3600 + 10 * 60)
        at_midnight = datetime(2026, 6, 3, 0, 5, tzinfo=ZoneInfo("Europe/Kiev"))
        assert terminus_seconds_until_departure(TRIP_ID, at_midnight, late) == 300.0


# ---------------------------------------------------------------------------
# Feature 1: per-horizon prediction uncertainty (GTFS-RT StopTimeEvent.uncertainty)
# ---------------------------------------------------------------------------

class TestUncertainty:
    def test_lookup_clamps_to_widest_band(self):
        table = {1: 20, 2: 35, 3: 60}
        assert _uncertainty_for(table, 1) == 20
        assert _uncertainty_for(table, 3) == 60
        # Beyond the largest measured horizon → reuse the widest band, not None.
        assert _uncertainty_for(table, 9) == 60
        # No table → field omitted entirely.
        assert _uncertainty_for(None, 2) is None
        assert _uncertainty_for({}, 2) is None

    def test_encode_emits_uncertainty_by_horizon(self):
        now = datetime.now(tz=timezone.utc)
        updates = [{
            "vehicle_id": "v1", "trip_id": TRIP_ID, "route_id": ROUTE_ID,
            "snap_ts": now,
            "predictions": [
                {"stop_id": "stop1", "stop_sequence": 2, "stops_ahead": 1, "seconds": 100.0},
                {"stop_id": "stop2", "stop_sequence": 3, "stops_ahead": 2, "seconds": 200.0},
            ],
        }]
        from google.transit import gtfs_realtime_pb2
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(
            encode_trip_updates(updates, int(now.timestamp()), {1: 15, 2: 40})
        )
        stus = feed.entity[0].trip_update.stop_time_update
        assert stus[0].arrival.uncertainty == 15
        assert stus[0].departure.uncertainty == 15
        assert stus[1].arrival.uncertainty == 40

    def test_uncertainty_absent_when_no_table(self):
        now = datetime.now(tz=timezone.utc)
        updates = [{
            "vehicle_id": "v1", "trip_id": TRIP_ID, "route_id": ROUTE_ID,
            "snap_ts": now,
            "predictions": [
                {"stop_id": "stop1", "stop_sequence": 2, "seconds": 100.0},
            ],
        }]
        from google.transit import gtfs_realtime_pb2
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(encode_trip_updates(updates, int(now.timestamp())))
        arr = feed.entity[0].trip_update.stop_time_update[0].arrival
        assert not arr.HasField("uncertainty")


# ---------------------------------------------------------------------------
# Feature 2: cleaned VehiclePositions feed
# ---------------------------------------------------------------------------

class TestVehiclePositions:
    def _model(self):
        return {"route_to_int": {ROUTE_ID: 0}, "baseline": 300.0, "trees": []}

    def test_run_inference_returns_both_feeds(self, gtfs):
        from google.transit import gtfs_realtime_pb2
        data = _compact_data(gtfs)
        now = datetime.now(timezone.utc)
        vp = _vp_bytes(LAT, _lon_at(1200.0), ts=now, bearing=90.0)

        out = run_inference(data, self._model(), {}, vp, with_vehicle_positions=True)
        assert isinstance(out, tuple) and len(out) == 2
        tu_bytes, vp_bytes = out

        vpos = gtfs_realtime_pb2.FeedMessage()
        vpos.ParseFromString(vp_bytes)
        assert len(vpos.entity) == 1
        v = vpos.entity[0].vehicle
        assert v.trip.trip_id == TRIP_ID          # corrected match carried through
        assert v.vehicle.id == "v1"
        assert v.position.latitude == pytest.approx(LAT)
        assert v.HasField("current_stop_sequence")
        # Next stop after 1200 m is stop3 (at 1500 m) → IN_TRANSIT_TO it.
        assert v.current_status == gtfs_realtime_pb2.VehiclePosition.IN_TRANSIT_TO

    def test_withheld_vehicle_still_gets_position(self, gtfs):
        """A vehicle whose ETAs are withheld — idling short of the terminus
        stop — still has its cleaned position published: the whole point of
        the VP feed."""
        from google.transit import gtfs_realtime_pb2
        data = _compact_data(gtfs)
        data["stop_distances"][(SHAPE_ID, "stop0")] = 400.0
        now = datetime.now(timezone.utc)
        vp = _vp_bytes(LAT, _lon_at(2.0), ts=now)

        tu_bytes, vp_bytes = run_inference(
            data, self._model(), {}, vp, with_vehicle_positions=True
        )
        tu = gtfs_realtime_pb2.FeedMessage(); tu.ParseFromString(tu_bytes)
        vpos = gtfs_realtime_pb2.FeedMessage(); vpos.ParseFromString(vp_bytes)
        assert len(tu.entity) == 0    # ETA withheld (short of the terminus)
        assert len(vpos.entity) == 1  # position still served

    def test_default_call_returns_only_trip_updates(self, gtfs):
        """Backward-compat: without the flag, the return type is plain bytes."""
        data = _compact_data(gtfs)
        now = datetime.now(timezone.utc)
        vp = _vp_bytes(LAT, _lon_at(1200.0), ts=now)
        out = run_inference(data, self._model(), {}, vp)
        assert isinstance(out, (bytes, bytearray))

    def test_encode_vehicle_positions_congestion(self):
        from google.transit import gtfs_realtime_pb2
        VP = gtfs_realtime_pb2.VehiclePosition
        records = [{
            "vehicle_id": "v1", "trip_id": TRIP_ID, "route_id": ROUTE_ID,
            "lat": LAT, "lon": LON0, "bearing": 90.0,
            "stop_id": "stop3", "stop_sequence": 3, "status": VP.STOPPED_AT,
            "congestion": VP.SEVERE_CONGESTION,
        }]
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(encode_vehicle_positions(records, int(datetime.now(timezone.utc).timestamp())))
        v = feed.entity[0].vehicle
        assert v.current_status == VP.STOPPED_AT
        assert v.congestion_level == VP.SEVERE_CONGESTION
        assert v.stop_id == "stop3"


# ---------------------------------------------------------------------------
# Staleness anchoring (upstream clock skew)
# ---------------------------------------------------------------------------

def _skewed_feed(feed_ts, vehicle_ages):
    """A VehiclePositions feed whose vehicles are *ages* seconds behind feed_ts."""
    from google.transit import gtfs_realtime_pb2
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.header.gtfs_realtime_version = "2.0"
    feed.header.timestamp = feed_ts
    for i, age in enumerate(vehicle_ages):
        ent = feed.entity.add()
        ent.id = f"v{i}"
        ent.vehicle.vehicle.id = f"v{i}"
        if age is not None:
            ent.vehicle.timestamp = feed_ts - age
    return feed


class TestStalenessReference:
    """Per-vehicle staleness is judged against the feed's newest fix, not the
    header, so an upstream clock skew doesn't drop the entire fleet (#live
    2026-07-31: every one of 590 vehicles 9+ min behind the header, positions
    still moving, and we published an empty feed for it)."""

    FEED_TS = 1_785_490_000

    def test_normal_feed_anchors_to_the_header(self):
        """Ordinary reporting jitter must not move the reference at all."""
        feed = _skewed_feed(self.FEED_TS, [2, 12, 60])
        ref, skew = staleness_reference(feed, self.FEED_TS)
        assert ref == self.FEED_TS
        assert skew == 2

    def test_anchor_holds_until_the_fleet_is_wholly_stale(self):
        feed = _skewed_feed(self.FEED_TS, [STALE_VEHICLE_MAX_AGE_SEC, 400])
        ref, _ = staleness_reference(feed, self.FEED_TS)
        assert ref == self.FEED_TS

    def test_fleet_wide_skew_slides_the_anchor(self):
        feed = _skewed_feed(self.FEED_TS, [544, 556, 600])
        ref, skew = staleness_reference(feed, self.FEED_TS)
        assert skew == 544
        assert ref == self.FEED_TS - 544

    def test_slid_anchor_keeps_the_fleet_and_still_drops_stragglers(self):
        # The live shape of the outage: a skewed fleet plus one month-old fix.
        ages = [544, 556, 600, 30 * 86_400]
        feed = _skewed_feed(self.FEED_TS, ages)
        ref, _ = staleness_reference(feed, self.FEED_TS)
        kept = [a for a in ages if ref - (self.FEED_TS - a) <= STALE_VEHICLE_MAX_AGE_SEC]
        assert kept == [544, 556, 600]
        # Measured against the header instead, every one of them would go.
        assert not [a for a in ages if a <= STALE_VEHICLE_MAX_AGE_SEC]

    def test_skew_past_the_cap_falls_back_to_the_header(self):
        """Past the cap we publish nothing rather than trust an ancient feed."""
        age = MAX_FEED_CLOCK_SKEW_SEC + 60
        feed = _skewed_feed(self.FEED_TS, [age, age + 10])
        ref, skew = staleness_reference(feed, self.FEED_TS)
        assert ref == self.FEED_TS
        assert skew == age

    def test_future_timestamps_do_not_slide_the_anchor(self):
        """A vehicle clock ahead of the header must not widen the age window."""
        feed = _skewed_feed(self.FEED_TS, [-120, 30])
        ref, skew = staleness_reference(feed, self.FEED_TS)
        assert ref == self.FEED_TS
        assert skew == 0

    def test_feed_without_vehicle_timestamps(self):
        feed = _skewed_feed(self.FEED_TS, [None, None])
        assert staleness_reference(feed, self.FEED_TS) == (self.FEED_TS, 0)


class TestSkewedFeedInference:
    """End-to-end: a skewed-but-live feed is still served, and the pass reports
    what it saw so /health can name the upstream as the cause."""

    def _model(self):
        return {"route_to_int": {ROUTE_ID: 0}, "baseline": 300.0, "trees": []}

    def _vp(self, now, *, vehicle_age):
        from google.transit import gtfs_realtime_pb2
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(_vp_bytes(LAT, _lon_at(1200.0), ts=now))
        feed.entity[0].vehicle.timestamp = int(now.timestamp()) - vehicle_age
        return feed.SerializeToString()

    def _entities(self, data, vp, stats=None):
        from google.transit import gtfs_realtime_pb2
        out = run_inference(data, self._model(), {}, vp, stats=stats)
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(out)
        return len(feed.entity)

    def test_whole_feed_skewed_is_still_served(self, gtfs):
        data = _compact_data(gtfs)
        now = datetime.now(timezone.utc)
        stats = {}
        assert self._entities(data, self._vp(now, vehicle_age=600), stats) == 1
        assert stats["feed_skew_sec"] == 600
        assert stats["vehicles_stale"] == 0

    def test_stats_report_a_fully_dropped_fleet(self, gtfs):
        """Skew past the cap: nothing served, and the counts say why."""
        data = _compact_data(gtfs)
        now = datetime.now(timezone.utc)
        stats = {}
        age = MAX_FEED_CLOCK_SKEW_SEC + 600
        assert self._entities(data, self._vp(now, vehicle_age=age), stats) == 0
        assert stats == {
            "vehicles_in": 1,
            "vehicles_stale": 1,
            "feed_skew_sec": age,
        }

    def test_stats_are_optional(self, gtfs):
        data = _compact_data(gtfs)
        now = datetime.now(timezone.utc)
        assert self._entities(data, self._vp(now, vehicle_age=5)) == 1


# ---------------------------------------------------------------------------
# Trip-instance identity, horizon fencing, dwell and past-due clamping
# ---------------------------------------------------------------------------

def _encode_one(predictions, **extra):
    """Encode a single trip update and return its parsed TripUpdate."""
    from google.transit import gtfs_realtime_pb2
    now = datetime.now(tz=timezone.utc)
    updates = [{
        "vehicle_id": "v1", "trip_id": TRIP_ID, "route_id": ROUTE_ID,
        "snap_ts": now, "predictions": predictions, **extra,
    }]
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(
        encode_trip_updates(updates, int(now.timestamp()),
                            extra.pop("_bands", None))
    )
    return feed


class TestPastDueClamping:
    """An already-elapsed prediction is published as 'arriving now', not dropped.

    Dropping it removed the most useful arrival there is and renumbered every
    later stop in the emitted list — which is the horizon the quality scorer
    reads, so serving and calibration keyed off different horizons.
    """

    def test_elapsed_prediction_is_kept_and_clamped(self):
        feed = _encode_one([
            {"stop_id": "stop1", "stop_sequence": 2, "stops_ahead": 1, "seconds": -300.0},
            {"stop_id": "stop2", "stop_sequence": 3, "stops_ahead": 2, "seconds": 400.0},
        ])
        stus = feed.entity[0].trip_update.stop_time_update
        assert len(stus) == 2, "the elapsed stop must not vanish from the feed"
        now = feed.header.timestamp
        assert stus[0].arrival.time > now
        assert stus[0].arrival.time <= now + 2, "elapsed arrival should read as 'now'"

    def test_emitted_position_matches_true_horizon(self):
        """With nothing dropped, list position *is* stops_ahead — which is what
        src.scoring._parse_prediction_feed measures the horizon by."""
        preds = [
            {"stop_id": f"stop{i}", "stop_sequence": i + 1, "stops_ahead": i,
             "seconds": -100.0 + 60.0 * i}
            for i in range(1, 5)
        ]
        stus = _encode_one(preds).entity[0].trip_update.stop_time_update
        assert len(stus) == len(preds)
        for emitted_position, pred in enumerate(preds, start=1):
            assert pred["stops_ahead"] == emitted_position


class TestHorizonFence:
    def test_no_data_sentinel_appended(self):
        feed = _encode_one(
            [{"stop_id": "stop1", "stop_sequence": 2, "stops_ahead": 1, "seconds": 100.0}],
            horizon_end={"stop_id": "stop9", "stop_sequence": 9},
        )
        from google.transit import gtfs_realtime_pb2
        stus = feed.entity[0].trip_update.stop_time_update
        assert len(stus) == 2
        fence = stus[-1]
        assert (fence.schedule_relationship
                == gtfs_realtime_pb2.TripUpdate.StopTimeUpdate.NO_DATA)
        assert fence.stop_id == "stop9" and fence.stop_sequence == 9
        # A time on the fence would defeat its purpose.
        assert not fence.HasField("arrival")
        assert not fence.HasField("departure")

    def test_absent_without_a_stop_past_the_horizon(self):
        feed = _encode_one(
            [{"stop_id": "stop1", "stop_sequence": 2, "stops_ahead": 1, "seconds": 100.0}]
        )
        assert len(feed.entity[0].trip_update.stop_time_update) == 1

    def test_stop_after_horizon_is_the_eleventh_upcoming_stop(self, gtfs):
        from src.inference import stop_after_horizon
        data = _compact_data(gtfs)
        # The fake route has only N_STOPS (5) stops — nothing past the horizon.
        assert stop_after_horizon(TRIP_ID, 0.0, data) is None

        # A trip long enough to overflow the cap fences at stop index 10 (0-based)
        # of the stops still ahead.
        n = MAX_STOPS_AHEAD + 4
        data["trip_index"]["long"] = {
            "route_id": ROUTE_ID, "shape_id": "L",
            "stop_times": [(f"s{i}", i + 1, 60.0 * i) for i in range(n)],
            "start_sec": None,
        }
        data["stop_distances"].update({("L", f"s{i}"): 100.0 * i for i in range(n)})
        # s0 sits exactly at the vehicle, so the stops *ahead* are s1..s13 and
        # the ten predicted ones are s1..s10 — the fence lands on s11, matching
        # build_features' own `d_target <= v_dist` skip.
        fence = stop_after_horizon("long", 0.0, data)
        assert fence == {"stop_id": f"s{MAX_STOPS_AHEAD + 1}",
                         "stop_sequence": MAX_STOPS_AHEAD + 2}
        predicted = build_features("long", 0.0, 5.0,
                                   datetime.now(timezone.utc), data)
        assert [stop_id for _row, stop_id, _seq in predicted][-1] == f"s{MAX_STOPS_AHEAD}"


class TestTripInstanceFields:
    def test_encoder_publishes_instance_identity(self):
        tu = _encode_one(
            [{"stop_id": "stop1", "stop_sequence": 2, "stops_ahead": 1, "seconds": 100.0}],
            start_date="20260809", start_time="07:15:00", direction_id=1,
            trip_ts=1786000000, delay=-42,
        ).entity[0].trip_update
        assert tu.trip.start_date == "20260809"
        assert tu.trip.start_time == "07:15:00"
        assert tu.trip.direction_id == 1
        assert tu.timestamp == 1786000000
        assert tu.delay == -42

    def test_fields_omitted_when_unknown(self):
        """An export predating these keys must produce the old feed exactly."""
        tu = _encode_one(
            [{"stop_id": "stop1", "stop_sequence": 2, "stops_ahead": 1, "seconds": 100.0}]
        ).entity[0].trip_update
        assert not tu.trip.HasField("start_date")
        assert not tu.trip.HasField("start_time")
        assert not tu.trip.HasField("direction_id")
        assert not tu.HasField("timestamp")
        assert not tu.HasField("delay")

    def test_context_resolves_the_service_day(self, gtfs):
        from src.inference import trip_instance_context
        data = _compact_data(gtfs, start_sec=7 * 3600.0)
        snap = datetime(2026, 8, 9, 5, 30, tzinfo=timezone.utc)  # 08:30 Kyiv
        start_date, start_time, sched = trip_instance_context(
            TRIP_ID, snap, data, 8.5 * 3600, ZoneInfo("Europe/Kiev")
        )
        assert start_date == "20260809"
        assert start_time == "07:00:00"
        # 60 s per stop from a 07:00 local start.
        assert sched[1] == pytest.approx(sched[2] - 60.0)

    def test_context_is_empty_without_a_schedule(self, gtfs):
        from src.inference import trip_instance_context
        data = _compact_data(gtfs)  # start_sec is None
        assert trip_instance_context(
            TRIP_ID, datetime.now(timezone.utc), data, 0.0, timezone.utc
        ) == (None, None, {})

    def test_delay_only_for_modes_whose_timetable_holds(self):
        from src.inference import publishes_delay
        assert publishes_delay(0) is True     # tram
        assert publishes_delay(11) is True    # trolleybus
        assert publishes_delay(3) is False    # bus — Lviv timetable is noise
        assert publishes_delay(None) is False


class TestDwell:
    def test_measured_table_beats_the_flat_fallback(self):
        from src.inference import _DWELL_SECS, _dwell_for
        table = {0: 32, 3: 27, "_global": 28}
        assert _dwell_for(table, 0) == 32
        assert _dwell_for(table, 3) == 27
        assert _dwell_for(table, 11) == 28       # unmeasured type → global
        assert _dwell_for(None, 0) == _DWELL_SECS  # older export → old behaviour

    def test_encoder_uses_the_supplied_dwell(self):
        stus = _encode_one(
            [{"stop_id": "stop1", "stop_sequence": 2, "stops_ahead": 1, "seconds": 100.0}],
            dwell=32,
        ).entity[0].trip_update.stop_time_update
        assert stus[0].departure.time - stus[0].arrival.time == 32


class TestMatcherMemoisation:
    """The per-push caches and the bbox prune are optimisations: they must not
    move a single prediction."""

    def _many_trips(self):
        data = {
            "shapes": {
                "near": _pack_shape([(0.0, 0.0), (1000.0, 0.0)]),
                "far":  _pack_shape([(0.0, 50000.0), (1000.0, 50000.0)]),
            },
            "shape_lengths": {"near": 1000.0, "far": 1000.0},
            "stop_distances": {("near", "A"): 0.0, ("near", "B"): 1000.0,
                               ("far", "A"): 0.0, ("far", "B"): 1000.0},
            "trip_index": {}, "route_trips": {"R": []},
        }
        for i in range(20):
            shape = "near" if i % 2 else "far"
            tid = f"t{i}"
            data["trip_index"][tid] = {
                "route_id": "R", "shape_id": shape, "start_sec": float(i * 60),
                "stop_times": [("A", 1, 0.0), ("B", 2, 600.0)],
            }
            data["route_trips"]["R"].append(tid)
        return data

    def test_bbox_prune_does_not_change_the_match(self):
        import copy
        plain = self._many_trips()
        boxed = copy.deepcopy(plain)
        boxed["shape_bboxes"] = {
            "near": (0.0, 0.0, 1000.0, 0.0),
            "far":  (0.0, 50000.0, 1000.0, 50000.0),
        }
        for now_sec in (None, 300.0, 1200.0):
            assert infer_trip("R", None, 500.0, 0.0, 90.0, plain, now_sec=now_sec) == \
                   infer_trip("R", None, 500.0, 0.0, 90.0, boxed, now_sec=now_sec)

    def test_expected_dist_cache_matches_the_uncached_result(self):
        from src.inference import _expected_dist_along
        data = self._many_trips()
        cache: dict = {}
        for tid in data["trip_index"]:
            uncached = _expected_dist_along(tid, 400.0, data)
            assert _expected_dist_along(tid, 400.0, data, cache) == uncached
            # Second hit comes from the cache and must agree with itself.
            assert _expected_dist_along(tid, 400.0, data, cache) == uncached

    def test_caches_do_not_leak_between_data_dicts(self):
        """The memos live on the data dict, so two exports never mix."""
        from src.inference import _trip_sched_arrays
        a = self._many_trips()
        b = self._many_trips()
        b["trip_index"]["t1"]["start_sec"] = None
        assert _trip_sched_arrays("t1", a) is not None
        assert _trip_sched_arrays("t1", b) is None
