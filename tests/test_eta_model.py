"""
Unit tests for the snapshot-anchored training data and the compact
production inference path.

The regression these tests guard: a bus sitting *at* a stop must get a
near-zero remaining distance (and therefore a near-zero ETA from a model
trained on snapshot-anchored rows), never the full previous-segment time.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from src.features import (
    FEATURE_COLS,
    SPEED_UNKNOWN,
    TARGET_COL,
    compute_features_for_inference,
    compute_features_for_training,
    sched_sec_at_dist,
)
from src.gtfs_static import StopTime, TripInfo
from src.inference import (
    MAX_STOPS_AHEAD,
    build_features,
    encode_trip_updates,
    progress_speed,
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
        for col in FEATURE_COLS + [TARGET_COL, "date"]:
            assert col in feats.columns, col
        assert len(feats) == len(rows)
        assert feats[FEATURE_COLS[1:]].notna().all().all()

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

def _compact_data(gtfs: FakeGTFS) -> dict:
    import struct
    coords = list(gtfs._shape.coords)
    shape_bytes = struct.pack(f"{2 * len(coords)}d", *(v for xy in coords for v in xy))
    # Cumulative scheduled seconds: 60 s per stop (matches FakeGTFS times)
    stop_times = [(f"stop{i}", i + 1, 60.0 * i) for i in range(N_STOPS)]
    return {
        "shapes": {SHAPE_ID: shape_bytes},
        "shape_lengths": {SHAPE_ID: gtfs._shape.length},
        "stop_distances": dict(gtfs._stop_dists),
        "trip_index": {
            TRIP_ID: {"route_id": ROUTE_ID, "shape_id": SHAPE_ID, "stop_times": stop_times}
        },
        "route_trips": {ROUTE_ID: [TRIP_ID]},
    }


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
        assert feat[9] == pytest.approx(0.6, abs=0.5)         # sched_remaining_sec
        assert feat[10] == 8.0                                # progress_speed_mps
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
        df = pd.DataFrame({
            "route_id": rng.choice(["10", "25", "117"], n),
            "stop_sequence": rng.integers(1, 60, n),
            "stops_ahead": rng.integers(1, 11, n),
            "hour": rng.integers(5, 23, n),
            "day_of_week": rng.integers(0, 7, n),
            "month": rng.integers(1, 13, n),
            "is_weekend": rng.integers(0, 2, n),
            "is_holiday": np.zeros(n, dtype=int),
            "remaining_dist_m": rng.uniform(0, 5000, n),
            "sched_remaining_sec": rng.uniform(0, 1200, n),
            "progress_speed_mps": rng.uniform(-1, 15, n),
            "stops_remaining": rng.integers(1, 40, n),
            "trip_progress_frac": rng.uniform(0, 1, n),
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
