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
    MAX_STOPS_AHEAD,
    _isotonic,
    build_features,
    encode_trip_updates,
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
        "route_hour_priors": {
            f"{ROUTE_ID}:7": (8.0, 45.0),  # (hist_speed_mps, hist_time_per_stop_sec)
            "_global": (5.0, 40.0),
        },
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
    """An idling vehicle at the origin gets no predictions; a moving one does (#6)."""

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

    def test_origin_idle_withheld_midroute_served(self, gtfs):
        self._gtfs_data = _compact_data(gtfs)
        now = datetime.now(timezone.utc)
        # At the shape start, first sighting → speed unknown, v_dist ≈ 0 → gated.
        at_origin = _vp_bytes(LAT, _lon_at(2.0), ts=now)
        assert self._entities(at_origin) == 0
        # Mid-route, first sighting (speed still unknown) but v_dist large → served.
        mid = _vp_bytes(LAT, _lon_at(1200.0), ts=now)
        assert self._entities(mid) == 1
