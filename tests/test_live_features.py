"""Tests for the live_v2 model path: LiveState semantics, the training replay
driver, the flattened-tree walker against sklearn, and run_inference."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.live_features import (
    LIVE_COLS, MISSING, LiveState, LiveTables, TripGeom, replay_day, side_paths,
    stationary_runs,
)

C = {c: i for i, c in enumerate(LIVE_COLS)}
T0 = datetime(2026, 9, 21, 9, 0, tzinfo=timezone.utc).timestamp()  # Mon 12:00 Kyiv
GEOM = TripGeom([0.0, 500.0, 1000.0, 1500.0, 2000.0], ["s0", "s1", "s2", "s3", "s4"])


def _geom(trip_id):
    return GEOM if trip_id in ("t1", "t2") else None


def _drive(st, vid, trip, t0, speed_mps, t_end, step=10.0, start=0.0):
    """Feed one vehicle moving at constant speed; returns its last (t, d)."""
    t, d = t0, start
    while t <= t_end:
        st.observe_feed(vid, t, t - 5, speed_mps, (10_000 + d) / 1000.0)
        st.observe_position(vid, trip, t, d, _geom(trip))
        t += step
        d += speed_mps * step
    return t - step, d - speed_mps * step


class TestLinkStore:
    def test_traversals_are_detected_and_summed_along_the_path(self):
        st = LiveState()
        # vehicle A at 10 m/s: every 500 m link takes 50 s
        _drive(st, "A", "t1", T0, 10.0, T0 + 200)
        assert set(st.links) == {"s0>s1", "s1>s2", "s2>s3", "s3>s4"}
        for lst in st.links.values():
            assert lst[-1][2] == pytest.approx(50.0)
        # vehicle B just past s0 asks for s2 (index 2) and s4 (index 4)
        t = T0 + 300
        st.observe_feed("B", t, t, 5.0, 1.0)
        st.observe_position("B", "t1", t, 250.0, GEOM)
        rows = st.features("B", "t1", t, 250.0, GEOM, [2, 4], 0.0)
        # half of link s0>s1 left + one more link = 1.5 links x 50 s
        assert rows[0][C["live_path_sec_m3"]] == pytest.approx(75.0)
        assert rows[1][C["live_path_sec_m5"]] == pytest.approx(175.0)
        assert rows[0][C["live_path_cov"]] == pytest.approx(1.0)
        assert rows[0][C["live_speed_mps"]] == pytest.approx(750.0 / 75.0)
        assert rows[0][C["cur_link_frac"]] == pytest.approx(0.5)
        assert rows[0][C["cur_link_live"]] == pytest.approx(50.0)
        # no historical tables, no own lap
        assert rows[0][C["hist_path_sec"]] == MISSING
        assert rows[0][C["hist_path_cov"]] == 0.0
        assert rows[0][C["lap_path_cov"]] == 0.0

    def test_traversal_detected_this_snapshot_is_not_visible_yet(self):
        st = LiveState()
        last_t, _ = _drive(st, "A", "t1", T0, 10.0, T0 + 50)   # crosses s1 at T0+50
        rows = st.features("B", "t1", last_t, 100.0, GEOM, [1], 0.0)
        assert rows[0][C["live_path_cov"]] == 0.0
        rows = st.features("B", "t1", last_t + 10, 100.0, GEOM, [1], 0.0)
        assert rows[0][C["live_path_cov"]] == pytest.approx(1.0)

    def test_stale_traversals_expire(self):
        st = LiveState()
        _drive(st, "A", "t1", T0, 10.0, T0 + 60)
        rows = st.features("B", "t1", T0 + 60 + 1900, 100.0, GEOM, [1], 0.0)
        assert rows[0][C["live_path_cov"]] == 0.0

    def test_own_previous_lap_and_vehicle_ratio(self):
        tables = LiveTables(hist_k={"s0>s1": 100.0, "s1>s2": 100.0, "s2>s3": 100.0, "s3>s4": 100.0})
        st = LiveState(tables)
        _drive(st, "A", "t1", T0, 10.0, T0 + 200)
        # next lap of A on another trip id over the same stops
        t = T0 + 900
        st.observe_feed("A", t, t, 0.0, 12.0)
        st.observe_position("A", "t2", t, 100.0, GEOM)
        rows = st.features("A", "t2", t + 10, 100.0, GEOM, [1, 2], 0.0)
        assert rows[1][C["lap_path_sec"]] == pytest.approx((0.8 + 1.0) * 50.0)
        assert rows[1][C["lap_path_cov"]] == pytest.approx(1.0)
        # own links ran at 50 s against a 100 s history
        assert rows[0][C["veh_hist_ratio60"]] == pytest.approx(0.5)
        assert rows[0][C["veh_hist_n60"]] == 4.0
        assert rows[1][C["hist_path_sec"]] == pytest.approx(180.0)


class TestVehicleFeatures:
    def test_own_speed_feed_and_odometer(self):
        st = LiveState()
        last_t, last_d = _drive(st, "A", "t1", T0, 5.0, T0 + 1300)
        rows = st.features("A", "t1", last_t, last_d, GEOM, [4], 0.0)
        r = rows[0]
        assert r[C["own_speed_60"]] == pytest.approx(5.0)
        assert r[C["own_speed_300"]] == pytest.approx(5.0)
        assert r[C["pos_age_s"]] == pytest.approx(5.0)
        assert r[C["pos_age_med600"]] == pytest.approx(5.0)
        assert r[C["pos_age_dist"]] == pytest.approx(25.0)
        assert r[C["fs_now"]] == pytest.approx(5.0)
        assert r[C["fs_mean60"]] == pytest.approx(5.0)
        assert r[C["fs_zero_frac600"]] == 0.0
        assert r[C["odo_spd_60"]] == pytest.approx(5.0)
        assert r[C["odo_spd_1200"]] == pytest.approx(5.0)
        assert r[C["odo_m_300"]] == pytest.approx(1500.0)

    def test_cold_vehicle_is_all_missing_except_counts(self):
        st = LiveState()
        r = st.features("X", "t1", T0, 100.0, GEOM, [1], 0.0)[0]
        zero = {"live_path_cov", "hist_path_cov", "lap_path_cov", "veh_hist_n60", "dwell_n"}
        for c, i in C.items():
            if c in zero:
                assert r[i] == 0.0, c
            elif c != "cur_link_frac":
                assert r[i] == MISSING, c

    def test_dwell_table(self):
        tables = LiveTables(dwell={"s1@0": np.array([10, 20, 30, 40, 60, 200, 300], dtype=np.float32)})
        st = LiveState(tables)
        r = st.features("A", "t1", T0, 520.0, GEOM, [2], 25.0)[0]
        # runs longer than 25 s: 30, 40, 60, 200, 300 -> median 60, minus 25
        assert r[C["dwell_n"]] == 5.0
        assert r[C["dwell_rem_med"]] == pytest.approx(35.0)
        assert r[C["dwell_p_more120"]] == pytest.approx(2 / 5)
        assert r[C["dwell_long_share"]] == pytest.approx(2 / 7)


class TestPersistence:
    def test_dump_load_round_trip_answers_identically(self):
        st = LiveState()
        _drive(st, "A", "t1", T0, 10.0, T0 + 200)
        _drive(st, "B", "t1", T0 + 100, 7.0, T0 + 260)
        blob = st.dump()
        import pickle
        st2 = LiveState()
        st2.load(pickle.loads(pickle.dumps(blob)), _geom)
        for s in (st, st2):
            s.observe_feed("B", T0 + 270, T0 + 268, 7.0, 11.2)
            s.observe_position("B", "t1", T0 + 270, 1190.0, GEOM)
        a = st.features("B", "t1", T0 + 280, 1190.0, GEOM, [3, 4], 0.0)
        b = st2.features("B", "t1", T0 + 280, 1190.0, GEOM, [3, 4], 0.0)
        assert a == b

    def test_prune_drops_dead_vehicles(self):
        st = LiveState()
        _drive(st, "A", "t1", T0, 10.0, T0 + 200)
        st.prune(T0 + 4 * 3600)
        assert not st.veh and not st.links


class TestReplay:
    def test_replay_matches_incremental_state(self, tmp_path):
        day = "2026-09-21"
        feed, pos = [], []
        for vid, t0, v in (("A", T0, 10.0), ("B", T0 + 100, 7.0)):
            t, d = t0, 0.0
            while d <= 2000.0:
                feed.append((vid, t, t - 3, v, (5000 + d) / 1000))
                pos.append((vid, "t1", t, d, 0))
                t += 10.0
                d += v * 10.0
        f = pd.DataFrame(feed, columns=["vehicle_id", "t", "vehicle_ts", "speed", "odometer"])
        p = pd.DataFrame(pos, columns=["vehicle_id", "trip_id", "t", "dist_along_m", "run"])
        fp, pp = side_paths(tmp_path, day)
        fp.parent.mkdir(parents=True)
        f.to_parquet(fp)
        p.to_parquet(pp)
        q = pd.DataFrame({"vehicle_id": ["B", "B"], "trip_id": ["t1", "t1"], "t": [T0 + 240] * 2,
                          "dist_along_m": [980.0] * 2, "stop_dist_along_m": [1000.0, 2000.0],
                          "stationary_sec": [0.0, 0.0]})
        got, links = replay_day(tmp_path, day, _geom, None, queries=q, collect_links=True)
        # the same streams fed by hand, the way the daemon sees them
        st = LiveState()
        for t in sorted(set(f["t"]) | set(p["t"])):
            if t > T0 + 240:
                break
            for r in f[f["t"] == t].itertuples():
                st.observe_feed(r.vehicle_id, r.t, r.vehicle_ts, r.speed, r.odometer)
            for r in p[p["t"] == t].itertuples():
                st.observe_position(r.vehicle_id, r.trip_id, r.t, r.dist_along_m, GEOM)
        want = st.features("B", "t1", T0 + 240, 980.0, GEOM, [2, 4], 0.0)
        np.testing.assert_array_equal(got, np.array(want))
        assert len(links) == 7 and set(links["vehicle_id"]) == {"A", "B"}

    def test_stationary_runs(self):
        pos = pd.DataFrame({"vehicle_id": "A", "trip_id": "t1",
                            "t": [0, 10, 20, 30, 40, 50],
                            "dist_along_m": [500, 505, 510, 600, 600, 700]})
        r = stationary_runs(pos, _geom)
        assert list(r["key"]) == ["s1@0", "s1@2"] and list(r["D"]) == [20.0, 10.0]


class TestFlatTrees:
    """predict_live must reproduce sklearn exactly, categorical splits included."""

    def test_parity_with_sklearn(self):
        from sklearn.ensemble import HistGradientBoostingRegressor

        from src.inference import predict_live
        from src.train_live import flatten_model

        rng = np.random.default_rng(0)
        n = 4000
        route = rng.integers(0, 12, n).astype(float)
        stop = rng.integers(0, 255, n).astype(float)
        x2 = rng.normal(size=n)
        x3 = np.where(rng.random(n) < 0.2, -1.0, rng.random(n) * 100)
        y = (route % 3) * 50 + (stop % 7) * 10 + x2 * 20 + np.maximum(x3, 0) + rng.normal(size=n)
        X = np.column_stack([route, stop, x2, x3])
        model = HistGradientBoostingRegressor(categorical_features=[0, 1], max_iter=60,
                                              max_leaf_nodes=31, loss="absolute_error",
                                              random_state=0).fit(X, y)
        flat = flatten_model({"model": model, "cols": ["a", "b", "c", "d"],
                              "route_codes": {}, "stop_codes": {}})
        Xt = X[:500].copy()
        Xt[:20, 0] = -1      # unknown route -> missing
        Xt[20:40, 0] = 40    # code never seen in training -> missing
        np.testing.assert_allclose(predict_live(flat, Xt), model.predict(Xt), rtol=0, atol=1e-6)


class TestRunInference:
    """A live_v2 model is served through run_inference with a LiveState."""

    def test_live_model_serves_and_updates_state(self):
        from google.transit import gtfs_realtime_pb2
        from sklearn.ensemble import HistGradientBoostingRegressor

        from src.inference import run_inference
        from src.live_features import N_LIVE, geom_from_worker_data
        from src.train_live import MODEL_COLS, flatten_model
        from tests.test_eta_model import LAT, FakeGTFS, _compact_data, _lon_at

        rng = np.random.default_rng(1)
        X = rng.normal(size=(500, len(MODEL_COLS)))
        X[:, 0] = 0
        X[:, 1] = rng.integers(0, 3, 500)
        model = HistGradientBoostingRegressor(categorical_features=[0, 1], max_iter=5,
                                              random_state=0).fit(X, 200 + 10 * X[:, 2])
        from tests.test_eta_model import ROUTE_ID
        md = flatten_model({"model": model, "cols": MODEL_COLS,
                            "route_codes": {ROUTE_ID: 0}, "stop_codes": {}})
        md["live_tables"] = {}
        data = _compact_data(FakeGTFS())
        live = LiveState(LiveTables())
        live.geom = geom_from_worker_data(data)

        def vp(meters, ts):
            feed = gtfs_realtime_pb2.FeedMessage()
            feed.header.gtfs_realtime_version = "2.0"
            feed.header.timestamp = int(ts)
            ent = feed.entity.add()
            ent.id = "v1"
            v = ent.vehicle
            v.vehicle.id = "v1"
            v.trip.trip_id = "t1"
            v.trip.route_id = ROUTE_ID
            v.position.latitude = LAT
            v.position.longitude = _lon_at(meters)
            v.position.speed = 8.0
            v.position.odometer = 1234.5 + meters / 1000
            v.timestamp = int(ts) - 4
            return feed.SerializeToString()

        now = datetime.now(timezone.utc).timestamp()
        trackers: dict = {}
        out = None
        for k, m in enumerate((600.0, 680.0, 760.0)):
            out = run_inference(data, md, trackers, vp(m, now + 10 * k), live=live)
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(out)
        assert len(feed.entity) == 1
        assert "v1" in live.veh and len(live.veh["v1"].feed) == 3 and len(live.veh["v1"].pos) == 3
        assert N_LIVE == len(LIVE_COLS)


class TestDaemonPersistence:
    """push_feed carries the LiveState across its ~5.5 min restarts through R2."""

    def _push_feed(self):
        import importlib.util
        import os
        from pathlib import Path
        for k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
            os.environ.setdefault(k, "test")
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("push_feed_live", root / "scripts" / "push_feed.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    class _R2:
        def __init__(self):
            self.objects = {}

        def put_object(self, Bucket, Key, Body, **kw):
            self.objects[Key] = Body

        def get_object(self, Bucket, Key):
            import io
            if Key not in self.objects:
                raise KeyError(Key)
            return {"Body": io.BytesIO(self.objects[Key])}

    def test_round_trip_and_legacy_model_has_no_state(self):
        import time as _time
        pf = self._push_feed()
        assert pf._make_live_state({"trees": []}, {}) is None
        now = _time.time()
        st = LiveState()
        st.geom = _geom
        _drive(st, "A", "t1", now - 300, 10.0, now - 100)
        r2 = self._R2()
        pf._save_live_state(r2, st)
        st2 = LiveState()
        st2.geom = _geom
        pf._load_live_state(r2, st2)
        assert set(st2.links) == set(st.links) and set(st2.veh) == {"A"}
        q = dict(vid="B", trip_id="t1", t=now - 90, d=100.0, geom=GEOM, targets=[1, 3], stationary_sec=0.0)
        assert st.features(**q) == st2.features(**q)

    def test_unreadable_state_starts_cold(self):
        pf = self._push_feed()
        r2 = self._R2()
        r2.objects[pf.LIVE_STATE_KEY] = b"not gzip"
        st = LiveState()
        st.geom = _geom
        pf._load_live_state(r2, st)
        assert not st.veh and not st.links
