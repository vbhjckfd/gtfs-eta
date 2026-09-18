"""Unit tests for GTFSStatic._clean_shapes (retraced-loop removal).

Some operator shapes run back and forth over streets the bus drives once --
route 126's shape 38045 is 44 km for a 14.6 km run -- and stops then snap onto
a later pass, inflating remaining distance by kilometres. The cleaner cuts a
stretch that returns to where it left, unless that removes a stop's only pass.
"""

from __future__ import annotations

from shapely.geometry import LineString, Point

from src.gtfs_static import GTFSStatic, StopInfo, StopTime, TripInfo


def _gtfs(coords, stops: dict[str, tuple[float, float]]) -> GTFSStatic:
    g = GTFSStatic()
    g._shapes = {"s": LineString(coords)}
    g._shape_lengths = {"s": g._shapes["s"].length}
    g._stops = {
        sid: StopInfo(sid, sid, 0.0, 0.0, x, y) for sid, (x, y) in stops.items()
    }
    g._trip_index = {
        "t": TripInfo("t", "R", "svc", "s", 0, [
            StopTime(sid, i + 1, "08:00:00", "08:00:00", None)
            for i, sid in enumerate(stops)
        ]),
    }
    g._clean_shapes()
    g._build_stop_distances()
    return g


def test_a_plain_shape_is_left_exactly_as_it_was():
    coords = [(0, 0), (1000, 0), (2000, 0)]
    g = _gtfs(coords, {"A": (0, 0), "B": (1000, 0), "C": (2000, 0)})
    assert list(g._shapes["s"].coords) == [(0, 0), (1000, 0), (2000, 0)]
    assert g._cleaned_shapes == {}


def test_a_stop_free_out_and_back_is_cut():
    # Drives to x=1000, detours 1 km up a street and straight back, carries on.
    coords = [(0, 0), (1000, 0), (1000, 1000), (1000, 0), (2000, 0)]
    g = _gtfs(coords, {"A": (0, 0), "B": (1000, 0), "C": (2000, 0)})
    assert g._shapes["s"].length < 2100
    assert g._stop_distances[("s", "C")] < 2100


def test_a_spur_serving_its_own_stop_is_kept():
    # Same detour, but a stop sits at the end of it: the bus really goes there.
    coords = [(0, 0), (1000, 0), (1000, 1000), (1000, 0), (2000, 0)]
    g = _gtfs(coords, {"A": (0, 0), "V": (1000, 1000), "C": (2000, 0)})
    assert g._shapes["s"].length > 3900
    assert g._cleaned_shapes == {}


def test_a_spur_is_kept_up_to_a_stop_half_way_along_it():
    # A stop half-way up the retraced street: the spur must still reach it --
    # not stop 40 m short on the way up -- while the empty stretch beyond,
    # which serves nothing, may go.
    coords = [(0, 0), (1000, 0), (1000, 1000), (1000, 0), (2000, 0)]
    g = _gtfs(coords, {"A": (0, 0), "M": (1000, 500), "C": (2000, 0)})
    assert g._shapes["s"].distance(Point(1000, 500)) <= 20
    # C is reached via the trip up to M and back.
    assert g._stop_distances[("s", "C")] > 2900


def test_stops_after_a_cut_loop_no_longer_snap_to_a_later_pass():
    # Route 126's failure: the shape runs A->B, back to A, then A->B again
    # and on to C. Every stop on A..B has two passes; nearest-point picked the
    # second. After cleaning there is one pass, and C sits ~2 km along.
    coords = [(0, 0), (1000, 0), (0, 0), (1000, 0), (2000, 0)]
    g = _gtfs(coords, {"A": (0, 0), "B": (1000, 0), "C": (2000, 0)})
    assert g._stop_distances[("s", "B")] < 1100
    assert g._stop_distances[("s", "C")] < 2100
