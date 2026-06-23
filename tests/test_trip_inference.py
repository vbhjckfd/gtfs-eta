"""Trip-matching scoring + bearing-gated stickiness (issue #3 direction fix)."""
import types

from shapely.geometry import LineString

from src import trip_inference as ti


class _FakeGTFS:
    def __init__(self, shape):
        self._shape = shape

    def get_shape_linestring(self, _shape_id):
        return self._shape


# A horizontal shape running west→east; its tangent is ~90° (due east).
_EAST_SHAPE = LineString([(0.0, 0.0), (100.0, 0.0)])


def test_score_trip_uses_only_spatial_and_bearing():
    g = _FakeGTFS(_EAST_SHAPE)
    trip = types.SimpleNamespace(shape_id="s")
    # Vehicle 5 m off the line at x=50: spatial_score = 5/100 = 0.05.
    aligned, _ = ti.score_trip(trip, 50.0, 5.0, 90.0, g)   # heading east, matches tangent
    opposed, _ = ti.score_trip(trip, 50.0, 5.0, 270.0, g)  # heading west, opposes
    # SPATIAL_WEIGHT*0.05 + BEARING_WEIGHT*0.0 with no schedule term.
    assert abs(aligned - ti.SPATIAL_WEIGHT * 0.05) < 1e-6
    # Opposed heading adds the full bearing penalty (diff 180/180 = 1.0).
    assert abs(opposed - (ti.SPATIAL_WEIGHT * 0.05 + ti.BEARING_WEIGHT)) < 1e-6
    assert opposed > aligned


def test_stickiness_holds_when_close_and_aligned():
    assert ti._stickiness_dist(_EAST_SHAPE, 50.0, 5.0, 90.0, best_dist=50.0) is not None


def test_stickiness_released_on_reverse_direction():
    # Close to the shape but heading the opposite way (terminus reversal) → released,
    # so the matcher won't pin the vehicle to the outbound trip after it turns around.
    assert ti._stickiness_dist(_EAST_SHAPE, 50.0, 5.0, 270.0, best_dist=50.0) is None


def test_stickiness_released_when_beyond_hysteresis():
    assert ti._stickiness_dist(_EAST_SHAPE, 50.0, 200.0, 90.0, best_dist=500.0) is None


def test_stickiness_released_when_farther_than_fresh_winner():
    # prev_dist 30 > best_dist 10 → the fresh match is closer, don't hold.
    assert ti._stickiness_dist(_EAST_SHAPE, 50.0, 30.0, 90.0, best_dist=10.0) is None


def test_stickiness_no_bearing_falls_back_to_distance_only():
    # bearing unavailable → gate on distance alone (still holds when close).
    assert ti._stickiness_dist(_EAST_SHAPE, 50.0, 5.0, None, best_dist=50.0) is not None
