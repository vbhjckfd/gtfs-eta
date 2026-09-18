"""Unit tests for src/labeling.py's run splitting.

A vehicle can carry the same trip_id on two back-to-back runs (the operator
feed keeps reporting it, and infer_trips' fast path trusts it). _split_runs
must separate them, or every stop's "first crossing" comes from the first run
only and the second run loses its actual arrivals and training rows.
"""

from __future__ import annotations

import os

for _k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
    os.environ.setdefault(_k, "test")

import pandas as pd  # noqa: E402

from src import labeling  # noqa: E402


def _traj(dists: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-09-16 06:00", periods=len(dists), freq="30s", tz="UTC"),
        "dist_along": dists,
    })


def _run_dists(traj: pd.DataFrame) -> list[list[float]]:
    return [r["dist_along"].tolist() for r in labeling._split_runs(traj, "dist_along", 1000.0)]


def test_a_single_run_is_left_whole():
    assert _run_dists(_traj([0, 200, 400, 700, 1000])) == [[0, 200, 400, 700, 1000]]


def test_jitter_backwards_does_not_split():
    # Projection jitter / a brief backward snap, well under half the shape.
    assert len(_run_dists(_traj([0, 300, 250, 600, 520, 900]))) == 1


def test_back_to_back_runs_split_at_the_reset():
    assert _run_dists(_traj([100, 500, 950, 1000, 20, 120, 300, 900])) == [
        [100, 500, 950, 1000], [20, 120, 300, 900],
    ]


def test_three_runs():
    assert len(_run_dists(_traj([0, 900, 10, 50, 90, 950, 5, 40, 60, 600]))) == 3


def test_split_is_measured_from_the_furthest_point_reached():
    # Slow drift down from the peak still adds up to a reset.
    assert len(_run_dists(_traj([0, 900, 700, 450, 300, 250, 200]))) == 2


def test_a_short_backward_blip_does_not_split():
    # One or two misprojections onto another occurrence of a self-intersecting
    # shape, then back on track: the same run.
    assert len(_run_dists(_traj([0, 400, 800, 50, 60, 850, 950]))) == 1


def test_the_split_starts_at_the_first_snapshot_of_the_reset():
    runs = _run_dists(_traj([0, 950, 10, 30, 60, 200]))
    assert runs == [[0, 950], [10, 30, 60, 200]]


def test_empty_trajectory():
    assert labeling._split_runs(_traj([]), "dist_along", 1000.0) == []
