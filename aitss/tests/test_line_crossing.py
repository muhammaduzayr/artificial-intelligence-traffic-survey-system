"""Tests for the geometry helpers in zone_counter.py.

These are the lowest-level building blocks of the counting logic — every
crossing decision in the whole pipeline reduces to _line_side and
_check_crossing. A bug here doesn't crash, it silently miscounts.
"""

import pytest

from aitss.zone_counter import ZoneCounter, _segment_intersection, _vector_angle


# ---------------------------------------------------------------------- #
# _segment_intersection
# ---------------------------------------------------------------------- #

def test_segment_intersection_simple_cross():
    # Two segments crossing in a clean X shape.
    pt = _segment_intersection((0, 0), (10, 10), (0, 10), (10, 0))
    assert pt == pytest.approx((5, 5))


def test_segment_intersection_parallel_lines_no_cross():
    assert _segment_intersection((0, 0), (10, 0), (0, 5), (10, 5)) is None


def test_segment_intersection_outside_segment_bounds():
    # Lines would cross if extended to infinity, but not within either
    # actual segment.
    assert _segment_intersection((0, 0), (1, 1), (5, 0), (5, 1)) is None


def test_segment_intersection_touching_endpoint():
    pt = _segment_intersection((0, 0), (10, 0), (5, 0), (5, 10))
    assert pt == pytest.approx((5, 0))


# ---------------------------------------------------------------------- #
# _vector_angle
# ---------------------------------------------------------------------- #

def test_vector_angle_same_direction_is_zero():
    assert _vector_angle((1, 0), (5, 0)) == pytest.approx(0.0)


def test_vector_angle_perpendicular_is_90():
    assert _vector_angle((1, 0), (0, 1)) == pytest.approx(90.0)


def test_vector_angle_opposite_is_180():
    assert _vector_angle((1, 0), (-1, 0)) == pytest.approx(180.0)


def test_vector_angle_degenerate_vector_returns_zero():
    # A near-zero-magnitude vector (e.g. a queued vehicle barely moving)
    # must not blow up or report a spurious large angle — see the
    # ROUTE_CHANGE_ANGLE_THRESHOLD comment in zone_counter.py about this
    # exact case being noisy on stationary traffic.
    assert _vector_angle((0.5, 0.5), (10, 0)) == 0.0


# ---------------------------------------------------------------------- #
# ZoneCounter._line_side / _check_crossing
# ---------------------------------------------------------------------- #

HORIZONTAL_LINE = ((0, 100), (100, 100))


def test_line_side_above_and_below():
    above = ZoneCounter._line_side((50, 50), HORIZONTAL_LINE)
    below = ZoneCounter._line_side((50, 150), HORIZONTAL_LINE)
    assert above != 0
    assert below != 0
    assert above == -below


def test_line_side_within_dead_zone_is_uncertain():
    on_line = ZoneCounter._line_side((50, 100), HORIZONTAL_LINE, dead_zone=4)
    assert on_line == 0


def test_check_crossing_detects_side_change():
    counter = ZoneCounter(start_lines={}, finish_lines={})
    cross = counter._check_crossing((50, 50), (50, 150), HORIZONTAL_LINE)
    assert cross is not None
    assert cross == pytest.approx((50, 100))


def test_check_crossing_same_side_is_none():
    counter = ZoneCounter(start_lines={}, finish_lines={})
    assert counter._check_crossing((50, 50), (60, 60), HORIZONTAL_LINE) is None


def test_check_crossing_fast_jump_over_line_still_detected():
    # A fast vehicle / large FRAME_SKIP gap can jump clean over the line
    # between two detections without ever landing near it — side-based
    # detection must still catch this (this is the whole reason the
    # implementation isn't pure segment-intersection).
    counter = ZoneCounter(start_lines={}, finish_lines={})
    cross = counter._check_crossing((10, 10), (90, 190), HORIZONTAL_LINE)
    assert cross is not None


def test_check_crossing_rejects_crossing_outside_segment_extent():
    # A side-flip that happens well past the drawn segment's endpoints is
    # a different lane's line crossing its infinite extension, not a real
    # crossing of this counting line.
    counter = ZoneCounter(start_lines={}, finish_lines={})
    far_outside = counter._check_crossing((500, 50), (500, 150), HORIZONTAL_LINE)
    assert far_outside is None
