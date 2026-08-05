"""Tests for the pure label-parsing logic in tmc_export.py.

parse_zone_label decides which TMC sheet/column a counted crossing lands
in — a silent regression here doesn't crash anything, it just quietly
drops counts into the wrong movement or the "unplaced" bucket, which is
exactly the kind of bug that's invisible until someone cross-checks the
Excel output by hand.
"""

from aitss.tmc_export import parse_zone_label


def test_valid_labels():
    assert parse_zone_label("N1") == ("N", 1)
    assert parse_zone_label("S2") == ("S", 2)
    assert parse_zone_label("E3") == ("E", 3)
    assert parse_zone_label("W4") == ("W", 4)


def test_lowercase_is_normalized():
    assert parse_zone_label("n1") == ("N", 1)


def test_whitespace_is_tolerated():
    assert parse_zone_label(" N1 ") == ("N", 1)
    assert parse_zone_label("N 1") == ("N", 1)


def test_movement_out_of_range_rejected():
    # Only movements 1-4 exist in the template's per-arm blocks.
    assert parse_zone_label("N5") == (None, None)
    assert parse_zone_label("N0") == (None, None)


def test_invalid_direction_letter_rejected():
    assert parse_zone_label("X1") == (None, None)


def test_non_string_input_rejected():
    assert parse_zone_label(None) == (None, None)
    assert parse_zone_label(42) == (None, None)


def test_malformed_labels_rejected():
    # Real risk: a hand-picked zone label that doesn't follow the N/S/E/W
    # + digit convention (e.g. a custom lane name) must be reported as
    # "unplaced" by the caller, not silently mis-parsed into a movement.
    assert parse_zone_label("North") == (None, None)
    assert parse_zone_label("N12") == (None, None)
    assert parse_zone_label("") == (None, None)
