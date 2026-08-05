"""End-to-end tests of ZoneCounter.update() against a synthetic straight-
line vehicle path (no YOLO/video involved — just the counting state
machine that decides whether a start->finish crossing becomes a count).

Covers the three failure modes that matter most for a survey's accuracy:
  1. A normal crossing produces exactly one count, with the right lane.
  2. A track that crosses a start line but never reaches a finish line is
     expired, not silently left dangling or double-counted later.
  3. Track-identity bridging (_relink_orphans) — when the tracker's
     internal ID changes mid-crossing (occlusion, a brief ID swap), the
     vehicle must still be counted exactly once. This is the exact area
     TUNING_NOTES.md / CUSTOM_MODEL_AND_REID_NOTES.md describe as having
     shipped real bugs (undercounting, flicker) — a regression here is
     easy to miss just by eyeballing debug video.
"""

from aitss import config
from aitss.zone_counter import ZoneCounter

START_LINES = {"N": [(0, 100), (100, 100)]}
# Finish-line labels encode the ENTRY arm + movement (e.g. "N2" = a vehicle
# that entered via the North start line, movement 2) — not the exit
# direction. Using "N2" here (matching the "N" start line) keeps this a
# clean crossing with no origin/finish-label mismatch; see
# test_relink_refuses_to_bridge_different_categories for the mismatch path.
FINISH_LINES = {"N2": [(0, 200), (100, 200)]}


def make_det(track_id, cx, cy, category="Car", age=10, conf=0.8, predicted=False):
    return {
        "track_id": track_id,
        "category": category,
        "centroid": (cx, cy),
        "conf": conf,
        "age": age,
        "predicted": predicted,
    }


def test_straight_line_crossing_counts_exactly_once():
    counter = ZoneCounter(START_LINES, FINISH_LINES)

    # Build enough track history below MIN_TRACK_AGE first so the
    # back-check path (triggered the first frame age qualifies) doesn't
    # accidentally register the real crossing.
    counter.update([make_det(1, 50, 50, age=1)], frame_idx=1, fps=25)
    counter.update([make_det(1, 50, 60, age=9)], frame_idx=2, fps=25)  # age now qualifies
    events = counter.update([make_det(1, 50, 150, age=10)], frame_idx=3, fps=25)
    assert events == []  # crossed start, not finish yet

    counter.update([make_det(1, 50, 160, age=11)], frame_idx=4, fps=25)
    counter.update([make_det(1, 50, 180, age=12)], frame_idx=5, fps=25)
    events = counter.update([make_det(1, 50, 220, age=13)], frame_idx=6, fps=25)

    assert len(events) == 1
    event = events[0]
    assert event.lane == "N2"
    assert event.direction == "N2"
    assert event.origin == "N"
    assert event.category == "Car"
    assert not event.origin_mismatch

    # Continuing to feed the same (now-counted) track must never produce
    # a second count.
    more_events = counter.update([make_det(1, 50, 250, age=14)], frame_idx=7, fps=25)
    assert more_events == []
    assert counter.get_lane_counts()["N2"] == 1


def test_track_that_never_reaches_finish_line_is_not_counted():
    counter = ZoneCounter(START_LINES, FINISH_LINES)

    counter.update([make_det(2, 50, 50, age=1)], frame_idx=1, fps=25)
    counter.update([make_det(2, 50, 60, age=9)], frame_idx=2, fps=25)
    events = counter.update([make_det(2, 50, 150, age=10)], frame_idx=3, fps=25)
    assert events == []
    assert 2 in counter._crossed_start  # crossed the start line...

    # ...but then goes stale: last seen at frame 3, never reaches the
    # finish line. Feed a large enough frame_idx jump for a DIFFERENT
    # track to trigger _stale_cleanup past STALE_TRACK_TIMEOUT.
    far_future_frame = 3 + config.STALE_TRACK_TIMEOUT + 10
    counter.update([make_det(99, 500, 500, age=10)], frame_idx=far_future_frame, fps=25)

    assert 2 not in counter._crossed_start
    assert counter.get_lane_counts()["N2"] == 0
    assert counter._diag_stats["stale_cleanup"] == 1


def test_identity_bridging_across_a_track_id_switch_still_counts():
    """Simulates the tracker assigning a NEW track_id to the same physical
    vehicle mid-crossing (the exact scenario _relink_orphans exists for).
    """
    counter = ZoneCounter(START_LINES, FINISH_LINES)

    # Old track_id=3 crosses the start line.
    counter.update([make_det(3, 50, 50, age=1)], frame_idx=1, fps=25)
    counter.update([make_det(3, 50, 60, age=9)], frame_idx=2, fps=25)
    events = counter.update([make_det(3, 50, 150, age=10)], frame_idx=3, fps=25)
    assert events == []
    assert 3 in counter._crossed_start

    # track_id=3 vanishes this frame (not present in detections) — gets
    # orphaned. A different, unrelated track is present so the update()
    # loop has something to iterate.
    counter.update([make_det(98, 900, 900, age=10)], frame_idx=4, fps=25)
    assert 3 in counter._orphaned_starts

    # A brand-new track_id=4 appears close to track 3's last position,
    # same category, well within ZONE_RELINK_MAX_FRAMES/DIST_PX — this is
    # the tracker's ID-swap continuing the same physical vehicle.
    counter.update([make_det(4, 55, 155, age=10)], frame_idx=5, fps=25)
    assert counter._diag_stats["relinked"] == 1
    assert 3 not in counter._crossed_start
    assert 4 in counter._crossed_start

    # track_id=4 continues on to the finish line — must still count,
    # attributed to the NEW id, with the ORIGINAL start-line origin
    # preserved.
    counter.update([make_det(4, 55, 165, age=11)], frame_idx=6, fps=25)
    counter.update([make_det(4, 55, 185, age=12)], frame_idx=7, fps=25)
    events = counter.update([make_det(4, 55, 225, age=13)], frame_idx=8, fps=25)

    assert len(events) == 1
    assert events[0].track_id == 4
    assert events[0].origin == "N"
    assert events[0].relinked is True
    assert counter.get_lane_counts()["N2"] == 1


def test_relink_refuses_to_bridge_different_categories():
    """A Car's orphaned start must never be bridged onto a Motorcycle's
    new track — that would silently mislabel the counted vehicle's
    category. See the category gate in _relink_orphans.
    """
    counter = ZoneCounter(START_LINES, FINISH_LINES)

    counter.update([make_det(5, 50, 50, age=1, category="Car")], frame_idx=1, fps=25)
    counter.update([make_det(5, 50, 60, age=9, category="Car")], frame_idx=2, fps=25)
    counter.update([make_det(5, 50, 150, age=10, category="Car")], frame_idx=3, fps=25)
    assert 5 in counter._crossed_start

    counter.update([make_det(97, 900, 900, age=10)], frame_idx=4, fps=25)

    # New track at nearly the same position, but a different category.
    counter.update([make_det(6, 55, 155, age=10, category="Motorcycle")], frame_idx=5, fps=25)

    assert counter._diag_stats["relinked"] == 0
    assert 5 in counter._orphaned_starts
    assert 6 not in counter._crossed_start
