"""
Line-based vehicle counting with finish-line direction labels.

Multiple START_LINES (config.START_LINES, one per entry approach) mark
where vehicles enter the counting area.  When a tracked vehicle's centroid
crosses any start line, the system records the crossing.

When that same vehicle subsequently crosses one of the FINISH_LINES, it is
counted with:
  - lane = the finish line's key (e.g. "N1", "S2")
  - direction = the finish line's key (the direction label IS the lane)

Each track_id generates at most one count.  Tracks that cross a start
line but never reach a finish line within _STALE_TRACK_TIMEOUT frames are
silently expired (not counted — we don't know their exit).
"""

import csv
import math
import os

import cv2
from collections import defaultdict, deque
from dataclasses import dataclass

from . import config


def _segment_intersection(p1, p2, p3, p4):
    """Return (x, y) where segment (p1→p2) crosses (p3→p4), or None."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    if not (0.0 <= t <= 1.0):
        return None

    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    if not (0.0 <= u <= 1.0):
        return None

    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def _vector_angle(va, vb):
    dot = va[0] * vb[0] + va[1] * vb[1]
    ma = (va[0] ** 2 + va[1] ** 2) ** 0.5
    mb = (vb[0] ** 2 + vb[1] ** 2) ** 0.5
    if ma < 1 or mb < 1:
        return 0.0
    cos_a = max(-1.0, min(1.0, dot / (ma * mb)))
    return math.degrees(math.acos(cos_a))


@dataclass
class CountEvent:
    frame_idx: int
    timestamp_sec: float
    track_id: int
    lane: str
    line_name: str  # matches lane for line counting
    direction: str
    category: str
    bbox: tuple = None
    conf: float = None  # confidence at lock time (start-line crossing)
    relinked: bool = False  # whether this track's identity was bridged
                             # across an ID switch via _relink_orphans
    origin: str = None  # name of the start line this track actually crossed
    origin_mismatch: bool = False  # True if `origin` doesn't match the arm
                                    # implied by the finish line's own label
                                    # (e.g. started via "S" but finished via
                                    # "N1") — surfaced instead of silently
                                    # trusting the finish label alone.


class ZoneCounter:
    def __init__(self, start_lines, finish_lines, lock_category_fn=None,
                 retire_track_fn=None, protect_track_fn=None,
                 relink_kf_fn=None, diagnostic_csv_path=None):
        """
        start_lines  : dict of {approach_label: [(x1,y1), (x2,y2)]}
        finish_lines : dict of {lane_name: [(x1,y1), (x2,y2)]}
        lock_category_fn : callable(track_id, category) or None.
            Called when a track crosses a start line so the detector can
            permanently lock that track's category for display and counting.
        retire_track_fn : callable(track_id) or None.
            Called after a track is counted (crosses a finish line) so the
            detector can prevent it from ever matching again — the root
            cause of double counting.
        protect_track_fn : callable(track_id) or None.
            Called when a track crosses a start line so the detector can
            apply stricter matching (lower cost threshold) to prevent ID
            swaps while the track is between start and finish.
        relink_kf_fn : callable(old_track_id, new_track_id) or None.
            Called when _relink_orphans bridges an ID switch, so the
            detector can carry its Kalman-filter state (position, velocity)
            from the old id onto the new one — otherwise the relink keeps
            the *count* correct but the displayed position still snaps,
            since the new track's filter would start cold.
        diagnostic_csv_path : str or None.
            If given, one row is appended per counted crossing (track_id,
            category, timestamp, frame_idx, confidence at lock time,
            whether it was relinked) directly to this CSV as counting
            happens — independent of whatever report-generation step runs
            afterward, so the raw why-did-this-count-end-up-wrong trail is
            on disk even if that later step fails. Rows are appended
            immediately rather than buffered, so a crash mid-run doesn't
            lose earlier crossings.
        """
        self._start_lines = {k: tuple(v) for k, v in start_lines.items()}
        self._finish_lines = {k: tuple(v) for k, v in finish_lines.items()}
        self._lock_category_fn = lock_category_fn
        self._retire_track_fn = retire_track_fn
        self._protect_track_fn = protect_track_fn
        self._relink_kf_fn = relink_kf_fn

        self._diagnostic_csv_path = diagnostic_csv_path
        self._diagnostic_csv_header_written = False
        # _write_diagnostic_row appends (so a mid-run crash doesn't lose
        # earlier rows) — but that means a stale file from a PREVIOUS run at
        # the same output-dir would otherwise silently mix its rows in with
        # this run's, making track_ids (which restart from 1 every run)
        # collide and look like nonsensical duplicate/conflicting crossings.
        # Clear any leftover file now, at the start of this run, once.
        if self._diagnostic_csv_path and os.path.exists(self._diagnostic_csv_path):
            try:
                os.remove(self._diagnostic_csv_path)
            except OSError as exc:
                print(f"WARNING: could not clear stale diagnostic CSV "
                      f"({self._diagnostic_csv_path}): {exc}")

        # Stale track timeout from config (frames).  Tracks that crossed a
        # start line but never reach a finish line within this window are
        # silently expired.
        self._stale_track_timeout = config.STALE_TRACK_TIMEOUT

        # Per-track state: track_id -> (start_label, crossing_centroid)
        self._crossed_start = {}
        # track_id -> frame_idx when start was crossed
        self._start_cross_frame = {}
        # track_id -> (ts, fi, cat, bbox, conf)
        self._start_entry = {}

        # Per-track state: track_id -> finish_lane_name once finish line is
        # crossed.
        self._crossed_finish = {}
        # track_id -> frame_idx when finish was crossed
        self._finish_cross_frame = {}

        self._track_counted = set()

        self._last_centroid = {}
        # track_id -> last real detection's appearance descriptor (HSV
        # histogram from VehicleDetector._compute_appearance). Mirrors
        # _last_centroid exactly (same update/clear sites) - see
        # _relink_orphans for how this gates identity bridging.
        self._last_appearance = {}
        # Per-track/per-line hysteresis state. The last stable side and its
        # point are retained while the centroid is inside the dead zone, so a
        # vehicle must genuinely emerge on the opposite side before a line
        # crossing is accepted.
        self._line_side_state = {}
        self._line_last_stable_point = {}
        self._track_centroids = defaultdict(
            lambda: deque(maxlen=config.ROUTE_CENTROID_WINDOW)
        )
        self._track_last_seen = {}
        self._track_route_info = {}

        # Track-identity bridging (see config.ZONE_RELINK_* for the full
        # explanation): track_id -> {sl_name, cross_pt, entry, centroid,
        # orphaned_at}. Populated when a track that already crossed a
        # start line disappears; consumed when a brand-new track_id shows
        # up nearby shortly after.
        self._orphaned_starts = {}

        # track_id -> True once its identity has been bridged across an ID
        # switch via _relink_orphans, so _count_track can record it on the
        # resulting CountEvent for the diagnostic trail.
        self._track_relinked = set()

        self._diag_stats = {"counted": 0, "stale_cleanup": 0,
                               "disp_reject": 0, "gap_reject": 0,
                               "relinked": 0,
                               "crossing_dedup_skipped": 0,
                               "raw_smoothed_side_mismatch": 0,
                               "kf_vs_raw_crossing_mismatch": 0,
                               "origin_mismatch": 0}
        self._lane_counts = {k: 0 for k in finish_lines}

        # Raw-centroid history (KF centroid used for crossing detection).
        # Used to back-check start-line crossings that happened before
        # MIN_TRACK_AGE was satisfied (vehicle entered frame close to line).
        self._raw_centroid_buffer = defaultdict(
            lambda: deque(maxlen=config.ROUTE_CENTROID_WINDOW)
        )
        # Tracks whose history has already been back-checked.
        self._min_age_back_checked = set()

        # ── Crossing-direction auto-detection ─────────────────────────
        # For each line, determine which normal direction is "forward"
        # (pointing toward the counting zone interior).  +1 means vehicles
        # must move in the left-hand normal direction (-dy, dx); -1 means
        # the right-hand normal (+dy, -dx).  If LINE_CROSSING_DIRECTIONS
        # has an explicit entry for a line, use that; otherwise auto-detect
        # by checking which normal points toward the finish-area center.
        self._line_expected_dir = {}
        finish_centers = []
        for fname, fline in self._finish_lines.items():
            fm = ((fline[0][0] + fline[1][0]) / 2,
                  (fline[0][1] + fline[1][1]) / 2)
            finish_centers.append(fm)
        all_lines = {}
        all_lines.update({k: v for k, v in self._start_lines.items()})
        all_lines.update({k: v for k, v in self._finish_lines.items()})
        for name, line in all_lines.items():
            if name in config.LINE_CROSSING_DIRECTIONS:
                self._line_expected_dir[name] = config.LINE_CROSSING_DIRECTIONS[name]
            elif finish_centers:
                p1, p2 = line
                ldx = p2[0] - p1[0]
                ldy = p2[1] - p1[1]
                mid_x = (p1[0] + p2[0]) / 2
                mid_y = (p1[1] + p2[1]) / 2
                ref_cx = sum(p[0] for p in finish_centers) / len(finish_centers)
                ref_cy = sum(p[1] for p in finish_centers) / len(finish_centers)
                dot = (ref_cx - mid_x) * (-ldy) + (ref_cy - mid_y) * ldx
                self._line_expected_dir[name] = 1 if dot > 0 else -1
            else:
                self._line_expected_dir[name] = 0

    # ------------------------------------------------------------------ #
    #  Line-crossing helpers                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _signed_line_distance(point, line):
        p1, p2 = line
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = (dx * dx + dy * dy) ** 0.5
        if length <= 0:
            return 0.0
        return (dx * (point[1] - p1[1]) - dy * (point[0] - p1[0])) / length

    @classmethod
    def _line_side(cls, point, line, dead_zone=None):
        """Return +1 or -1 indicating which side of the line the point is on,
        or 0 if the point is within *dead_zone* pixels of the line (uncertain).

        Uses the sign of the cross product (p2-p1) × (point-p1).  The 1 px
        dead zone catches only the truly ambiguous case of a centroid landing
        exactly on the line (rare with FRAME_SKIP > 1), while still allowing
        fast-moving centroids that pass near the line between detection frames
        to register a crossing.
        """
        if dead_zone is None:
            dead_zone = config.COUNTING_LINE_DEAD_ZONE_PX
        distance = cls._signed_line_distance(point, line)
        if abs(distance) < dead_zone:
            return 0
        return 1 if distance > 0 else -1

    def _check_crossing(self, prev_pt, curr_pt, line):
        """Return (x, y) if the centroid crossed the line between frames, or
        None.

        Uses **side-based detection** as the primary method, with a segment-
        intersection fallback when one endpoint is within the dead zone:

          1. Determine which side of the line each point is on (with a 1 px
             dead zone — points within 1 px of the line are uncertain).
          2. If the side changed → a crossing occurred.
          3. If one side is uncertain, fall back to direct segment-segment
             intersection to decide whether a crossing actually happened.

        The side-based method catches fast vehicles whose centroid jumps
        OVER the line in one frame (common with FRAME_SKIP > 1).  The
        segment-intersection fallback prevents the dead zone from silently
        swallowing legitimate crossings when a centroid lands near the line.
        """
        side_prev = self._line_side(prev_pt, line)
        side_curr = self._line_side(curr_pt, line)

        # Both uncertain — no crossing possible.
        if side_prev == 0 and side_curr == 0:
            return None

        # One side uncertain — fall back to segment intersection.
        if side_prev == 0 or side_curr == 0:
            cross = _segment_intersection(prev_pt, curr_pt, line[0], line[1])
            if cross is not None:
                return cross
            return None

        # Both sides clear — side change = crossing.
        if side_prev != side_curr:
            mid = ((prev_pt[0] + curr_pt[0]) / 2,
                   (prev_pt[1] + curr_pt[1]) / 2)
            p1, p2 = line
            ldx = p2[0] - p1[0]
            ldy = p2[1] - p1[1]
            length_sq = ldx * ldx + ldy * ldy + 1e-10
            t = ((mid[0] - p1[0]) * ldx + (mid[1] - p1[1]) * ldy) / length_sq
            # Reject side-flips that happen outside the drawn segment —
            # those are a different lane/vehicle crossing the line's
            # infinite extension, not a real crossing of the counting
            # zone.  5 % margin lets the segment endpoints not be
            # pixel-perfect while still keeping the gate tight.
            margin = 0.05
            if t < -margin or t > 1 + margin:
                return None
            t = max(0.0, min(1.0, t))
            return (p1[0] + t * ldx, p1[1] + t * ldy)

        return None

    # ------------------------------------------------------------------ #
    #  Route tracking (angle-based)                                       #
    # ------------------------------------------------------------------ #

    def _segment_vector(self, seg):
        n = len(seg)
        if n < 3:
            return (0.0, 0.0)
        k = min(3, n // 3 or 1)
        sx = sum(p[0] for p in seg[:k]) / k
        sy = sum(p[1] for p in seg[:k]) / k
        ex = sum(p[0] for p in seg[-k:]) / k
        ey = sum(p[1] for p in seg[-k:]) / k
        return (ex - sx, ey - sy)

    def _check_route_change(self, track_id, frame_idx, timestamp_sec):
        centroids = list(self._track_centroids.get(track_id, []))
        if len(centroids) < 6:
            return
        window = centroids[-config.ROUTE_CENTROID_WINDOW:]
        mid = len(window) // 2
        if mid < 3:
            return
        vec_early = self._segment_vector(window[:mid])
        vec_late = self._segment_vector(window[mid:])
        angle = _vector_angle(vec_early, vec_late)
        info = self._track_route_info.get(track_id)
        if info is None:
            return
        if angle > config.ROUTE_CHANGE_ANGLE_THRESHOLD and not info["route_changed"]:
            info["route_changed"] = True
            info["change_frame"] = frame_idx
            info["change_timestamp"] = timestamp_sec
            print(f"[route] CHANGE track={track_id} angle={angle:.0f}deg "
                  f"at frame={frame_idx} ts={timestamp_sec:.0f}s")

    def get_route_info(self, track_id):
        return self._track_route_info.get(track_id)

    def get_all_route_info(self):
        return dict(self._track_route_info)

    def get_lane_counts(self):
        return dict(self._lane_counts)

    @property
    def diagnostic_csv_path(self):
        """Path to the sidecar per-crossing diagnostic CSV, or None if disabled."""
        return self._diagnostic_csv_path

    # ------------------------------------------------------------------ #
    #  Counting                                                           #
    # ------------------------------------------------------------------ #

    def _write_diagnostic_row(self, event):
        """Append one row to the sidecar diagnostic CSV for a counted crossing.

        Best-effort: a failure here (e.g. unwritable output dir) is logged
        and swallowed rather than interrupting counting — this is a
        diagnostic aid, not part of the counting logic itself.
        """
        if not self._diagnostic_csv_path:
            return
        try:
            is_new_file = not os.path.exists(self._diagnostic_csv_path)
            with open(self._diagnostic_csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if is_new_file or not self._diagnostic_csv_header_written:
                    if is_new_file:
                        writer.writerow([
                            "track_id", "category", "timestamp_sec", "frame_idx",
                            "confidence_at_lock", "relinked", "lane", "direction",
                            "origin", "origin_mismatch",
                        ])
                    self._diagnostic_csv_header_written = True
                writer.writerow([
                    event.track_id, event.category, f"{event.timestamp_sec:.3f}",
                    event.frame_idx, event.conf, event.relinked,
                    event.lane, event.direction,
                    event.origin, event.origin_mismatch,
                ])
        except OSError as exc:
            print(f"WARNING: could not write diagnostic CSV row "
                  f"({self._diagnostic_csv_path}): {exc}")

    def _count_track(self, track_id, events, fps, finish_label):
        if track_id in self._track_counted:
            return
        if track_id not in self._crossed_start:
            return
        entry = self._start_entry.get(track_id)
        if entry is None:
            return

        ts, fi, cat, bbox, conf = entry
        direction = finish_label  # finish line key IS the direction

        # The actual start line this track crossed — tracked in
        # _crossed_start all along but previously discarded here, leaving
        # the finish label as the ONLY thing that decided lane/direction/
        # movement (see tmc_export.parse_zone_label). Surface it on the
        # event so a South-origin vehicle that somehow got counted as N1
        # is auditable instead of silently absorbed into the count.
        start_label, _ = self._crossed_start[track_id]
        expected_origin = None
        if finish_label and finish_label[0].upper() in "NSEW":
            expected_origin = finish_label[0].upper()
        origin_mismatch = expected_origin is not None and start_label != expected_origin
        if origin_mismatch:
            print(f"WARNING: [origin-mismatch] track={track_id} crossed START line "
                  f"'{start_label}' but FINISH line '{finish_label}' implies origin "
                  f"'{expected_origin}' — movement may be misclassified in the TMC report.")
            self._diag_stats["origin_mismatch"] += 1

        route_info = self._track_route_info.get(track_id)
        route_suffix = ""
        if route_info:
            if route_info.get("route_changed"):
                route_suffix += f" ROUTE_CHANGED@{route_info['change_frame']}"

        self._track_counted.add(track_id)
        self._diag_stats["counted"] += 1
        self._lane_counts[finish_label] = self._lane_counts.get(finish_label, 0) + 1

        # Retire the track — it can never match again, preventing the
        # counted-track-steals-new-vehicle double-count bug.
        if self._retire_track_fn is not None:
            self._retire_track_fn(track_id)

        events.append(CountEvent(
            frame_idx=fi,
            timestamp_sec=ts,
            track_id=track_id,
            lane=finish_label,
            line_name=finish_label,
            direction=direction,
            category=cat,
            bbox=bbox,
            conf=conf,
            relinked=track_id in self._track_relinked,
            origin=start_label,
            origin_mismatch=origin_mismatch,
        ))
        self._write_diagnostic_row(events[-1])
        self._track_relinked.discard(track_id)

        print(
            f"[counted] {ts:.0f}s "
            f"lane={finish_label} "
            f"track={track_id} category={cat} "
            f"direction={direction} origin={start_label}"
            f"{route_suffix}"
        )

    def _orphan_disappeared_tracks(self, frame_idx, current_ids):
        """Move tracks that crossed a start line but vanished THIS frame
        (their track_id isn't in the current frame's detections) into the
        orphan pool, so a brief tracker ID switch has a chance to be
        bridged in _relink_orphans before the much longer
        STALE_TRACK_TIMEOUT gives up on them for good.
        """
        for tid in list(self._crossed_start.keys()):
            if tid in self._track_counted or tid in current_ids or tid in self._orphaned_starts:
                continue
            centroid = self._last_centroid.get(tid)
            if centroid is None:
                continue
            # Fetch the category recorded at start-line crossing so
            # _relink_orphans can verify the new track's category matches
            # before bridging identities — prevents a Light Truck's
            # orphaned start from being dumped onto a nearby Car's track.
            entry = self._start_entry.get(tid)
            orphan_cat = entry[2] if entry else None
            self._orphaned_starts[tid] = {
                "centroid": centroid,
                "orphaned_at": frame_idx,
                "category": orphan_cat,
                "appearance": self._last_appearance.get(tid),
            }

    def _relink_orphans(self, new_track_id, first_centroid, category, frame_idx,
                         appearance=None):
        """If a brand-new track_id's first detection appears close to an
        orphaned start (recently-vanished track that already crossed a
        start line), AND both carry the same category, AND (when both have
        one) their appearance signatures are similar enough, treat it as
        the same physical vehicle continuing: carry the start-crossing
        state over to the new track_id.

        Requires same category because in dense traffic several different
        vehicles are legitimately within 130 px of each other — bridging
        purely on position swaps categories between vehicles (e.g. a Light
        Truck's locked identity onto a Car's track). The appearance check
        catches the case category alone can't: two different vehicles of
        the SAME category passing close together (e.g. two cars in a queue)
        — see config.ZONE_RELINK_MIN_APPEARANCE_SIMILARITY.

        Returns True if a relink happened.
        """
        best_tid = None
        best_dist = None
        for old_tid, orphan in self._orphaned_starts.items():
            age = frame_idx - orphan["orphaned_at"]
            if age > config.ZONE_RELINK_MAX_FRAMES:
                continue
            ocx, ocy = orphan["centroid"]
            dist = ((first_centroid[0] - ocx) ** 2 + (first_centroid[1] - ocy) ** 2) ** 0.5
            if dist > config.ZONE_RELINK_MAX_DIST_PX:
                continue
            # Category gate: don't bridge a Light Truck onto a Car.
            # The orphan's category was recorded when it crossed its start
            # line and is stored alongside its centroid.
            orphan_cat = orphan.get("category")
            if orphan_cat is not None and orphan_cat != category:
                continue
            # Appearance gate: don't bridge two different same-category
            # vehicles just because they passed close together. Fails open
            # (no rejection) when either side is missing a signature — a
            # degenerate crop shouldn't block an otherwise-good relink.
            orphan_app = orphan.get("appearance")
            if orphan_app is not None and appearance is not None:
                similarity = cv2.compareHist(orphan_app, appearance, cv2.HISTCMP_CORREL)
                if similarity < config.ZONE_RELINK_MIN_APPEARANCE_SIMILARITY:
                    continue
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_tid = old_tid

        if best_tid is None:
            return False

        # Carry the old track's start-crossing state over to the new ID.
        # Use safe-get + pop(None) instead of bare pop() because there is
        # a narrow window where _stale_cleanup (tier 2) removes the old
        # track from _crossed_start after it was orphaned but before
        # _relink_orphans finishes — the orphan survives tier 1's 45-frame
        # cycle check but the crossed_start entry was cleaned at the much
        # longer STALE_TRACK_TIMEOUT boundary in the same call.  If the
        # old track is already gone, just clean up the stale orphan and
        # return.
        old_state = self._crossed_start.pop(best_tid, None)
        if old_state is None:
            self._orphaned_starts.pop(best_tid, None)
            return False
        self._crossed_start[new_track_id] = old_state
        if best_tid in self._start_cross_frame:
            self._start_cross_frame[new_track_id] = self._start_cross_frame.pop(best_tid)
        if best_tid in self._start_entry:
            self._start_entry[new_track_id] = self._start_entry.pop(best_tid)
        # Carry over route/origin info too, so the report's route tracking
        # stays continuous across the ID switch instead of restarting.
        if best_tid in self._track_route_info:
            self._track_route_info[new_track_id] = self._track_route_info.pop(best_tid)

        self._orphaned_starts.pop(best_tid, None)
        self._crossed_finish.pop(best_tid, None)
        self._finish_cross_frame.pop(best_tid, None)
        self._track_centroids.pop(best_tid, None)
        self._track_last_seen.pop(best_tid, None)
        self._last_centroid.pop(best_tid, None)
        self._last_appearance.pop(best_tid, None)

        self._track_relinked.discard(best_tid)
        self._track_relinked.add(new_track_id)

        # Carry the Kalman filter itself across the ID switch — without
        # this the count is right but the displayed position still snaps,
        # since the new track's filter would otherwise start cold at the
        # new detection's raw position instead of predicting forward from
        # where the old track left off.
        if self._relink_kf_fn is not None:
            self._relink_kf_fn(best_tid, new_track_id)

        self._diag_stats["relinked"] += 1
        print(f"[relink] track={best_tid} -> track={new_track_id} "
              f"dist={best_dist:.0f}px at frame={frame_idx}")
        return True

    def _stale_cleanup(self, frame_idx):
        """Expire orphaned starts and stale tracks.

        Two tiers:
          1. Orphans beyond ZONE_RELINK_MAX_FRAMES are dropped immediately.
          2. Tracks that crossed a start line but never reached a finish
             line within STALE_TRACK_TIMEOUT frames are silently expired
             (not counted — we don't know their exit).
          3. Tracks that never crossed a start line and haven't been seen
             for STALE_TRACK_TIMEOUT frames are also cleaned up to prevent
             unbounded memory growth (every unique track_id accumulates
             centroid/route state forever otherwise).
        """
        for tid in list(self._orphaned_starts.keys()):
            if frame_idx - self._orphaned_starts[tid]["orphaned_at"] > config.ZONE_RELINK_MAX_FRAMES:
                del self._orphaned_starts[tid]

        for tid in list(self._crossed_start.keys()):
            if tid in self._track_counted:
                continue
            last_seen = self._track_last_seen.get(tid)
            if last_seen is not None and (frame_idx - last_seen) > self._stale_track_timeout:
                self._diag_stats["stale_cleanup"] += 1
                self._clear_track_state(tid)

        # Tier 3: never-crossed-start tracks — clean up so their centroid
        # and route state doesn't accumulate forever.
        for tid in list(self._track_last_seen.keys()):
            if tid in self._crossed_start or tid in self._track_counted:
                continue
            last_seen = self._track_last_seen.get(tid)
            if last_seen is not None and (frame_idx - last_seen) > self._stale_track_timeout:
                self._clear_track_state(tid)

    def _clear_track_state(self, tid):
        self._crossed_start.pop(tid, None)
        self._start_entry.pop(tid, None)
        self._start_cross_frame.pop(tid, None)
        self._crossed_finish.pop(tid, None)
        self._finish_cross_frame.pop(tid, None)
        self._track_centroids.pop(tid, None)
        self._track_last_seen.pop(tid, None)
        self._last_centroid.pop(tid, None)
        self._last_appearance.pop(tid, None)
        self._track_route_info.pop(tid, None)
        self._orphaned_starts.pop(tid, None)
        self._raw_centroid_buffer.pop(tid, None)
        self._min_age_back_checked.discard(tid)

    # ------------------------------------------------------------------ #
    #  Main update                                                        #
    # ------------------------------------------------------------------ #

    def update(self, detections, frame_idx, fps):
        events = []
        timestamp_sec = frame_idx / fps if fps else 0.0

        self._stale_cleanup(frame_idx)

        current_ids = {det["track_id"] for det in detections}
        self._orphan_disappeared_tracks(frame_idx, current_ids)

        for det in detections:
            track_id = det["track_id"]
            category = det["category"]
            is_predicted = det.get("predicted", False)

            # Primary crossing centroid: use the bbox bottom-center (road
            # contact point) to minimise perspective-induced drift — when a
            # vehicle moves toward the camera its apparent box height grows,
            # pulling the box centre upward even though the vehicle hasn't
            # moved.  The bottom-center stays on the road surface.
            #
            # Both the centroid (cx, cy) and the bbox are median-smoothed
            # (detector.py _smooth_centroids + _smooth_bboxes), so this
            # value is as jitter-free as det["centroid"] but more
            # geometrically stable for line-crossing math.
            bbox = det.get("bbox")
            if bbox is not None:
                centroid_raw = (det["centroid"][0], bbox[3])  # (cx, road_y)
            else:
                centroid_raw = det["centroid"]
            centroid_plain_raw = det.get("centroid_raw", det["centroid"])  # for diagnostic only
            centroid = det["centroid"]                                     # smoothed for route/display
            conf = det.get("conf")
            age = det.get("age", 0)

            # Track-identity bridging: if this is the first frame we've
            # ever seen this track_id, check whether it's really a
            # continuation of a track that crossed a start line and then
            # vanished nearby a moment ago (see config.ZONE_RELINK_*).
            is_new_track = track_id not in self._track_route_info
            if is_new_track:
                self._relink_orphans(track_id, centroid_raw, category, frame_idx,
                                      appearance=det.get("appearance"))

            self._track_centroids[track_id].append(centroid)          # smoothed for route direction
            # On predicted frames (non-inference) the centroid is a KF
            # projection that can drift — do NOT update _last_centroid
            # or _raw_centroid_buffer, and skip crossing checks entirely.
            # The next inference frame's segment from prev_centroid (last
            # real centroid) to current centroid will still catch any
            # crossing that occurred during the gap.
            if not is_predicted:
                prev_centroid = self._last_centroid.get(track_id)
                self._last_centroid[track_id] = centroid_raw
                self._raw_centroid_buffer[track_id].append(centroid_raw)
                appearance = det.get("appearance")
                if appearance is not None:
                    self._last_appearance[track_id] = appearance
            else:
                prev_centroid = None
            self._track_last_seen[track_id] = frame_idx

            # Route tracking
            if track_id not in self._track_route_info:
                self._track_route_info[track_id] = {
                    "origin": None,
                    "destination": None,
                    "route_changed": False,
                    "change_frame": None,
                    "change_timestamp": None,
                }
                print(f"[route] INIT track={track_id}")
            self._check_route_change(track_id, frame_idx, timestamp_sec)

            # ---- Line crossing detection (inference frames only) ----
            if prev_centroid is not None and track_id not in self._track_counted:
                # ── Gate 1: Start lines ──────────────────────────────
                if track_id not in self._crossed_start:
                    min_age = config.MIN_TRACK_AGE.get(category, config.MIN_TRACK_AGE.get("default", 0))
                    if age < min_age:
                        continue

                    # Back-check: if this is the FIRST frame where age
                    # satisfies min_age, walk the entire raw centroid
                    # history for crossings the track may have made
                    # before the gate opened (vehicle entered frame
                    # already close to the start line).
                    if track_id not in self._min_age_back_checked:
                        self._min_age_back_checked.add(track_id)
                        buf = self._raw_centroid_buffer.get(track_id)
                        if buf is not None and len(buf) >= 2:
                            for i in range(len(buf) - 1):
                                for sl_name, sl_line in self._start_lines.items():
                                    cross_pt = self._check_crossing(buf[i], buf[i + 1], sl_line)
                                    if cross_pt is not None:
                                        self._crossed_start[track_id] = (sl_name, cross_pt)
                                        self._start_cross_frame[track_id] = frame_idx
                                        self._start_entry[track_id] = (
                                            timestamp_sec, frame_idx, category, bbox, conf,
                                        )
                                        if self._lock_category_fn is not None:
                                            self._lock_category_fn(track_id, category)
                                        if self._protect_track_fn is not None:
                                            self._protect_track_fn(track_id)
                                        print(f"[line] START(BACK-CHECK) track={track_id} via {sl_name} "
                                              f"at ({cross_pt[0]:.0f},{cross_pt[1]:.0f})")
                                        break
                                if track_id in self._crossed_start:
                                    break
                        # Whether found or not, skip the normal check below
                        # so we don't double-register on this frame.
                        continue

                    for sl_name, sl_line in self._start_lines.items():
                        cross_pt = self._check_crossing(prev_centroid, centroid_raw, sl_line)
                        if cross_pt is not None:
                            # Dedup: skip crossing if another track already
                            # crossed this same start line within the last
                            # FRAME_SKIP*5 frames at nearly the same point
                            # (< 60 px).  This prevents double-counting
                            # when the detector swaps IDs across the line.
                            is_dup = False
                            for other_tid, (other_sl, other_pt) in self._crossed_start.items():
                                if other_sl != sl_name:
                                    continue
                                gap = frame_idx - self._start_cross_frame.get(other_tid, -1)
                                if gap > config.FRAME_SKIP * 5:
                                    continue
                                dx = cross_pt[0] - other_pt[0]
                                dy = cross_pt[1] - other_pt[1]
                                if (dx * dx + dy * dy) ** 0.5 < 60:
                                    is_dup = True
                                    self._diag_stats["crossing_dedup_skipped"] = self._diag_stats.get("crossing_dedup_skipped", 0) + 1
                                    print(f"[line] DEDUP-SKIP track={track_id} {sl_name} at "
                                          f"({cross_pt[0]:.0f},{cross_pt[1]:.0f}) — "
                                          f"track={other_tid} already crossed at "
                                          f"({other_pt[0]:.0f},{other_pt[1]:.0f})")
                                    break
                                if is_dup:
                                    break
                            if is_dup:
                                continue

                            # Regression check: would smoothed centroid have also crossed?
                            sm_cross = self._check_crossing(prev_centroid, centroid, sl_line)
                            if sm_cross is None:
                                self._diag_stats["raw_smoothed_side_mismatch"] += 1
                            # KF-vs-raw comparison: would plain raw have crossed?
                            plain_cross = self._check_crossing(prev_centroid, centroid_plain_raw, sl_line)
                            if plain_cross is None and centroid_plain_raw != centroid_raw:
                                self._diag_stats["kf_vs_raw_crossing_mismatch"] += 1

                            self._crossed_start[track_id] = (sl_name, cross_pt)
                            self._start_cross_frame[track_id] = frame_idx
                            self._start_entry[track_id] = (
                                timestamp_sec, frame_idx, category, bbox, conf,
                            )
                            # Lock this track's category at start-line
                            # crossing — the detector will return this
                            # category for every subsequent frame.
                            if self._lock_category_fn is not None:
                                self._lock_category_fn(track_id, category)
                            # Protect track from cheap ID swaps between
                            # start and finish — if the detector swaps
                            # IDs here, the crossing state is orphaned
                            # and the count is lost.
                            if self._protect_track_fn is not None:
                                self._protect_track_fn(track_id)
                            print(f"[line] START track={track_id} via {sl_name} "
                                  f"at ({cross_pt[0]:.0f},{cross_pt[1]:.0f}) "
                                  f"centroid_raw=({centroid_raw[0]:.0f},{centroid_raw[1]:.0f})")
                            break

                # ── Gate 2: Finish lines ─────────────────────────────
                # Must have crossed a start line first.
                # Minimum 3-frame gap between start and finish crossing
                # prevents flicker-induced false positives (a detection
                # that flickers across both lines within 1-2 frames is
                # almost certainly noise).
                # A displacement gate (scaled by FRAME_SKIP) blocks
                # finish-line checks when the centroid jumps unrealistically
                # far between detections, which typically indicates a
                # tracking ID switch to a different vehicle.
                if (track_id in self._crossed_start
                        and not self._crossed_finish.get(track_id)):
                    start_fi = self._start_cross_frame.get(track_id, -1)
                    if (frame_idx - start_fi) >= config.MIN_CROSSING_GAP_FRAMES:

                        # Displacement sanity: if the centroid jumped
                        # more than 200 px per frame-skip step the track
                        # was likely lost and re-acquired on a different
                        # vehicle.  Log the warning but still check the
                        # finish line — a fast legitimate vehicle might
                        # exceed the threshold too, and using `continue`
                        # here would permanently skip its count.
                        dx = centroid_raw[0] - prev_centroid[0]
                        dy = centroid_raw[1] - prev_centroid[1]
                        displacement = (dx * dx + dy * dy) ** 0.5
                        disp_max = 200 * config.FRAME_SKIP
                        if displacement > disp_max:
                            self._diag_stats["disp_reject"] += 1
                            # Do NOT continue — still check finish line

                        for fin_name, fin_line in self._finish_lines.items():
                            fin_cross = self._check_crossing(prev_centroid, centroid_raw, fin_line)
                            if fin_cross is not None:
                                # Regression check: would smoothed have also crossed?
                                sm_fin = self._check_crossing(prev_centroid, centroid, fin_line)
                                if sm_fin is None:
                                    self._diag_stats["raw_smoothed_side_mismatch"] += 1
                                # KF-vs-raw: would plain raw have crossed?
                                plain_fin = self._check_crossing(prev_centroid, centroid_plain_raw, fin_line)
                                if plain_fin is None and centroid_plain_raw != centroid_raw:
                                    self._diag_stats["kf_vs_raw_crossing_mismatch"] += 1

                                self._crossed_finish[track_id] = fin_name
                                self._finish_cross_frame[track_id] = frame_idx
                                print(f"[line] FINISH track={track_id} via {fin_name} "
                                      f"at frame={frame_idx}")

                                self._count_track(track_id, events, fps, fin_name)
                                self._crossed_start.pop(track_id, None)
                                self._start_cross_frame.pop(track_id, None)
                                self._start_entry.pop(track_id, None)
                                self._crossed_finish.pop(track_id, None)
                                self._finish_cross_frame.pop(track_id, None)
                                break
                    else:
                        self._diag_stats["gap_reject"] += 1

        if events:
            print(f"[line] COUNTED {len(events)} event(s) at frame={frame_idx} "
                  f"ts={timestamp_sec:.0f}s")

        return events

    def print_diag_stats(self):
        s = self._diag_stats
        print(f"[line-stats] counted={s['counted']} "
              f"stale_cleanup={s['stale_cleanup']} "
              f"disp_reject={s['disp_reject']} "
              f"gap_reject={s['gap_reject']} "
              f"relinked={s['relinked']} "
              f"crossing_dedup_skipped={s['crossing_dedup_skipped']}")
        if s["raw_smoothed_side_mismatch"]:
            print(f"[regression] RAW vs SMOOTHED crossing disagreement: "
                  f"{s['raw_smoothed_side_mismatch']} frame(s)")
        if s["kf_vs_raw_crossing_mismatch"]:
            print(f"[regression] KF vs PLAIN-RAW crossing disagreement: "
                  f"{s['kf_vs_raw_crossing_mismatch']} frame(s)")
        if s["origin_mismatch"]:
            print(f"[origin] {s['origin_mismatch']} counted crossing(s) had a start-line "
                  f"origin that didn't match the finish line's implied arm — see "
                  f"crossing_diagnostics.csv 'origin'/'origin_mismatch' columns.")
        n_routes = len(self._track_route_info)
        n_route_changed = sum(1 for v in self._track_route_info.values()
                              if v.get("route_changed"))
        print(f"[line-stats] routes_tracked={n_routes} route_changed={n_route_changed}")
