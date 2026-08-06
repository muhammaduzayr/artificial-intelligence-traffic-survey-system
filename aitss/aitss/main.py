"""
AITSS core pipeline entry point.

Usage:
    python -m aitss.main --video path/to/survey.mp4 --start-time "2026-07-11 08:00:00"

This runs detection + tracking + zone-based counting over the whole
video. Two report files are written incrementally every 15 simulated
minutes of video, and finalized at the end:
  - output/<Enumerator>/traffic_survey_report.xlsx  (summary/raw-events)
  - output/<Enumerator>/tmc_report.xlsx             (turning-movement
    workbook matching your standard survey template)
If --enumerator is omitted, results go to output/.
Use --output-dir to override the base directory entirely.
Pressing Ctrl+C also saves both reports with everything counted so far
instead of losing it.
"""

import argparse
import contextlib
import os
import sys
import time
from datetime import datetime, timedelta

import cv2
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
import numpy as np

from . import config
from .detector import VehicleDetector
from .video_io import get_video_info, open_video, release_video
from .zone_counter import ZoneCounter
from .aggregator import ReportAggregator
from .tmc_export import export_tmc_report, check_raw_df_for_double_counts
from .zone_picker import load_zones, load_occluders


@contextlib.contextmanager
def _suppress_stderr():
    old_stderr = sys.stderr
    try:
        with open(os.devnull, "w") as devnull:
            sys.stderr = devnull
            yield
    finally:
        sys.stderr = old_stderr


def _sanitize_enumerator_dirname(enumerator):
    """Turn a free-typed enumerator name into a single safe directory
    segment.

    enumerator comes straight from a CLI flag / GUI text field and is
    joined onto output_dir to build the report path (see run() below).
    Strip path separators and ".." so a value like "../../Windows" can't
    make the report land outside output_dir.
    """
    name = enumerator.strip()
    name = name.replace("\\", "_").replace("/", "_")
    name = name.replace("..", "_")
    return name.strip(" ._")


def _rects_overlap(a, b):
    """Check if two (x1,y1,x2,y2) rects overlap (including touching)."""
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _bbox_iou(a, b):
    """IoU of two (x1,y1,x2,y2) boxes."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter) if (area_a + area_b - inter) > 0 else 0.0


def _cluster_by_overlap(detections, iou_threshold=0.15):
    """Group detections whose bboxes overlap significantly.
    Returns list of lists (each inner list is a cluster)."""
    if not detections:
        return []
    n = len(detections)
    assigned = [False] * n
    clusters = []
    for i in range(n):
        if assigned[i]:
            continue
        cluster = [detections[i]]
        assigned[i] = True
        for j in range(i + 1, n):
            if assigned[j]:
                continue
            iou = _bbox_iou(detections[i]["bbox"], detections[j]["bbox"])
            if iou > iou_threshold:
                cluster.append(detections[j])
                assigned[j] = True
        clusters.append(cluster)
    return clusters


CATEGORY_COLORS = {
    "Car":          (0, 255, 0),      # green
    "Motorcycle":   (0, 200, 255),    # yellow-orange
    "Light Truck":  (255, 140, 0),    # orange
    "Truck":        (255, 140, 0),    # orange (same as Light Truck)
    "Bus":          (0, 0, 255),      # red
    "default":      (200, 200, 200),  # grey fallback for anything unlisted
}


def _color_for_category(category):
    """Stable color keyed by vehicle category, not track ID.

    A Car is always green whether it's track #4 or #400 — and a wrong
    color instantly flags a misclassification to your eye instead of
    being per-ID noise you have to mentally filter out.
    """
    return CATEGORY_COLORS.get(category, CATEGORY_COLORS["default"])


def _place_label(bbox, label, font_scale, thickness, placed_rects, frame_shape):
    """Place a label near the bbox, nudging to avoid collisions.
    Tries candidates in order—each at increasing vertical offsets to
    stack labels that target the same area.  Returns (bg_rect, text_origin)
    or None if every candidate is out of bounds."""
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x1, y1, x2, y2 = map(int, bbox)
    bw = x2 - x1
    fw, fh = frame_shape[1], frame_shape[0]

    # Candidate positions relative to the bbox:
    #   (dx_to_anchor, dy_to_anchor, anchor_is_y1|y2, text_align)
    # anchor_is_y1 → dy offsets from top edge; y2 → from bottom edge
    # text_align 0 = left-align with x1, 1 = right-align with x2, 2 = center
    candidates = [
        # above bbox
        (0, -5, "top", 0),
        (-tw // 2, -5, "top", 2),
        (tw // 2, -5, "top", 1),
        # left of bbox
        (-tw - 6, 0, "top", 0),
        # right of bbox
        (bw + 4, 0, "top", 0),
        # below bbox
        (0, bw // 2 + 6, "bottom", 0),
    ]

    for dx, dy_base, anchor, align in candidates:
        for stack_offset in range(0, 80, 20):  # try vertical stacking
            dy = dy_base - stack_offset if anchor == "top" else dy_base + stack_offset
            if anchor == "top":
                lx = x1 + dx
                ly = max(0, y1 + dy)
            else:
                lx = x1 + dx
                ly = min(fh - th - 4, y2 + dy)

            if align == 1:
                lx = x2 - tw - 2
            elif align == 2:
                lx = x1 + (x2 - x1) // 2 - tw // 2

            rect = (lx, ly - th - 2, lx + tw + 4, ly + 2)
            origin = (lx + 2, ly)

            if rect[0] < 0 or rect[2] > fw or rect[1] < 0 or rect[3] > fh:
                continue
            if not any(_rects_overlap(rect, pr) for pr in placed_rects):
                return (rect, origin)
    # Fallback: above bbox even if colliding
    lx = x1
    ly = max(0, y1 - 5)
    rect = (lx, ly - th - 2, lx + tw + 4, ly + 2)
    return (rect, (lx + 2, ly))


def draw_debug_frame(frame, detections, start_lines, finish_lines,
                     route_info=None, source_timestamp=None, frame_idx=None, fps=None,
                     lane_counts=None, display_id_fn=None, display_category_fn=None):
    # display_id_fn maps a track's internal track_id to the number actually
    # shown on screen. Defaults to showing track_id as-is. When the caller
    # passes VehicleDetector.display_id, a vehicle whose internal ID was
    # bridged across a tracker ID switch (see detector.relink_kf) keeps
    # showing its original number instead of visibly flipping to the new
    # one — route_info/color-shift lookups still use the raw track_id
    # since that's what the rest of the pipeline keys state by.
    if display_id_fn is None:
        display_id_fn = lambda tid: tid
    # display_category_fn(tid, category) maps to the category actually
    # shown/colored on screen. Defaults to the detection's own category
    # (today's behavior). VehicleDetector.display_category returns a
    # locked category's stable value instead, so a vehicle doesn't
    # flicker to a different class label/color for a frame or two right
    # after an ID switch.
    if display_category_fn is None:
        display_category_fn = lambda tid, category: category
    # --- source-time overlay ---
    if source_timestamp is not None:
        elapsed_video = frame_idx / fps if frame_idx is not None and fps else 0.0
        elapsed_label = str(timedelta(seconds=max(0, int(elapsed_video))))
        timestamp_label = source_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        fps_label = f"{fps:.4f}" if fps else "n/a"
        time_text = (
            f"SOURCE TIME {timestamp_label} | VIDEO +{elapsed_label} | "
            f"FRAME {frame_idx if frame_idx is not None else 'n/a'} | FPS {fps_label}"
        )
        cv2.rectangle(frame, (8, 8), (min(frame.shape[1] - 8, 900), 40), (0, 0, 0), -1)
        cv2.putText(
            frame, time_text, (14, 30), cv2.FONT_HERSHEY_DUPLEX,
            0.55, (0, 255, 255), 2, cv2.LINE_AA,
        )

    # --- start lines (green), labeled by approach ---
    for name, pts in start_lines.items():
        p1 = tuple(map(int, pts[0]))
        p2 = tuple(map(int, pts[1]))
        cv2.line(frame, p1, p2, (0, 255, 0), 3)
        label = f"START {name}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        lx, ly = p1[0] + 4, p1[1] - 4
        cv2.rectangle(frame, (lx, ly - th - 2), (lx + tw + 4, ly + 2), (0, 0, 0), -1)
        cv2.putText(frame, label, (lx + 2, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # --- finish lines (red), labeled by name with live count ---
    for name, pts in finish_lines.items():
        p1 = tuple(map(int, pts[0]))
        p2 = tuple(map(int, pts[1]))
        cv2.line(frame, p1, p2, (0, 0, 255), 3)
        count = lane_counts.get(name, 0) if lane_counts else 0
        label = f"{name} ({count})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        lx, ly = p1[0] + 4, p1[1] - 4
        cv2.rectangle(frame, (lx, ly - th - 2), (lx + tw + 4, ly + 2), (0, 0, 0), -1)
        cv2.putText(frame, label, (lx + 2, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # --- suppress freshly-spawned tracks from the overlay ---
    # A track this young either becomes a real, confirmed vehicle (and
    # starts being drawn within a few more frames) or gets silently
    # rejected as static/noise a bit later — detector.py's static-motion
    # filter needs STATIC_MIN_SAMPLES real detections before it can tell
    # the two apart, so every transient false-positive blip (foliage,
    # reflections, a parked vehicle's shadow) otherwise flashes on screen
    # for its few-frame grace period before being rejected. Reusing the
    # same MIN_TRACK_AGE gate already used to decide counting eligibility
    # means a real vehicle's box just appears with a short, deliberate
    # delay instead.
    detections = [
        d for d in detections
        if d.get("age", 0) >= config.MIN_TRACK_AGE.get(
            d["category"], config.MIN_TRACK_AGE.get("default", 0)
        )
    ]

    # --- detection boxes with collision-free label placement ---
    if not detections:
        return frame

    placed_label_rects = []

    # Cluster by bbox overlap (IoU > 0.15) rather than centre distance.
    clusters = _cluster_by_overlap(detections, iou_threshold=0.15)

    for cluster in clusters:
        cluster = sorted(cluster, key=lambda d: d["bbox"][1])  # top→bottom

        if len(cluster) >= 3:
            # --- Group label for dense clusters ---
            cats = {}
            for d in cluster:
                cat = display_category_fn(d["track_id"], d["category"])
                cats[cat] = cats.get(cat, 0) + 1
            group_parts = [f"{n} {c}" for c, n in sorted(cats.items())]
            group_label = ", ".join(group_parts)

            boxes = [d.get("display_bbox", d["bbox"]) for d in cluster]
            xs = [b[0] for b in boxes] + [b[2] for b in boxes]
            ys = [b[1] for b in boxes] + [b[3] for b in boxes]
            ux1, uy1, ux2, uy2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

            # Draw each box with a distinct thin color
            for i, d in enumerate(cluster):
                bx1, by1, bx2, by2 = map(int, d.get("display_bbox", d["bbox"]))
                tid = d["track_id"]
                color = _color_for_category(display_category_fn(tid, d["category"]))
                # Shift overlapping boxes by 1-2 px per track so they
                # don't render flush on top of each other. Keyed by
                # display_id (not the raw track_id) so a relinked vehicle's
                # shift doesn't itself jump by 1-2px when its internal ID
                # switches.
                shift = (display_id_fn(tid) % 5) - 2
                cv2.rectangle(frame, (bx1 + shift, by1 + shift),
                              (bx2 + shift, by2 + shift), color, 1)

            # Group label above union box
            (gtw, gth), _ = cv2.getTextSize(group_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            glx = int(ux1 + (ux2 - ux1) / 2 - gtw / 2)
            gly = max(0, uy1 - 8)
            bg = (glx - 2, gly - gth - 2, glx + gtw + 4, gly + 2)
            if not any(_rects_overlap(bg, pr) for pr in placed_label_rects):
                cv2.rectangle(frame, (bg[0], bg[1]), (bg[2], bg[3]), (0, 0, 0), -1)
                cv2.putText(frame, group_label, (glx, gly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                placed_label_rects.append(bg)
        else:
            # --- Individual rendering with collision avoidance ---
            for d in cluster:
                tid = d["track_id"]
                display_bbox = d.get("display_bbox", d["bbox"])
                x1, y1, x2, y2 = map(int, display_bbox)
                ri = route_info.get(tid) if route_info else None
                route_changed = ri is not None and ri["route_changed"]

                shown_category = display_category_fn(tid, d["category"])
                base_color = _color_for_category(shown_category)
                color = (0, 0, 255) if route_changed else base_color
                thickness = 3 if route_changed else 2
                # Shift box slightly per track to separate overlapping boxes.
                # Keyed by display_id — see the group-label branch above.
                shown_id = display_id_fn(tid)
                shift = (shown_id % 5) - 2
                cv2.rectangle(frame, (x1 + shift, y1 + shift),
                              (x2 + shift, y2 + shift), color, thickness)

                label = f'#{shown_id} {shown_category}'
                if route_changed:
                    label += " !"

                route_label = ""
                if ri is not None:
                    o = ri.get("origin")
                    d_ = ri.get("destination")
                    if o and d_:
                        route_label = f"{o}->{d_}"
                    elif o:
                        route_label = o

                # Place main label with collision avoidance
                placed = _place_label(display_bbox, label, 0.5, 1,
                                      placed_label_rects, frame.shape)
                if placed:
                    bg_rect, origin = placed
                    cv2.rectangle(frame, (bg_rect[0], bg_rect[1]),
                                  (bg_rect[2], bg_rect[3]), (0, 0, 0), -1)
                    cv2.putText(frame, label, origin,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    placed_label_rects.append(bg_rect)

                if route_label:
                    (rw, rh), _ = cv2.getTextSize(
                        route_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    rlx = max(0, x1)
                    rly = max(0, origin[1] - 16) if placed else max(0, y1 - 24)
                    rrect = (rlx, rly - rh - 2, rlx + rw + 4, rly + 2)
                    if not any(_rects_overlap(rrect, pr) for pr in placed_label_rects):
                        cv2.rectangle(frame, (rrect[0], rrect[1]),
                                      (rrect[2], rrect[3]), (0, 0, 0), -1)
                        cv2.putText(frame, route_label, (rlx + 2, rly),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
                        placed_label_rects.append(rrect)

        # --- centroid marker: draw for every detection, regardless of label path ---
        for d in cluster:
            ckf = d.get("centroid_kf_smooth", d.get("centroid_kf", d["centroid"]))
            cxx, cyy = int(round(ckf[0])), int(round(ckf[1]))
            cr = 4
            # Neutral, low-contrast marker — informative without adding visual noise
            cv2.line(frame, (cxx - cr, cyy), (cxx + cr, cyy), (255, 255, 255), 1)
            cv2.line(frame, (cxx, cyy - cr), (cxx, cyy + cr), (255, 255, 255), 1)
            cv2.circle(frame, (cxx, cyy), cr, (180, 180, 180), 1)  # flat grey ring, no per-track color

    return frame


def format_eta(seconds):
    if seconds is None or seconds < 0:
        return "unknown"
    return str(timedelta(seconds=int(seconds)))


def run(video_path, survey_start_time, save_debug_video=False, output_dir=None, live_preview=False,
        model_path=None, conf=None, imgsz=None, show_detections=False,
        enumerator="", junction="", junction_name="", device=None,
        survey_end_time=None, fps_override=None):
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    safe_enumerator = _sanitize_enumerator_dirname(enumerator)
    if safe_enumerator:
        output_dir = os.path.join(output_dir, safe_enumerator)
    os.makedirs(output_dir, exist_ok=True)

    # Per-video zone lines: sidecar .zones.json takes priority;
    # fall back to config.py global defaults.
    start_lines, finish_lines = load_zones(video_path)
    if start_lines is None:
        start_lines = config.START_LINES
        finish_lines = config.FINISH_LINES
        print(f"Using global zones from config.py")
    else:
        print(f"Using per-video zones from {video_path}.zones.json")

    # Per-video known-occluder boxes (poles, signs, fixed foliage) — same
    # sidecar file as the zone lines, since both are specific to this
    # camera's fixed framing. Falls back to config.KNOWN_OCCLUDERS (empty
    # by default) when the sidecar has none defined.
    occluders = load_occluders(video_path)
    if occluders:
        print(f"Using {len(occluders)} per-video known occluder(s) from {video_path}.zones.json")
    else:
        occluders = config.KNOWN_OCCLUDERS
        print(f"Using global KNOWN_OCCLUDERS from config.py ({len(occluders)} zone(s))")

    detector = VehicleDetector(model_path=model_path, conf=conf, imgsz=imgsz, device=device,
                               occluders=occluders)

    counter = ZoneCounter(start_lines, finish_lines,
                           lock_category_fn=detector.lock_category,
                           retire_track_fn=detector.retire_track,
                           protect_track_fn=detector.protect_track,
                           relink_kf_fn=detector.relink_kf,
                           diagnostic_csv_path=os.path.join(output_dir, "crossing_diagnostics.csv"))
    aggregator = ReportAggregator(survey_start_time)

    print(f"START_LINES: {list(start_lines.keys())}")
    for name, pts in start_lines.items():
        print(f"  {name}: {pts}")
    print(f"FINISH_LINES: {list(finish_lines.keys())}")
    for name, pts in finish_lines.items():
        print(f"  {name}: {pts}")

    cap = open_video(video_path)
    reported_fps, total_frames = get_video_info(cap)

    # Cross-check the reported FPS against the video's actual duration.
    # CCTV/NVR exports are frequently variable-frame-rate (VFR) — the
    # file's metadata reports one "average" FPS, but real elapsed time
    # per frame isn't constant. Since every timestamp in this pipeline
    # (report intervals, TMC time-of-day rows, debug video playback
    # speed) comes from frame_idx / fps, trusting a wrong reported value
    # would silently drift for the whole survey. We measure the actual
    # duration by seeking near the end and reading its real timestamp,
    # then use whichever FPS is more consistent with that.
    #
    # IMPORTANT LIMITATION: this only catches drift the file is honest
    # about internally. Some NVR exports mislabel their true capture rate
    # entirely — in that case the file's OWN internal timestamps are
    # consistently wrong too, so this check finds nothing unusual even
    # though the fps is genuinely off. That's what survey_end_time below
    # is for: it doesn't trust the file at all, and derives fps purely
    # from a real-world clock fact you already know.
    fps = reported_fps
    fps_source = "file metadata"
    if total_frames and total_frames > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        ok = cap.grab()
        actual_duration_sec = (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0) if ok else None
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if actual_duration_sec and actual_duration_sec > 0:
            measured_fps = total_frames / actual_duration_sec
            drift_pct = abs(measured_fps - reported_fps) / reported_fps * 100
            if drift_pct > 2:
                print(f"WARNING: video FPS mismatch detected — file metadata reports "
                      f"{reported_fps:.2f} fps, but actual duration implies ~{measured_fps:.2f} fps "
                      f"({drift_pct:.1f}% difference). This usually means the video is variable "
                      f"frame rate (common with CCTV/NVR exports), which throws off both timestamp "
                      f"accuracy and playback speed. Using the measured value ({measured_fps:.2f} fps) "
                      f"for all timing in this run. For a permanent fix, re-encode to constant frame "
                      f"rate: ffmpeg -i input.mp4 -vsync cfr -r {measured_fps:.0f} -c:v libx264 "
                      f"-crf 18 -c:a copy output_cfr.mp4")
                fps = measured_fps
                fps_source = "measured from the file's own internal timestamps"
    release_video(cap)

    # Ground-truth override: a known real-world start AND end clock time
    # beats anything the file itself claims, because it doesn't depend on
    # the file being honest about its own capture rate at all. Manual
    # --fps takes top priority if you already know the correct number.
    if fps_override is not None:
        fps = fps_override
        fps_source = "manual --fps override"
        print(f"Using manually specified fps: {fps:.4f} (overrides file metadata/timestamps entirely).")
    elif survey_end_time is not None and total_frames:
        real_elapsed_sec = (survey_end_time - survey_start_time).total_seconds()
        if real_elapsed_sec <= 0:
            raise ValueError("--end-time must be AFTER --start-time.")
        derived_fps = total_frames / real_elapsed_sec
        close_to_metadata = abs(derived_fps - reported_fps) / reported_fps < 0.02
        print(f"Deriving true fps from known recording start/end: {total_frames} frames over "
              f"{format_eta(real_elapsed_sec)} of real time = {derived_fps:.4f} fps "
              f"(file metadata claimed {reported_fps:.2f} fps — "
              f"{'matches closely, good sign' if close_to_metadata else 'a real mismatch — using the derived value for ALL timing in this run'}).")
        fps = derived_fps
        fps_source = "derived from known --start-time/--end-time"

    print(f"Using fps={fps:.4f} for all timing in this run (source: {fps_source}).")
    print(f"Enumerator: {enumerator or '(not set)'} | Junction: {junction or '(not set)'} | Junction name: {junction_name or '(not set)'}")
    print(f"Output directory: {os.path.abspath(output_dir)}")

    print(f"Model: {detector.model.ckpt_path if hasattr(detector.model, 'ckpt_path') else (model_path or config.MODEL_PATH)} "
          f"| imgsz: {detector.imgsz} | conf: {detector.conf}")

    if total_frames:
        video_duration_sec = total_frames / fps
        print(f"Video: ~{total_frames} frames, {fps:.4f} fps, "
              f"~{format_eta(video_duration_sec)} of footage.")
    else:
        print("Video: frame count unavailable, progress % won't be shown.")

    writer = None
    if save_debug_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        debug_path = os.path.join(output_dir, "debug_annotated.mp4")

    preview_window = "AITSS live preview (press q in this window to stop early)"
    if live_preview:
        cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)

    total_events = 0
    last_exported_bucket = -1
    start_wall_time = time.time()
    stopped_early = False

    def export_reports():
        raw_df = aggregator.raw_dataframe()
        total_events = len(raw_df)
        if total_events == 0:
            print("[WARNING] export_reports called with ZERO count events in raw_df. "
                  "TMC report will be empty.")
        else:
            print(f"[export] exporting {total_events} count events to TMC report...")
            # Automated double-count self-check.
            check_raw_df_for_double_counts(
                raw_df, config.ID_RETIREMENT_COOLDOWN_SECONDS
            )
        excel_path, csv_path = aggregator.export(output_dir=output_dir)
        tmc_path = os.path.join(output_dir, "tmc_report.xlsx")
        try:
            _, unplaced = export_tmc_report(
                raw_df, survey_start_time, tmc_path,
                interval_minutes=config.INTERVAL_MINUTES,
                enumerator=enumerator, junction=junction, junction_name=junction_name,
            )
        except PermissionError:
            # Excel keeps an open workbook locked on Windows. Preserve the
            # completed run by writing a timestamped copy instead of failing
            # the whole export. The original file can still be updated after
            # the user closes Excel.
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            fallback_tmc_path = os.path.join(output_dir, f"tmc_report_{timestamp}.xlsx")
            print(f"[WARNING] Cannot overwrite {tmc_path!r}; it may be open in Excel.")
            print(f"[WARNING] Saving TMC report to {fallback_tmc_path!r} instead.")
            _, unplaced = export_tmc_report(
                raw_df, survey_start_time, fallback_tmc_path,
                interval_minutes=config.INTERVAL_MINUTES,
                enumerator=enumerator, junction=junction, junction_name=junction_name,
            )
            tmc_path = fallback_tmc_path
        return excel_path, csv_path, tmc_path, unplaced

    # Real-time pacing target for live preview: one frame every 1/fps
    # seconds of WALL-CLOCK time, matching the video's own natural pace.
    # Without this, the preview just displays each frame the instant
    # processing finishes it — if your hardware is fast enough to process
    # faster than real-time, the preview visibly races ahead of normal
    # playback speed instead of looking natural. This can only slow a
    # too-fast loop DOWN to match real speed; if per-frame processing
    # itself takes longer than 1/fps (the loop is behind), there's
    # nothing to pace against and it simply displays as soon as ready,
    # same as before — no software fix changes how long inference itself
    # takes.
    frame_budget_sec = 1.0 / fps if fps else None
    last_display_wall_time = None

    try:
        for frame_idx, frame, detections in detector.track_stream(video_path):
            if show_detections and detections and frame_idx % 30 == 0:
                event_time = survey_start_time + timedelta(seconds=frame_idx / fps)
                detection_strings = []
                for d in detections:
                    tid = d["track_id"]
                    ri = counter.get_route_info(tid)
                    route_s = ""
                    if ri:
                        o = ri.get("origin") or "?"
                        d_ = ri.get("destination") or "?"
                        route_s = f" {o}->{d_}"
                        if ri["route_changed"]:
                            route_s += " !"
                    detection_strings.append(
                        f"#{tid} {d['category']} conf={d['conf']:.2f}{route_s}"
                    )
                print(f"[detections] {event_time.strftime('%H:%M:%S')} ({len(detections)} objs) {', '.join(detection_strings)}")

            # Only real (non-predicted) detections are fed to the zone
            # counter — predicted detections are for smooth display only.
            real_dets = [d for d in detections if not d.get("predicted")]
            events = counter.update(real_dets, frame_idx, fps)
            aggregator.add_events(events)
            total_events += len(events)

            annotated = None
            if save_debug_video or live_preview:
                annotated = draw_debug_frame(
                    frame.copy(), detections, start_lines, finish_lines,
                    route_info=counter.get_all_route_info(),
                    source_timestamp=survey_start_time + timedelta(seconds=frame_idx / fps),
                    frame_idx=frame_idx,
                    fps=fps,
                    lane_counts=counter.get_lane_counts(),
                    display_id_fn=detector.display_id,
                    display_category_fn=detector.display_category,
                )

            if save_debug_video:
                if writer is None:
                    h, w = annotated.shape[:2]
                    writer = cv2.VideoWriter(debug_path, fourcc, fps, (w, h))
                    # Write 2 black frames to prime the encoder so the
                    # first real frame doesn't come out garbled/blocky
                    # (a known issue with mp4v on some systems).
                    black = np.zeros((h, w, 3), dtype=np.uint8)
                    writer.write(black)
                    writer.write(black)
                writer.write(annotated)

            if live_preview:
                if frame_budget_sec and last_display_wall_time is not None:
                    elapsed = time.time() - last_display_wall_time
                    remaining = frame_budget_sec - elapsed
                    if remaining > 0:
                        time.sleep(remaining)
                last_display_wall_time = time.time()

                cv2.imshow(preview_window, annotated)
                # waitKey is required for the window to actually redraw.
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stopped early by user (q pressed in preview window).")
                    stopped_early = True
                    break

            # --- Incremental report: flush to disk every 15 simulated minutes ---
            video_time_sec = frame_idx / fps
            current_bucket = int(video_time_sec // (config.INTERVAL_MINUTES * 60))
            if current_bucket > last_exported_bucket:
                excel_path, csv_path, tmc_path, _ = export_reports()
                elapsed_video_time = survey_start_time + timedelta(seconds=video_time_sec)
                print(f"  [partial report updated] up to ~{elapsed_video_time.strftime('%H:%M')} "
                      f"-> {excel_path}")
                last_exported_bucket = current_bucket

            if frame_idx % 100 == 0:
                elapsed_wall = time.time() - start_wall_time
                progress_str = ""
                eta_str = ""
                speed_str = ""
                if total_frames:
                    pct = 100 * frame_idx / total_frames
                    frames_per_sec_real = frame_idx / elapsed_wall if elapsed_wall > 0 else 0
                    remaining_frames = total_frames - frame_idx
                    eta_sec = remaining_frames / frames_per_sec_real if frames_per_sec_real > 0 else None
                    progress_str = f" ({pct:.1f}%)"
                    eta_str = f" | ETA: {format_eta(eta_sec)}"
                    # Explicit "how does processing speed compare to the
                    # video's own real-time pace" readout — this is the
                    # actual number that tells you whether the preview
                    # LOOKING slow means anything, rather than eyeballing
                    # it. >=1.0x means processing is keeping up with (or
                    # beating) real-time; below that is how many times
                    # slower than real-time it currently is. This has NO
                    # bearing on report accuracy (see run()'s docstring/
                    # comments on fps) — it's purely a speed diagnostic.
                    if frames_per_sec_real > 0:
                        speed_ratio = frames_per_sec_real / fps
                        speed_str = f" | speed: {speed_ratio:.2f}x real-time ({frames_per_sec_real:.1f} fps processed vs {fps:.1f} fps video)"
                print(f"[frame {frame_idx}{progress_str}] counted so far: {total_events}"
                      f" | elapsed: {format_eta(elapsed_wall)}{eta_str}{speed_str}")
    except KeyboardInterrupt:
        print("\nCtrl+C detected — stopping now and saving report for everything counted so far...")
        stopped_early = True

    if live_preview:
        with _suppress_stderr():
            cv2.destroyAllWindows()

    if writer is not None:
        with _suppress_stderr():
            writer.release()
        print(f"Debug video saved to {debug_path}")

    excel_path, csv_path, tmc_path, unplaced_labels = export_reports()
    peak_start, peak_vol = aggregator.peak_hour()
    counter.print_diag_stats()

    print("\n--- AITSS Survey Complete" + (" (stopped early)" if stopped_early else "") + " ---")
    print(f"Total vehicles counted: {total_events}")
    if peak_start is not None:
        print(f"Peak hour: {peak_start.strftime('%H:%M')} ({peak_vol} vehicles)")
    print(f"Report (Excel): {excel_path}")
    print(f"Report (CSV):   {csv_path}")
    print(f"TMC Report (North/East/South/West): {tmc_path}")
    print(f"Crossing diagnostics: {counter.diagnostic_csv_path}")
    if unplaced_labels:
        print(f"WARNING: these zone labels didn't match the N/S/E/W + 1-4 pattern and "
              f"were NOT included in the TMC report: {unplaced_labels}")


def _warn_if_stale_pyc():
    """Guard against silently testing stale compiled bytecode.

    This package's directory can end up with standalone
    <module>_cpython-*.pyc files sitting next to their .py sources (e.g.
    from a packaging/build step) — a different layout from the normal
    __pycache__ folder Python manages itself. If anything in the workflow
    ever runs one of those .pyc files directly instead of the .py source,
    an edit made during fast tuning iteration on detector.py/zone_counter.py
    /etc. can go untested with no indication anything is wrong.

    This is a cheap, non-fatal check: it warns (doesn't block) when a
    source file is newer than a sibling compiled file of the same name, so
    a stale .pyc doesn't go unnoticed. The reliable fix is still to always
    run via `python -m aitss.main` from source rather than any .pyc, so
    Python's own mtime-based invalidation stays in control — this is a
    backstop for the case where that discipline slips.
    """
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        entries = os.listdir(pkg_dir)
    except OSError:
        return
    py_files = [e for e in entries if e.endswith(".py")]
    pyc_files = [e for e in entries if e.endswith(".pyc")]
    for py_name in py_files:
        stem = py_name[:-3]
        py_path = os.path.join(pkg_dir, py_name)
        matching_pyc = [p for p in pyc_files if p.startswith(stem + "_cpython-")]
        for pyc_name in matching_pyc:
            pyc_path = os.path.join(pkg_dir, pyc_name)
            try:
                stale = os.path.getmtime(py_path) > os.path.getmtime(pyc_path)
            except OSError:
                continue
            if stale:
                print(f"WARNING: {py_name} has been edited more recently than "
                      f"its compiled {pyc_name} — if anything in this workflow "
                      f"runs that .pyc directly, it will silently test old "
                      f"logic. Run via `python -m aitss.main` from source, or "
                      f"delete stale __pycache__/*.pyc before testing.")


def main():
    _warn_if_stale_pyc()
    parser = argparse.ArgumentParser(description="AITSS core AI pipeline")
    parser.add_argument("--video", required=True, help="Path to recorded traffic video")
    parser.add_argument(
        "--start-time",
        required=True,
        help='Wall-clock time the recording started, e.g. "2026-07-11 08:00:00"',
    )
    parser.add_argument(
        "--end-time",
        default=None,
        help='Known wall-clock time the recording actually ENDS, e.g. "2026-07-11 22:00:00". '
             "If given, the TRUE fps is derived as total_frames / real elapsed seconds, "
             "overriding whatever frame rate the video file's own metadata claims. Use this "
             "when you know the real start/end clock times and suspect the file's declared "
             "fps is wrong (common with NVR exports) — file metadata can't be trusted to "
             "flag this itself, since if it's wrong once it's usually wrong consistently.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Directly override the fps used for ALL timing in the run, if you already know "
             "the correct value. Takes priority over --end-time and the file's own metadata.",
    )
    parser.add_argument("--debug-video", action="store_true", help="Save an annotated debug video")
    parser.add_argument(
        "--live-preview",
        action="store_true",
        help="Show a live window with detection boxes and zones while processing. "
             "Press 'q' in that window to stop early and still get a report of what was counted so far.",
    )
    parser.add_argument("--model", default=None,
                        help=f"YOLO checkpoint (default: {config.MODEL_PATH}). "
                             f"Available: {', '.join(config.AVAILABLE_MODELS)}")
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=None,
        help="Override the confidence threshold used by the detector.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Override inference resolution (e.g. 1280 for more detail on small/night objects like motorcycles). "
             "Higher = more accurate, slower. Default comes from config.IMGSZ.",
    )
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument(
        "--show-detections",
        action="store_true",
        help="Print detected objects per frame in the terminal for debugging.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='Override which device runs inference: "cuda:0" to force GPU, "cpu" to force CPU. '
             "Default (omit this) auto-picks a GPU if available, else CPU — see config.DEVICE.",
    )
    parser.add_argument("--enumerator", default="", help="Enumerator name, written into the TMC report header")
    parser.add_argument("--junction", default="", help="Junction code (e.g. JC15), written into the TMC report header")
    parser.add_argument("--junction-name", default="", help="Junction name, written into the TMC report header")
    args = parser.parse_args()

    survey_start_time = datetime.strptime(args.start_time, "%Y-%m-%d %H:%M:%S")
    survey_end_time = datetime.strptime(args.end_time, "%Y-%m-%d %H:%M:%S") if args.end_time else None
    run(
        args.video,
        survey_start_time,
        save_debug_video=args.debug_video,
        output_dir=args.output_dir,
        live_preview=args.live_preview,
        model_path=args.model,
        conf=args.conf_threshold,
        imgsz=args.imgsz,
        show_detections=args.show_detections,
        enumerator=args.enumerator,
        junction=args.junction,
        junction_name=args.junction_name,
        device=args.device,
        survey_end_time=survey_end_time,
        fps_override=args.fps,
    )


if __name__ == "__main__":
    main()
