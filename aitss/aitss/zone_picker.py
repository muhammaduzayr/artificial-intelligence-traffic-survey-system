"""
Shared line-picking core: navigate to a frame, click 2 points to define
a line, get a label for it. Used by both tools/pick_zone.py (terminal)
and run_gui.py (dialog), so the picking logic lives in one place.

Labels with a single letter (S, N, E, W) go into START_LINES.
Labels with a letter+digit (N1, S2, E4, W4) go into FINISH_LINES.
"""

import os
import re
import sys
import numpy as np
import cv2
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

from .video_io import get_video_info, open_video, release_video

WINDOW_NAME = "AITSS Line Picker"
MAX_DISPLAY_WIDTH = 1600
MAX_DISPLAY_HEIGHT = 900


def _zones_path(video_path):
    """Derive a sidecar .zones.json path from a video path."""
    base, _ = os.path.splitext(video_path)
    return base + ".zones.json"


def load_zones(video_path):
    """Load per-video zone lines from a sidecar JSON file.

    Returns (start_lines, finish_lines) where each is a dict
    {label: [(x1,y1), (x2,y2)], ...}, or (None, None) if no file exists.
    """
    path = _zones_path(video_path)
    if not os.path.exists(path):
        return None, None
    import json
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    start = {}
    for k, v in data.get("start_lines", {}).items():
        start[k] = [tuple(p) for p in v]
    finish = {}
    for k, v in data.get("finish_lines", {}).items():
        finish[k] = [tuple(p) for p in v]
    return start, finish


def save_zones(video_path, start_lines, finish_lines):
    """Save per-video zone lines to a sidecar JSON file.

    *start_lines* and *finish_lines* are dicts
    {label: [(x1,y1), (x2,y2)], ...}.

    Preserves an existing "known_occluders" entry in the sidecar file (set
    via save_occluders) rather than clobbering it — the line-picker and
    occluder-picker write to the same file independently.
    """
    import json
    path = _zones_path(video_path)
    existing_occluders = None
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing_occluders = json.load(f).get("known_occluders")
        except (OSError, json.JSONDecodeError):
            existing_occluders = None
    data = {
        "start_lines": {k: [list(p) for p in v] for k, v in start_lines.items()},
        "finish_lines": {k: [list(p) for p in v] for k, v in finish_lines.items()},
    }
    if existing_occluders is not None:
        data["known_occluders"] = existing_occluders
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


def load_occluders(video_path):
    """Load per-video known-occluder boxes from the sidecar JSON file.

    Returns a list of (x1, y1, x2, y2) tuples, or None if the sidecar file
    doesn't exist or has no "known_occluders" entry — callers should fall
    back to config.KNOWN_OCCLUDERS (the single-camera default) in that case.

    Kept in the same sidecar file as start/finish lines (one file per
    camera/video) rather than in config.py, because a fixed obstruction
    like a pole or tree is specific to one camera's framing — a global
    list in config.py can't correctly serve more than one camera at once.
    """
    path = _zones_path(video_path)
    if not os.path.exists(path):
        return None
    import json
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    occluders = data.get("known_occluders")
    if occluders is None:
        return None
    return [tuple(box) for box in occluders]


def save_occluders(video_path, occluders):
    """Save per-video known-occluder boxes to the sidecar JSON file.

    *occluders* is a list of (x1, y1, x2, y2) tuples. Preserves any
    existing start_lines/finish_lines already in the sidecar file.
    """
    import json
    path = _zones_path(video_path)
    data = {"start_lines": {}, "finish_lines": {}}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    data["known_occluders"] = [list(box) for box in occluders]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


def is_frame_blank(frame, threshold=15):
    return np.std(frame) <= threshold


def read_frame_at(cap, frame_idx, total_frames, current_fps=25.0, last_idx=None):
    if total_frames:
        frame_idx = max(0, min(frame_idx, total_frames - 1))
    else:
        frame_idx = max(0, frame_idx)

    # Fast path: stepping forward a short distance from where we already
    # are (covers both the ',''.' +-1 nudge and the a/d +-30 jump). A
    # cap.set() seek forces the decoder back to the nearest keyframe and
    # re-decodes everything forward from there -- on long-GOP HEVC/H.265
    # footage that can mean decoding dozens/hundreds of frames just to
    # show the next one, which is what made every keypress feel slow.
    # Sequential cap.grab() calls skip the decode step for frames we
    # don't need, so they're much cheaper than reseeking.
    if last_idx is not None and 0 <= frame_idx - last_idx <= 30:
        ok = True
        for _ in range(frame_idx - last_idx):
            ok = cap.grab()
            if not ok:
                break
        if ok:
            ok, frame = cap.retrieve()
            if ok and frame is not None:
                return frame_idx, frame
        # Sequential path failed (e.g. hit EOF) -- fall through to a seek.

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        return frame_idx, None
    return frame_idx, frame


def setup_window(frame_shape):
    h, w = frame_shape[:2]
    scale = min(MAX_DISPLAY_WIDTH / w, MAX_DISPLAY_HEIGHT / h, 1.0)
    display_w, display_h = int(w * scale), int(h * scale)
    # Use WINDOW_NORMAL + WINDOW_GUI_NORMAL for better Windows compatibility.
    # WINDOW_GUI_NORMAL enables the native Windows title bar and frame,
    # which avoids the "no system menu" issue that can make the window
    # appear only as a hidden taskbar entry on some setups.
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, display_w, display_h)
    cv2.waitKey(1)  # flush window events
    # Verify the window actually exists
    prop = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)
    return scale, prop


def terminal_label_prompt(line_number, hint="label"):
    while True:
        raw = input(
            f"  Label for {hint} {line_number} "
            f"(e.g. S, N, E, W for start lines, or N1, S2 for finish lines, "
            f"Enter for default): "
        ).strip().upper()
        if raw == "":
            return f"Line{line_number}"
        if len(raw) == 1 and raw in "NSEW":
            return raw
        if len(raw) == 2 and raw[0] in "NSEW" and raw[1] in "1234":
            return raw
        print("  Use single letter (S/N/E/W) for start lines, or letter+digit (N1/S2) for finish lines.")


def pick_lines_interactive(video_path, start_frame=50,
                            label_prompt_fn=terminal_label_prompt,
                            status_fn=print,
                            collect_labels_after=False,
                            on_window_open=None):
    """
    Opens an interactive window: navigate to a frame (a/d = +-30 frames,
    ',' '.' = +-1 frame, ENTER = lock it in), then click 2 points to draw
    a line (left-click = add point, right-click = close line and prompt
    for a label, 'q' = finish).

    When collect_labels_after=True, label_prompt_fn is NOT called during
    drawing — lines are stored with None labels. The caller should prompt
    for labels after the window closes (useful when the OpenCV window
    and tkinter run on different threads).

    *on_window_open* is an optional zero-argument callable invoked right
    after the OpenCV window is successfully created.  run_gui.py uses it
    to hide the Tk root (self.root.withdraw) only once the OpenCV window
    is confirmed visible.

    Returns a dict with two lists: {"start_lines": [(label, [p1, p2]), ...],
                                    "finish_lines": [(label, [p1, p2]), ...]}
    Or an empty dict if the user quit before locking a frame.
    """
    print(f"[AITSS] pick_lines_interactive: opening {video_path}", file=sys.stderr)

    lines = {"start_lines": [], "finish_lines": []}
    current_points = []
    notification = {"text": "", "frames_left": 0}

    display_scale = 1.0

    def to_source_point(x, y):
        return (int(round(x / display_scale)), int(round(y / display_scale)))

    def to_display_point(point):
        return (int(round(point[0] * display_scale)), int(round(point[1] * display_scale)))

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append(to_source_point(x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(current_points) == 2:
                if collect_labels_after:
                    lines["unlabeled"].append((None, list(current_points)))
                    total = len(lines["unlabeled"])
                    notification["text"] = f"Line saved ({total} total) - label later"
                    notification["frames_left"] = 30
                    status_fn(f"Line: {current_points[0]} -> {current_points[1]} (unlabeled)")
                else:
                    label = label_prompt_fn(
                        len(lines["start_lines"]) + len(lines["finish_lines"]) + 1
                    )
                    if len(label) == 1 and label in "NSEW":
                        lines["start_lines"].append((label, list(current_points)))
                        status_fn(f"START line '{label}': {current_points[0]} -> {current_points[1]}")
                    else:
                        lines["finish_lines"].append((label, list(current_points)))
                        status_fn(f"FINISH line '{label}': {current_points[0]} -> {current_points[1]}")
                current_points.clear()
            else:
                status_fn("Need exactly 2 points before closing a line.")

    try:
        cap = open_video(video_path)
        print(f"[AITSS] open_video succeeded: cap={cap}", file=sys.stderr)
    except Exception as exc:
        print(f"[AITSS] open_video FAILED: {exc}", file=sys.stderr)
        status_fn(f"ERROR opening video: {exc}")
        return {}
    video_fps, total_frames = get_video_info(cap)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[AITSS] Video info: {w}x{h}, {video_fps} fps, {total_frames} frames", file=sys.stderr)
    status_fn(f"Video: {w}x{h}, {video_fps:.2f} fps, {total_frames} frames")

    frame_idx = start_frame
    last_idx = None
    frame_idx, frame = read_frame_at(cap, frame_idx, total_frames, video_fps, last_idx)
    print(f"[AITSS] Initial frame read: idx={frame_idx}, frame={type(frame).__name__ if frame is not None else 'None'}", file=sys.stderr)
    last_idx = frame_idx if frame is not None else None
    # Light blank-skip: only skip frames that are extremely dark/black
    # (stddev < 5).  This avoids discarding valid low-contrast frames
    # from outdoor/night recordings while still skipping the pure-black
    # first frames some NVRs write before the camera activates.
    attempts = 0
    while frame is not None and is_frame_blank(frame, threshold=5) and attempts < 300:
        frame_idx += 1
        frame_idx, frame = read_frame_at(cap, frame_idx, total_frames, video_fps, last_idx)
        last_idx = frame_idx if frame is not None else None
        attempts += 1
    if attempts > 0:
        print(f"[AITSS] Skipped {attempts} blank frames, final idx={frame_idx}, frame={type(frame).__name__ if frame is not None else 'None'}", file=sys.stderr)

    if frame is None:
        release_video(cap)
        print(f"[AITSS] No valid frame found in video", file=sys.stderr)
        status_fn("ERROR: Could not read any frame from the video. The file may be unsupported or corrupt.")
        return {}

    if attempts > 0:
        status_fn(f"Skipped {attempts} dark/blank frames, showing frame {frame_idx}.")

    print(f"[AITSS] Creating window with frame shape {frame.shape}", file=sys.stderr)
    display_scale, win_prop = setup_window(frame.shape)
    print(f"[AITSS] Window created, scale={display_scale:.3f}, WND_PROP_VISIBLE={win_prop}", file=sys.stderr)
    if win_prop < 1:
        print(f"[AITSS] WARNING: window visibility prop={win_prop} — window may not be visible", file=sys.stderr)
        status_fn("WARNING: OpenCV window may not be visible on this system.")
    status_fn(f"Window created at {int(frame.shape[1] * display_scale)}x{int(frame.shape[0] * display_scale)}")
    status_fn("Step 1: navigate (a/d = +-30, ',' '.' = +-1, ENTER = lock, q = quit).")

    # --- Step 1: navigation mode ---
    mode = "navigate"
    while mode == "navigate":
        display = cv2.resize(frame, None, fx=display_scale, fy=display_scale,
                             interpolation=cv2.INTER_AREA) if display_scale != 1 else frame.copy()
        label = f"frame {frame_idx}" + (f" / {total_frames}" if total_frames else "")
        cv2.rectangle(display, (0, 0), (480, 40), (0, 0, 0), -1)
        cv2.putText(display, label + "  |  ENTER=lock  q=quit", (10, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 32):
            mode = "draw"
        elif key == ord("q"):
            cv2.destroyAllWindows()
            release_video(cap)
            status_fn("Quit without locking a frame.")
            return {}
        elif key == ord("d") or key == 83:
            frame_idx += 30
            frame_idx, frame = read_frame_at(cap, frame_idx, total_frames, video_fps, last_idx)
        elif key == ord("a") or key == 81:
            frame_idx -= 30
            frame_idx, frame = read_frame_at(cap, frame_idx, total_frames, video_fps, last_idx)
        elif key == ord("."):
            frame_idx += 1
            frame_idx, frame = read_frame_at(cap, frame_idx, total_frames, video_fps, last_idx)
        elif key == ord(","):
            frame_idx -= 1
            frame_idx, frame = read_frame_at(cap, frame_idx, total_frames, video_fps, last_idx)

        if frame is None:
            frame_idx, frame = read_frame_at(
                cap, max(0, frame_idx - 1), total_frames, video_fps, last_idx
            )

        if frame is not None:
            last_idx = frame_idx

    release_video(cap)
    status_fn(f"Locked frame {frame_idx}.")
    status_fn("Step 2: left-click 2 points per line, right-click to close a line, 'q' to finish.")

    # In deferred mode, unlabeled lines go to a separate list.
    if collect_labels_after:
        lines["unlabeled"] = []

    # --- Step 2: drawing mode ---
    def all_drawn():
        return lines["start_lines"] + lines["finish_lines"] + (
            lines.get("unlabeled", [])
        )

    def refresh_display(display):
        # Draw all saved lines
        for lbl, pts in all_drawn():
            dp1 = to_display_point(pts[0])
            dp2 = to_display_point(pts[1])
            cv2.line(display, dp1, dp2, (0, 0, 255), 3)
            label_text = lbl if lbl else "?"
            cv2.putText(display, label_text, (dp1[0] + 4, dp1[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # Draw current points
        for p in current_points:
            cv2.circle(display, to_display_point(p), 4, (0, 255, 0), -1)
        if len(current_points) == 2:
            dp1 = to_display_point(current_points[0])
            dp2 = to_display_point(current_points[1])
            cv2.line(display, dp1, dp2, (0, 255, 0), 2)
        cv2.rectangle(display, (0, 0), (480, 40), (0, 0, 0), -1)
        cv2.putText(display, "left-click=point  right-click=close line  q=finish", (10, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # Flash notification (fades after ~30 frames)
        if notification["frames_left"] > 0:
            ntext = notification["text"]
            (nw, nh), _ = cv2.getTextSize(ntext, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            nx = (display.shape[1] - nw) // 2
            ny = display.shape[0] - 30
            cv2.rectangle(display, (nx - 8, ny - nh - 8), (nx + nw + 8, ny + 8), (0, 0, 0), -1)
            cv2.putText(display, ntext, (nx, ny), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            notification["frames_left"] -= 1

    cv2.setMouseCallback(WINDOW_NAME, on_click)

    while True:
        display = cv2.resize(frame, None, fx=display_scale, fy=display_scale,
                             interpolation=cv2.INTER_AREA) if display_scale != 1 else frame.copy()
        refresh_display(display)

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    return lines


_VALID_LINE_LABEL = re.compile(r"^[NSEW][1-4]?$")


def write_lines_to_config(lines, config_path):
    """
    Writes START_LINES and FINISH_LINES dicts into config.py by replacing
    the existing dict definitions in place.

    lines: dict with keys "start_lines" and "finish_lines", each a list
           of (label, [p1, p2]) tuples as returned by pick_lines_interactive().

    Returns True on success, False if the config file doesn't exist.

    Raises ValueError if any label isn't a plain N/S/E/W (+ optional
    1-4) code. Labels are spliced directly into config.py's source below,
    which is then imported and executed on every run — callers already
    enforce this same pattern when prompting the user (see
    terminal_label_prompt / run_gui.py's _bulk_label_dialog), but this
    function is where the write itself happens, so it re-checks rather
    than trusting every caller to have validated first.
    """
    if not os.path.exists(config_path):
        return False

    with open(config_path, "r", encoding="utf-8") as f:
        src = f.read()

    def _replace_dict(src, name, items):
        pattern = re.compile(
            rf'^({name}\s*=\s*)'  # capture "START_LINES = "
            r'\{'                 # opening brace
            r'(?:[^{}]|\{[^{}]*\})*'  # body (allows one level of nesting)
            r'\}',                # closing brace
            re.MULTILINE
        )
        body_items = []
        if items:
            for label, pts in items:
                if not _VALID_LINE_LABEL.match(label):
                    raise ValueError(
                        f"Invalid line label {label!r} — must be a single "
                        f"letter (N/S/E/W) or letter+digit (N1-N4/S1-S4/"
                        f"E1-E4/W1-W4)."
                    )
                x1, y1 = (int(v) for v in pts[0])
                x2, y2 = (int(v) for v in pts[1])
                body_items.append(f'    "{label}": [({x1}, {y1}), ({x2}, {y2})],')
        body = "\n".join(body_items) if body_items else ""
        replacement = f'{name} = {{\n{body}\n}}' if body else f'{name} = {{}}'
        return pattern.sub(replacement, src, count=1)

    src = _replace_dict(src, "START_LINES", lines.get("start_lines", []))
    src = _replace_dict(src, "FINISH_LINES", lines.get("finish_lines", []))

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(src)
    return True
