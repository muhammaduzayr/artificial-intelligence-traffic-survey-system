"""
AITSS desktop launcher — pick a video, mark zones, run the full
pipeline, no command line needed.

Usage:
    python run_gui.py

Place this file in the project root (same folder as requirements.txt),
NOT inside the aitss/ package folder — it needs to sit next to the
"aitss" package so `import aitss` resolves correctly.
"""

import os
import sys
import threading
import traceback
import queue
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from datetime import datetime

# Suppress harmless but noisy HEVC/H.265 decoder warnings from OpenCV
# ("PPS id out of range", "No start code is found", etc.) that flood
# the console during video seeking in zone_picker.
import cv2
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

import contextlib


@contextlib.contextmanager
def _suppress_stderr():
    old_stderr = sys.stderr
    devnull = None
    try:
        devnull = open(os.devnull, "w")
        sys.stderr = devnull
        yield
    finally:
        if devnull is not None:
            devnull.close()
        sys.stderr = old_stderr


from PIL import Image, ImageTk

from aitss.main import run as run_pipeline
from aitss.zone_picker import write_lines_to_config, save_zones, load_zones
from aitss.video_io import get_video_info, open_video, release_video
from aitss import config as aitss_config

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aitss", "config.py")


class TextRedirector:
    """Redirects print() output into a queue so it can be safely picked
    up by the Tkinter main thread (Tkinter widgets aren't safe to touch
    directly from a background thread, which is where the pipeline runs
    so the GUI doesn't freeze while processing)."""
    def __init__(self, out_queue):
        self.out_queue = out_queue

    def write(self, text):
        if text.strip():
            self.out_queue.put(text)

    def flush(self):
        pass


class AitssGui:
    def __init__(self, root):
        self.root = root
        root.title("AITSS - AI Traffic Survey")
        root.geometry("760x720")

        self.video_path = tk.StringVar()
        self.start_time = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.end_time = tk.StringVar()
        self.fps_override = tk.StringVar()
        self.enumerator = tk.StringVar()
        self.junction = tk.StringVar()
        self.junction_name = tk.StringVar()
        self.model = tk.StringVar(value=aitss_config.MODEL_PATH)
        self.imgsz = tk.StringVar(value="960")
        self.debug_video = tk.BooleanVar(value=True)
        # Defaults to OFF, not True: the pipeline runs in a background
        # thread (see worker() below) so the GUI itself doesn't freeze
        # while processing, but OpenCV's live-preview window
        # (cv2.imshow/cv2.waitKey) isn't reliably safe to drive from a
        # non-main thread on Windows — it can appear to work for a while
        # and then simply stop repainting/responding, which looks exactly
        # like the whole thing "got stuck." The log box below already
        # shows the same frame-by-frame progress via the redirected
        # print() output, with none of that risk. If you specifically
        # want the visual preview window, run from the command line
        # instead (python -m aitss.main --live-preview), where it's on
        # the actual main thread.
        self.live_preview = tk.BooleanVar(value=False)
        self.show_detections = tk.BooleanVar(value=False)
        self.zone_status = tk.StringVar()

        self.out_queue = queue.Queue()
        self._build_ui()
        self._refresh_zone_status()
        self.root.after(100, self._poll_queue)

    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}

        frm = ttk.Frame(self.root)
        frm.pack(fill="x", **pad)

        ttk.Label(frm, text="1. Video file:").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.video_path, width=55).grid(row=0, column=1, sticky="we")
        ttk.Button(frm, text="Browse...", command=self._browse).grid(row=0, column=2)

        ttk.Label(frm, text="2. Mark zones:").grid(row=1, column=0, sticky="w")
        self.mark_zones_button = ttk.Button(frm, text="Mark Zones on Selected Video", command=self._mark_zones)
        self.mark_zones_button.grid(row=1, column=1, sticky="w")

        ttk.Label(frm, textvariable=self.zone_status, foreground="#0a6").grid(
            row=2, column=0, columnspan=3, sticky="w", padx=8)

        ttk.Separator(frm, orient="horizontal").grid(row=3, column=0, columnspan=3, sticky="we", pady=6)

        ttk.Label(frm, text="Recording start time:").grid(row=4, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.start_time, width=30).grid(row=4, column=1, sticky="w")
        ttk.Label(frm, text="(YYYY-MM-DD HH:MM:SS)").grid(row=4, column=2, sticky="w")

        ttk.Label(frm, text="Recording end time (optional):").grid(row=5, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.end_time, width=30).grid(row=5, column=1, sticky="w")
        ttk.Label(frm, text="if known, derives the TRUE fps").grid(row=5, column=2, sticky="w")

        ttk.Label(frm, text="Manual fps override (optional):").grid(row=6, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.fps_override, width=30).grid(row=6, column=1, sticky="w")
        ttk.Label(frm, text="skips end-time calc if you already know it").grid(row=6, column=2, sticky="w")

        ttk.Label(frm, text="Enumerator:").grid(row=7, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.enumerator, width=30).grid(row=7, column=1, sticky="w")

        ttk.Label(frm, text="Junction code:").grid(row=8, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.junction, width=30).grid(row=8, column=1, sticky="w")

        ttk.Label(frm, text="Junction name:").grid(row=9, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.junction_name, width=30).grid(row=9, column=1, sticky="w")

        ttk.Label(frm, text="Model:").grid(row=10, column=0, sticky="w")
        from aitss.config import AVAILABLE_MODELS
        model_box = ttk.Combobox(frm, textvariable=self.model, width=27,
                                  values=AVAILABLE_MODELS)
        model_box.grid(row=10, column=1, sticky="w")

        ttk.Label(frm, text="Image size:").grid(row=11, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.imgsz, width=10).grid(row=11, column=1, sticky="w")

        opts = ttk.Frame(self.root)
        opts.pack(fill="x", **pad)
        ttk.Checkbutton(opts, text="Save debug video", variable=self.debug_video).pack(side="left")
        ttk.Checkbutton(opts, text="Live preview window", variable=self.live_preview).pack(side="left")
        ttk.Label(opts, text="(can freeze — see tooltip)", foreground="#a60").pack(side="left", padx=(0, 12))
        ttk.Checkbutton(opts, text="Show detections in log", variable=self.show_detections).pack(side="left")

        self.run_button = ttk.Button(self.root, text="3. Start Analysis", command=self._start)
        self.run_button.pack(pady=8)

        self.log_box = tk.Text(self.root, height=20, bg="black", fg="#00FF66")
        self.log_box.pack(fill="both", expand=True, **pad)

        frm.columnconfigure(1, weight=1)

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select traffic survey video",
            filetypes=[
                ("Common video files",
                 "*.mp4 *.mov *.avi *.mkv *.m4v *.ts *.mts *.m2ts *.webm *.wmv *.flv"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.video_path.set(path)

    def _log(self, text):
        self.log_box.insert("end", text + ("\n" if not text.endswith("\n") else ""))
        self.log_box.see("end")

    def _refresh_zone_status(self):
        try:
            import importlib
            importlib.reload(aitss_config)
            lines = aitss_config.FINISH_LINES
            labels = ", ".join(lines.keys())
            self.zone_status.set(f"Finish lines: {labels}" if lines else
                                  "No finish lines defined — click 'Mark Zones' to set START_LINE + FINISH_LINES.")
        except Exception:
            self.zone_status.set("Could not read current lines from config.py.")

    def _mark_zones(self):
        video = self.video_path.get().strip()
        if not video or not os.path.exists(video):
            messagebox.showerror("No video selected", "Select a video file first (step 1) before marking lines.")
            return

        self.mark_zones_button.config(state="disabled", text="Opening video...")
        self.root.update_idletasks()

        # Suppress OpenCV/FFmpeg stderr noise ("no start code is found",
        # "PPS id out of range", etc.) that is harmless but alarming.
        with _suppress_stderr():
            try:
                cap = open_video(video)
            except Exception as exc:
                self.mark_zones_button.config(state="normal", text="Mark Zones on Selected Video")
                messagebox.showerror("Cannot open video", str(exc))
                self._log(f"ERROR: {exc}")
                return
        video_fps, total_frames = get_video_info(cap)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._log(f"Video: {w}x{h}, {video_fps:.2f} fps, {total_frames} frames")

        # Create a Toplevel window for the picker
        picker = tk.Toplevel(self.root)
        picker.title("AITSS Zone Picker")
        picker.transient(self.root)
        picker.grab_set()

        # Scale the display to fit screen
        MAX_W, MAX_H = 1400, 850
        scale = min(MAX_W / w, MAX_H / h, 1.0)
        disp_w, disp_h = int(w * scale), int(h * scale)

        # Canvas for the video frame
        canvas = tk.Canvas(picker, width=disp_w, height=disp_h, bg="black",
                           highlightthickness=0, cursor="crosshair")
        canvas.pack(pady=4)

        # Status bar
        status_var = tk.StringVar()
        status_bar = ttk.Label(picker, textvariable=status_var, relief="sunken",
                               anchor="w", padding=(4, 1))
        status_bar.pack(fill="x", padx=4, pady=(0, 4))

        # ── Sequential frame reader ──────────────────────────────────────
        # Uses only cap.grab() + cap.retrieve() — NEVER cap.set() — because
        # cap.set() blocks indefinitely on HEVC videos without a decoder.
        # grab() is a lightweight packet-skip; only retrieve() decodes.
        # We re-open from scratch for backward seeks.
        class _SeqReader:
            def __init__(self, video_path, cap, total_frames):
                self._video_path = video_path
                self._cap = cap
                self._total = total_frames
                self._pos = 0  # last successfully retrieved frame index

            def _reopen(self):
                self._cap = open_video(self._video_path)
                self._pos = 0

            def close(self):
                release_video(self._cap)

            def goto(self, target):
                """Advance (or rewind+advance) to *target*, return (frame, index)."""
                target = max(0, min(target, self._total - 1 if self._total else 999999))

                # Going backward: re-open and grab forward from 0
                if target < self._pos:
                    with _suppress_stderr():
                        self._reopen()

                # Grab forward from current position
                if target > self._pos:
                    diff = target - self._pos
                    with _suppress_stderr():
                        ok = True
                        for _ in range(diff):
                            ok = self._cap.grab()
                            if not ok:
                                break
                        if ok:
                            ok, frame = self._cap.retrieve()
                            if ok and frame is not None:
                                self._pos = target
                                return frame, target
                            # retrieve failed — try sequential from 0
                            self._reopen()

                # If we're already at the target or backward-seeking failed,
                # grab one frame sequentially from position 0
                if self._pos != target:
                    with _suppress_stderr():
                        self._reopen()
                with _suppress_stderr():
                    ok = True
                    for _ in range(target - self._pos):
                        ok = self._cap.grab()
                        if not ok:
                            break
                    if ok:
                        ok, frame = self._cap.retrieve()
                        if ok and frame is not None:
                            self._pos = target
                            return frame, target
                return None, target

            def current_pos(self):
                return self._pos

        # State
        reader = _SeqReader(video, cap, total_frames)
        state = {
            "reader": reader,
            "frame_idx": 0,
            "frame": None,
            "photo": None,
            "scale": scale,
            "mode": "navigate",
            "points": [],
            "lines": [],
        }

        def to_source(x, y):
            return (int(round(x / scale)), int(round(y / scale)))

        def to_display(px):
            return (int(round(px[0] * scale)), int(round(px[1] * scale)))

        def show_frame():
            """Read and display frame at state['frame_idx'] using sequential access."""
            target = state["frame_idx"]
            # Ensure no None frame displayed — if current frame is None,
            # scan forward until we find one.
            while True:
                frame, idx = reader.goto(target)
                if frame is not None:
                    state["frame_idx"] = idx
                    break
                # Try next frame
                target += 1
                if total_frames and target >= total_frames:
                    status_var.set("ERROR: No readable frame found")
                    return

            state["frame"] = frame

            # Convert to RGB and then to PhotoImage
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if scale < 1.0:
                disp = cv2.resize(rgb, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
            else:
                disp = rgb
            pil_img = Image.fromarray(disp)
            photo = ImageTk.PhotoImage(pil_img)
            state["photo"] = photo
            canvas.delete("all")
            canvas.create_image(0, 0, anchor="nw", image=photo, tags="img")

            # Draw saved lines
            for lbl, pts in state["lines"]:
                dp1, dp2 = to_display(pts[0]), to_display(pts[1])
                canvas.create_line(dp1[0], dp1[1], dp2[0], dp2[1],
                                   fill="red", width=3, tags="line")
                canvas.create_text(dp1[0] + 6, dp1[1] - 6, text=lbl or "?",
                                   fill="red", anchor="w",
                                   font=("TkDefaultFont", 10, "bold"), tags="line")

            # Draw current points
            for p in state["points"]:
                dp = to_display(p)
                canvas.create_oval(dp[0] - 4, dp[1] - 4, dp[0] + 4, dp[1] + 4,
                                   fill="lime", outline="white", tags="pts")
            if len(state["points"]) == 2:
                dp1, dp2 = to_display(state["points"][0]), to_display(state["points"][1])
                canvas.create_line(dp1[0], dp1[1], dp2[0], dp2[1],
                                   fill="lime", width=2, tags="pts")

            # Overlay info
            info = f"frame {reader.current_pos()}"
            if total_frames:
                info += f" / {total_frames}"
            info += f"  |  Mode: {state['mode'].upper()}"
            canvas.create_rectangle(2, 2, 340, 28, fill="black", outline="", tags="info")
            canvas.create_text(8, 15, text=info, fill="lime",
                               anchor="w", font=("TkFixedFont", 10), tags="info")

        def on_key(event):
            """Unified key handler for both modes."""
            key = event.keysym
            if state["mode"] == "navigate":
                old_idx = state["frame_idx"]
                if key in ("Right", "d", "D"):
                    state["frame_idx"] = min(state["frame_idx"] + 5,
                                             total_frames - 1 if total_frames else 999999)
                elif key in ("Left", "a", "A"):
                    state["frame_idx"] = max(0, state["frame_idx"] - 5)
                elif key == "period":
                    state["frame_idx"] = min(state["frame_idx"] + 1,
                                             total_frames - 1 if total_frames else 999999)
                elif key == "comma":
                    state["frame_idx"] = max(0, state["frame_idx"] - 1)
                elif key in ("Return", "space"):
                    state["mode"] = "draw"
                    status_var.set("Step 2: left-click 2 points, right-click to save, Esc to quit")
                    show_frame()
                    return
                elif key in ("Escape", "q"):
                    reader.close()
                    picker.destroy()
                    return
                if state["frame_idx"] != old_idx:
                    show_frame()
            elif state["mode"] == "draw":
                if key in ("Escape", "q"):
                    reader.close()
                    picker.destroy()

        def on_canvas_click(event):
            """Handle mouse clicks in draw mode."""
            if state["mode"] != "draw":
                return
            # Return keyboard focus to the picker window
            picker.focus_set()
            src_pt = to_source(event.x, event.y)
            if event.num == 1:  # left click
                state["points"].append(src_pt)
                show_frame()
                if len(state["points"]) == 2:
                    status_var.set(f"Line from {state['points'][0]} to {state['points'][1]}. "
                                   "Right-click to save, or left-click to redo.")
            elif event.num == 3:  # right click
                if len(state["points"]) == 2:
                    state["lines"].append((None, list(state["points"])))
                    total = len(state["lines"])
                    status_var.set(f"Line saved ({total} total)")
                    self._log(f"Line: {state['points'][0]} -> {state['points'][1]} (unlabeled)")
                    state["points"] = []
                    show_frame()
                else:
                    status_var.set("Need exactly 2 points first (left-click)")

        # Bind keyboard events to both picker and canvas
        picker.bind("<Key>", on_key)
        canvas.bind("<Key>", on_key)
        canvas.bind("<Button-1>", on_canvas_click)
        canvas.bind("<Button-3>", on_canvas_click)
        canvas.bind("<FocusIn>", lambda e: picker.focus_set())

        picker.focus_set()

        # Show initial frame (frame 0 — already validated by open_video).
        state["frame_idx"] = 0
        show_frame()
        status_var.set("Step 1: navigate (Right/Left = ±5, ,/. = ±1, Enter = lock, Esc = quit)")
        self._log("Zone picker window opened. Navigate to the frame, then draw lines.")

        # Wait for the picker to close
        self.root.wait_window(picker)

        # Cleanup
        reader.close()
        self.mark_zones_button.config(state="normal", text="Mark Zones on Selected Video")

        # Convert lines to the format expected by _handle_picker_result
        # (list of (None, pts) tuples under the "unlabeled" key triggers
        # the bulk label dialog).
        lines_result = {
            "unlabeled": [(None, pts) for _, pts in state["lines"]],
        }
        self._handle_picker_result(lines_result, video)

    def _poll_picker_result(self, result_queue):
        try:
            while True:
                msg_type, payload = result_queue.get_nowait()
                if msg_type == "status":
                    self._log(payload)
                elif msg_type == "result":
                    self._handle_picker_result(payload)
                    return
        except queue.Empty:
            pass
        self.root.after(50, lambda: self._poll_picker_result(result_queue))

    def _handle_picker_result(self, lines, video_path=""):
        self.mark_zones_button.config(state="normal", text="Mark Zones on Selected Video")

        if not lines or (
            not lines.get("start_lines")
            and not lines.get("finish_lines")
            and not lines.get("unlabeled")
        ):
            self._log("No lines were drawn — config.py was not changed.")
            return

        unlabeled = lines.pop("unlabeled", [])
        if unlabeled:
            self._log(f"Prompting for {len(unlabeled)} line labels...")
            labels = self._bulk_label_dialog(unlabeled)
            if labels is None:
                self._log("Labeling cancelled — no lines saved.")
                return
            labeled_start = []
            labeled_finish = []
            for i, (label_str, (_, pts)) in enumerate(zip(labels, unlabeled)):
                label_str = label_str.strip().upper()
                if not label_str:
                    label_str = f"Line{i+1}"
                if len(label_str) == 1 and label_str in "NSEW":
                    labeled_start.append((label_str, pts))
                    self._log(f"  START line '{label_str}': {pts[0]} -> {pts[1]}")
                else:
                    labeled_finish.append((label_str, pts))
                    self._log(f"  FINISH line '{label_str}': {pts[0]} -> {pts[1]}")
            lines.setdefault("start_lines", []).extend(labeled_start)
            lines.setdefault("finish_lines", []).extend(labeled_finish)

        if not lines.get("start_lines") and not lines.get("finish_lines"):
            self._log("No lines were drawn — config.py was not changed.")
            return

        # Build dicts in the same format as config.py START_LINES / FINISH_LINES
        start_dict = {lbl: pts for lbl, pts in lines.get("start_lines", [])}
        finish_dict = {lbl: pts for lbl, pts in lines.get("finish_lines", [])}

        # Save per-video zone file alongside the video
        if video_path:
            zone_path = save_zones(video_path, start_dict, finish_dict)
            self._log(f"Saved zones to: {zone_path}")
        else:
            self._log("No video path — zones are only in memory, not saved.")

        # Also update config.py for backward compatibility with CLI usage
        ok = write_lines_to_config(lines, CONFIG_PATH)
        if ok:
            start_labels = list(start_dict.keys())
            finish_labels = list(finish_dict.keys())
            self._log(f"Saved start lines: {start_labels}, finish lines: {finish_labels}")
        else:
            self._log(f"Could not write to {CONFIG_PATH} — check the path/permissions.")

        self._refresh_zone_status()

    def _bulk_label_dialog(self, unlabeled_lines):
        dialog = tk.Toplevel(self.root)
        dialog.title("Label your lines")
        dialog.transient(self.root)
        dialog.grab_set()

        frame = ttk.Frame(dialog, padding=12)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Enter a label for each line:\n"
                  "S / N / E / W  for start lines\n"
                  "N1 / S2 / E4 / W4  for finish lines",
                  justify="left").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        entries = []
        for i, (_, pts) in enumerate(unlabeled_lines):
            ttk.Label(frame, text=f"Line {i+1} ({pts[0]},{pts[1]}):").grid(
                row=i+1, column=0, sticky="w", padx=(0, 8), pady=2)
            var = tk.StringVar()
            entry = ttk.Entry(frame, textvariable=var, width=12)
            entry.grid(row=i+1, column=1, sticky="w", pady=2)
            entries.append(var)

        result = [None]

        def on_save():
            labels = [v.get() for v in entries]
            for lbl in labels:
                raw = lbl.strip().upper()
                if raw and not (
                    (len(raw) == 1 and raw in "NSEW") or
                    (len(raw) == 2 and raw[0] in "NSEW" and raw[1] in "1234")
                ):
                    messagebox.showwarning("Invalid label",
                        f"'{lbl}' is not valid. Use S/N/E/W for start lines, "
                        f"N1/S2/E4/W4 for finish lines.", parent=dialog)
                    return
            result[0] = [v.get() for v in entries]
            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=len(unlabeled_lines)+1, column=0, columnspan=2, pady=(12, 0))
        ttk.Button(btn_frame, text="Save", command=on_save).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side="left", padx=4)

        dialog.update_idletasks()
        w = max(480, dialog.winfo_reqwidth())
        h = dialog.winfo_reqheight()
        dialog.minsize(w, h)

        dialog.wait_window()
        return result[0]

    def _poll_queue(self):
        try:
            while True:
                text = self.out_queue.get_nowait()
                self._log(text)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _start(self):
        video = self.video_path.get().strip()
        if not video:
            messagebox.showerror("Missing video", "Please select a video file first.")
            return
        if not os.path.exists(video):
            messagebox.showerror("File not found", f"Can't find:\n{video}")
            return

        self._refresh_zone_status()
        if not aitss_config.FINISH_LINES:
            proceed = messagebox.askyesno(
                "No finish lines defined",
                "No FINISH_LINES are defined in config.py, so results won't be split by "
                "lane/direction. Mark finish lines first?\n\nClick 'No' to run anyway (not recommended).",
            )
            if proceed:
                return

        try:
            survey_start_time = datetime.strptime(self.start_time.get().strip(), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            messagebox.showerror("Bad start time", "Use the format YYYY-MM-DD HH:MM:SS")
            return

        survey_end_time = None
        end_time_str = self.end_time.get().strip()
        if end_time_str:
            try:
                survey_end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                messagebox.showerror("Bad end time", "Use the format YYYY-MM-DD HH:MM:SS")
                return
            if survey_end_time <= survey_start_time:
                messagebox.showerror("Bad end time", "End time must be AFTER start time.")
                return

        fps_override_val = None
        fps_str = self.fps_override.get().strip()
        if fps_str:
            try:
                fps_override_val = float(fps_str)
            except ValueError:
                messagebox.showerror("Bad fps override", "Enter a plain number, e.g. 40.13")
                return

        if self.live_preview.get():
            proceed = messagebox.askyesno(
                "Live preview risk",
                "Live preview runs inside a background thread here, which can cause the "
                "preview window to freeze/stop responding partway through on some systems "
                "(this is an OpenCV/Windows limitation, not a bug in your data or settings). "
                "The log below already shows the same frame-by-frame progress safely.\n\n"
                "Continue with live preview anyway?",
            )
            if not proceed:
                return

        self.run_button.config(state="disabled", text="Running...")
        self.log_box.delete("1.0", "end")

        imgsz_val = None
        if self.imgsz.get().strip():
            try:
                imgsz_val = int(self.imgsz.get().strip())
            except ValueError:
                imgsz_val = None

        def worker():
            old_stdout = sys.stdout
            sys.stdout = TextRedirector(self.out_queue)
            try:
                run_pipeline(
                    video,
                    survey_start_time,
                    save_debug_video=self.debug_video.get(),
                    live_preview=self.live_preview.get(),
                    model_path=self.model.get().strip() or None,
                    imgsz=imgsz_val,
                    show_detections=self.show_detections.get(),
                    enumerator=self.enumerator.get(),
                    junction=self.junction.get(),
                    junction_name=self.junction_name.get(),
                    survey_end_time=survey_end_time,
                    fps_override=fps_override_val,
                )
            except Exception as e:
                import traceback
                self.out_queue.put(f"\n[ERROR] {traceback.format_exc()}\n")
            finally:
                sys.stdout = old_stdout
                self.out_queue.put("\n--- Done. You can close this window or run another video. ---\n")
                self.root.after(0, lambda: self.run_button.config(state="normal", text="3. Start Analysis"))

        threading.Thread(target=worker, daemon=True).start()


def main():
    root = tk.Tk()
    AitssGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
