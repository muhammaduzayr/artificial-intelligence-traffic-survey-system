"""
Thin wrapper around Ultralytics YOLO for detection + ByteTrack tracking.

Reads frames manually via cv2.VideoCapture (rather than handing the video
path straight to model.track()) so an optional contrast-enhancement step
can run on each frame BEFORE detection — this is the standard
frame-by-frame streaming pattern Ultralytics itself documents for
webcam/live sources, and behaves identically for tracking continuity
(persist=True) as the whole-video path did.

NOTE: The PyTorch .pt model is always used for inference (not OpenVINO),
because Ultralytics' model.track() with an OpenVINO backend has a known
bug where numpy array inputs trigger an internal C++ image-loading
fallback that fails (see the OpenVINO DLL error).  When a CUDA GPU is
available the pipeline uses it automatically; on CPU the yolo12n model
at imgsz=960 provides usable throughput.
"""

import math
import os
from collections import defaultdict, deque
from itertools import count as count_iter

import cv2
import numpy as np
from ultralytics import YOLO

from . import config
from .video_io import open_video, release_video

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from scipy.optimize import linear_sum_assignment
    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - scipy ships with ultralytics' own
    # dependencies, but fall back to the old greedy behavior rather than
    # crash if it's somehow missing.
    _SCIPY_AVAILABLE = False


_clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT, tileGridSize=config.CLAHE_TILE_GRID_SIZE)

_gamma_lut = None
_daytime_lut = None
_l_mean_ema = None
_l_mean_alpha = 0.15


def _get_gamma_lut():
    global _gamma_lut
    if _gamma_lut is None:
        # Gamma < 1.0 brightens shadows: out = (in/255)^gamma * 255
        g = config.GAMMA_VALUE
        _gamma_lut = (np.array([(i / 255.0) ** g for i in range(256)], dtype=np.float32) * 255).astype(np.uint8)
    return _gamma_lut


def _get_daytime_lut():
    g = 0.90  # mild brightening for daytime — lifts midtones without washing highlights
    return (np.array([(i / 255.0) ** g for i in range(256)], dtype=np.float32) * 255).astype(np.uint8)

_daytime_lut = None

def _enhance_frame(frame):
    """Contrast enhancement + sharpening for CCTV footage.

    Pipeline: gamma correction → CLAHE → unsharp-mask sharpening.

    Gamma correction uses a temporally-smoothed luminance estimate (EMA)
    to avoid frame-to-frame oscillation at dusk/dawn.  Sharpening recovers
    edges degraded by motion blur and compression, improving detection
    confidence for small/distant vehicles.
    """
    global _l_mean_ema
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    global _daytime_lut
    l_mean = l_channel.mean()
    if _l_mean_ema is None:
        _l_mean_ema = l_mean
    else:
        _l_mean_ema = _l_mean_alpha * l_mean + (1 - _l_mean_alpha) * _l_mean_ema
    if _l_mean_ema < config.GAMMA_CORRECT_THRESHOLD:
        l_channel = cv2.LUT(l_channel, _get_gamma_lut())
    else:
        if _daytime_lut is None:
            _daytime_lut = _get_daytime_lut()
        l_channel = cv2.LUT(l_channel, _daytime_lut)

    l_channel = _clahe.apply(l_channel)
    lab = cv2.merge((l_channel, a_channel, b_channel))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Unsharp-mask sharpening — brings out edges blurred by motion/compression.
    if config.ENABLE_SHARPEN:
        blurred = cv2.GaussianBlur(enhanced, (0, 0), config.SHARPEN_SIGMA)
        enhanced = cv2.addWeighted(
            enhanced, 1.0 + config.SHARPEN_STRENGTH,
            blurred, -config.SHARPEN_STRENGTH, 0,
        )
    return enhanced


def _resolve_device(requested_device):
    if requested_device and requested_device != "auto":
        return requested_device
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _in_ignore_zone(cx, cy):
    """Return whether a detection centroid is in a permanent static zone."""
    return any(x1 <= cx <= x2 and y1 <= cy <= y2
               for x1, y1, x2, y2 in config.IGNORE_ZONES)


class _PerTrackKF:
    """Constant-velocity Kalman filter for centroid smoothing.

    State: [cx, cy, vx, vy]
    Measurement: [cx, cy] (bottom-center of detection bbox)

    Predicts the position forward by dt frames (dt can be >1 for
    FRAME_SKIP gaps).  Updates incorporate a bottom-center measurement
    with confidence-weighted noise; low-confidence or abnormally-sized
    detections can be gated out (call predict-only, skip update).
    """
    def __init__(self, cx, cy, dt=1.0, measurement_noise=25.0):
        self.x = np.array([cx, cy, 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 20.0
        # Process noise: higher values = filter trusts measurements more,
        # responds faster to turns.  Q[0:2]=position, Q[2:4]=velocity.
        # Increased from 4/2 to 10/5 so the KF catches turning vehicles
        # rather than coasting straight through the turn.
        self.Q = np.diag([10.0, 10.0, 5.0, 5.0]).astype(np.float64)
        # Base measurement noise, overridable per-model via the tuning
        # profile (lower = trust each detection more; higher = smoother,
        # slower to react).
        self.R_base = np.diag([measurement_noise, measurement_noise]).astype(np.float64)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        self.I = np.eye(4, dtype=np.float64)
        self.last_dt = dt

    def predict(self, dt=None):
        if dt is None:
            dt = self.last_dt
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q * dt
        self.last_dt = dt
        return (float(self.x[0]), float(self.x[1]))

    def peek(self, dt=None):
        """Return predicted position WITHOUT mutating state.
        
        Used by _match_detections and _predict_detections to get a
        matching/display position without advancing the filter — only
        _kalman_step should ever call predict() and update().
        """
        if dt is None:
            dt = self.last_dt
        return (float(self.x[0] + self.x[2] * dt),
                float(self.x[1] + self.x[3] * dt))

    def update(self, cx, cy, conf=1.0):
        R = self.R_base / max(conf, 0.1)
        z = np.array([cx, cy], dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P
        return (float(self.x[0]), float(self.x[1]))

    def update_clamped(self, cx, cy, conf, max_jump):
        """Update the KF with a position-clamped measurement.

        If the measurement is farther than *max_jump* from the predicted
        position, pull the filter only partway — enough to keep the KF's
        anchor from drifting away from the real vehicle during a run of
        suspicious frames, but not enough to let a genuine ID-swap
        teleport it across the frame.
        """
        pred_x, pred_y = self.x[0], self.x[1]
        dx, dy = cx - pred_x, cy - pred_y
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > max_jump and dist > 0:
            scale = max_jump / dist
            cx, cy = pred_x + dx * scale, pred_y + dy * scale
        return self.update(cx, cy, conf)


class VehicleDetector:
    def __init__(self, model_path=None, conf=None, classes=None, imgsz=None, device=None):
        print(f"Optimal (Hungarian) track assignment: "
              f"{'ON' if _SCIPY_AVAILABLE else 'OFF (scipy missing — using greedy fallback, see requirements.txt)'}")

        self.device = _resolve_device(device or config.DEVICE)

        # Always load the native PyTorch model for tracking — OpenVINO's
        # model.track() has a known bug where numpy array inputs trigger
        # an internal image loading fallback that fails with "Cannot read
        # 'image.png' (this model does not support image input)".  The
        # PyTorch model handles numpy array inputs correctly in track().
        pt_path = model_path or config.MODEL_PATH
        self.tuning = config.get_tuning_profile(pt_path)
        print(f"Using tuning profile for {os.path.basename(str(pt_path))}: {self.tuning}")
        self.model = YOLO(pt_path, task="detect")
        self.conf = conf or config.CONFIDENCE_THRESHOLD
        self.classes = classes or config.VEHICLE_CLASS_IDS

        # Sanity check: the global conf floor passed to YOLO must sit at
        # or below every per-class floor in MIN_CONF_BY_CLASS, or a
        # category's "more permissive" exception is silently unreachable
        # — this exact mismatch (global 0.20 vs Motorcycle's 0.15) was a
        # real bug that discarded low-confidence motorcycles before our
        # own per-class logic ever got to evaluate them. Loud warning
        # instead of a silent miss if a future tuning pass reintroduces it.
        lowest_class_floor = min(config.MIN_CONF_BY_CLASS.values())
        if self.conf > lowest_class_floor:
            print(f"WARNING: CONFIDENCE_THRESHOLD ({self.conf}) is HIGHER than the lowest "
                  f"MIN_CONF_BY_CLASS value ({lowest_class_floor}). Detections in that gap will "
                  f"be discarded by YOLO before per-class filtering ever runs, silently making "
                  f"that category's lower floor meaningless. Lower CONFIDENCE_THRESHOLD in "
                  f"config.py to {lowest_class_floor} or below.")
        self.imgsz = imgsz or config.IMGSZ

        self.half = self.device.startswith("cuda") and config.USE_HALF_PRECISION

        if self.device == "cpu":
            print(f"Running on CPU — {self.model.ckpt_path or 'yolo12n'} at {self.imgsz}px. "
                  f"Install GPU PyTorch for 10-20x speedup (see README).")
        else:
            gpu_name = torch.cuda.get_device_name(0) if _TORCH_AVAILABLE else self.device
            print(f"Using GPU: {gpu_name} (half precision: {self.half})")
            if self.half:
                try:
                    self.model.model.half()
                except Exception as e:
                    print(f"Could not enable half precision ({e}) — continuing at full precision.")
                    self.half = False

        # History stores (category, conf) tuples for category smoothing.
        self._category_history = defaultdict(lambda: deque(maxlen=15))

        # Lightweight tracker state for FRAME_SKIP > 1 mode.
        self._next_track_id = count_iter(1)
        self._active_tracks = {}       # track_id -> dict of last state
        self._track_age = defaultdict(int)
        self._locked_category = {}     # track_id -> locked category (permanent once set)

        # Track lifecycle tracking (for ID-reuse / double-count diagnostics).
        self._track_created_at = {}
        self._track_last_seen = {}
        self._track_centroid_history = defaultdict(lambda: deque(maxlen=15))
        # Location history catches static false positives that repeatedly
        # reappear with different track IDs after each short-lived track
        # expires. Buckets are coarse enough to tolerate small bbox jitter.
        self._location_static_history = defaultdict(lambda: deque(maxlen=30))

        # Bounding-box size history for bbox smoothing.
        self._track_bbox_history = defaultdict(lambda: deque(maxlen=5))

        # Centroid smoothing: progressive median-based smoothing (see
        # _smooth_centroids).  Uses _track_centroid_history from the
        # static filter above as the raw centroid source.

        # Kalman filters per track for smooth centroid visualisation.
        self._kf = {}
        self._kf_last_frame = {}  # frame_idx of last KF predict (for correct dt)
        # Output EMA smoother on centroid_kf for display (extra-smooth crosshair).
        self._kf_smooth = {}      # tid -> (sx, sy)

        # Per-track flag: set by _kalman_step when the confidence/area gate
        # rejects an update.  Read by the next frame's _smoothed_category
        # to down-weight that frame's category vote.
        self._kf_update_rejected = {}

        # ── Per-run model-characterization metrics ───────────────────
        # Collects the three numbers needed to fill in real per-model
        # tuning profiles (see config.MODEL_TUNING_PROFILES):
        #   1. % of inference frames with ZERO detections (coasting-gap
        #      frequency) — higher means the tracker must predict through
        #      longer gaps, so the KF needs higher process noise.
        #   2. Mean per-frame confidence of matched detections — drives
        #      kf_measurement_noise (a model with low average confidence
        #      should use higher noise).
        #   3. Variance of frame-to-frame centroid displacement for
        #      straight-line-moving vehicles — the "noise floor" the
        #      smoothing is trying to remove.
        self._metrics = {
            "inference_frames": 0,
            "empty_inference_frames": 0,
            "matched_conf_sum": 0.0,
            "matched_conf_n": 0,
            "disp_samples": [],  # straight-line displacement magnitudes (px)
        }
        # tid -> normalized previous displacement direction, for the
        # straight-line gate in _filter_static_tracks.
        self._prev_disp_dir = {}

        # Tracks that have been counted and must never match again.
        self._retired_tracks = set()
        # Tracks that crossed a start line — use stricter matching to
        # prevent an ID swap between start-line and finish-line (which
        # would orphan the crossing state and lose the count).
        self._protected_tracks = set()

        # Bbox area history per track for detection-gating in the KF update.
        self._track_area_history = defaultdict(lambda: deque(maxlen=7))

        # Diagnostic counters for detection pipeline analysis.
        self._diag = {"frames": 0, "yolo_raw": 0, "cls_reject": 0,
                       "bbox_reject": 0, "class_area_reject": 0,
                       "cross_suppress": 0, "passed": 0,
                       "yolo_raw_mc": 0, "cls_reject_mc": 0,
                       "ignore_zone_reject": 0,
                       "track_created": 0,
                       "track_near_disappeared": 0}
        # KF update-rate diagnostics per category: how often a detection
        # for that category actually passed _detection_is_good and
        # corrected the filter, vs. how many detections were seen at all.
        # A low ratio for a category means its KF is coasting on
        # prediction most of the time -- exactly the failure mode behind
        # the "motorcycle centroid jumps + doesn't get counted" bug.
        self._kf_update_attempts = defaultdict(int)
        self._kf_update_passed = defaultdict(int)

    @staticmethod
    def _bbox_iou(a, b):
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter) if (area_a + area_b - inter) > 0 else 0.0

    _CLASS_AREA_HINTS = {
        # (threshold, penalty) — weight is multiplied by penalty when area
        # goes AGAINST the expected size for that category.
        "Motorcycle":   (60000, 0.05),   # area > 60000 → almost certainly not a MC
        "Bus":          (800,   0.05),   # area < 800   → almost certainly not a Bus
        "Light Truck":  (800,   0.05),   # area < 800   → almost certainly not a Truck
    }

    def _smoothed_category(self, track_id, category, conf, bbox_area=None):
        raw_category = category
        history = self._category_history[track_id]
        history.append((category, conf, bbox_area))

        # If locked externally (via lock_category, called by ZoneCounter
        # at start-line crossing), return the locked category permanently.
        if track_id in self._locked_category:
            # Periodic re-evaluation: every 15 frames after locking, check
            # if the last 12 history entries consistently vote for a
            # different category with 2× the weight — this corrects cases
            # where the lock was set during a brief misclassification
            # streak.
            locked = self._locked_category[track_id]
            if len(history) >= 15 and len(history) % 15 == 0:
                recent = list(history)[-12:]
                weights = defaultdict(float)
                for cat, c, area in recent:
                    w = c
                    if area is not None and cat in self._CLASS_AREA_HINTS:
                        th, pen = self._CLASS_AREA_HINTS[cat]
                        if (th > 1000 and area > th) or (th < 1000 and area < th):
                            w *= pen
                    weights[cat] += w
                if weights:
                    best = max(weights, key=weights.get)
                    if best != locked and weights[best] > weights[locked] * 2.0:
                        self._locked_category[track_id] = best
                        print(f"[cat-relock] track={track_id} {locked}→{best} "
                              f"(w_best={weights[best]:.1f} w_locked={weights[locked]:.1f})")
            return self._locked_category[track_id]

        # Confidence-weighted voting from recent history.
        recent = list(history)[-8:]
        weights = defaultdict(float)
        for cat, c, area in recent:
            w = c
            if area is not None and cat in self._CLASS_AREA_HINTS:
                threshold, penalty = self._CLASS_AREA_HINTS[cat]
                # Penalise when area contradicts the category:
                #   * threshold > 1000 → upper bound (area too large)
                #   * threshold < 1000 → lower bound (area too small)
                if (threshold > 1000 and area > threshold) or (threshold < 1000 and area < threshold):
                    w *= penalty
            weights[cat] += w

        best = max(weights, key=weights.get)

        if best != raw_category and len(recent) >= 2:
            print(f"[cat-vote] track={track_id} raw={raw_category}->voted={best} "
                  f"(w={weights[best]:.1f}/{sum(weights.values()):.1f} n={len(recent)})")
        return best

    def lock_category(self, track_id, category=None):
        """Lock the category for this track permanently.
        
        Called by ZoneCounter when the track crosses a start line.
        If *category* is None, the current voted category from the
        track's history is locked.  If a string is given, that exact
        category is locked.
        
        When config.FORCE_MAX_CONF_CATEGORY is True, the locked
        category is overridden by the highest-confidence single-frame
        detection in the track's history — this fixes cases where
        early low-confidence frames voted the wrong category and the
        smoothed vote can't recover.
        
        Once locked, _smoothed_category returns this category for
        every subsequent frame — the count and the display both use
        the locked value.
        """
        # Gather all history entries for this track
        history = self._category_history.get(track_id)
        if history is None:
            return

        if config.FORCE_MAX_CONF_CATEGORY:
            # Use the single frame with the highest confidence
            best_entry = max(history, key=lambda x: x[1])
            best_cat = best_entry[0]
            print(f"[cat-lock-force] track={track_id} max_conf={best_entry[1]:.3f} "
                  f"cat={best_cat} (overriding smoothed vote)")
            self._locked_category[track_id] = best_cat
            return

        if category is None:
            recent = list(history)[-8:]
            weights = defaultdict(float)
            for cat, c, _ in recent:
                weights[cat] += c
            if weights:
                best = max(weights, key=weights.get)
                self._locked_category[track_id] = best
                print(f"[cat-lock-start] track={track_id} locked={best} "
                      f"(w={weights[best]:.1f}/{sum(weights.values()):.1f} n={len(recent)})")
        else:
            self._locked_category[track_id] = category
            print(f"[cat-lock-start] track={track_id} locked={category} (explicit)")

    def retire_track(self, tid):
        """Mark a track as retired — it will never match to a detection again.
        
        Called by ZoneCounter after the track has been counted (crossed a
        finish line).  The track is kept alive (predicted) until it naturally
        expires, so it remains visible in the overlay, but it cannot steal
        the detection of another vehicle, which was the root cause of double
        counting.
        """
        self._retired_tracks.add(tid)

    def protect_track(self, tid):
        """Protect a track from cheap ID swaps.
        
        Called by ZoneCounter when this track crosses a start line.  Until
        it reaches a finish line, matching uses a stricter cost threshold
        (_PROTECTED_MATCH_COST = 0.5 vs normal _MAX_MATCH_COST = 0.7), so
        the Hungarian algorithm will reject a marginal assignment even if
        it reduces the global sum — better to coast on prediction and wait
        for a confident re-match than to accidentally swap IDs with a
        nearby vehicle and orphan the counting state.
        """
        self._protected_tracks.add(tid)

    def _smooth_centroids(self, detections):
        """Smooth centroid positions with progressive strength.

        Activate smoothing as soon as the track has 2+ raw centroid samples
        in _track_centroid_history (populated by _filter_static_tracks which
        runs before this method):

          * 2 samples   → average (equal weight, mild smoothing)
          * 3-4 samples → median of all available samples (robust to 1 outlier)
          * 5+ samples  → median of last 7 (completely immune to 3 outlier frames)

        The median approach is inherently flicker-proof — a single frame
        where YOLO's bbox is abnormally tall/short has zero effect on the
        median value — while genuine movement over consecutive frames
        shifts the median naturally.
        """
        for d in detections:
            tid = d["track_id"]
            raw = d["centroid"]
            hist = self._track_centroid_history.get(tid)

            if hist is None or len(hist) < 2:
                d["centroid_raw"] = raw
                continue

            n = len(hist)
            window = list(hist)[-min(n, 7):]
            nw = len(window)

            d["centroid_raw"] = raw  # preserve raw for zero-delay line crossing

            if nw >= 3:
                xs = sorted(p[0] for p in window)
                ys = sorted(p[1] for p in window)
                mid = nw // 2
                d["centroid"] = (xs[mid], ys[mid])
            elif nw == 2:
                d["centroid"] = (
                    (window[0][0] + window[1][0]) / 2,
                    (window[0][1] + window[1][1]) / 2,
                )

    def _smooth_bboxes(self, detections):
        """Smooth bounding boxes using median center + median size.

        Uses the already-smoothed centroid (from _smooth_centroids which
        runs before this method) as the box center, and computes a median
        width/height from _track_bbox_history.  This prevents the drawn
        rectangle from visibly jittering frame to frame while still
        following the vehicle's genuine movement.
        """
        for d in detections:
            tid = d["track_id"]
            raw_bbox = d["bbox"]
            d["bbox_raw"] = raw_bbox

            cx, cy = d["centroid"]  # already median-smoothed

            x1, y1, x2, y2 = raw_bbox
            hist = self._track_bbox_history[tid]
            hist.append((x2 - x1, y2 - y1))

            if len(hist) >= 3:
                ws = sorted(p[0] for p in hist)
                hs = sorted(p[1] for p in hist)
                mid = len(ws) // 2
                mw = ws[mid]
                mh = hs[mid]
                d["bbox"] = (
                    cx - mw / 2,
                    cy - mh / 2,
                    cx + mw / 2,
                    cy + mh / 2,
                )

    def _predict_position(self, state, gap=None):
        """Predict current centroid, scaling velocity by the actual frame gap."""
        cx, cy = state["centroid"]
        if gap is None:
            gap = state.get("gap", config.FRAME_SKIP)
        if "prev_centroid" in state:
            prev_gap = state.get("prev_gap", gap)
            vx = (cx - state["prev_centroid"][0]) / prev_gap * gap
            vy = (cy - state["prev_centroid"][1]) / prev_gap * gap
            if "prev_prev_centroid" in state and "prev_prev_gap" in state:
                prev_vx = (state["prev_centroid"][0] - state["prev_prev_centroid"][0]) / state["prev_prev_gap"] * gap
                prev_vy = (state["prev_centroid"][1] - state["prev_prev_centroid"][1]) / state["prev_prev_gap"] * gap
                ax = vx - prev_vx
                ay = vy - prev_vy
                return (cx + vx + int(ax * 0.5), cy + vy + int(ay * 0.5))
            return (cx + vx, cy + vy)
        return (cx, cy)

    def _kalman_step(self, detections, frame_idx):
        """Apply Kalman predict (and optionally update) to matched detections.

        Called every inference frame.  For each detection:
          * Computes *dt = frame_idx - last_update* so the KF always advances
            by the correct number of video frames regardless of FRAME_SKIP.
          * Predicts forward by *dt*.
          * Updates with the bottom-centre measurement *only* when the
            detection passes the gating check (*_detection_is_good*).
          * Stores the filtered position in *det['centroid_kf']*.

        When a detection is gated out (low confidence or abnormal bbox area),
        the KF runs predict-only.  The KF then coasts on its velocity model
        until a good measurement resumes it — this prevents the centroid from
        snapping to occlusion fragments or merged boxes.

        Unmatched tracks (still in _active_tracks but not in *detections*)
        are not advanced here — they are advanced in *predict_detections*
        on non-inference frames.
        """
        for d in detections:
            tid = d["track_id"]
            bottom = d.get("centroid_bottom", d["centroid"])
            conf = d["conf"]
            # ── Metric: mean per-frame confidence of matched detections ──
            self._metrics["matched_conf_sum"] += conf
            self._metrics["matched_conf_n"] += 1
            bbox = d["bbox"]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

            self._track_area_history[tid].append(area)

            if tid not in self._kf:
                self._kf[tid] = _PerTrackKF(bottom[0], bottom[1],
                                             measurement_noise=self.tuning["kf_measurement_noise"])
                self._kf_last_frame[tid] = frame_idx
            else:
                dt = max(1, frame_idx - self._kf_last_frame[tid])
                if dt > 0:
                    kf = self._kf[tid]
                    kf.predict(dt)
                    self._kf_last_frame[tid] = frame_idx
                cat = d.get("category")
                self._kf_update_attempts[cat] += 1
                if self._detection_is_good(tid, area, conf, category=cat):
                    self._kf_update_passed[cat] += 1
                    self._kf_update_rejected.pop(tid, None)
                    # Clamped update: always correct toward the measurement
                    # but cap the per-frame displacement.  This prevents a
                    # swapped detection from teleporting the KF while still
                    # tracking the real vehicle through turns and acceleration.
                    speed = (kf.x[2] ** 2 + kf.x[3] ** 2) ** 0.5
                    max_jump = max(60.0, speed * 5)
                    self._kf[tid].update_clamped(bottom[0], bottom[1], conf, max_jump)
                else:
                    self._kf_update_rejected[tid] = True
                    if tid in self._category_history and self._category_history[tid]:
                        last_cat, last_conf, last_area = self._category_history[tid][-1]
                        self._category_history[tid][-1] = (last_cat, last_conf * 0.5, last_area)

            ckf = (float(self._kf[tid].x[0]), float(self._kf[tid].x[1]))
            d["centroid_kf"] = ckf

            # Output EMA smoother for display — keeps the visual crosshair
            # extra-smooth without affecting the crossing math.
            # Low alpha heavily damps single-frame KF corrections, so a
            # turn-induced snap is spread over many frames instead of
            # hitting the display in one frame.  Pure cosmetics: this field
            # is never used by zone_counter's counting logic.
            # Per-model alpha from the tuning profile.
            alpha = self.tuning["kf_smooth_alpha"]
            prev = self._kf_smooth.get(tid, ckf)
            blended = (
                prev[0] * (1 - alpha) + ckf[0] * alpha,
                prev[1] * (1 - alpha) + ckf[1] * alpha,
            )
            # Hard cap on visual step size — belt-and-suspenders against
            # any single-frame jump regardless of what feeds the KF.
            max_visual_step = 15  # px per frame
            ddx = blended[0] - prev[0]
            ddy = blended[1] - prev[1]
            dist = (ddx * ddx + ddy * ddy) ** 0.5
            if dist > max_visual_step:
                scale = max_visual_step / dist
                blended = (prev[0] + ddx * scale, prev[1] + ddy * scale)
            self._kf_smooth[tid] = blended
            d["centroid_kf_smooth"] = blended

    def _detection_is_good(self, tid, area, conf, category=None):
        """Return True when a detection is reliable enough for a KF update.

        Rejects:
          * Confidence below the per-class KF-update floor
            (config.KF_UPDATE_MIN_CONF). This is intentionally separate
            from MIN_CONF_BY_CLASS (which only gates tracker admission) —
            see the comment on KF_UPDATE_MIN_CONF in config.py for why a
            flat threshold across all classes was starving motorcycle
            tracks of real corrections.
          * Detections whose bbox area differs a lot from the running
            median area — these are almost always occlusion fragments or
            merged boxes that would pull the centroid off the real
            vehicle. Uses a HYBRID relative+absolute gate: small boxes
            (motorcycles) swing a lot in relative terms from just a few
            pixels of detection noise, so relative-only unfairly rejected
            valid small-vehicle updates. Both the relative AND absolute
            change must be large before rejecting.

        Position-plausibility is handled by the clamped update in
        _kalman_step, not by a binary reject gate here.
        """
        min_conf = config.KF_UPDATE_MIN_CONF.get(category, 0.3) if category else 0.3
        if conf < min_conf:
            return False
        hist = self._track_area_history.get(tid)
        if hist is None or len(hist) < 3:
            return True  # not enough history to judge
        sorted_ = sorted(hist)
        median_area = sorted_[len(sorted_) // 2]
        if median_area <= 0:
            return True
        rel_change = abs(area - median_area) / median_area
        abs_change = abs(area - median_area)
        if rel_change > self.tuning["area_change_threshold"] and abs_change > 400:
            return False
        return True

    def _predict_detections(self, frame_idx):
        """Generate predicted detections from active tracks for the current frame.

        Uses velocity (and acceleration if available) to estimate each track's
        position at the current frame.  These are yielded on non-inference
        frames so that:
          - the displayed tracker bounding box is visible every frame (no flicker)
          - the zone counter can check for line crossings every frame (no missed
            vehicles between detection frames)

        Only predicts for tracks whose gap since last detection is within one
        FRAME_SKIP interval — stale buffer-only tracks are not predicted.
        Also skips tracks identified as static false positives.
        """
        detections = []
        for tid, state in self._active_tracks.items():
            # Skip retired tracks — they've been counted and should not
            # appear in the overlay or be considered for any matching.
            if tid in self._retired_tracks:
                continue
            actual_gap = frame_idx - state["last_frame"]
            if actual_gap < 1 or actual_gap > config.FRAME_SKIP:
                continue

            # Skip tracks whose centroid history shows no genuine movement
            # (they are static false positives like foliage or debris)
            hist = self._track_centroid_history.get(tid)
            if hist is not None and len(hist) >= config.STATIC_MIN_SAMPLES:
                hlist = list(hist)
                max_pairwise_sq = 0.0
                for i in range(len(hlist)):
                    for j in range(i + 1, len(hlist)):
                        d_sq = (hlist[i][0] - hlist[j][0]) ** 2 + (hlist[i][1] - hlist[j][1]) ** 2
                        if d_sq > max_pairwise_sq:
                            max_pairwise_sq = d_sq
                pairwise = max_pairwise_sq ** 0.5
                net_dx = hlist[-1][0] - hlist[0][0]
                net_dy = hlist[-1][1] - hlist[0][1]
                net = (net_dx * net_dx + net_dy * net_dy) ** 0.5
                if (pairwise < config.STATIC_MOTION_THRESHOLD_PX and
                        net < config.STATIC_NET_DISPLACEMENT_PX):
                    continue

            # Use Kalman filter predict if available; fall back to kinematic.
            # Use peek() NOT predict() — only _kalman_step should advance the
            # KF state.  This keeps the prediction consistent with the actual
            # frame gap on every inference frame.
            if tid in self._kf:
                kf = self._kf[tid]
                dt = max(1, frame_idx - self._kf_last_frame.get(tid, frame_idx - 1))
                pred_cx, pred_cy = kf.peek(dt)
            else:
                pred_cx, pred_cy = self._predict_position(state, gap=actual_gap)

            # Smooth the crosshair across ALL frames (inference + non-inference)
            # using the peeked KF position.  On inference frames _kalman_step
            # also updates _kf_smooth with the same alpha and step cap,
            # keeping the blend consistent regardless of which method ran last.
            prev_smooth = self._kf_smooth.get(tid)
            if prev_smooth is None:
                self._kf_smooth[tid] = (pred_cx, pred_cy)
            else:
                alpha = self.tuning["kf_smooth_alpha"]
                blended = (
                    prev_smooth[0] * (1 - alpha) + pred_cx * alpha,
                    prev_smooth[1] * (1 - alpha) + pred_cy * alpha,
                )
                # Same hard cap on visual step size as _kalman_step
                max_visual_step = 15  # px per frame
                ddx = blended[0] - prev_smooth[0]
                ddy = blended[1] - prev_smooth[1]
                dist = (ddx * ddx + ddy * ddy) ** 0.5
                if dist > max_visual_step:
                    scale = max_visual_step / dist
                    blended = (prev_smooth[0] + ddx * scale,
                               prev_smooth[1] + ddy * scale)
                self._kf_smooth[tid] = blended

            bbox = state["bbox"]
            dx = pred_cx - state["centroid"][0]
            dy = pred_cy - state["centroid"][1]
            pred_bbox = (
                bbox[0] + dx,
                bbox[1] + dy,
                bbox[2] + dx,
                bbox[3] + dy,
            )

            detections.append({
                "track_id": tid,
                "centroid": (pred_cx, pred_cy),
                "centroid_kf": (pred_cx, pred_cy),
                "centroid_kf_smooth": self._kf_smooth.get(tid, (pred_cx, pred_cy)),
                "bbox": pred_bbox,
                "category": state["category"],
                "conf": state["conf"],
                "age": self._track_age.get(tid, 0),
                "predicted": True,
            })
        return detections

    def _record_track_birth(self, tid, detection, frame_idx, frame_gap,
                            excluded_track_ids=()):
        """Log a new ID and flag likely churn against recently unseen tracks.

        This is diagnostic-only: it does not alter matching, relinking, or
        count behavior.  ``_active_tracks`` deliberately retains unmatched
        tracks in a buffer, so a newly minted ID can be compared with the
        last position/frame of a track that disappeared even when that old
        track did not cross a counting line and ZoneCounter never considered
        it for relinking.
        """
        self._diag["track_created"] += 1
        cx, cy = detection["centroid"]
        category = detection.get("category", "?")
        print(f"[TRACK-CREATE] frame={frame_idx} track={tid} cat={category} "
              f"centroid=({cx:.0f},{cy:.0f}) frame_gap={frame_gap}")

        excluded = set(excluded_track_ids)
        nearby = []
        for old_tid, state in self._active_tracks.items():
            if old_tid in excluded or old_tid in self._retired_tracks:
                continue
            last_frame = state.get("last_frame")
            if last_frame is None:
                continue
            gap = frame_idx - last_frame
            if gap <= 0 or gap > config.ZONE_RELINK_MAX_FRAMES:
                continue
            old_cx, old_cy = state["centroid"]
            dist = ((cx - old_cx) ** 2 + (cy - old_cy) ** 2) ** 0.5
            if dist <= config.ZONE_RELINK_MAX_DIST_PX:
                nearby.append((dist, old_tid, gap, state.get("category", "?")))

        if not nearby:
            return

        self._diag["track_near_disappeared"] += 1
        for dist, old_tid, gap, old_category in sorted(nearby):
            print(f"[TRACK-CHURN] frame={frame_idx} new={tid} prior={old_tid} "
                  f"gap={gap} frames dist={dist:.0f}px "
                  f"cat={category}/{old_category}")

    # Matches at or above this combined cost are rejected outright — it's
    # better to end that track and start a fresh one than to force a poor
    # association onto it.  Under the old greedy matcher, a locally-best
    # pairing could be "taken" early even when the total cost across all
    # detections/tracks was worse than an alternative assignment — this
    # is one of the main ways a nearby vehicle stole another one's track
    # ID, especially in dense clusters or when a turning vehicle's motion
    # prediction drifted off the real vehicle.
    _MAX_MATCH_COST = 0.6

    # Stricter threshold for tracks that have crossed a start line (see
    # protect_track).  These are "in flight" between start and finish —
    # an ID swap at this point would orphan the crossing state and lose
    # the count.  Better to coast on prediction and reject a marginal
    # match than to accidentally swap.
    _PROTECTED_MATCH_COST = 0.5

    def _match_detections(self, detections, frame_gap, frame_idx):
        """Match raw detections to tracks using Kalman + IoU + distance.

        Uses a cost matrix combining centroid distance and bbox similarity,
        then assigns via the Hungarian algorithm (optimal assignment).  On
        the detection side uses the raw centroid (box centre); on the track
        side uses the Kalman-filtered prediction when available (which gives
        a smooth, velocity-consistent position) falling back to the
        kinematic model.  The cost function:

          * When boxes overlap (IoU > 0.1) — weighted (1 - IoU) + distance.
            Overlap is a strong signal that these are the same vehicle even
            when detections are close together.

          * When boxes don't overlap — distance (70 %) + bbox-size
            similarity (30 %).  Size helps disambiguate nearby vehicles of
            different types (car vs motorcycle) when their bounding boxes
            don't touch.

        Pairs whose centroid distance exceeds 150 px per frame-skip step
        are excluded from the cost matrix (cost = 10.0) and will not be
        matched even if the Hungarian algorithm is forced to pick them.
        """
        if not self._active_tracks or not detections:
            result = []
            for d in detections:
                tid = next(self._next_track_id)
                self._record_track_birth(tid, d, frame_idx, frame_gap)
                bbox = d["bbox"]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                cat = self._smoothed_category(tid, d["category"], d["conf"], bbox_area=area)
                self._locked_category[tid] = cat
                result.append({**d, "track_id": tid, "category": cat, "age": 1})
            return result

        # Build track predictions — prefer Kalman filter, fall back to kinematic.
        # IMPORTANT: Use peek() NOT predict() — we must NOT mutate the KF
        # state here; only _kalman_step (called right after) should advance
        # the filter.  If predict() is called here then _kalman_step computes
        # dt=0 (or dt=1 if clamped) and predicts an extra frame, causing the
        # KF position to drift ahead of the real vehicle.
        track_items = []  # (tid, state, (px, py), protected, (vx, vy))
        for tid, s in self._active_tracks.items():
            if tid in self._retired_tracks:
                continue
            if tid in self._kf:
                kf = self._kf[tid]
                dt = max(1, frame_idx - self._kf_last_frame.get(tid, frame_idx - 1))
                px, py = kf.peek(dt)
                vx = float(kf.x[2])
                vy = float(kf.x[3])
            else:
                px, py = self._predict_position(s, gap=frame_gap)
                # Kinematic velocity from state history
                if "prev_centroid" in s:
                    gap = s.get("gap", 1)
                    vx = (s["centroid"][0] - s["prev_centroid"][0]) / gap
                    vy = (s["centroid"][1] - s["prev_centroid"][1]) / gap
                else:
                    vx, vy = 0.0, 0.0
            protected = tid in self._protected_tracks
            track_items.append((tid, s, (px, py), protected, (vx, vy)))

        max_dist = 150 * frame_gap

        n_det = len(detections)
        n_trk = len(track_items)

        cost_matrix = np.full((n_det, n_trk), 10.0, dtype=np.float64)
        any_valid = False
        for di, det in enumerate(detections):
            dcx, dcy = det["centroid"]
            dbox = det["bbox"]
            det_w = dbox[2] - dbox[0]
            det_h = dbox[3] - dbox[1]
            det_area = det_w * det_h
            for ti, (tid, state, (px, py), _protected, (vx, vy)) in enumerate(track_items):
                dist = ((dcx - px) ** 2 + (dcy - py) ** 2) ** 0.5
                if dist > max_dist:
                    continue
                dist_score = dist / max_dist
                iou = self._bbox_iou(dbox, state["bbox"])
                if iou > 0.1:
                    # Boxes overlap — IoU is a strong signal
                    combined = (1.0 - iou) * 0.6 + dist_score * 0.4
                else:
                    # No overlap — use distance + bbox size similarity
                    trk_w = state["bbox"][2] - state["bbox"][0]
                    trk_h = state["bbox"][3] - state["bbox"][1]
                    trk_area = trk_w * trk_h
                    area_ratio = (min(det_area, trk_area) / max(det_area, trk_area)
                                  if max(det_area, trk_area) > 0 else 0.0)
                    # Base cost: distance dominates, size helps disambiguate.
                    combined = dist_score * 0.7 + (1.0 - area_ratio) * 0.3
                    # Direction consistency ONLY for protected tracks
                    # (those that crossed a start line).  An ID swap here
                    # would orphan the crossing state and lose the count.
                    # Regular tracks need the freedom to match across turns
                    # or they get lost at intersections.
                    if _protected:
                        direction_penalty = 0.0
                        v_norm_sq = vx * vx + vy * vy
                        if v_norm_sq > 2.0:
                            dx_pred = dcx - px
                            dy_pred = dcy - py
                            dot = vx * dx_pred + vy * dy_pred
                            if dot < 0:
                                direction_penalty = 0.35
                        combined = (dist_score * 0.45
                                    + (1.0 - area_ratio) * 0.20
                                    + direction_penalty * 0.35)
                cost_matrix[di, ti] = combined
                any_valid = True

        matched = []
        if any_valid and _SCIPY_AVAILABLE:
            det_idx, trk_idx = linear_sum_assignment(cost_matrix)
            for di, ti in zip(det_idx, trk_idx):
                cost = cost_matrix[di, ti]
                _, _, _, protected, _ = track_items[ti]
                threshold = self._PROTECTED_MATCH_COST if protected else self._MAX_MATCH_COST
                if cost >= threshold:
                    continue
                tid = track_items[ti][0]
                matched.append((di, tid))
        elif any_valid:
            # Fallback if scipy isn't installed: old greedy sorted matching,
            # now with the same bad-match rejection gate at least.
            costs = []
            for di in range(n_det):
                for ti in range(n_trk):
                    c = cost_matrix[di, ti]
                    _, _, _, protected, _ = track_items[ti]
                    threshold = self._PROTECTED_MATCH_COST if protected else self._MAX_MATCH_COST
                    if c < threshold:
                        costs.append((c, di, ti))
            costs.sort(key=lambda x: x[0])
            assigned_dets = set()
            assigned_tracks = set()
            for cost, di, ti in costs:
                if di in assigned_dets or ti in assigned_tracks:
                    continue
                assigned_dets.add(di)
                assigned_tracks.add(ti)
                matched.append((di, track_items[ti][0]))

        result = []
        matched_di = set(di for di, _ in matched)
        matched_track_ids = set(tid for _, tid in matched)

        for di, tid in matched:
            det = detections[di]
            bbox = det["bbox"]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            cat = self._smoothed_category(tid, det["category"], det["conf"], bbox_area=area)
            self._track_age[tid] += 1
            result.append({**det, "track_id": tid, "category": cat, "age": self._track_age[tid]})

        # ByteTrack-style second pass: unmatched low-confidence detections
        # can still recover unmatched tracks (occlusion recovery).
        # Build a list of tracks that were NOT used in the first pass.
        unmatched_tracks = [(ti, t) for ti, t in enumerate(track_items)
                            if t[0] not in matched_track_ids]

        for di, det in enumerate(detections):
            if di in matched_di:
                continue
            conf = det["conf"]

            # ── ByteTrack low-confidence recovery pass ──────────────
            if conf < self.tuning["bytetrack_high_conf_threshold"] and unmatched_tracks:
                dcx, dcy = det["centroid"]
                best_dist = max_dist * 1.2  # relaxed for recovery
                best_tid = None
                for _, (tid, _, (px, py), _, _) in unmatched_tracks:
                    dist = ((dcx - px) ** 2 + (dcy - py) ** 2) ** 0.5
                    if dist < best_dist:
                        best_dist = dist
                        best_tid = tid
                if best_tid is not None:
                    matched_di.add(di)
                    matched_track_ids.add(best_tid)
                    bbox = det["bbox"]
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    cat = self._smoothed_category(best_tid, det["category"], det["conf"], bbox_area=area)
                    self._track_age[best_tid] += 1
                    result.append({**det, "track_id": best_tid, "category": cat,
                                   "age": self._track_age[best_tid]})
                    continue
                # ── Low-conf unmatched → drop (never spawns a new track) ──
                continue

            # ── High-confidence unmatched → dedup re-adoption or new track ──
            centroid = det["centroid"]
            # Dedup: before minting a new track_id, check if this
            # detection is suspiciously close to a recently-active
            # track that we failed to match (e.g. because of a
            # temporary cost-threshold rejection).  Re-adopt the
            # old ID rather than spawning a duplicate.
            reuse_tid = None
            for tid, s in self._active_tracks.items():
                if tid in matched_track_ids or tid in self._retired_tracks:
                    continue
                gap = frame_idx - s["last_frame"]
                if gap <= config.FRAME_SKIP * 2:
                    sc = s["centroid"]
                    dist = ((centroid[0] - sc[0]) ** 2 +
                            (centroid[1] - sc[1]) ** 2) ** 0.5
                    if dist < 60:
                        reuse_tid = tid
                        break
            if reuse_tid is not None:
                tid = reuse_tid
                if tid in self._track_age:
                    self._track_age[tid] += 1
                cat = self._smoothed_category(tid, det["category"], det["conf"],
                                              bbox_area=None)
                result.append({**det, "track_id": tid, "category": cat,
                               "age": self._track_age.get(tid, 0)})
            else:
                tid = next(self._next_track_id)
                self._record_track_birth(tid, det, frame_idx, frame_gap,
                                         excluded_track_ids=matched_track_ids)
                self._track_age[tid] += 1
                bbox = det["bbox"]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                cat = self._smoothed_category(tid, det["category"], det["conf"], bbox_area=area)
                # NOTE: category is NOT locked here — let the vote
                # stabilize over the first few frames.  Locking is only
                # done at start-line crossing (ZoneCounter calls
                # lock_category), when we have enough history to be
                # confident.
                result.append({**det, "track_id": tid, "category": cat, "age": self._track_age[tid]})

        return result

    def _update_active_tracks(self, detections, frame_idx):
        """Update internal tracker state, keeping velocity history and frame gaps."""
        new_tracks = {}
        for d in detections:
            tid = d["track_id"]
            prev = self._active_tracks.get(tid)
            if prev is None:
                self._track_created_at.setdefault(tid, frame_idx)
            self._track_last_seen[tid] = frame_idx
            entry = {
                "centroid": d["centroid"],
                "bbox": d["bbox"],
                "category": d["category"],
                "conf": d["conf"],
                "last_frame": frame_idx,
            }
            if prev is not None:
                # Store the actual number of video frames since last detection
                gap = frame_idx - prev["last_frame"]
                entry["gap"] = gap
                entry["prev_centroid"] = prev["centroid"]
                if "gap" in prev:
                    entry["prev_gap"] = prev["gap"]
                if "prev_centroid" in prev:
                    entry["prev_prev_centroid"] = prev["prev_centroid"]
                    if "prev_gap" in prev:
                        entry["prev_prev_gap"] = prev["prev_gap"]
            new_tracks[tid] = entry

        track_buffer_frames = 300  # ~12 s at 25 fps — keeps tracks through brief occlusion / angle dropout
        expired_tracks = []
        for tid, state in self._active_tracks.items():
            if tid not in new_tracks:
                if frame_idx - state["last_frame"] < track_buffer_frames:
                    new_tracks[tid] = state
                else:
                    expired_tracks.append(tid)

        if expired_tracks and self._diag["frames"] % 500 == 0:
            lifespan = frame_idx - self._track_created_at.get(expired_tracks[0], frame_idx)
            print(f"[tid-lifecycle] EXPIRE {len(expired_tracks)} track(s) at frame={frame_idx}, "
                  f"e.g. tid={expired_tracks[0]} lifespan={lifespan} frames")

        self._active_tracks = new_tracks

        # Clean up per-track history for expired tracks — without this the
        # centroid/bbox/category dictionaries grow unbounded with every
        # unique track_id that ever appears in the video, eventually
        # exhausting memory on long surveys.
        for tid in expired_tracks:
            self._track_centroid_history.pop(tid, None)
            self._category_history.pop(tid, None)
            self._track_age.pop(tid, None)
            self._track_bbox_history.pop(tid, None)
            self._track_last_seen.pop(tid, None)
            self._track_created_at.pop(tid, None)
            self._locked_category.pop(tid, None)
            self._kf.pop(tid, None)
            self._kf_last_frame.pop(tid, None)
            self._kf_smooth.pop(tid, None)
            self._kf_update_rejected.pop(tid, None)
            self._track_area_history.pop(tid, None)
            self._retired_tracks.discard(tid)
            self._protected_tracks.discard(tid)

    @staticmethod
    def _cross_class_nms(detections, iou_threshold=None):
        """Cross-class NMS: if two different classes overlap heavily, keep the higher-confidence one.
        
        YOLO's built-in NMS runs per class, so the same bus can output both
        Bus (cls=5) and Motorcycle (cls=3) detections — both survive,
        the tracker creates two tracks, and one gets counted as "Motorcycle".
        This pass suppresses the lower-confidence detection of any overlapping pair
        regardless of class label, preventing that confusion.

        iou_threshold: IoU threshold above which the lower-conf detection is
        suppressed.  Defaults to config.CROSS_CLASS_NMS_IOU (0.5).
        """
        if iou_threshold is None:
            iou_threshold = config.CROSS_CLASS_NMS_IOU
        if len(detections) < 2:
            return detections
        sorted_dets = sorted(detections, key=lambda d: -d["conf"])
        keep = [True] * len(sorted_dets)
        for i in range(len(sorted_dets)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(sorted_dets)):
                if not keep[j]:
                    continue
                if sorted_dets[i]["category"] == sorted_dets[j]["category"]:
                    continue
                iou = VehicleDetector._bbox_iou(sorted_dets[i]["bbox"], sorted_dets[j]["bbox"])
                if iou > iou_threshold:
                    keep[j] = False
        return [d for i, d in enumerate(sorted_dets) if keep[i]]

    def _filter_static_tracks(self, detections, frame_idx):
        """
        Remove detections whose track centroids show no genuine movement —
        these are almost certainly static-scene false positives (foliage,
        debris, road furniture) that flicker on and off at the same pixel
        location as the detector's confidence wavers above/below threshold.

        Uses TWO complementary metrics, both of which must fall below
        threshold for a track to be rejected as static:

          1. Max pairwise distance (any two centroids) — catches
             circular/zigzag paths where start and end happen to overlap
             (e.g. a vehicle turning around, or wind-blown foliage).

          2. Net displacement (first centroid → last centroid) — catches
             oscillating false positives (tree branches swaying back and
             forth) that have large pairwise spread but near-zero net
             progress.  A genuinely moving vehicle shows both metrics
             above threshold.

        Only filters tracks old enough (8+ centroid samples) so a
        momentarily stopped vehicle at a red light is still counted
        before the gate activates.
        """
        if not detections:
            return detections

        result = []
        for d in detections:
            tid = d["track_id"]
            centroid = d["centroid"]
            hist = self._track_centroid_history[tid]
            hist.append(centroid)

            # ── Metric: straight-line displacement variance ──────────
            # Displacement between consecutive inference frames, kept only
            # while the track moves in a roughly constant direction
            # (direction change < ~15°).  These samples form the "noise
            # floor" for the current model — high variance means the
            # detector's box centers jitter a lot even on straight roads.
            if len(hist) >= 2:
                pc = hist[-2]
                dx = centroid[0] - pc[0]
                dy = centroid[1] - pc[1]
                disp = (dx * dx + dy * dy) ** 0.5
                if disp > 0:
                    nx, ny = dx / disp, dy / disp
                    prev_dir = self._prev_disp_dir.get(tid)
                    if prev_dir is not None:
                        dot = nx * prev_dir[0] + ny * prev_dir[1]
                        if dot > 0.966:  # cos(15°)
                            self._metrics["disp_samples"].append(disp)
                    self._prev_disp_dir[tid] = (nx, ny)

            # A static false positive can churn through many track IDs, so
            # keep a second history keyed by its coarse image location.
            grid_key = (int(centroid[0]) // 20, int(centroid[1]) // 20)
            loc_hist = self._location_static_history[grid_key]
            loc_hist.append(frame_idx)
            if len(loc_hist) >= 10 and (loc_hist[-1] - loc_hist[0]) < 900:
                print(f"[REJECTED-LOCATION] frame={frame_idx} grid={grid_key} "
                      f"track={tid} cat={d['category']}")
                continue

            if len(hist) >= config.STATIC_MIN_SAMPLES:
                # Max pairwise distance
                max_pairwise_sq = 0.0
                for i in range(len(hist)):
                    for j in range(i + 1, len(hist)):
                        dx = hist[i][0] - hist[j][0]
                        dy = hist[i][1] - hist[j][1]
                        d_sq = dx * dx + dy * dy
                        if d_sq > max_pairwise_sq:
                            max_pairwise_sq = d_sq
                pairwise = max_pairwise_sq ** 0.5

                # Net displacement (first → last centroid)
                net_dx = hist[-1][0] - hist[0][0]
                net_dy = hist[-1][1] - hist[0][1]
                net = (net_dx * net_dx + net_dy * net_dy) ** 0.5

                if (pairwise < config.STATIC_MOTION_THRESHOLD_PX and
                        net < config.STATIC_NET_DISPLACEMENT_PX):
                    print(f"[REJECTED-STATIC] frame={frame_idx} track={tid} "
                          f"cat={d['category']} centroid=({centroid[0]:.0f},{centroid[1]:.0f}) "
                          f"pairwise={pairwise:.1f}px net={net:.1f}px  n={len(hist)}")
                    continue
            result.append(d)
        return result

    def _apply_detection_filters(self, raw_detections):
        """Shared filter pass for both FRAME_SKIP=1 and FRAME_SKIP>1 paths.

        Applies bbox size, aspect-ratio, class-vs-aspect, and class-vs-area
        filters to a list of raw detector outputs.
        Each dict in raw_detections must have keys: bbox (x1,y1,x2,y2),
        centroid (cx, cy), and cls_id (YOLO class integer).
        Returns the filtered list.
        """
        out = []
        for d in raw_detections:
            x1, y1, x2, y2 = d["bbox"]
            w = x2 - x1
            h = y2 - y1
            cls_id = d.get("cls_id")

            # Minimum bbox size
            if w < config.MIN_BBOX_WIDTH or h < config.MIN_BBOX_HEIGHT:
                continue

            # Aspect-ratio sanity
            aspect = w / h if h > 0 else 0
            if aspect < config.MIN_BBOX_ASPECT or aspect > config.MAX_BBOX_ASPECT:
                continue

            # Class-vs-aspect sanity
            if cls_id is not None:
                cls_aspect = config.CLASS_ASPECT_RANGE.get(cls_id)
                if cls_aspect is not None:
                    min_a, max_a = cls_aspect
                    if min_a is not None and aspect < min_a:
                        continue
                    if max_a is not None and aspect > max_a:
                        continue

            # Class-vs-area sanity
            area = w * h
            max_area = config.CLASS_MAX_BBOX_AREA.get(cls_id) if cls_id is not None else None
            min_area = config.CLASS_MIN_BBOX_AREA.get(cls_id) if cls_id is not None else None
            if max_area is not None and area > max_area:
                continue
            if min_area is not None and area < min_area:
                continue

            out.append(d)
        return out

    def _inference_frame(self, frame):
        """Run YOLO detection and return filtered detections with NMS tuning."""
        model_input = _enhance_frame(frame) if config.ENABLE_FRAME_ENHANCEMENT else frame
        # Ensure the array is C-contiguous — OpenCV enhancement steps
        # (split/merge/LUT/addWeighted) can produce non-contiguous arrays,
        # which trigger an internal Ultralytics code path that tries to
        # save the image as a temp PNG and reload it, causing
        # "Cannot read 'image.png'" errors.
        model_input = np.ascontiguousarray(model_input)
        try:
            results = self.model(
                [model_input],
                conf=self.conf,
                classes=self.classes,
                imgsz=self.imgsz,
                iou=config.YOLO_NMS_IOU,
                augment=config.ENABLE_TTA,
                device=self.device,
                verbose=False,
            )
        except Exception as e:
            # Known Ultralytics bug: if a non-contiguous array or certain
            # internal export paths are triggered, it tries to save the
            # frame as a temp PNG and reload it, failing with
            # "Cannot read 'image.png'".  Fall back to passing the raw
            # BGR frame (no enhancement) as a plain numpy array.
            print(f"[WARNING] model inference failed ({e}), retrying with un-enhanced frame...")
            model_input = np.ascontiguousarray(frame)
            try:
                results = self.model(
                    [model_input],
                    conf=self.conf,
                    classes=self.classes,
                    imgsz=self.imgsz,
                    iou=config.YOLO_NMS_IOU,
                    augment=config.ENABLE_TTA,
                    device=self.device,
                    verbose=False,
                )
            except Exception as e2:
                print(f"[ERROR] model inference failed even with raw frame ({e2}). "
                      f"Returning empty detections for this frame.")
                return []
        result = results[0]

        raw_detections = []
        boxes = result.boxes
        n_yolo_raw = 0
        n_cls_reject = 0
        n_cls_reject_mc = 0
        n_bbox_reject = 0
        n_class_area_reject = 0
        n_yolo_raw_mc = 0
        if boxes is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            n_yolo_raw = len(xyxy)

            for box, cls_id, conf in zip(xyxy, cls_ids, confs):
                conf = float(conf)
                if cls_id == 3:
                    n_yolo_raw_mc += 1

                # Per-class confidence floor.
                min_conf = config.MIN_CONF_BY_CLASS.get(int(cls_id), config.CONFIDENCE_THRESHOLD)
                if conf < min_conf:
                    n_cls_reject += 1
                    if cls_id == 3:
                        n_cls_reject_mc += 1
                    continue

                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1

                # Minimum bbox size.
                if w < config.MIN_BBOX_WIDTH or h < config.MIN_BBOX_HEIGHT:
                    n_bbox_reject += 1
                    continue

                # Aspect-ratio sanity.
                aspect = w / h if h > 0 else 0
                if aspect < config.MIN_BBOX_ASPECT or aspect > config.MAX_BBOX_ASPECT:
                    n_bbox_reject += 1
                    continue

                # Class-vs-aspect sanity: reject when the bbox shape contradicts
                # the assigned class (e.g. a super-narrow box classified as Car
                # is almost certainly a Motorcycle, and a super-wide box
                # classified as Motorcycle is almost certainly a Car/Truck).
                cls_aspect = config.CLASS_ASPECT_RANGE.get(int(cls_id))
                if cls_aspect is not None:
                    min_a, max_a = cls_aspect
                    if min_a is not None and aspect < min_a:
                        n_class_area_reject += 1
                        continue
                    if max_a is not None and aspect > max_a:
                        n_class_area_reject += 1
                        continue

                # Bbox-area class sanity: reject physically impossible
                # class+size combinations (e.g. a 10-pixel box labeled "Bus").
                area = w * h
                max_area = config.CLASS_MAX_BBOX_AREA.get(int(cls_id))
                min_area = config.CLASS_MIN_BBOX_AREA.get(int(cls_id))
                if max_area is not None and area > max_area:
                    n_class_area_reject += 1
                    continue
                if min_area is not None and area < min_area:
                    n_class_area_reject += 1
                    continue

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                # Permanent masks are applied before cross-class NMS and
                # before any detection can reach the tracker/matcher.
                if _in_ignore_zone(cx, cy):
                    self._diag["ignore_zone_reject"] += 1
                    continue

                category = config.COCO_TO_CATEGORY.get(int(cls_id), "Car")

                # Refine COCO class 7: bbox area > threshold → Truck.
                # Threshold is resolution-adaptive: scaled by (imgsz/1280)²
                # so a 960px run uses ~0.56× the default threshold.
                if int(cls_id) == 7:
                    scale_factor = (self.imgsz / 1280.0) ** 2
                    truck_thresh = config.TRUCK_REFINE_AREA_THRESHOLD * scale_factor
                    if area > truck_thresh:
                        category = "Truck"

                raw_detections.append({
                    "centroid": (cx, cy),  # box center — kept for counting
                    "centroid_bottom": ((x1 + x2) / 2, float(y2)),  # contact-point for KF
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),
                    "bbox_raw": (float(x1), float(y1), float(x2), float(y2)),  # pre-smoothing
                    "category": category,
                    "conf": conf,
                    "cls_id": int(cls_id),
                })

        n_before_xnms = len(raw_detections)
        # Cross-class NMS: suppress lower-confidence detection when two
        # different classes (e.g. Bus + Motorcycle) overlap on the same vehicle.
        raw_detections = self._cross_class_nms(raw_detections)
        n_cross_suppress = n_before_xnms - len(raw_detections)

        self._diag["frames"] += 1
        self._diag["yolo_raw"] += n_yolo_raw
        self._diag["yolo_raw_mc"] += n_yolo_raw_mc
        self._diag["cls_reject"] += n_cls_reject
        self._diag["cls_reject_mc"] += n_cls_reject_mc
        self._diag["bbox_reject"] += n_bbox_reject
        self._diag["class_area_reject"] += n_class_area_reject
        self._diag["cross_suppress"] += n_cross_suppress
        self._diag["passed"] += len(raw_detections)

        if self._diag["frames"] % 500 == 0 and self._diag["frames"] > 0:
            d = self._diag
            mc_survival = (d["yolo_raw_mc"] - d["cls_reject_mc"] - d["cross_suppress"]) / max(d["yolo_raw_mc"], 1) * 100
            print(f"[DIAG] {d['frames']} inf frames: YOLO raw={d['yolo_raw']} MC={d['yolo_raw_mc']} "
                  f"cls_reject={d['cls_reject']}(MC={d['cls_reject_mc']}) "
                  f"bbox_reject={d['bbox_reject']} class_area={d['class_area_reject']} "
                  f"cross_nms={d['cross_suppress']} "
                  f"passed={d['passed']} MC survival={mc_survival:.0f}%")

        return raw_detections

    def print_diag(self):
        d = self._diag
        print(f"[DIAG FINAL] {d['frames']} inf frames:")
        print(f"  YOLO raw:              {d['yolo_raw']} ({d['yolo_raw_mc']} MC)")
        print(f"  Per-class filter:      {d['cls_reject']} ({d['cls_reject_mc']} MC)")
        print(f"  Bbox size/aspect:      {d['bbox_reject']}")
        print(f"  Class-area mismatch:   {d['class_area_reject']}")
        print(f"  Ignore-zone filter:    {d['ignore_zone_reject']}")
        print(f"  Cross-class NMS:       {d['cross_suppress']}")
        print(f"  Passed to tracker:     {d['passed']}")
        print(f"  Track births:          {d['track_created']}")
        print(f"  Births near unseen ID: {d['track_near_disappeared']} "
              f"(within relink window/radius)")
        total_reject = (d['cls_reject'] + d['bbox_reject'] + d['class_area_reject']
                        + d['ignore_zone_reject'] + d['cross_suppress'])
        print(f"  Total rejection rate:  {100*total_reject/max(d['yolo_raw'],1):.0f}%")
        if d['yolo_raw_mc']:
            mc_survive = d['yolo_raw_mc'] - d['cls_reject_mc'] - d['cross_suppress']
            print(f"  MC pipeline: {mc_survive}/{d['yolo_raw_mc']} = {100*mc_survive/d['yolo_raw_mc']:.0f}% survival")
        print(f"  KF update rate by category (how often a detection actually")
        print(f"  corrected the filter, vs. predict-only coasting):")
        for cat in sorted(self._kf_update_attempts):
            attempts = self._kf_update_attempts[cat]
            passed = self._kf_update_passed[cat]
            rate = 100 * passed / attempts if attempts else 0
            print(f"    {cat}: {passed}/{attempts} = {rate:.0f}%")
        print(f"  Track lifespan range:  {min(self._track_created_at.values()) if self._track_created_at else 0}-"
              f"{max(self._track_last_seen.values()) if self._track_last_seen else 0}")

    def print_model_metrics(self):
        """Print the three per-model characterization numbers at end of run.

        These are the inputs needed to fill in real starting values for
        config.MODEL_TUNING_PROFILES instead of the current estimates:
          1. % of inference frames with zero detections (coasting gaps)
          2. Mean per-frame confidence of matched detections
          3. Variance of straight-line centroid displacement (noise floor)
        """
        m = self._metrics
        n_inf = m["inference_frames"]
        empty_pct = (m["empty_inference_frames"] / n_inf * 100.0) if n_inf else 0.0
        avg_conf = (m["matched_conf_sum"] / m["matched_conf_n"]) if m["matched_conf_n"] else 0.0
        samples = m["disp_samples"]
        if len(samples) >= 2:
            mean = sum(samples) / len(samples)
            var = sum((s - mean) ** 2 for s in samples) / (len(samples) - 1)
            std = var ** 0.5
        else:
            mean = var = std = float("nan")
        print("\n" + "=" * 70)
        print(f"MODEL CHARACTERIZATION — {os.path.basename(str(self.model.ckpt_path or config.MODEL_PATH))}")
        print(f"  1. Coasting-gap frequency : {empty_pct:.1f}%  "
              f"({m['empty_inference_frames']}/{n_inf} inference frames with zero detections)")
        print(f"  2. Mean matched confidence : {avg_conf:.3f}  "
              f"({m['matched_conf_n']} matched detections)")
        print(f"  3. Straight-line centroid  : mean={mean:.1f} px/frame, "
              f"std={std:.1f}, var={var:.1f} px²  ({len(samples)} samples)")
        print("=" * 70 + "\n")

    def track_stream(self, video_path):
        """
        Yields (frame_idx, frame_bgr, detections) for every frame.

        Uses model.predict() + a lightweight centroid tracker for ALL frame
        rates (FRAME_SKIP=1, 2, 3, …).  This avoids the known Ultralytics bug
        where model.track(source=numpy_array) internally saves to a temp
        PNG file and then tries to load it, triggering "Cannot read
        'image.png' (this model does not support image input)" — the
        predict() path handles numpy array inputs correctly in all backends.

        When FRAME_SKIP > 1, detection only runs every Nth frame and the
        lightweight tracker uses velocity prediction to bridge gaps.

        detections: list of dicts:
            track_id (int), category (str), centroid (x, y), bbox (x1,y1,x2,y2), conf (float), age (int)

        frame_bgr is always the ORIGINAL frame (not the enhanced one).
        """
        cap = open_video(video_path)

        frame_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_idx % config.FRAME_SKIP == 0:
                    raw_dets = self._inference_frame(frame)
                    # ── Metric: coasting-gap frequency ───────────────
                    self._metrics["inference_frames"] += 1
                    if not raw_dets:
                        self._metrics["empty_inference_frames"] += 1
                    frame_gap = config.FRAME_SKIP
                    detections = self._match_detections(raw_dets, frame_gap, frame_idx)
                    self._kalman_step(detections, frame_idx)
                    detections = self._filter_static_tracks(detections, frame_idx)
                    self._update_active_tracks(detections, frame_idx)
                    self._smooth_centroids(detections)
                    self._smooth_bboxes(detections)
                    # Update active tracks with smoothed positions so that
                    # predictions on non-inference frames are consistent
                    # with what the display just showed.
                    for d in detections:
                        tid = d["track_id"]
                        if tid in self._active_tracks:
                            self._active_tracks[tid]["centroid"] = d["centroid"]
                            self._active_tracks[tid]["bbox"] = d["bbox"]
                else:
                    detections = self._predict_detections(frame_idx)

                yield frame_idx, frame, detections
                frame_idx += 1
        finally:
            self.print_model_metrics()
            release_video(cap)
