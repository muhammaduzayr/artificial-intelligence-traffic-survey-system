"""
AITSS configuration.

START_LINES and FINISH_LINES are in pixel space of the source video.
Use `python -m aitss.tools.pick_zone` to click points on a frame.
"""

import os


# ---------------------------------------------------------------------------
# Detection / tracking
# ---------------------------------------------------------------------------

# Available YOLO model checkpoints.  Ultralytics auto-downloads any
# standard model name on first use if not found locally.
# Usage: set MODEL_PATH below, or override via `--model` (CLI) or the
#        Model combobox in the GUI.
AVAILABLE_MODELS = [
    "yolo12s.pt",   # YOLOv12 small (default, balanced, ~18 MB)
    "yolo12n.pt",   # YOLOv12 nano (fastest CPU, ~5 MB)
    "yolo12m.pt",   # YOLOv12 medium (most accurate, ~50 MB)
    "yolo11s.pt",   # YOLOv11 small
    "yolo11n.pt",   # YOLOv11 nano
    "yolov8s.pt",   # YOLOv8 small (fast CPU, ~22 MB)
    "yolov8n.pt",   # YOLOv8 nano (fastest CPU, ~6 MB)
]

MODEL_PATH = "yolov8n.pt"

# When both YOLO class IDs and track votes disagree, force the track to use the
# HIGHEST confidence single-frame detection as its category, instead of a
# smoothed vote.  This helps when YOLO briefly misclassifies a vehicle on
# early frames (e.g. distant Car → Motorcycle) and the vote locks the wrong
# answer before better evidence arrives.  Set True to enable "max conf wins"
# override at count time.
FORCE_MAX_CONF_CATEGORY = True

# Test-time augmentation: runs each frame through the model multiple
# times (flipped/scaled variants) and averages the results — a genuine
# accuracy boost (catches some detections a single pass misses), at a
# real cost of roughly 2-3x slower inference PER FRAME on top of
# whatever the model size above already costs. OFF by default because
# it makes CPU inference prohibitively slow. Turn on only if accuracy
# matters more than turnaround time (e.g. short clip, overnight run).
ENABLE_TTA = False

# GLOBAL floor passed straight to the model.  Raised from 0.01 to 0.03
# after diagnostics confirmed that 83% of raw YOLO output at conf=0.01 was
# sub-threshold noise immediately discarded by the per-class filter —
# processing 17139 boxes every 500 frames only to reject 14218 of them
# wastes tracker capacity and inflates false-positive detections on static
# scene elements (foliage, shadows, road markings).
# Motorcycle MIN_CONF also raised to 0.03 (same as global) — the ~174
# MCs/frame at conf=0.01-0.02 are predominantly noise detections that never
# survive min_track_age to be counted, so raising the floor costs nothing
# in true-positive counts.
CONFIDENCE_THRESHOLD = 0.03

# ByteTrack-style two-stage matching threshold.  Detections with confidence
# >= this value are matched first (high-conf pass), then the remaining tracks
# are offered to low-confidence detections.  Low-conf detections that still
# go unmatched are dropped — they never create new tracks, preventing ghost
# tracks from false-positive fragments.
BYTETRACK_HIGH_CONF_THRESHOLD = 0.50

# Per-class confidence floor.  The global threshold (0.03) gates YOLO output;
# these per-class floors are identical for MC (same as global) and slightly
# higher for larger vehicles where noise is less likely to be legitimate.
MIN_CONF_BY_CLASS = {
    2: 0.07,   # Car — 0.07 catches distant/blurry/night/angled cars (was 0.10)
    3: 0.03,   # Motorcycle — matches global floor; catches small/distant MCs (was 0.05)
    5: 0.30,   # Bus
    7: 0.30,   # Truck
}

# Confidence floor required before a detection is trusted enough to CORRECT
# a track's Kalman filter position estimate (vs. predict-only / coasting).
# Deliberately separate from MIN_CONF_BY_CLASS above, which only gates
# whether a detection is admitted to the tracker at all. A detection can
# be good enough to keep a track alive without being good enough to trust
# for a position correction.
#
# Per-class because motorcycles are legitimately detected at lower typical
# confidence than cars even when the detection is correct (smaller target,
# more often partially occluded by nearby vehicles). Gating every class at
# the same flat value (this used to be a hardcoded 0.3 in
# detector.py._detection_is_good for ALL classes) meant most real
# motorcycle detections - which routinely land between the 0.03 admission
# floor and 0.3 - never actually corrected the Kalman filter. The KF just
# coasted on stale/zero velocity for long stretches, which produced both
# symptoms seen in testing: the displayed centroid visibly "jumping" once
# a rare >0.3 detection finally forced a large correction, and motorcycles
# not registering line crossings because the KF point never actually
# tracked the real vehicle's position through the counting zone.
KF_UPDATE_MIN_CONF = {
    "Motorcycle": 0.12,
    "Car": 0.30,
    "Bus": 0.50,
    "Light Truck": 0.60,
    "Truck": 0.70,
}

# Bounding-box sanity filter: reject detections whose box is unrealistically
# small or has an extreme aspect ratio (tiny specks and thread-like boxes
# are virtually always noise from foliage / compression artifacts).
MIN_BBOX_WIDTH = 8
MIN_BBOX_HEIGHT = 8
MIN_BBOX_ASPECT = 0.2    # width/height lower bound (e.g. very tall thin boxes)
MAX_BBOX_ASPECT = 6.0    # width/height upper bound (e.g. very wide flat boxes)

# Bbox-area class constraints: reject physically impossible class+size
# combinations.  A 10×20 px box cannot be a Bus (too small), and a
# 100×200 px box cannot be a Motorcycle (too large).  These are loose
# bounds — they only catch extreme misclassifications, not borderline cases.
# Areas are in pixel² at the inference resolution (1920×1080 source).
CLASS_MAX_BBOX_AREA = {
    2: 250000,  # Car — a ~500×500 box at IMGSZ (largest plausible single car)
    3: 30000,   # Motorcycle — ~173×173 box at IMGSZ (up from 6000 which rejected close MCs)
    5: 500000,  # Bus — ~707×707 box at IMGSZ (covers nearly full frame height)
    7: 500000,  # Truck — same as Bus
}
CLASS_MIN_BBOX_AREA = {
    2: 40,      # Car — ~6×7 box at IMGSZ (was 60)
    3: 20,      # Motorcycle — ~4×5 box at IMGSZ (tiny distant MCs, was 30)
    5: 300,     # Bus — ~17×17 box (was 400)
    7: 300,     # Truck — same as Bus
}

# Static-object rejection (MAX PAIRWISE): if a track's centroid history's
# farthest two points are closer than this threshold, it is treated as a
# static false positive (foliage, debris, road furniture) and suppressed.
# Tightened from 6/4 to aggressively kill noise specks that flicker in the
# same pixel region without ever genuinely moving.
STATIC_MOTION_THRESHOLD_PX = 4
STATIC_NET_DISPLACEMENT_PX = 3
STATIC_MIN_SAMPLES = 6

# Permanent ignore zones for confirmed static false-positive hotspots.
# Coordinates are source-video pixels and apply to detection centroids before
# they enter matching/tracking. The first zone covers the recurring static
# hotspot observed around grid cells (47, 5) and (47, 6) in the CCTV sample.
# Grid cells are 20 px wide/high and the zone bounds are inclusive, so the
# exact range is x=940..959, y=100..139.
IGNORE_ZONES = [
    (940, 100, 959, 139),
]

# Per-class aspect-ratio sanity bounds: reject a detection when its
# bbox aspect ratio (width/height) is physically implausible for the
# assigned class.  A "Car" that is taller than it is wide (aspect < 0.5)
# is almost certainly a motorcycle misclassified as Car; a "Motorcycle"
# that is extremely wide (aspect > 3.0) is almost certainly a Car or
# Light Truck misclassified as MC.  None values = no constraint.
# Values widened for v2 to catch more camera angles (head-on cars,
# side-view motorcycles).
CLASS_ASPECT_RANGE = {
    2: (0.25, None),          # Car: lower 0.3→0.25 (head-on cars)
    3: (None, 3.0),           # Motorcycle: upper 2.5→3.0 (side-view MCs)
    7: (0.4, None),           # Light Truck: unchanged
}

# --- Route tracking ---

# Angle threshold (degrees) for detecting a route change.  The vehicle's
# centroid history is split into two halves (earlier vs later); if the
# movement vectors of the two halves differ by more than this angle, the
# track is flagged route_changed = True.  35 degrees catches turns and
# lane changes while ignoring mild weaving.
ROUTE_CHANGE_ANGLE_THRESHOLD = 35

# How many recent centroids to keep per track for route-change angle
# computation.  30 points at FRAME_SKIP=3 and 25 fps covers ~3.6 s of
# real time — enough to detect a turn without holding stale data.
ROUTE_CENTROID_WINDOW = 30


# Track ID retirement cooldown (seconds): used by the export-time double-count
# self-check. If the same track_id appears in raw_events twice within this
# window, a warning is logged with both event details.  Set to 30 minutes —
# well within a survey session, so any reuse is almost certainly a bug.
ID_RETIREMENT_COOLDOWN_SECONDS = 1800

# Road-area filter: disabled (full frame) — the pipeline no longer
# restricts detections to a sub-region, so vehicles entering from any
# angle or distance are picked up.  The existing bbox-size, aspect-ratio,
# class-vs-area, and static-motion filters are sufficient to reject
# false positives from sky/trees/foliage.
ROAD_Y_MIN = 0
ROAD_Y_MAX = 99999
ROAD_X_MIN = 0
ROAD_X_MAX = 99999

# Inference resolution. YOLO downsizes every frame to this before detecting.
# 960 is the default (good balance of speed vs accuracy).  Set 1280 for
# better distant/small motorcycle detection (~3× more distant MCs caught)
# at ~1.6× the wall time.  Override via --imgsz on CLI.
IMGSZ = 1280

# YOLO NMS IoU threshold.  0.7 is permissive (keeps overlapping boxes for
# dense groups like motorcycle clusters).  0.65 is slightly tighter,
# reducing near-duplicate boxes that confuse the tracker.  Highly unlikely
# to suppress legitimate detections — same-class overlap above 0.65
# usually means the same physical vehicle anyway.
YOLO_NMS_IOU = 0.65

# Cross-class NMS IoU threshold: if two detections of DIFFERENT classes
# (e.g. Bus + Motorcycle on the same physical vehicle) overlap by more
# than this, the lower-confidence one is suppressed.  0.5 is a moderate
# threshold — aggressive enough to catch same-vehicle double detections
# but permissive enough not to suppress nearby different-class vehicles
# (e.g. a car and motorcycle passing side-by-side).
CROSS_CLASS_NMS_IOU = 0.5

# Process every Nth frame for inference. 1 = process all frames.
# 2 = process every other frame (half the inference calls, ~2x faster).
# Motorcycles that only appear in skipped frames will be missed, but
# ByteTrack's track_buffer keeps existing tracks alive across gaps.
# Tune this based on your speed vs accuracy needs.
FRAME_SKIP = 3

# ByteTrack config — switched back from BoT-SORT after benchmarking.
# ByteTrack is significantly faster on CPU (no optical-flow GMC per
# frame), and with the tuned match_thresh (0.6, down from 0.8) and
# longer track_buffer (90) in aitss_bytetrack.yaml, it handles turning
# vehicles nearly as well as BoT-SORT without the per-frame compute
# overhead.  If you have GPU inference available, try aitss_botsort.yaml
# with with_reid: True for the best tracking through tight turns.
# aitss_botsort.yaml is left in place for GPU users.
#
# *** CURRENTLY NOT WIRED UP ***  detector.py does not call Ultralytics'
# model.track() (it explains why in its own module docstring — an
# OpenVINO/numpy-input bug), so this file — and aitss_bytetrack.yaml /
# aitss_botsort.yaml themselves — are not read by anything right now.
# Tracking is instead done by a small custom tracker inside detector.py
# (see VehicleDetector._match_detections). Tune track-association
# behavior there (and see config.ZONE_RELINK_* below for the
# counting-side safety net), not in these yaml files, until/unless
# detector.py is changed to actually use model.track().
TRACKER_CONFIG = os.path.join(os.path.dirname(__file__), "aitss_bytetrack.yaml")

# --- Speed / hardware ---

# "auto" picks an NVIDIA GPU (cuda:0) if torch can see one, else falls
# back to CPU — this is by far the single biggest speed lever available.
# CPU inference is typically 5-20x slower than GPU for the same model, so
# if processing feels slow, check whether this is actually landing on
# "cpu" (detector.py prints which one it picked, and a warning + fix
# pointer if it's CPU). Force "cpu" or "cuda:0" here to override.
DEVICE = "auto"

# Half-precision (fp16) inference — roughly 2x faster on a GPU with
# negligible accuracy impact. Ignored automatically when running on CPU
# (fp16 isn't faster there, sometimes slower).
USE_HALF_PRECISION = True

# --- Frame enhancement ---

# Applies CLAHE (adaptive contrast enhancement) to each frame before
# detection — essential for dark/low-light footage where motorcycles
# (small targets) blend into the road.
ENABLE_FRAME_ENHANCEMENT = True
CLAHE_CLIP_LIMIT = 5.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Unsharp-mask sharpening applied after CLAHE.  Enhances edges degraded
# by motion blur, compression artifacts, or upscaling — directly improves
# detection confidence for small/distant vehicles whose fine details
# would otherwise blend into the background.
ENABLE_SHARPEN = True
SHARPEN_STRENGTH = 0.3    # weight of the sharpened overlay; 0.3 is
                          # subtle enough not to amplify noise visibly
SHARPEN_SIGMA = 2.0       # Gaussian blur sigma for the mask

# Gamma correction applied before CLAHE on frames with low average
# brightness (night footage).  < 1.0 brightens shadows.
# 0.45 is very aggressive — lifts barely-visible night traffic into
# detectable range.
GAMMA_CORRECT_THRESHOLD = 100   # activate night gamma when LAB-L < 100
GAMMA_VALUE = 0.45

# COCO class ids we care about (COCO names in comments).
# 3 = motorcycle, 2 = car, 5 = bus, 7 = truck
VEHICLE_CLASS_IDS = [2, 3, 5, 7]

# COCO doesn't distinguish van / light truck / heavy truck out of the box.
# For v1 we map every COCO "truck" to LIGHT_TRUCK by default; swapping this
# for a custom-trained model (Phase 2+) will let you split light vs heavy.
COCO_TO_CATEGORY = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Light Truck",   # baseline; refined to "Truck" by bbox-area rule below
}

# Bbox area threshold (px² at 1280 inference resolution) for refining
# COCO class 7 ("Truck") into Light Truck vs Truck.  A detection whose
# bbox area exceeds this threshold is classified as "Truck" (heavy /
# large cargo truck, dump truck, etc.); below the threshold it stays as
# "Light Truck" (pickup, van, small delivery truck).  The threshold is
# automatically scaled for other inference resolutions via
# (imgsz / 1280) ** 2.
TRUCK_REFINE_AREA_THRESHOLD = 120000

ALL_CATEGORIES = ["Motorcycle", "Car", "Bus", "Light Truck", "Truck"]


# ---------------------------------------------------------------------------
# Counting (line-based)
# ---------------------------------------------------------------------------

# How many frames a track must be observed for before it's allowed to
# trigger a count (filters out short-lived/flickery phantom detections).
# With FRAME_SKIP=3 at 25 fps, each detection ≈ 3 frames = 0.12 s.
# MC=6 → ~0.72 s, default=8 → ~0.96 s of observation before counting.
MIN_TRACK_AGE = {
    "Motorcycle": 6,
    "default": 8,
}

# How many frames a track that crossed a start line can go without being seen
# before it is silently expired (not counted).  Tracks that take longer than
# this to cross a finish line are treated as lost.  120 frames at 25 fps
# with FRAME_SKIP=3 gives ~4.8 seconds of real time.
STALE_TRACK_TIMEOUT = 240

# Minimum real-frame gap between start-line crossing and finish-line crossing.
# This prevents false counts from centroid jitter — a vehicle must be tracked
# for at least this many frames BETWEEN the two events, proving it actually
# travelled through the counting zone.  3 frames at 25 fps with FRAME_SKIP=3
# = ~0.12 seconds (1 inference cycle).
MIN_CROSSING_GAP_FRAMES = 3

# Line-crossing stability gates. A centroid must leave this many pixels of
# perpendicular distance from a line before its side is considered stable.
# This creates hysteresis around the virtual line, so detector jitter does not
# repeatedly change sides while fast vehicles can still jump across cleanly.
COUNTING_LINE_DEAD_ZONE_PX = 4

# Minimum perpendicular movement required to accept a crossing. This rejects
# tiny side flips caused by detector noise without imposing a maximum speed.
MIN_CROSSING_PERP_MOTION_PX = 4

# Optional directional validation per line. Values are +1 or -1, where +1
# means motion in the line's left-hand normal direction (-dy, dx), and -1 is
# the opposite. Leave empty to accept traffic from either direction.
# Example: {"N": -1, "N2": 1}
LINE_CROSSING_DIRECTIONS = {}

# --- Track-identity bridging (counting-side safety net) ---
#
# A vehicle is only counted if the SAME track_id crosses a start line and
# later a finish line. If the tracker's internal ID switches mid-crossing
# (occlusion, a turn breaking the association, two vehicles passing close
# together) the vehicle would otherwise just silently vanish from the
# count with no error. ZoneCounter guards against this: when a track that
# already crossed a start line disappears, it's kept as an "orphan" for a
# short window; if a brand-new track_id then appears close to the
# orphan's last known position within that window, it's treated as the
# same physical vehicle continuing on, and the start-crossing state is
# carried over to the new ID.
#
# How many frames an orphaned start stays eligible to be re-linked to a
# new track_id.  Kept short and local (relative to STALE_TRACK_TIMEOUT)
# so it only bridges genuine short ID-switch gaps, not long absences that
# are more likely a different vehicle entirely.
ZONE_RELINK_MAX_FRAMES = 45

# How close (pixels, in source video space) a brand-new track_id's first
# detected position must be to an orphan's last known position to be
# considered the same vehicle.  Tightened relative to the tracker's own
# max_dist gate (150px * FRAME_SKIP) since this is a coarser, last-resort
# check running on sparser information (no bbox/IoU, just position) —
# keeping it tighter reduces the risk of merging two different vehicles.
ZONE_RELINK_MAX_DIST_PX = 130


# START_LINES — one line segment per entry approach, keyed by a label
# (e.g. "S" for south entry, "N" for north entry).  When a tracked
# vehicle's centroid crosses any of these lines, the system records the
# crossing and begins tracking it for a future count.
# Use python -m aitss.tools.pick_zone to click two endpoints per line.
START_LINES = {
    "N": [(754, 230), (794, 317)],
    "S": [(1548, 243), (1651, 354)],
}

# FINISH_LINES — one line segment per exit direction, keyed by the lane
# name that will appear in the Excel report (e.g. "N1", "S2", "E4", "W4").
# When a vehicle that crossed a start line later crosses one of these, it
# is counted with:
#   lane = the finish line's key (e.g. "N1")
#   direction = the finish line's key (the direction label IS the lane)
FINISH_LINES = {
    "S2": [(1137, 191), (1185, 266)],
    "N2": [(1252, 266), (1385, 439)],
}

# Aggregation interval in minutes (per the proposal: 15-minute reports).
INTERVAL_MINUTES = 15

# Where results get written (absolute path, so it's always
# <project-root>/output/ regardless of working directory).
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "output")

# Path to the official counting template (the exact Counting_Custom_Rev1.xlsx
# file) that tmc_export.py copies and fills in for every survey run. Put
# the real template file at this path — a "templates" folder next to
# config.py. tmc_export.py edits a COPY of this file directly via raw XML
# patching (not a normal openpyxl load+save), because openpyxl silently
# drops this template's per-sheet arrow-icon diagrams on save — this path
# preserves them exactly.
TMC_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "templates", "Counting_Custom_Rev1.xlsx")

# --- Per-model tuning profiles ---------------------------------------
# Different YOLO models produce detections at different density/noise
# characteristics (e.g. v11n detects more consistently than v8n at awkward
# angles, meaning fewer coasting gaps and more frequent small per-frame
# noise). The KF/EMA smoothing tuned for one model's detection pattern
# isn't automatically right for another's.  Keyed by model filename so
# switching models in the GUI automatically switches tuning too.
#
# Fields:
#   kf_smooth_alpha          EMA alpha for the display crosshair
#                            (centroid_kf_smooth).  Lower = smoother but
#                            more lag.  Display-only, never touches counting.
#   area_change_threshold    relative bbox-area change that rejects a KF
#                            update as an occlusion fragment / merged box.
#   kf_measurement_noise     base measurement noise (R) for the per-track
#                            Kalman filter.  Lower = filter trusts each
#                            detection measurement more; higher = smoother
#                            but slower to react to genuine motion.
#   bytetrack_high_conf_threshold
#                            ByteTrack-style two-stage matching split point.
#                            Detections with conf >= this are matched first
#                            and may spawn new tracks; lower-confidence
#                            detections only recover existing unmatched
#                            tracks and are dropped otherwise.  Too high
#                            silently discards real vehicles whose models
#                            calibrate confidence low (e.g. v11n on this
#                            footage); too low lets noise spawn ghost tracks.
MODEL_TUNING_PROFILES = {
    "yolov8n.pt": {
        "kf_smooth_alpha": 0.20,
        "area_change_threshold": 0.65,
        "kf_measurement_noise": 8.0,
        "bytetrack_high_conf_threshold": 0.50,
    },
    "yolov8s.pt": {
        "kf_smooth_alpha": 0.18,
        "area_change_threshold": 0.65,
        "kf_measurement_noise": 9.0,
        "bytetrack_high_conf_threshold": 0.50,
    },
    "yolo11n.pt": {
        "kf_smooth_alpha": 0.12,       # denser detections need heavier EMA damping
        "area_change_threshold": 0.65,
        "kf_measurement_noise": 12.0,  # trust each individual measurement a bit less
        # v11n's confidence calibration runs lower than v8n's for the same
        # true positives on this footage -- 0.50 was silently dropping real
        # vehicles before they could ever spawn a track (see detector.py's
        # "low-conf unmatched -> drop" branch).  Starting point; validate
        # against a real confidence histogram and adjust.
        "bytetrack_high_conf_threshold": 0.35,
    },
    "yolo11s.pt": {
        "kf_smooth_alpha": 0.12,
        "area_change_threshold": 0.65,
        "kf_measurement_noise": 12.0,
        "bytetrack_high_conf_threshold": 0.35,
    },
    "yolo12n.pt": {
        "kf_smooth_alpha": 0.12,
        "area_change_threshold": 0.65,
        "kf_measurement_noise": 12.0,
        "bytetrack_high_conf_threshold": 0.35,
    },
    "yolo12s.pt": {
        "kf_smooth_alpha": 0.12,
        "area_change_threshold": 0.65,
        "kf_measurement_noise": 12.0,
        "bytetrack_high_conf_threshold": 0.35,
    },
    "yolo12m.pt": {
        "kf_smooth_alpha": 0.15,
        "area_change_threshold": 0.65,
        "kf_measurement_noise": 10.0,
        "bytetrack_high_conf_threshold": 0.40,
    },
}

DEFAULT_TUNING_PROFILE = {
    "kf_smooth_alpha": 0.20,
    "area_change_threshold": 0.65,
    "kf_measurement_noise": 8.0,
    "bytetrack_high_conf_threshold": 0.50,
}


def get_tuning_profile(model_path):
    """Look up tuning constants for a given model file, by basename.

    Falls back to DEFAULT_TUNING_PROFILE with a warning for any model
    not yet profiled — so new/untested models still run, just without
    model-specific tuning until someone characterizes them.
    """
    name = os.path.basename(str(model_path))
    profile = MODEL_TUNING_PROFILES.get(name)
    if profile is None:
        print(f"WARNING: no tuning profile for '{name}' — using defaults. "
              f"Centroid smoothness may not be optimal until this model is profiled.")
        return DEFAULT_TUNING_PROFILE
    return profile
