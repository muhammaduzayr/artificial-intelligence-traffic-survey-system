# AITSS — Counting/Crossing Logic: Findings & Roadmap

Context for whoever picks this up: AITSS is a traffic-survey pipeline (`detector.py` +
`zone_counter.py` + `main.py` + calibration tools `pick_line.py`/`pick_zone.py`/`zone_picker.py`)
that detects and tracks vehicles with YOLO, counts them crossing manually-drawn
`START_LINES`/`FINISH_LINES` per camera, and exports a turning-movement-count (TMC) Excel report
(`Counting_Custom_Rev1.xlsx` template — North/East/South/West sheets, movement codes like `N1`-`N4`,
vehicle classes C/L/LL/B/M, 15-min bins). This doc is the output of a code review plus a design
session for one specific improvement (auto-suggesting where to draw lines). Two files referenced
by `main.py` — `aggregator.py` and `tmc_export.py` — were not reviewed because they weren't shared;
re-check item 1 below once they're available.

## 1. Priority fixes (found in review of config.py, detector.py, main.py, video_io.py,
##    zone_counter.py, zone_picker.py, pick_line.py, pick_zone.py, track_quality.py)

### 1.1 [HIGH] Origin is tracked but discarded — destination label alone decides the movement
`ZoneCounter._crossed_start` stores `(start_line_name, cross_point)` per track, but
`_count_track` builds `CountEvent` using only the finish line's name for both `lane` and
`direction` (see the module docstring: "direction = the finish line's key"). The TMC export
step then parses that finish-line name for the `N/S/E/W` + digit pattern (per the warning
string in `main.py`: "these zone labels didn't match the N/S/E/W + 1-4 pattern") to decide the
arm-sheet and movement column — it never checks that the vehicle actually crossed the
corresponding start line.

**Fix:** add the actual start-line name to `CountEvent` (new field, e.g. `origin`). Either
cross-check it against the finish line's expected origin (parsed from the finish label's
first letter) and flag/log a mismatch, or at minimum surface it in the diagnostic CSV so
mismatches are auditable instead of silently absorbed into a possibly-wrong count.

Also add `origin` to `_write_diagnostic_row`'s CSV columns — right now the diagnostic CSV can't
answer "did a South-origin vehicle ever get counted as N1" even by hand.

### 1.2 [MEDIUM] Calibration tools have no conflict-zone awareness
`pick_line.py` and `pick_zone.py`/`zone_picker.py` let you click line endpoints anywhere on the
frame with no warning if that point sits inside a high-occlusion junction/roundabout area.
Placing a finish line there is the single most common calibration mistake (confirmed on the
Jln Bagan Jermal footage — the `N2` finish line sat right at a roundabout curve). Short-term
fix: visual reminder/overlay in the picker UI. Long-term fix: see Section 3 (auto-suggested
lines derived from real trajectory curvature, not eyeballing).

### 1.3 [LOW-MEDIUM] Dead/duplicate calibration path
`pick_line.py` writes to `config.COUNTING_LINES` (a list of `CountingLine` dataclasses) — a
structure nothing in `zone_counter.py` reads. The actual runtime only consumes
`START_LINES`/`FINISH_LINES`, produced by `pick_zone.py`/`zone_picker.py`. Either wire
`pick_line.py`'s output into the real pipeline or delete it — as-is it's a trap for whoever
runs it expecting it to calibrate the system.

### 1.4 [LOW] Calibration is incomplete
`config.py`'s global `START_LINES`/`FINISH_LINES` only define the North arm (`"N"` and `"N2"`).
Not a bug if this is mid-setup, but the other three arms need lines before a real 4-way count
can run from the global config (per-video `.zones.json` sidecars may already cover this —
confirm before assuming a gap).

### 1.5 [LOW] No homography / perspective correction
The only oblique-angle mitigation is using bbox bottom-center as the crossing point (documented
in `zone_counter.py`'s comment on "perspective-induced drift"). A real top-down warp before
line-crossing math would be more robust, especially for lines drawn far from the camera where
pixel-space distances compress. Not urgent — the bottom-center trick is a legitimate partial
fix — but worth a future pass if false positives cluster on far-side lines.

### 1.6 Confirmed OK — no action needed
Side-based crossing check with dead zone + segment-intersection fallback, `MIN_TRACK_AGE`
gating with retroactive back-check, `MIN_CROSSING_GAP_FRAMES` debounce, displacement-jump
rejection, category locking at start-crossing, track retirement after counting, and the
`_relink_orphans` ID-bridging safety net are all correctly designed and shouldn't be touched
without a specific reason.

## 2. New feature: system-suggested, human-confirmed line placement

**Goal:** replace "user clicks two points on a blank frame" with "system proposes candidate
start/finish lines from real observed vehicle trajectories; user reviews, drags, relabels."
This also directly fixes 1.2, because the suggestion step actively avoids curving/merging
regions instead of relying on a human's eyeballed guess.

No ML/training data needed for this feature (unlike the detector, which is a legitimate CVAT/
fine-tuning candidate — see Section 3). This is closed-form geometry + one unsupervised
clustering call, computed fresh from each camera's own footage, not learned from labeled
examples.

### 2.1 Pipeline

1. **Harvest trajectories.** Run `VehicleDetector.track_stream()` over a sample window (5-10 min)
   with no counting logic attached — a lightweight collector (structurally a stripped-down
   `ZoneCounter`) accumulates each track's bottom-center points into
   `{track_id: [(x,y), ...]}`. Reuse `config.STATIC_MOTION_THRESHOLD_PX` /
   `STATIC_NET_DISPLACEMENT_PX` to drop non-moving/noise tracks. Require a minimum point count
   (~15) before a trajectory is eligible.

2. **Extract entry/exit descriptors** per trajectory, using the same window size k as
   `zone_counter._segment_vector` (`k = min(3, n//3 or 1)`):
   - Entry point `E` = mean of first k points; entry heading `d_E` = normalize(mean(points[k:2k]) − E)
   - Exit point `X` = mean of last k points; exit heading `d_X` = normalize(X − mean(points[-2k:-k]))

3. **Cluster** on the combined descriptor `(E, d_E, X, d_X)` using
   `sklearn.cluster.DBSCAN(metric="precomputed")` over a custom distance:

   ```
   d(i,j) = sqrt( w_pos * (||Ei-Ej||^2 + ||Xi-Xj||^2)
                + w_ang * (angle(dEi,dEj)^2 + angle(dXi,dXj)^2) )
   ```

   angle(a,b) = acos(a·b) in [0,π]. Scale `w_ang` so a 180° disagreement contributes comparably
   to a "large" positional gap, e.g. `w_ang = (L/π)^2` with L ≈ typical lane width in px
   (40-80px). `eps` ≈ that same lane-width scale; `min_samples` ≈ 5-10 so one mistracked
   vehicle can't manufacture a fake movement. Each cluster = one real observed movement.

4. **Fit a candidate line per cluster** from its entry points (and separately from its exit
   points for the finish line):
   - Mean heading `d̄` = normalize(Σdᵢ); perpendicular `n = (−d̄_y, d̄_x)`
   - Centroid `C` = mean(Eᵢ); project each point: `tᵢ = (Eᵢ−C)·n`
   - Line endpoints: `C + (min(tᵢ)−margin)·n` to `C + (max(tᵢ)+margin)·n` (margin ~10-20px,
     since centroids sit near lane center and understate true road width)

5. **Curvature-based conflict-zone avoidance.** Resample the cluster's trajectories to fixed
   arc-length steps (`numpy.interp` on cumulative distance) and average corresponding steps
   into one representative centerline `c(s)`. Local curvature:
   `κ(s) = |angle(heading(s−δ→s), heading(s→s+δ))| / (2δ)`. Starting from the candidate line's
   position, walk along `c(s)` away from the counting zone (upstream for start lines,
   downstream for finish) until κ(s) stays below a threshold for a sustained run (30-50px), not
   just one low sample. Re-fit the line (step 4) centered at that corrected position. Ground the
   curvature threshold and "sustained run" distance in real measurements from known
   straight/turning stretches on actual footage before trusting a default number.

6. **Human confirms.** Extend `zone_picker.py`'s interactive window: open with the trajectory
   cloud drawn faintly (2D density accumulation, one `cv2`/numpy histogram) and candidate lines
   pre-drawn/draggable, instead of a blank frame. The one thing that must stay manual: labeling
   which arm/movement each line is (`S`, `N1`, etc.) — only a human knows the compass
   orientation and what each movement code is supposed to mean. Save through the existing
   `save_zones()` — `.zones.json` format and `ZoneCounter` don't change at all.

### 2.2 Tools/libraries
`numpy` (all the geometry — projections, resampling, curvature), `scikit-learn` (DBSCAN only —
the one new dependency), `cv2` (already used — sample collection + review UI). No new ML model,
no GPU beyond what detection already uses, no homography required for this feature specifically.

## 3. Where ML/CVAT-labeled training data actually helps (separate track of work)
Not the line-placement problem above — that's closed-form. The legitimate ML investment is
improving the **detector's** class resolution, already flagged in `config.py`'s own comments:
COCO's single "truck" class gets bucketed into `Light Truck` by default and split into `Truck`
via a bbox-area heuristic (`TRUCK_REFINE_AREA_THRESHOLD`) rather than real classification.
`config.py` literally says: "swapping this for a custom-trained model (Phase 2+) will let you
split light vs heavy." That's the right place for CVAT-labeled data and fine-tuning — plus any
other class the current COCO-based detector approximates rather than truly resolves (e.g.
motorcycle detection at extreme angles, bus/light-truck confusion).
