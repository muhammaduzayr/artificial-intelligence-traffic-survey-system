# Custom Model Training + Appearance Re-ID Session — 2026-08-02 to 2026-08-05

Working notes for picking this back up in a fresh conversation. This session directly followed up on two items `TUNING_NOTES.md` had flagged as deferred: gap #4 (custom YOLO training) and gap #2 (appearance-based re-ID). Both are now built. **One real bug from the re-ID work is still open — see "Known bug, not yet fixed" below, that's the immediate next thing to do.**

## What happened this session, in order

1. **Built a full labeling pipeline.** Extracted frames from every video in `Desktop/atur traffic/` (132 source videos — S49, L04 F1/F2, L18 F1/F2, all recorded days), auto-labeled them with `yolo12m.pt`, and stood up a local CVAT instance (Docker Desktop + WSL2, freshly installed this session) to hand-correct them. CVAT runs at `http://localhost:8080`, login `admin` / `AitssCvat2026!`, containers managed via `docker compose` in `Desktop/cvat/`.
2. **User hand-corrected all 8 CVAT tasks** (1,626 images total, every camera/angle/day). Exported via `cvat_sdk` into `training_data/dataset_v4/` (1,383 train / 243 val, YOLO format, 4 classes: Motorcycle/Car/Bus/Truck).
3. **Fine-tuned `yolov8n.pt` on the full corrected dataset** → `training_data/runs/v4_dataset_v4/weights/best.pt`. Survived two mid-training interruptions (a laptop shutdown, a user pause/resume) by resuming from `last.pt` each time — see `training_data/resume_training.py`. Final validation: overall mAP50-95 0.642 (Car 0.82, Bus 0.85, Motorcycle 0.38, Truck 0.53 — Truck/Motorcycle are the weak classes, more corrected data would help most there).
4. **Made it the production default.** Copied to `aitss/aitss_v4.pt`, set as `config.MODEL_PATH`. Old baseline (`yolov8n.pt`) still available via `--model yolov8n.pt`.
5. **Found and fixed a real correctness bug this surfaced.** `detector.py` had hardcoded COCO class-id assumptions (id 2=Car, 3=Motorcycle, 5=Bus, 7=Truck) baked into `VEHICLE_CLASS_IDS`, `COCO_TO_CATEGORY`, `MIN_CONF_BY_CLASS`, `CLASS_MAX_BBOX_AREA`, `CLASS_MIN_BBOX_AREA`, `CLASS_ASPECT_RANGE`, and the truck-refine check. `aitss_v4.pt` has its own head (0=Motorcycle, 1=Car, 2=Bus, 3=Truck) — naively swapping the checkpoint silently dropped Car/Motorcycle entirely (filtered out by the id-based `classes=` filter) and mislabeled Bus as Car, Truck as Motorcycle. **Fixed**: `VehicleDetector.__init__` now reads `self.model.names` and picks the right id→category mapping at runtime (`self._cls_id_to_category`); every downstream config lookup (`MIN_CONF_BY_CLASS` etc.) is now keyed by **category name**, not raw id, so it works for any checkpoint. Confirmed with a real before/after test on held-out footage: coasting-gap frequency 35%→0%, mean confidence 0.29→0.61.
6. **Added a real tuning profile for `aitss_v4.pt`** in `config.MODEL_TUNING_PROFILES`, grounded in an actual `print_model_metrics()` characterization run (not just inference/estimate like the other profiles) — see the comment block there for the numbers. Flagged as first-pass/not fully validated (single clip).
7. **Scoped and built appearance-based re-ID** (`TUNING_NOTES.md` gap #2 — was the last unsolved piece of the original "centroid jumping" complaint):
   - `detector.py`: `_compute_appearance()` — cheap HSV color-histogram crop descriptor, computed once per real detection, attached as `det["appearance"]`.
   - `detector.py`: `relink_kf(old_tid, new_tid)` — meant to carry Kalman-filter state across a confirmed relink so the *displayed* position doesn't snap, only the identity does.
   - `zone_counter.py`: `_relink_orphans` now has a third gate (on top of distance + category): appearance similarity via `cv2.compareHist(..., HISTCMP_CORREL)`, threshold `config.ZONE_RELINK_MIN_APPEARANCE_SIMILARITY = 0.5`. Calls `relink_kf_fn` (wired to `detector.relink_kf` in `main.py`) on a confirmed relink.
   - Also fixed a real bug found in passing: `crossing_diagnostics.csv` was opened in append mode with nothing ever truncating it between runs — if you pointed two runs at the same `--output-dir`, rows from both runs silently mixed together (confusing, looked like impossible data e.g. one track_id with two different categories/timestamps). Now cleared at the start of every `ZoneCounter.__init__`.
8. **Removed confirmed dead code** (verified zero callers, repo-wide grep, before and after): `detector.py`'s `_apply_detection_filters` and `print_diag`, `zone_counter.py`'s `_crossing_point`.
9. **User tested the re-ID build and reported two symptoms**: centroid "blinking and flickering" (worse than before), and miscounting. Diagnosed both (see below) but **did not fix yet** — that's the next session's first task.

## Known bug, not yet fixed — do this first

### Bug 1: `relink_kf` overwrites a fresh, correct KF with a stale one (causes the flicker)

**Root cause**: `main.py`'s loop calls `detector.track_stream()` (which runs the full per-frame KF step, creating/updating `self._kf[tid]` for every track *including brand-new ones*) **before** calling `counter.update()` (where `_relink_orphans` decides whether to bridge an identity and calls `relink_kf`). So by the time `relink_kf(old_tid, new_tid)` runs, `self._kf[new_tid]` already exists and is already correctly positioned at the new track's real, current detection. `relink_kf` then blindly does `self._kf[new_tid] = self._kf.pop(old_tid)` — overwriting that correct state with the *old* track's KF, which reflects wherever the vehicle was up to `ZONE_RELINK_MAX_FRAMES` (45 frames) in the past. The transferred `_kf_last_frame[new_tid]` is also stale, so the next `predict(dt)` call uses a large `dt` and can way overshoot — then the next real detection yanks it back via the clamped update. Stale jump forward, snap back = the flicker.

**Fix**: don't overwrite position. `relink_kf` should keep the new track's already-correct current position/state, and only transfer *velocity* from the old track (for motion continuity across the identity swap), and set `_kf_last_frame[new_tid]` to the *current* frame_idx (not the old one). Concretely, in `detector.py`:

```python
def relink_kf(self, old_tid, new_tid):
    if old_tid in self._kf and new_tid in self._kf:
        old_vx, old_vy = self._kf[old_tid].x[2], self._kf[old_tid].x[3]
        self._kf[new_tid].x[2] = old_vx
        self._kf[new_tid].x[3] = old_vy
        # do NOT touch self._kf[new_tid].x[0:2] (position) or _kf_last_frame —
        # those are already correct from this frame's real detection.
    self._kf.pop(old_tid, None)
    self._kf_last_frame.pop(old_tid, None)
    self._kf_smooth.pop(old_tid, None)
    # _track_heading / _track_area_history: still fine to transfer as before,
    # they don't have this staleness problem.
    if old_tid in self._track_heading:
        self._track_heading[new_tid] = self._track_heading.pop(old_tid)
    if old_tid in self._track_area_history:
        self._track_area_history[new_tid] = self._track_area_history.pop(old_tid)
```

Handle the case where `new_tid` isn't in `self._kf` yet too (shouldn't happen given the call ordering above, but don't assume — fall back to the old behavior of just moving the old KF over if so).

### Bug 2 (likely cause of miscount): `ZONE_RELINK_MIN_APPEARANCE_SIMILARITY = 0.5` is probably too strict

Flagged as an unvalidated guess when added. A plain HSV histogram is noisy — a quick sanity check this session showed same-vehicle-ish Car-vs-Car similarity ranging all over 0.13–0.83, so a fixed 0.5 floor plausibly rejects a lot of genuine same-vehicle relinks (different angle/lighting after an occlusion), which means that vehicle's identity never gets bridged and it silently doesn't get counted — reads as undercounting. Next step: lower the threshold (try 0.3 first) and/or re-test to see if miscounting improves before/after, since we don't have ground-truth counted totals to validate against yet.

## Files changed this session
- `aitss/aitss/config.py` — `MODEL_PATH`/`AVAILABLE_MODELS` point at `aitss_v4.pt`; `MIN_CONF_BY_CLASS`/`CLASS_MAX_BBOX_AREA`/`CLASS_MIN_BBOX_AREA`/`CLASS_ASPECT_RANGE` re-keyed to category names; added `aitss_v4.pt` tuning profile; added `ZONE_RELINK_MIN_APPEARANCE_SIMILARITY`.
- `aitss/aitss/detector.py` — dynamic `_cls_id_to_category` resolution in `__init__`; `relink_kf()` (needs the fix above); `_compute_appearance()`; removed `_apply_detection_filters`, `print_diag`.
- `aitss/aitss/zone_counter.py` — appearance gate + `relink_kf_fn` call in `_relink_orphans`; diagnostic-CSV truncation in `__init__`; removed `_crossing_point`.
- `aitss/aitss/main.py` — wires `relink_kf_fn=detector.relink_kf` into `ZoneCounter(...)`.
- `aitss/aitss_v4.pt` — the production checkpoint (new file, not from git).
- `training_data/` — whole new directory (gitignored): `dataset_v4/`, `runs/v4_dataset_v4/`, `bootstrap/` (extracted frames + original auto-labels), `resume_training.py`/`.bat`, `train_v1.py`.

## Repeatable test setup (same one used throughout this session)
Held-out clip (never in any training set — L18 F2, a camera with no `.zones.json`, so counts will read 0/unreliable but detection/tracking quality is fully visible):
```bash
ffmpeg -ss 300 -i "<L18 F2 source video>" -t 90 -c:v libx264 -preset veryfast -crf 20 -an clip.mp4
```
Then:
```bash
python -m aitss.main --video clip.mp4 --start-time "2026-08-05 10:00:00" --model aitss_v4.pt --debug-video --output-dir out/
```
Check the `MODEL CHARACTERIZATION` block (coasting-gap %, mean confidence, centroid variance) and `[relink]`/`relinked=` counts in the console output, and pull matching frame numbers from `out/debug_annotated.mp4` before/after a change to compare directly.

## Docker/CVAT
Still running from this session — `docker compose` in `Desktop/cvat/` manages it, restarts automatically with Docker Desktop. All 8 tasks' corrected annotations live in CVAT's Postgres volume; `training_data/dataset_v4/` is just an exported snapshot. If more correction happens later (there's more Truck/Bus data worth adding per the validation numbers above), re-export via the same `cvat_sdk` approach used this session and retrain.
