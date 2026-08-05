# Tracking Tuning Session — 2026-08-01

Working notes for the centroid-jumping / occlusion / dense-traffic tuning pass, so this can be picked back up later without re-deriving context. Test footage used: `S49 F1 Simpangjlnbgnjermal-Jlntgtokong ...mp4` and `S49 P1 Spgjlnbgnjermal-Jlntgtokong ...mp4` in `Desktop/atur traffic/`.

## Original complaint

Centroid not static to vehicles — jumps/flickers, especially when:
- a vehicle's view is blocked by a tree/pole/fixed object
- there are too many vehicles on the same route or on opposite/crossing routes (dense traffic)

Also requested (deferred, see below): train YOLO to recognize every vehicle angle and route from the user's own footage.

## Environment change

- **CUDA-enabled PyTorch installed.** The venv had `torch==2.13.0+cpu` even though the machine has an NVIDIA MX450 (2GB VRAM, driver supports CUDA 13.0). Replaced with the matching `+cu126` build:
  ```
  pip install --index-url https://download.pytorch.org/whl/cu126 "torch==2.13.0+cu126" "torchvision==0.28.0+cu126"
  ```
  Verified: `torch.cuda.is_available() == True`, device = `NVIDIA GeForce MX450`.
- Benchmarked GPU inference at `imgsz=1280`: **~7 fps** on real footage (vs. much slower on CPU). Good enough for offline survey processing, not real-time at `FRAME_SKIP=1`.

## Root causes found and fixed

### 1. `config.KNOWN_OCCLUDERS` was empty (the big one for "blocked by tree/pole")
The detector already had a full mechanism for "this pixel region is a fixed obstruction — a track predicted to be there is hidden, not gone" (extends stale-track buffer 3x via `OCCLUDER_BUFFER_MULTIPLIER`, widens re-acquisition distance on the far side). Nobody had ever populated it, and it can't be a single global list anyway since each camera has different obstructions.

**Fix:** extended the existing per-video `.zones.json` sidecar mechanism (same file next to each video that already stores `start_lines`/`finish_lines`) to also carry `known_occluders`. New functions in `aitss/zone_picker.py`: `load_occluders(video_path)` / `save_occluders(video_path, occluders)`. `VehicleDetector.__init__` now takes an `occluders=` param (falls back to `config.KNOWN_OCCLUDERS` if not given); `main.py` loads per-video occluders the same way it already loads zones and passes them in.

Populated real coordinates by extracting frames from the actual videos and reading pixel positions off a grid overlay:
- **F1** (`S49 F1 ....zones.json`): street-light pole `(590,0,625,340)`, STOP-sign pole `(695,150,730,520)`, twin-head traffic signal `(785,115,875,270)`, traffic-signal pole/box near camera bottom-right `(1600,250,1920,650)`.
- **P1** (`S49 P1 ....zones.json`): median tree canopy + palm cluster `(780,10,1400,680)` — this is the big one, a large planted median directly between the two carriageways — plus a traffic-signal pole near camera bottom-right `(1750,750,1920,1080)`.

If you add cameras/movements later: run a frame through the same "extract + grid overlay" trick (see `scratchpad` approach below) to pick new pole/tree pixel boxes, then call `save_occluders()` or hand-edit the `.zones.json`.

### 2. `_filter_static_tracks`'s location-churn filter was rejecting real idling vehicles (the big one for "dense traffic flicker")
In `aitss/detector.py`, the static-false-positive filter keyed by coarse grid cell was supposed to catch a false detection "churning through many track IDs" at the same spot (foliage, glare). The actual implementation just counted **any** revisit to a 20px cell — 10 hits within 900 frames (~36s) — regardless of whether it was the same track ID or different ones. A real vehicle idling at a red light for >36s (extremely common in dense/queued traffic) revisits its own cell that often *by itself*, so it got its own legitimate detections rejected outright. That silently kills the track mid-life → new ID spawns when it's finally re-detected → visible "jump."

**Fix:** `loc_hist` now stores `(frame_idx, track_id)` pairs, and rejection only fires when **5+ distinct track IDs** show up in that window — actual churn, not one vehicle sitting still.

**Validated on the F1 dense-queue clip** (same clip, same timestamp, before vs. after): track IDs created by that point dropped from the ~1800s to the ~300s (**~5-6x fewer spurious new tracks**), and `relinked` count (track-identity-bridge events, a proxy for disruptive ID-swaps) nearly halved: 71 → 35.

### 3. No direction-consistency check for regular (pre-start-line) tracks
`_match_detections`'s cost function already penalized a candidate match that lies *behind* an established track's velocity vector — but only for "protected" tracks (already crossed a start line, since an ID swap there would lose the count). Ordinary tracks got no such check, so in dense or crossing traffic (opposite routes converging) a nearby vehicle's detection could look like a cheaper match than the correct one.

**Fix:** added a lighter version of the same check for all tracks with established velocity (`speed > 4px/frame`, to avoid triggering on queued-vehicle jitter): a detection lying behind the track's motion vector gets a small `+0.12` cost penalty instead of the strict `0.35` used for protected tracks.

### 4. `FRAME_SKIP` lowered 3 → 2
Enabled by the GPU speed. Smaller gap between real detections means the Kalman filter spends less time coasting on prediction alone before the next correction — directly shrinks how far a track can drift before snapping back, which is what "jumping" looks like on screen. Comment in `config.py` explains the tradeoff (drop to 1 if you have GPU headroom to spare, ~3.5 fps at imgsz=1280 on this card; raise back to 3 if ever running on CPU only).

## Files changed
- `aitss/aitss/config.py` — `FRAME_SKIP` 3→2 (with explanatory comment).
- `aitss/aitss/detector.py` — `VehicleDetector.__init__` takes `occluders=`; occluder check call site uses `self.occluders`; location-churn filter fix; direction-consistency penalty for regular tracks.
- `aitss/aitss/zone_picker.py` — added `load_occluders()` / `save_occluders()`; `save_zones()` now preserves an existing `known_occluders` entry instead of clobbering it.
- `aitss/aitss/main.py` — loads per-video occluders alongside zones, passes to `VehicleDetector`.
- `Desktop/atur traffic/S49 F1 ....zones.json` and `S49 P1 ....zones.json` — added `known_occluders` arrays (real user data, outside the repo).

None of this has been committed to git yet — it's all local working-tree changes.

## Validation method (repeatable)

Full hour-long survey videos are too slow to iterate on directly. Used short ffmpeg stream-copy clips instead (fast, lossless, no re-encode):
```bash
ffmpeg -ss <start_sec> -i "<source video>" -t 200 -c copy "<clip>.mp4"
cp "<source video>.zones.json" "<clip>.zones.json"   # keep the same start/finish/occluder config
```
Then run the real pipeline with debug video on the clip:
```bash
python -m aitss.main --video "<clip>.mp4" --start-time "<any ISO timestamp>" --debug-video --output-dir "<some out dir>"
```
Compare `REJECTED-LOCATION` / `relinked` / `stale_cleanup` counts in the console log, and pull frames from `debug_annotated.mp4` at the same timestamp before/after a change to compare track IDs and box stability directly.

## Known remaining gaps / next steps

1. **Not yet run on the full-length videos** — only ~3-minute clips from each camera were validated. Do a full run on both `S49 F1` and `S49 P1` (and the third file, `S49_F1_SimpangJlnBgnJermal..._1.mp4`, which has no `.zones.json` yet and currently falls back to the global `config.START_LINES`/`FINISH_LINES`/`KNOWN_OCCLUDERS` — check whether that's actually the same camera framing as F1 before trusting those defaults for it) before treating results as final.
2. **P1's heaviest traffic window still showed a lot of relinks** (186 relinks against 123 tracked routes / 94 counted vehicles in a single 200s clip). The safety net (`_relink_orphans` in `zone_counter.py`) keeps the *count* correct across an ID swap, but doesn't prevent the visual jump when the swap happens. True fix would need appearance-based re-ID (a visual embedding per track, matched on re-acquisition) — a real feature addition, not a config/tuning change. Worth doing if visual smoothness under extreme density still matters after the above fixes.
3. **Route-change flagging looked noisy on queued/near-stationary vehicles** in the F1 debug video (many boxes flagged red/"!" for `route_changed` while barely moving at a red light). Likely direction-vector noise on near-zero-magnitude movement. Not fixed this session (cosmetic/diagnostic only, doesn't affect counts or the centroid position itself) — see `ROUTE_CHANGE_ANGLE_THRESHOLD` and `_vector_angle`'s `ma < 1 or mb < 1` guard in `zone_counter.py` if it turns out to matter.
4. **Custom YOLO training on the user's own footage** (multi-angle, multi-route) — explicitly deferred this session in favor of tuning first. No labeled dataset exists yet. Real path: extract frames from the survey videos → auto-label with the current model at high confidence → spot-check/correct a sample → fine-tune a checkpoint (yolo11n/yolo12n realistic on a 2GB MX450; batch size will be small) → iterate. Multi-hour-to-multi-day effort, not a quick pass.

## Handy paths from this session
- Test clips + debug videos + reports: `%TEMP%\claude\...\scratchpad\clips\` (session-scratch, not permanent — re-cut from source video if needed later).
- Frame grids used to pick occluder pixel coordinates: same scratchpad, `frames\` subfolder.
