# AITSS — Core AI Pipeline (v1)

This is the detect → track → count → report pipeline described in Section 3
and Section 9 of the AITSS proposal. It processes a recorded traffic video
and produces a 15-minute-interval Excel/CSV report.

## How it works

1. **Detection**: YOLO (`ultralytics`, default `yolov8n.pt`) detects
   vehicles per frame (car, motorcycle, bus, truck — from COCO classes).
2. **Tracking**: a custom Kalman-filter + IoU/distance tracker in
   `detector.py` assigns a persistent `track_id` to each vehicle across
   frames (not Ultralytics' built-in `model.track()` — see the
   `TRACKER_CONFIG` comment in `config.py` for why).
3. **Counting**: `ZoneCounter` (in `zone_counter.py`) watches each track's
   centroid and fires a count event when it crosses one of your configured
   start lines, then later one of your finish lines — one count per
   vehicle, with lane and direction.
4. **Aggregation**: Count events are bucketed into 15-minute intervals and
   exported to Excel (with a summary sheet, raw events, and category
   totals/percentages), CSV, and a turning-movement (N/E/S/W) workbook.

## Setup

```bash
cd aitss
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The first run will auto-download `yolov8n.pt` (~6MB) — needs internet once.

## 1. Define your counting lines

Every survey video has a different camera angle, so you need to set line
coordinates per video. Easiest way:

```bash
python tools/pick_zone.py --video path/to/survey.mp4
```

Navigate to the frame you want, then click 2 points per line to define
start and finish lines interactively (see the script's own docstring for
the full keyboard/mouse controls). It saves a per-video `.zones.json`
sidecar next to the video, which `main.py` picks up automatically —
falling back to the `START_LINES`/`FINISH_LINES` defaults in
`aitss/config.py` if no sidecar exists.

## 2. Run the pipeline

Either from the command line:

```bash
python -m aitss.main --video path/to/survey.mp4 --start-time "2026-07-11 08:00:00" --debug-video
```

...or with the desktop GUI (no command line needed) — pick a video, mark
zones, and run the pipeline from one window:

```bash
python run_gui.py
```

- `--start-time`: wall-clock time the recording started, so report
  intervals show real times (e.g. 08:00–08:15) instead of video-relative
  seconds.
- `--debug-video`: optional — saves an annotated MP4 with boxes, track IDs,
  and your counting lines drawn in, so you can visually verify accuracy.

Output lands in `output/` (or `output/<enumerator>/` if `--enumerator` was
passed):
- `traffic_survey_report.xlsx` — 15-min summary, raw events, category totals
- `traffic_survey_report.csv` — 15-min summary only
- `tmc_report.xlsx` — turning-movement count workbook (North/East/South/West)
- `crossing_diagnostics.csv` — one row per counted crossing, for auditing
  why a specific count did/didn't happen
- `debug_annotated.mp4` — if `--debug-video` was passed

Reports are written incrementally every 15 simulated minutes of video (and
on Ctrl+C), so a long run never loses everything counted so far.

## Known v1 limitations (matches the proposal's Section 7/8 scope)

- **Van / Light Truck / Heavy Truck aren't distinguished yet** — COCO's
  "truck" class is mapped to Light Truck as a placeholder. A custom-trained
  or fine-tuned model is needed to split these properly (natural Phase 2/3
  work, alongside speed estimation etc.).
- **No live/CCTV camera support** — that's Version 2 per the roadmap;
  this reads pre-recorded video files only.
- **Counting lines are set manually per video** — `run_gui.py` and
  `tools/pick_zone.py` both let you draw them interactively (see "Define
  your counting lines" above), but there's still no automatic detection of
  a sensible default per camera angle.
- **No confidence-based "flag for review" queue yet** — the proposal's
  "engineer only verifies uncertain detections" (Section 3) needs a
  low-confidence-detection review UI; the pipeline already captures `conf`
  scores per detection so this can be added on top without re-architecting.

## Project structure

```
aitss/
  run_gui.py                  # desktop GUI launcher (no CLI needed)
  requirements.txt
  README.md
  yolo*.pt                    # auto-downloaded YOLO checkpoints (gitignored)
  aitss/
    __init__.py
    config.py                 # model, counting lines, class mapping, tuning
    detector.py                # YOLO detection + custom Kalman/IoU tracker
    video_io.py                  # OpenCV/PyAV video reader wrapper
    zone_counter.py                # start/finish line crossing + counting logic
    zone_picker.py                   # per-video .zones.json load/save helpers
    aggregator.py                      # 15-min bucketing + Excel/CSV export
    tmc_export.py                       # turning-movement (N/E/S/W) workbook export
    main.py                               # CLI entry point
    templates/
      Counting_Custom_Rev1.xlsx           # TMC report template (tracked in git)
  tools/
    pick_zone.py               # interactive frame nav + click-to-draw zones
    track_quality.py               # dev tool: per-track detection quality stats
```
