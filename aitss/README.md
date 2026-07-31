# AITSS — Core AI Pipeline (v1)

This is the detect → track → count → report pipeline described in Section 3
and Section 9 of the AITSS proposal. It processes a recorded traffic video
and produces a 15-minute-interval Excel/CSV report.

## How it works

1. **Detection**: YOLOv8 (`ultralytics`) detects vehicles per frame (car,
   motorcycle, bus, truck — from COCO classes).
2. **Tracking**: Ultralytics' built-in ByteTrack assigns a persistent
   `track_id` to each vehicle across frames.
3. **Counting**: `LineCounter` watches each track's centroid and fires a
   count event the first time it crosses one of your configured virtual
   lines — one count per vehicle per line, with direction.
4. **Aggregation**: Count events are bucketed into 15-minute intervals and
   exported to Excel (with a summary sheet, raw events, and category
   totals/percentages) and CSV.

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
python tools/pick_line.py --video path/to/survey.mp4
```

Click two points per lane on the popup window, press `q` when done — it
prints ready-to-paste `CountingLine(...)` entries. Put them in
`aitss/config.py` under `COUNTING_LINES`.

## 2. Run the pipeline

```bash
python -m aitss.main --video path/to/survey.mp4 --start-time "2026-07-11 08:00:00" --debug-video
```

- `--start-time`: wall-clock time the recording started, so report
  intervals show real times (e.g. 08:00–08:15) instead of video-relative
  seconds.
- `--debug-video`: optional — saves an annotated MP4 with boxes, track IDs,
  and your counting lines drawn in, so you can visually verify accuracy.

Output lands in `output/`:
- `traffic_survey_report.xlsx` — 15-min summary, raw events, category totals
- `traffic_survey_report.csv` — 15-min summary only
- `debug_annotated.mp4` — if `--debug-video` was passed

## Known v1 limitations (matches the proposal's Section 7/8 scope)

- **Van / Light Truck / Heavy Truck aren't distinguished yet** — COCO's
  "truck" class is mapped to Light Truck as a placeholder. A custom-trained
  or fine-tuned model is needed to split these properly (natural Phase 2/3
  work, alongside speed estimation etc.).
- **No live/CCTV camera support** — that's Version 2 per the roadmap;
  this reads pre-recorded video files only.
- **Counting lines are set manually per video** — a UI for drawing lines
  inside the desktop app (PySide6) is the natural next step once this core
  pipeline is validated.
- **No confidence-based "flag for review" queue yet** — the proposal's
  "engineer only verifies uncertain detections" (Section 3) needs a
  low-confidence-detection review UI; the pipeline already captures `conf`
  scores per detection so this can be added on top without re-architecting.

## Project structure

```
aitss/
  aitss/
    __init__.py
    config.py          # model, counting lines, class mapping, interval
    detector.py         # YOLO + ByteTrack wrapper
    line_counter.py      # virtual line crossing logic
    aggregator.py         # 15-min bucketing + Excel/CSV export
    main.py                # CLI entry point
  tools/
    pick_line.py             # click-to-get counting line coordinates
  requirements.txt
  README.md
```
