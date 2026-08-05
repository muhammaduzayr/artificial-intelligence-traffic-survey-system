"""AITSS — AI Traffic Survey System core pipeline.

Detect -> track -> count -> report over a recorded traffic video. See
README.md for setup and usage; run `python -m aitss.main --help` for the
CLI, or `python run_gui.py` for the desktop GUI.

Public surface (import directly from these submodules, not from the
package root — several tools under aitss/tools/ intentionally avoid
pulling in the full detection stack):
    aitss.detector.VehicleDetector       — YOLO detection + custom tracker
    aitss.zone_counter.ZoneCounter       — start/finish line crossing counts
    aitss.aggregator.ReportAggregator    — 15-min bucketing + Excel/CSV export
    aitss.video_io.open_video            — lightweight OpenCV/PyAV video reader
"""
