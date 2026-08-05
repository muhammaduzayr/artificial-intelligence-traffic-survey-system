"""Generates a blank manual-tally CSV for validating a survey run's counts
against a human's own count of the same video.

Usage:
    python tools/ground_truth_template.py --video path/to/survey.mp4 [--output-dir out/]

Produces <output-dir>/ground_truth_template.csv with one row per
(lane, category) pair from that video's configured finish lines (falling
back to config.FINISH_LINES if the video has no .zones.json sidecar) —
matching exactly how the pipeline's own output is grouped, so the two are
directly comparable via compare_ground_truth.py.

How to use it: open the source video in any player, watch it at whatever
speed is comfortable, and manually tally each vehicle that crosses a
finish line into the matching (lane, category) cell's "manual_count"
column. This is deliberately independent of the pipeline itself — it's
the ground truth the pipeline's own numbers get checked against.
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aitss import config
from aitss.zone_picker import load_zones


def build_template(video_path, output_dir):
    _, finish_lines = load_zones(video_path)
    if finish_lines is None:
        finish_lines = config.FINISH_LINES
        print(f"No .zones.json for {video_path} — using global config.FINISH_LINES.")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "ground_truth_template.csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lane", "category", "manual_count"])
        for lane in finish_lines:
            for category in config.ALL_CATEGORIES:
                writer.writerow([lane, category, 0])

    print(f"Wrote {out_path}")
    print("Fill in the manual_count column while watching the source video, "
          "then run tools/compare_ground_truth.py against the pipeline's report.")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Path to the survey video (for its .zones.json)")
    parser.add_argument("--output-dir", default=".", help="Where to write ground_truth_template.csv")
    args = parser.parse_args()
    build_template(args.video, args.output_dir)


if __name__ == "__main__":
    main()
