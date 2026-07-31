"""
Interactively navigate to the exact frame you want, then click 2 points per
line to define START_LINES and FINISH_LINES, then print the config snippet.

Usage:
    python tools/pick_zone.py --video path/to/survey.mp4

Step 1 — Frame navigation:
    'd' / Right arrow  - jump forward 30 frames
    'a' / Left arrow   - jump back 30 frames
    '.'                - step forward 1 frame
    ','                - step back 1 frame
    ENTER / SPACE      - lock in the frame and start drawing lines
    'q'                - quit

Step 2 — Drawing lines:
    Left click   - add a point to the current line (exactly 2 needed)
    Right click  - close the line and prompt for a label
                   (S/N/E/W for start lines, N1/S2/E4/W4 for finish lines)
    'q'          - finish and print all completed lines
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aitss.zone_picker import pick_lines_interactive, terminal_label_prompt

VIDEO_EXTENSIONS = ("*.mp4", "*.mov", "*.avi", "*.mkv")


def suggest_nearby_videos():
    found = []
    for pattern in VIDEO_EXTENSIONS:
        found.extend(glob.glob(pattern))
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--start-frame",
        type=int,
        default=50,
        help="Frame index to start browsing from (default 50).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.video):
        nearby = suggest_nearby_videos()
        msg = f"Could not find video file: '{args.video}'\n"
        msg += "Check that: (1) the filename is EXACT, including the extension, "
        msg += "and (2) you're running this from the folder that actually contains the video.\n"
        if nearby:
            msg += "Video files found in the current folder:\n"
            for f in nearby:
                msg += f"  {f}\n"
        else:
            msg += "No video files found in the current folder."
        raise SystemExit(msg)

    lines = pick_lines_interactive(args.video, start_frame=args.start_frame,
                                    label_prompt_fn=terminal_label_prompt, status_fn=print)

    if not lines or (not lines.get("start_lines") and not lines.get("finish_lines")):
        print("No lines defined. Nothing to save.")
        return

    print("\n# Paste into config.py replacing the existing START_LINES / FINISH_LINES:\n")
    if lines.get("start_lines"):
        print("START_LINES = {")
        for label, pts in lines["start_lines"]:
            print(f'    "{label}": [{pts[0]}, {pts[1]}],')
        print("}")
    print()
    if lines.get("finish_lines"):
        print("FINISH_LINES = {")
        for label, pts in lines["finish_lines"]:
            print(f'    "{label}": [{pts[0]}, {pts[1]}],')
        print("}")


if __name__ == "__main__":
    main()
