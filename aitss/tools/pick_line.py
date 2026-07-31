import argparse
import os
import re

import numpy as np
import cv2

import sys

# Allow `python tools/pick_line.py` when launched from the aitss directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from aitss.video_io import open_video, release_video

SAVE_CONFIG_DEFAULT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "aitss", "config.py")
)

lines = []
current_points = []


def grab_usable_frame(video_path, start_frame=50, max_attempts=300):
    """
    Some codecs (notably HEVC/H.265) fail to decode frame 0 properly and
    return a blank/garbage frame instead of erroring out. Scan forward from
    `start_frame` until we find a frame with real image variation.
    """
    try:
        cap = open_video(video_path)
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame = None
    for _ in range(max_attempts):
        ok, candidate = cap.read()
        if not ok:
            break
        if np.std(candidate) > 15:
            frame = candidate
            break
        frame = candidate

    release_video(cap)
    return frame


def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))
        print(f"Point: ({x}, {y})")
        if len(current_points) == 2:
            lines.append(tuple(current_points))
            current_points.clear()


def find_matching_bracket(text, start_pos):
    depth = 0
    for i in range(start_pos, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                return i
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--start-frame",
        type=int,
        default=50,
        help="Frame index to start scanning from (helps skip broken HEVC first frames).",
    )
    parser.add_argument(
        "--config",
        default=SAVE_CONFIG_DEFAULT,
        help="Path to config.py to update when saving lines.",
    )
    args = parser.parse_args()

    frame = grab_usable_frame(args.video, start_frame=args.start_frame)
    if frame is None:
        raise SystemExit(
            "Could not find a usable frame. The video file may be corrupted, "
            "or try re-encoding it (see README troubleshooting section)."
        )

    cv2.namedWindow("pick_line - click 2 pts per line, q to finish")
    cv2.setMouseCallback("pick_line - click 2 pts per line, q to finish", on_click)

    while True:
        display = frame.copy()
        for a, b in lines:
            cv2.line(display, a, b, (0, 0, 255), 2)
        for p in current_points:
            cv2.circle(display, p, 4, (0, 255, 0), -1)

        instructions = "Click 2 pts per line | d=delete last line | c=clear all | q=finish"
        cv2.putText(display, instructions, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("pick_line - click 2 pts per line, q to finish", display)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        if key == ord("d"):
            if current_points:
                current_points.clear()
                print("Deleted unfinished point.")
            elif lines:
                lines.pop()
                print("Deleted last line.")
        if key == ord("c"):
            current_points.clear()
            lines.clear()
            print("Cleared all lines.")

    cv2.destroyAllWindows()

    if not lines:
        print("No lines defined. Nothing to save.")
        return

    print("\n# Paste into config.py COUNTING_LINES:")
    for i, (a, b) in enumerate(lines, start=1):
        print(f'CountingLine(name="Lane {i}", lane="L{i}", point_a={a}, point_b={b}),')

    save_config = input("Save these lines into config.py? [y/N]: ").strip().lower() == "y"
    if save_config:
        config_path = os.path.abspath(args.config)
        if not os.path.exists(config_path):
            print(f"Cannot save config: {config_path} does not exist.")
            return

        with open(config_path, "r", encoding="utf-8") as f:
            config_text = f.read()

        start_tag = "# BEGIN GENERATED COUNTING_LINES"
        end_tag = "# END GENERATED COUNTING_LINES"
        backup_start = "# BEGIN BACKUP COUNTING_LINES"
        backup_end = "# END BACKUP COUNTING_LINES"

        generated_lines = [
            "COUNTING_LINES = [\n",
        ]
        for i, (a, b) in enumerate(lines, start=1):
            generated_lines.append(
                f'    CountingLine(name="Lane {i}", lane="L{i}", point_a={a}, point_b={b}),\n'
            )
        generated_lines.append("]\n")
        generated_block = start_tag + "\n" + "".join(generated_lines) + end_tag

        if start_tag in config_text and end_tag in config_text:
            before, rest = config_text.split(start_tag, 1)
            _, after = rest.split(end_tag, 1)
            base_text = before + after
        else:
            base_text = config_text

        backup_block = ""
        m = re.search(r"^COUNTING_LINES\s*=\s*\[", base_text, re.MULTILINE)
        if m:
            lb = base_text.find("[", m.start())
            end_idx = find_matching_bracket(base_text, lb)
            if end_idx is not None:
                existing_block = base_text[m.start(): end_idx + 1]
                commented = "\n".join(["# " + line for line in existing_block.splitlines()])
                backup_block = backup_start + "\n" + commented + "\n" + backup_end + "\n\n"
                base_text = base_text[:m.start()] + base_text[end_idx + 1:]

        new_config = base_text.rstrip() + "\n\n" + backup_block + generated_block + "\n"

        with open(config_path, "w", encoding="utf-8") as f:
            f.write(new_config)

        print(f"Saved counting lines to {config_path} (backup added)")


if __name__ == "__main__":
    main()
