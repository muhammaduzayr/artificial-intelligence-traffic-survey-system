"""Inspect tracker quality over a short section of a survey video.

Dev/debugging aid — reaches into VehicleDetector's private frame-by-frame
methods directly (bypassing the public track_stream() API) to sample
per-track detection counts, category stability, and confidence range
without running a full survey. Not part of the report-generating pipeline.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2

# Allow `python track_quality.py` when launched from aitss/tools/.
AITSS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AITSS_DIR))

from aitss import config
from aitss.detector import VehicleDetector


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    videos = sorted(AITSS_DIR.glob("*.mp4"))
    parser.add_argument("--video", type=Path, default=videos[0] if videos else None)
    parser.add_argument("--start-frame", type=int, default=105000)
    parser.add_argument("--frames", type=int, default=800)
    parser.add_argument("--sample-step", type=int, default=5)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.video is None:
        raise FileNotFoundError(f"No .mp4 survey video found in {AITSS_DIR}")
    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if args.start_frame < 0 or args.frames <= 0 or args.sample_step <= 0:
        raise ValueError("start-frame must be >= 0; frames and sample-step must be > 0")

    model_path = args.model.expanduser().resolve() if args.model else AITSS_DIR / config.MODEL_PATH
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    detector = VehicleDetector(model_path=model_path, imgsz=args.imgsz)
    track_info = defaultdict(lambda: {"frames": [], "cats": set(), "confs": []})

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
        actual_start = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if actual_start < 0:
            raise RuntimeError(f"Could not seek to frame {args.start_frame}")

        sampled = 0
        for offset in range(args.frames):
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx = actual_start + offset
            if offset % args.sample_step != 0:
                continue

            raw = detector._inference_frame(frame)
            dets = detector._match_detections(raw, args.sample_step)
            detector._update_active_tracks(dets, frame_idx)
            for det in dets:
                tid = det["track_id"]
                track_info[tid]["frames"].append(frame_idx)
                track_info[tid]["cats"].add(det["category"])
                track_info[tid]["confs"].append(det["conf"])
            sampled += 1
            if sampled % 10 == 0:
                print(f"Processed {offset + 1}/{args.frames} frames ({sampled} sampled)", flush=True)
    finally:
        cap.release()

    cat_summary = defaultdict(list)
    for tid, info in track_info.items():
        if not info["cats"] or not info["confs"]:
            continue
        cats = info["cats"]
        cat = "MIXED(" + "/".join(sorted(cats)) + ")" if len(cats) > 1 else next(iter(cats))
        lifespan = info["frames"][-1] - info["frames"][0] if len(info["frames"]) > 1 else 0
        cat_summary[cat].append({
            "tid": tid, "n": len(info["frames"]), "life": lifespan,
            "conf_min": min(info["confs"]), "conf_max": max(info["confs"])
        })

    if not cat_summary:
        print("No detections found in the selected video section.")
        return

    for cat, tracks in sorted(cat_summary.items()):
        n = len(tracks)
        avg_n = sum(t["n"] for t in tracks) / n
        avg_life = sum(t["life"] for t in tracks) / n
        print(f"{cat}: {n} tracks, avg {avg_n:.1f} dets/track, {avg_life:.0f}f lifespan")
        for t in sorted(tracks, key=lambda x: -x["n"])[:3]:
            print(f"  #{t['tid']}: {t['n']} dets, {t['life']}f life, conf {t['conf_min']:.2f}-{t['conf_max']:.2f}")


if __name__ == "__main__":
    main()
