"""Compares a pipeline run's counted crossings against a hand-tallied
ground truth, per (lane, category) — the actual accuracy check this
project has never had (see ground_truth_template.py to produce the input
CSV, and TUNING_NOTES.md / CUSTOM_MODEL_AND_REID_NOTES.md for why this
was flagged as missing).

Usage:
    python tools/compare_ground_truth.py \\
        --pipeline-csv out/crossing_diagnostics.csv \\
        --ground-truth-csv out/ground_truth_template.csv

--pipeline-csv accepts either crossing_diagnostics.csv (one row per
counted crossing — preferred, most direct) or traffic_survey_report.csv
(the 15-min summary — also fine, just pre-aggregated).
"""

import argparse
import csv
from collections import defaultdict


def _load_pipeline_counts(path):
    counts = defaultdict(int)
    relinked_count = 0
    total = 0
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        is_diagnostics = "track_id" in fieldnames
        for row in reader:
            lane = row.get("lane")
            category = row.get("category")
            if is_diagnostics:
                counts[(lane, category)] += 1
                total += 1
                if str(row.get("relinked", "")).strip().lower() in ("true", "1"):
                    relinked_count += 1
            else:
                # traffic_survey_report.csv (15-min summary): already has
                # a per-row count column.
                n = int(row.get("count", 0))
                counts[(lane, category)] += n
                total += n
    return counts, total, relinked_count


def _load_ground_truth(path):
    counts = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lane = row["lane"]
            category = row["category"]
            counts[(lane, category)] = int(row["manual_count"])
    return counts


def compare(pipeline_csv, ground_truth_csv):
    pipeline_counts, pipeline_total, relinked_count = _load_pipeline_counts(pipeline_csv)
    manual_counts = _load_ground_truth(ground_truth_csv)

    all_keys = sorted(set(pipeline_counts) | set(manual_counts))
    manual_total = sum(manual_counts.values())

    print(f"{'lane':<8}{'category':<14}{'pipeline':>10}{'manual':>10}{'diff':>8}{'% error':>10}")
    print("-" * 60)
    for lane, category in all_keys:
        p = pipeline_counts.get((lane, category), 0)
        m = manual_counts.get((lane, category), 0)
        diff = p - m
        pct = (100 * diff / m) if m else (float("inf") if p else 0.0)
        pct_str = f"{pct:+.0f}%" if pct != float("inf") else "n/a (0 GT)"
        print(f"{lane:<8}{category:<14}{p:>10}{m:>10}{diff:>+8}{pct_str:>10}")

    print("-" * 60)
    overall_diff = pipeline_total - manual_total
    overall_pct = (100 * overall_diff / manual_total) if manual_total else float("nan")
    print(f"{'TOTAL':<22}{pipeline_total:>10}{manual_total:>10}{overall_diff:>+8}{overall_pct:>+9.1f}%")

    if pipeline_total:
        print(f"\nRelinked crossings: {relinked_count}/{pipeline_total} "
              f"({100 * relinked_count / pipeline_total:.1f}%) — a relinked crossing survived an "
              f"internal track-ID switch; a high fraction here means results lean more on the "
              f"identity-bridging safety net and are worth a closer manual spot-check.")

    if manual_total == 0:
        print("\nWARNING: ground-truth CSV has zero manual counts filled in — "
              "this comparison is not meaningful yet. Fill in ground_truth_template.csv first.")
    elif abs(overall_pct) > 15:
        print(f"\nOverall count is off by {overall_pct:+.1f}% — outside a rough 15% comfort "
              f"margin for a survey report. Check per-(lane, category) rows above for where "
              f"the error concentrates before presenting this run's numbers.")
    else:
        print(f"\nOverall count within {abs(overall_pct):.1f}% of manual ground truth.")

    return pipeline_counts, manual_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-csv", required=True,
                         help="crossing_diagnostics.csv or traffic_survey_report.csv from a pipeline run")
    parser.add_argument("--ground-truth-csv", required=True,
                         help="Hand-filled CSV from ground_truth_template.py")
    args = parser.parse_args()
    compare(args.pipeline_csv, args.ground_truth_csv)


if __name__ == "__main__":
    main()
