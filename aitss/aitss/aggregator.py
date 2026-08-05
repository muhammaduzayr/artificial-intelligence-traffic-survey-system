"""
Aggregates raw CountEvents into the 15-minute interval report described in
the AITSS proposal (Section 7: Reporting).
"""

import os
import time
from datetime import datetime, timedelta

import pandas as pd

from . import config


class ReportAggregator:
    def __init__(self, survey_start_time: datetime, interval_minutes=None):
        """
        survey_start_time: wall-clock time the video recording started.
            Needed so interval buckets read as real timestamps (e.g.
            08:00-08:15) instead of just "seconds since video start".
        """
        self.survey_start_time = survey_start_time
        self.interval_minutes = interval_minutes or config.INTERVAL_MINUTES
        self._rows = []

    def add_events(self, events):
        for e in events:
            actual_time = self.survey_start_time + timedelta(seconds=e.timestamp_sec)
            total_sec = actual_time.hour * 3600 + actual_time.minute * 60 + actual_time.second
            bucket_epoch = (total_sec // (self.interval_minutes * 60)) * (self.interval_minutes * 60)
            bucket_start = actual_time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(seconds=bucket_epoch)
            bucket_end = bucket_start + timedelta(minutes=self.interval_minutes)
            self._rows.append(
                {
                    "interval_start": bucket_start,
                    "interval_end": bucket_end,
                    "lane": e.lane,
                    "direction": e.direction,
                    "category": e.category,
                    "track_id": e.track_id,
                    "timestamp_sec": e.timestamp_sec,
                    "frame_idx": e.frame_idx,
                    "confidence_at_lock": e.conf,
                    "relinked": e.relinked,
                    "origin": e.origin,
                    "origin_mismatch": e.origin_mismatch,
                }
            )

    def raw_dataframe(self):
        return pd.DataFrame(self._rows)

    def summary_dataframe(self):
        """One row per (interval, lane, direction, category) with a count column,
        plus a Total column across categories."""
        df = self.raw_dataframe()
        if df.empty:
            return pd.DataFrame(
                columns=["interval_start", "interval_end", "lane", "direction", "category", "count"]
            )

        summary = (
            df.groupby(["interval_start", "interval_end", "lane", "direction", "category"])
            .size()
            .reset_index(name="count")
            .sort_values(["interval_start", "lane", "direction", "category"])
        )
        return summary

    def peak_hour(self):
        """Returns (peak_hour_start, peak_hour_volume) using a rolling hour
        over the 15-min buckets — matches Section 7's 'Peak Hour Volume'."""
        df = self.raw_dataframe()
        if df.empty:
            return None, 0

        counts_per_interval = (
            df.groupby("interval_start").size().sort_index()
        )
        # groupby() drops any interval with zero events entirely, so a
        # rolling window over its result can silently span more than
        # 4 x 15-min of real clock time whenever a bucket was empty.
        # Reindex over the full contiguous interval range first (filling
        # gaps with 0) so "1 hour" below always means a true clock hour.
        full_index = pd.date_range(
            start=counts_per_interval.index.min(),
            end=counts_per_interval.index.max(),
            freq=f"{self.interval_minutes}min",
        )
        counts_per_interval = counts_per_interval.reindex(full_index, fill_value=0)
        # rolling 1-hour window = 4 x 15-min buckets (adjust if interval changes)
        window = max(1, 60 // self.interval_minutes)
        rolling = counts_per_interval.rolling(window=window, min_periods=1).sum()
        peak_start = rolling.idxmax()
        return peak_start, int(rolling.max())

    def export(self, output_dir=None, basename="traffic_survey_report"):
        output_dir = output_dir or config.OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)

        summary = self.summary_dataframe()
        raw = self.raw_dataframe()

        excel_path = os.path.join(output_dir, f"{basename}.xlsx")
        csv_path = os.path.join(output_dir, f"{basename}.csv")

        def write_workbook(path):
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                summary.to_excel(writer, sheet_name="15min_summary", index=False)
                raw.to_excel(writer, sheet_name="raw_events", index=False)

                # Vehicle category totals + percentages
                if not raw.empty:
                    totals = raw["category"].value_counts().reset_index()
                    totals.columns = ["category", "count"]
                    totals["percentage"] = (100 * totals["count"] / totals["count"].sum()).round(1)
                    totals.to_excel(writer, sheet_name="category_totals", index=False)

        try:
            write_workbook(excel_path)
        except PermissionError:
            # Excel keeps an open workbook locked on Windows. Preserve the
            # completed run by writing a timestamped copy instead of losing
            # it entirely.
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            fallback_excel_path = os.path.join(output_dir, f"{basename}_{timestamp}.xlsx")
            write_workbook(fallback_excel_path)
            excel_path = fallback_excel_path

        summary.to_csv(csv_path, index=False)

        return excel_path, csv_path
