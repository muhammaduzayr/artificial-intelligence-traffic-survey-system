"""Tests for ReportAggregator's 15-min bucketing and peak-hour calculation.

peak_hour() has a real fixed regression covered explicitly here: groupby()
drops empty intervals, so a naive rolling window over its result can span
more real clock time than "1 hour" whenever a bucket had zero events. The
fix reindexes over the full contiguous interval range first — see the
comment in aggregator.py.
"""

from dataclasses import dataclass
from datetime import datetime

import pytest

from aitss.aggregator import ReportAggregator


@dataclass
class FakeEvent:
    frame_idx: int
    timestamp_sec: float
    track_id: int
    lane: str
    direction: str
    category: str
    conf: float = 0.8
    relinked: bool = False
    origin: str = None
    origin_mismatch: bool = False


def test_bucketing_assigns_correct_15min_interval():
    start = datetime(2026, 8, 5, 8, 0, 0)
    agg = ReportAggregator(start)

    # 7 minutes in -> still the 08:00-08:15 bucket.
    agg.add_events([FakeEvent(frame_idx=1, timestamp_sec=7 * 60, track_id=1,
                               lane="N1", direction="N1", category="Car")])
    # 16 minutes in -> the next bucket, 08:15-08:30.
    agg.add_events([FakeEvent(frame_idx=2, timestamp_sec=16 * 60, track_id=2,
                               lane="N1", direction="N1", category="Car")])

    raw = agg.raw_dataframe()
    assert list(raw["interval_start"]) == [
        datetime(2026, 8, 5, 8, 0, 0),
        datetime(2026, 8, 5, 8, 15, 0),
    ]


def test_bucketing_crosses_midnight_correctly():
    # Survey starting at 23:50 with an event 20 minutes later must land
    # on the NEXT day's 00:00-00:15 bucket, not wrap incorrectly.
    start = datetime(2026, 8, 5, 23, 50, 0)
    agg = ReportAggregator(start)
    agg.add_events([FakeEvent(frame_idx=1, timestamp_sec=20 * 60, track_id=1,
                               lane="N1", direction="N1", category="Car")])
    raw = agg.raw_dataframe()
    assert raw["interval_start"].iloc[0] == datetime(2026, 8, 6, 0, 0, 0)


def test_summary_dataframe_groups_by_interval_lane_direction_category():
    start = datetime(2026, 8, 5, 8, 0, 0)
    agg = ReportAggregator(start)
    agg.add_events([
        FakeEvent(frame_idx=1, timestamp_sec=60, track_id=1, lane="N1", direction="N1", category="Car"),
        FakeEvent(frame_idx=2, timestamp_sec=90, track_id=2, lane="N1", direction="N1", category="Car"),
        FakeEvent(frame_idx=3, timestamp_sec=120, track_id=3, lane="N1", direction="N1", category="Motorcycle"),
    ])
    summary = agg.summary_dataframe()
    row = summary[(summary["lane"] == "N1") & (summary["category"] == "Car")]
    assert row["count"].iloc[0] == 2
    row_mc = summary[(summary["lane"] == "N1") & (summary["category"] == "Motorcycle")]
    assert row_mc["count"].iloc[0] == 1


def test_peak_hour_with_no_events_returns_none():
    agg = ReportAggregator(datetime(2026, 8, 5, 8, 0, 0))
    peak_start, peak_vol = agg.peak_hour()
    assert peak_start is None
    assert peak_vol == 0


def test_peak_hour_ignores_gap_bucket_correctly():
    """Regression test for the reindex fix: a zero-count bucket in the
    middle must NOT let the rolling window silently absorb events from
    further away than a true clock hour.
    """
    start = datetime(2026, 8, 5, 8, 0, 0)
    agg = ReportAggregator(start)

    # 08:00-08:15: 1 event. 08:15-08:30: nothing (gap). 08:30-08:45: 1
    # event. 08:45-09:00: 10 events (the real spike).
    agg.add_events([FakeEvent(frame_idx=1, timestamp_sec=5 * 60, track_id=1,
                               lane="N1", direction="N1", category="Car")])
    agg.add_events([FakeEvent(frame_idx=2, timestamp_sec=35 * 60, track_id=2,
                               lane="N1", direction="N1", category="Car")])
    for i in range(10):
        agg.add_events([FakeEvent(frame_idx=100 + i, timestamp_sec=46 * 60 + i,
                                   track_id=10 + i, lane="N1", direction="N1", category="Car")])

    peak_start, peak_vol = agg.peak_hour()
    # The true busiest rolling hour is 08:00-09:00 (1 + 0 + 1 + 10 = 12),
    # spanning all 4 buckets including the empty one in the middle.
    assert peak_start == datetime(2026, 8, 5, 8, 0, 0)
    assert peak_vol == 12


def test_peak_hour_on_short_survey_never_predates_the_survey_start():
    """Regression test found via a real short (~2.5 min) test run: with
    only ONE 15-min bucket of data, the rolling window is necessarily
    partial (min_periods=1), so shifting back by a flat (window-1)
    buckets over-corrects and reports a peak hour starting BEFORE the
    survey itself began. A naive version of this fix reported "06:45" as
    the peak hour start for a clip that only covered 07:30-07:32.
    """
    start = datetime(2026, 2, 11, 7, 30, 0)
    agg = ReportAggregator(start)

    # All events land within the single 07:30-07:45 bucket (a 2.5-minute
    # clip can't span more than one 15-min interval).
    for i in range(5):
        agg.add_events([FakeEvent(frame_idx=i, timestamp_sec=30 + i, track_id=i,
                                   lane="N2", direction="N2", category="Car")])

    peak_start, peak_vol = agg.peak_hour()
    assert peak_start == datetime(2026, 2, 11, 7, 30, 0)
    assert peak_vol == 5
