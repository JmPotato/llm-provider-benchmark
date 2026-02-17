from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metrics import (
    LatencyTimeline,
    WindowSample,
    build_window_metrics,
    compute_latency_metrics,
    quantile_summary,
    summarize_window_quantiles,
)


def test_quantile_summary_empty_values() -> None:
    summary = quantile_summary([])
    assert summary["count"] == 0
    assert summary["p50"] is None
    assert summary["p99"] is None


def test_quantile_summary_expected_values() -> None:
    summary = quantile_summary([1.0, 2.0, 3.0, 4.0])
    assert summary["count"] == 4
    assert summary["p50"] == pytest.approx(2.5)
    assert summary["p90"] == pytest.approx(3.7)
    assert summary["p95"] == pytest.approx(3.85)
    assert summary["p99"] == pytest.approx(3.97)


def test_quantile_summary_single_value() -> None:
    summary = quantile_summary([7.5])
    assert summary["count"] == 1
    assert summary["p50"] == pytest.approx(7.5)
    assert summary["p99"] == pytest.approx(7.5)


def test_quantile_summary_handles_unsorted_input() -> None:
    summary = quantile_summary([9.0, 1.0, 5.0, 3.0])
    assert summary["p50"] == pytest.approx(4.0)


def test_compute_latency_metrics_with_multi_token_output() -> None:
    timeline = LatencyTimeline(
        request_sent_at=0.0,
        first_token_at=0.2,
        token_timestamps=[0.2, 0.3, 0.5],
        response_done_at=0.8,
    )
    metrics = compute_latency_metrics(timeline)
    assert metrics.ttft == pytest.approx(0.2)
    assert metrics.itl_values == pytest.approx([0.1, 0.2])
    assert metrics.tbt == pytest.approx(0.3)
    assert metrics.tpot == pytest.approx(0.15)
    assert metrics.e2e == pytest.approx(0.8)


def test_compute_latency_metrics_for_single_token() -> None:
    timeline = LatencyTimeline(
        request_sent_at=10.0,
        first_token_at=10.4,
        token_timestamps=[10.4],
        response_done_at=10.6,
    )
    metrics = compute_latency_metrics(timeline)
    assert metrics.itl_values == []
    assert metrics.tbt == pytest.approx(0.0)
    assert metrics.tpot == pytest.approx(0.0)


def test_compute_latency_metrics_without_token() -> None:
    timeline = LatencyTimeline(
        request_sent_at=1.0,
        first_token_at=None,
        token_timestamps=[],
        response_done_at=1.9,
    )
    metrics = compute_latency_metrics(timeline)
    assert metrics.ttft is None
    assert metrics.itl_values == []
    assert metrics.tbt is None
    assert metrics.tpot is None
    assert metrics.e2e == pytest.approx(0.9)


def test_build_window_metrics() -> None:
    windows = build_window_metrics(
        [
            WindowSample(ended_at=0.3, output_tokens=10, success=True, slo_passed=True),
            WindowSample(
                ended_at=0.7, output_tokens=6, success=False, slo_passed=False
            ),
            WindowSample(ended_at=1.2, output_tokens=4, success=True, slo_passed=False),
        ]
    )
    assert len(windows) == 2

    first = windows[0]
    assert first.rps == pytest.approx(2.0)
    assert first.tps == pytest.approx(16.0)
    assert first.goodput == pytest.approx(1.0)
    assert first.error_rate == pytest.approx(0.5)

    second = windows[1]
    assert second.rps == pytest.approx(1.0)
    assert second.tps == pytest.approx(4.0)
    assert second.goodput == pytest.approx(0.0)
    assert second.error_rate == pytest.approx(0.0)


def test_summarize_window_quantiles() -> None:
    windows = build_window_metrics(
        [
            WindowSample(ended_at=0.1, output_tokens=2, success=True, slo_passed=True),
            WindowSample(ended_at=1.1, output_tokens=6, success=True, slo_passed=False),
            WindowSample(
                ended_at=2.1, output_tokens=10, success=False, slo_passed=False
            ),
        ]
    )
    summary = summarize_window_quantiles(windows)
    assert "rps" in summary
    assert "tps" in summary
    assert "goodput" in summary
    assert "error_rate" in summary
    assert summary["rps"]["count"] == 3
    assert math.isfinite(summary["tps"]["p95"])


def test_build_window_metrics_with_no_samples() -> None:
    assert build_window_metrics([]) == []


def test_build_window_metrics_respects_custom_window_size() -> None:
    windows = build_window_metrics(
        [
            WindowSample(ended_at=0.3, output_tokens=2, success=True, slo_passed=True),
            WindowSample(ended_at=1.9, output_tokens=3, success=True, slo_passed=False),
            WindowSample(
                ended_at=2.1, output_tokens=4, success=False, slo_passed=False
            ),
        ],
        window_seconds=2.0,
    )
    assert [window.window_start for window in windows] == [0.0, 2.0]
    assert windows[0].rps == pytest.approx(1.0)
    assert windows[1].rps == pytest.approx(0.5)


def test_build_window_metrics_rejects_non_positive_window_size() -> None:
    with pytest.raises(ValueError, match="window_seconds"):
        build_window_metrics(
            [
                WindowSample(
                    ended_at=0.1, output_tokens=1, success=True, slo_passed=True
                )
            ],
            window_seconds=0.0,
        )
