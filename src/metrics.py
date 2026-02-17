from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor


@dataclass(slots=True)
class LatencyTimeline:
    request_sent_at: float
    first_token_at: float | None
    token_timestamps: list[float]
    response_done_at: float


@dataclass(slots=True)
class LatencyMetrics:
    ttft: float | None
    itl_values: list[float]
    tbt: float | None
    tpot: float | None
    e2e: float


@dataclass(slots=True)
class WindowSample:
    ended_at: float
    output_tokens: int
    success: bool
    slo_passed: bool


@dataclass(slots=True)
class WindowMetric:
    window_start: float
    request_count: int
    output_tokens: int
    success_count: int
    slo_passed_count: int
    rps: float
    tps: float
    goodput: float
    error_rate: float


def _quantile_cont(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])

    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * percentile
    lower_index = floor(position)
    upper_index = ceil(position)
    if lower_index == upper_index:
        return float(sorted_values[lower_index])

    left = sorted_values[lower_index]
    right = sorted_values[upper_index]
    fraction = position - lower_index
    return float(left + (right - left) * fraction)


def quantile_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "p50": _quantile_cont(values, 0.50),
        "p90": _quantile_cont(values, 0.90),
        "p95": _quantile_cont(values, 0.95),
        "p99": _quantile_cont(values, 0.99),
    }


def compute_latency_metrics(timeline: LatencyTimeline) -> LatencyMetrics:
    e2e = timeline.response_done_at - timeline.request_sent_at
    if timeline.first_token_at is None or not timeline.token_timestamps:
        return LatencyMetrics(ttft=None, itl_values=[], tbt=None, tpot=None, e2e=e2e)

    ordered_tokens = sorted(timeline.token_timestamps)
    itl_values: list[float] = []
    for index in range(1, len(ordered_tokens)):
        itl_values.append(ordered_tokens[index] - ordered_tokens[index - 1])

    ttft = timeline.first_token_at - timeline.request_sent_at
    tbt = ordered_tokens[-1] - ordered_tokens[0]
    denominator = max(len(ordered_tokens) - 1, 1)
    tpot = tbt / denominator

    return LatencyMetrics(
        ttft=ttft,
        itl_values=itl_values,
        tbt=tbt,
        tpot=tpot,
        e2e=e2e,
    )


def build_window_metrics(
    samples: list[WindowSample], window_seconds: float = 1.0
) -> list[WindowMetric]:
    if window_seconds <= 0:
        raise ValueError("window_seconds must be > 0")

    buckets: dict[float, list[WindowSample]] = {}
    for sample in samples:
        window_start = floor(sample.ended_at / window_seconds) * window_seconds
        # Keep deterministic keys for common decimal window sizes.
        window_start = round(window_start, 9)
        buckets.setdefault(window_start, []).append(sample)

    window_metrics: list[WindowMetric] = []
    for window_start in sorted(buckets):
        bucket_samples = buckets[window_start]
        request_count = len(bucket_samples)
        output_tokens = sum(sample.output_tokens for sample in bucket_samples)
        success_count = sum(1 for sample in bucket_samples if sample.success)
        slo_passed_count = sum(1 for sample in bucket_samples if sample.slo_passed)
        error_count = request_count - success_count

        window_metrics.append(
            WindowMetric(
                window_start=window_start,
                request_count=request_count,
                output_tokens=output_tokens,
                success_count=success_count,
                slo_passed_count=slo_passed_count,
                rps=request_count / window_seconds,
                tps=output_tokens / window_seconds,
                goodput=slo_passed_count / window_seconds,
                error_rate=(error_count / request_count if request_count else 0.0),
            )
        )
    return window_metrics


def summarize_window_quantiles(
    windows: list[WindowMetric],
) -> dict[str, dict[str, float | int | None]]:
    return {
        "rps": quantile_summary([window.rps for window in windows]),
        "tps": quantile_summary([window.tps for window in windows]),
        "goodput": quantile_summary([window.goodput for window in windows]),
        "error_rate": quantile_summary([window.error_rate for window in windows]),
    }
