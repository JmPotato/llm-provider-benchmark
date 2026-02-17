from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from providers import ProviderConfig
from records import RequestRecord, TokenEvent
from storage import BenchmarkStorage


def test_storage_can_persist_and_summarize_metrics(tmp_path: Path) -> None:
    db_path = tmp_path / "bench.duckdb"
    storage = BenchmarkStorage(db_path)
    run_id = "run-1"
    provider_name = "openai"

    storage.create_run(run_id=run_id, started_at=0.0, config_json="{}")
    storage.upsert_provider_snapshot(
        run_id=run_id,
        provider=ProviderConfig(
            name=provider_name,
            model="gpt-4o-mini",
            api_base="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
        ),
    )
    storage.insert_request_records(
        [
            RequestRecord(
                run_id=run_id,
                provider_name=provider_name,
                request_id="req-1",
                prompt="hello",
                request_started_at=0.0,
                first_token_at=0.2,
                response_done_at=0.9,
                output_tokens=3,
                success=True,
                error_type=None,
                error_message=None,
                ttft_s=0.2,
                tbt_s=0.5,
                tpot_s=0.25,
                e2e_s=0.9,
                slo_passed=True,
            ),
            RequestRecord(
                run_id=run_id,
                provider_name=provider_name,
                request_id="req-2",
                prompt="boom",
                request_started_at=0.4,
                first_token_at=None,
                response_done_at=1.3,
                output_tokens=0,
                success=False,
                error_type="RuntimeError",
                error_message="mock error",
                ttft_s=None,
                tbt_s=None,
                tpot_s=None,
                e2e_s=0.9,
                slo_passed=False,
            ),
        ]
    )
    storage.insert_token_events(
        [
            TokenEvent(
                run_id=run_id,
                provider_name=provider_name,
                request_id="req-1",
                token_index=0,
                token_timestamp=0.2,
                token_text="A",
            ),
            TokenEvent(
                run_id=run_id,
                provider_name=provider_name,
                request_id="req-1",
                token_index=1,
                token_timestamp=0.45,
                token_text="B",
            ),
            TokenEvent(
                run_id=run_id,
                provider_name=provider_name,
                request_id="req-1",
                token_index=2,
                token_timestamp=0.7,
                token_text="C",
            ),
        ]
    )
    storage.refresh_window_metrics(run_id=run_id, provider_name=provider_name)
    summary = storage.get_provider_summary(run_id=run_id, provider_name=provider_name)

    assert summary["latency"]["ttft"]["count"] == 1
    assert summary["latency"]["ttft"]["p50"] == pytest.approx(0.2)
    assert summary["latency"]["itl"]["count"] == 2
    assert summary["latency"]["itl"]["p50"] == pytest.approx(0.25)
    assert summary["latency"]["e2e"]["count"] == 2
    assert summary["throughput"]["rps"]["count"] >= 1
    assert summary["quality"]["error_rate"]["p50"] == pytest.approx(0.5)


def test_storage_lists_providers_for_run(tmp_path: Path) -> None:
    storage = BenchmarkStorage(tmp_path / "bench.duckdb")
    storage.create_run(run_id="run-1", started_at=0.0, config_json="{}")
    storage.upsert_provider_snapshot(
        run_id="run-1",
        provider=ProviderConfig(name="openai", model="gpt-4o-mini"),
    )
    storage.upsert_provider_snapshot(
        run_id="run-1",
        provider=ProviderConfig(name="openrouter", model="anthropic/claude-3-7-sonnet"),
    )
    assert storage.list_run_providers("run-1") == ["openai", "openrouter"]


def test_storage_lists_runs_with_stats(tmp_path: Path) -> None:
    storage = BenchmarkStorage(tmp_path / "bench.duckdb")
    run_id = "run-stats"
    provider_name = "openai"
    storage.create_run(run_id=run_id, started_at=1.0, config_json="{}")
    storage.upsert_provider_snapshot(
        run_id=run_id,
        provider=ProviderConfig(name=provider_name, model="gpt-4o-mini"),
    )
    storage.insert_request_records(
        [
            RequestRecord(
                run_id=run_id,
                provider_name=provider_name,
                request_id="req-ok",
                prompt="hello",
                request_started_at=1.0,
                first_token_at=1.1,
                response_done_at=1.2,
                output_tokens=2,
                success=True,
                error_type=None,
                error_message=None,
                ttft_s=0.1,
                tbt_s=0.0,
                tpot_s=0.0,
                e2e_s=0.2,
                slo_passed=True,
            ),
            RequestRecord(
                run_id=run_id,
                provider_name=provider_name,
                request_id="req-fail",
                prompt="boom",
                request_started_at=1.0,
                first_token_at=None,
                response_done_at=1.3,
                output_tokens=0,
                success=False,
                error_type="RuntimeError",
                error_message="mock",
                ttft_s=None,
                tbt_s=None,
                tpot_s=None,
                e2e_s=0.3,
                slo_passed=False,
            ),
        ]
    )

    runs = storage.list_runs_with_stats()
    assert len(runs) == 1
    run = runs[0]
    assert run["run_id"] == run_id
    assert run["provider_count"] == 1
    assert run["request_count"] == 2
    assert run["success_count"] == 1
    assert run["failed_count"] == 1


def test_storage_delete_run_removes_related_rows(tmp_path: Path) -> None:
    storage = BenchmarkStorage(tmp_path / "bench.duckdb")
    run_id = "run-delete"
    provider_name = "openai"
    storage.create_run(run_id=run_id, started_at=0.0, config_json="{}")
    storage.upsert_provider_snapshot(
        run_id=run_id,
        provider=ProviderConfig(name=provider_name, model="gpt-4o-mini"),
    )
    storage.insert_request_records(
        [
            RequestRecord(
                run_id=run_id,
                provider_name=provider_name,
                request_id="req-1",
                prompt="hello",
                request_started_at=0.0,
                first_token_at=0.1,
                response_done_at=0.2,
                output_tokens=1,
                success=True,
                error_type=None,
                error_message=None,
                ttft_s=0.1,
                tbt_s=0.0,
                tpot_s=0.0,
                e2e_s=0.2,
                slo_passed=True,
            )
        ]
    )
    storage.insert_token_events(
        [
            TokenEvent(
                run_id=run_id,
                provider_name=provider_name,
                request_id="req-1",
                token_index=0,
                token_timestamp=0.1,
                token_text="tok",
            )
        ]
    )
    storage.refresh_window_metrics(run_id=run_id, provider_name=provider_name)

    deleted = storage.delete_run(run_id)
    assert deleted is True
    with pytest.raises(KeyError):
        storage.get_run(run_id)
    assert storage.list_run_providers(run_id) == []
    assert storage.list_runs_with_stats() == []


def test_storage_delete_run_returns_false_for_missing_run(tmp_path: Path) -> None:
    storage = BenchmarkStorage(tmp_path / "bench.duckdb")
    deleted = storage.delete_run("missing")
    assert deleted is False


def test_storage_finish_run_sets_duration(tmp_path: Path) -> None:
    storage = BenchmarkStorage(tmp_path / "bench.duckdb")
    storage.create_run(run_id="run-1", started_at=3.0, config_json="{}")
    storage.finish_run(run_id="run-1", finished_at=8.5)
    run_row = storage.get_run("run-1")
    assert run_row["finished_at"] == pytest.approx(8.5)
    assert run_row["duration_s"] == pytest.approx(5.5)


def test_get_provider_summary_raises_for_unknown_provider(tmp_path: Path) -> None:
    storage = BenchmarkStorage(tmp_path / "bench.duckdb")
    storage.create_run(run_id="run-1", started_at=0.0, config_json="{}")
    with pytest.raises(KeyError):
        storage.get_provider_summary(run_id="run-1", provider_name="missing")


def test_get_provider_summary_with_no_requests_returns_empty_quantiles(tmp_path: Path) -> None:
    storage = BenchmarkStorage(tmp_path / "bench.duckdb")
    storage.create_run(run_id="run-1", started_at=0.0, config_json="{}")
    storage.upsert_provider_snapshot(
        run_id="run-1",
        provider=ProviderConfig(name="openai", model="gpt-4o-mini"),
    )
    storage.refresh_window_metrics(run_id="run-1", provider_name="openai")
    summary = storage.get_provider_summary(run_id="run-1", provider_name="openai")
    assert summary["latency"]["ttft"]["count"] == 0
    assert summary["latency"]["ttft"]["p50"] is None
    assert summary["throughput"]["rps"]["count"] == 0
    assert summary["quality"]["goodput"]["p99"] is None
