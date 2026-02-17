from __future__ import annotations

from pathlib import Path
from threading import Lock
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from providers import ProviderConfig
from runner import BenchmarkRunner, LiteLLMClient, SLOConfig
from storage import BenchmarkStorage


class StepClock:
    def __init__(self, start: float = 0.0, step: float = 0.1) -> None:
        self.current = start - step
        self.step = step

    def __call__(self) -> float:
        self.current += self.step
        return self.current


class FakeClient:
    def __init__(self, responses: dict[str, list[str] | Exception]) -> None:
        self.responses = responses

    def stream_completion(self, provider: ProviderConfig, prompt: str):
        value = self.responses[prompt]
        if isinstance(value, Exception):
            raise value
        for token in value:
            yield token


class ConcurrencyTrackingClient:
    def __init__(self, hold_s: float = 0.05) -> None:
        self.hold_s = hold_s
        self._lock = Lock()
        self.active_calls = 0
        self.max_active_calls = 0

    def stream_completion(self, provider: ProviderConfig, prompt: str):
        with self._lock:
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            time.sleep(self.hold_s)
            yield f"{provider.name}:{prompt}"
        finally:
            with self._lock:
                self.active_calls -= 1


def test_runner_records_successful_requests(tmp_path: Path) -> None:
    storage = BenchmarkStorage(tmp_path / "bench.duckdb")
    run_id = "run-success"
    provider = ProviderConfig(name="openai", model="gpt-4o-mini")
    storage.create_run(run_id=run_id, started_at=0.0, config_json="{}")
    storage.upsert_provider_snapshot(run_id=run_id, provider=provider)

    runner = BenchmarkRunner(
        storage=storage,
        client=FakeClient({"prompt-1": ["A", "B", "C"]}),
        clock=StepClock(start=0.0, step=0.1),
        sleep_fn=lambda _: None,
    )
    result = runner.run_provider(
        run_id=run_id,
        provider=provider,
        prompts=["prompt-1"],
        target_rps=None,
        slo=SLOConfig(max_ttft_s=1.0, max_e2e_s=2.0),
    )
    summary = storage.get_provider_summary(run_id=run_id, provider_name="openai")

    assert result.total_requests == 1
    assert result.success_requests == 1
    assert summary["latency"]["ttft"]["p50"] == pytest.approx(0.1)
    assert summary["latency"]["tbt"]["p50"] == pytest.approx(0.2)
    assert summary["latency"]["tpot"]["p50"] == pytest.approx(0.1)
    assert summary["quality"]["error_rate"]["p50"] == pytest.approx(0.0)


def test_runner_records_failed_requests(tmp_path: Path) -> None:
    storage = BenchmarkStorage(tmp_path / "bench.duckdb")
    run_id = "run-failure"
    provider = ProviderConfig(name="openai", model="gpt-4o-mini")
    storage.create_run(run_id=run_id, started_at=0.0, config_json="{}")
    storage.upsert_provider_snapshot(run_id=run_id, provider=provider)

    runner = BenchmarkRunner(
        storage=storage,
        client=FakeClient({"prompt-1": RuntimeError("boom")}),
        clock=StepClock(start=0.0, step=0.1),
        sleep_fn=lambda _: None,
    )
    result = runner.run_provider(
        run_id=run_id,
        provider=provider,
        prompts=["prompt-1"],
        target_rps=None,
        slo=SLOConfig(max_ttft_s=1.0, max_e2e_s=2.0),
    )
    summary = storage.get_provider_summary(run_id=run_id, provider_name="openai")

    assert result.total_requests == 1
    assert result.failed_requests == 1
    assert summary["quality"]["error_rate"]["p50"] == pytest.approx(1.0)
    assert summary["latency"]["ttft"]["count"] == 0


def test_runner_marks_slo_failures(tmp_path: Path) -> None:
    storage = BenchmarkStorage(tmp_path / "bench.duckdb")
    run_id = "run-slo"
    provider = ProviderConfig(name="openai", model="gpt-4o-mini")
    storage.create_run(run_id=run_id, started_at=0.0, config_json="{}")
    storage.upsert_provider_snapshot(run_id=run_id, provider=provider)

    runner = BenchmarkRunner(
        storage=storage,
        client=FakeClient({"prompt-1": ["A"]}),
        clock=StepClock(start=0.0, step=0.2),
        sleep_fn=lambda _: None,
    )
    result = runner.run_provider(
        run_id=run_id,
        provider=provider,
        prompts=["prompt-1"],
        target_rps=None,
        slo=SLOConfig(max_ttft_s=0.05, max_e2e_s=0.5),
    )
    summary = storage.get_provider_summary(run_id=run_id, provider_name="openai")

    assert result.slo_passed_requests == 0
    assert summary["quality"]["goodput"]["p50"] == pytest.approx(0.0)


def test_runner_applies_target_rps_sleep(tmp_path: Path) -> None:
    storage = BenchmarkStorage(tmp_path / "bench.duckdb")
    run_id = "run-rps"
    provider = ProviderConfig(name="openai", model="gpt-4o-mini")
    storage.create_run(run_id=run_id, started_at=0.0, config_json="{}")
    storage.upsert_provider_snapshot(run_id=run_id, provider=provider)

    sleep_calls: list[float] = []
    runner = BenchmarkRunner(
        storage=storage,
        client=FakeClient({"p1": ["A"], "p2": ["B"]}),
        clock=StepClock(start=0.0, step=0.01),
        sleep_fn=lambda duration: sleep_calls.append(duration),
    )
    runner.run_provider(
        run_id=run_id,
        provider=provider,
        prompts=["p1", "p2"],
        target_rps=1.0,
        slo=SLOConfig(max_ttft_s=1.0, max_e2e_s=2.0),
    )
    assert len(sleep_calls) >= 1
    assert max(sleep_calls) > 0.0


def test_runner_rejects_non_positive_target_rps(tmp_path: Path) -> None:
    storage = BenchmarkStorage(tmp_path / "bench.duckdb")
    run_id = "run-invalid-rps"
    provider = ProviderConfig(name="openai", model="gpt-4o-mini")
    storage.create_run(run_id=run_id, started_at=0.0, config_json="{}")
    storage.upsert_provider_snapshot(run_id=run_id, provider=provider)

    runner = BenchmarkRunner(
        storage=storage,
        client=FakeClient({"p1": ["A"]}),
        clock=StepClock(start=0.0, step=0.1),
        sleep_fn=lambda _: None,
    )
    with pytest.raises(ValueError, match="target_rps"):
        runner.run_provider(
            run_id=run_id,
            provider=provider,
            prompts=["p1"],
            target_rps=0.0,
            slo=SLOConfig(max_ttft_s=1.0, max_e2e_s=2.0),
        )


def test_runner_uses_token_counter_for_output_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = BenchmarkStorage(tmp_path / "bench.duckdb")
    run_id = "run-token-counter"
    provider = ProviderConfig(name="openai", model="gpt-4o-mini")
    storage.create_run(run_id=run_id, started_at=0.0, config_json="{}")
    storage.upsert_provider_snapshot(run_id=run_id, provider=provider)

    monkeypatch.setattr("litellm.token_counter", lambda **_: 11)
    runner = BenchmarkRunner(
        storage=storage,
        client=FakeClient({"p1": ["hello", " world"]}),
        clock=StepClock(start=0.0, step=0.1),
        sleep_fn=lambda _: None,
    )
    runner.run_provider(
        run_id=run_id,
        provider=provider,
        prompts=["p1"],
        target_rps=None,
        slo=SLOConfig(max_ttft_s=1.0, max_e2e_s=2.0),
    )

    row = storage.connection.execute(
        """
        SELECT output_tokens
        FROM requests
        WHERE run_id = ? AND provider_name = ? AND request_id = ?
        """,
        [run_id, provider.name, f"{provider.name}-0"],
    ).fetchone()
    assert row is not None
    assert int(row[0]) == 11


def test_runner_falls_back_to_chunk_count_when_token_counter_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = BenchmarkStorage(tmp_path / "bench.duckdb")
    run_id = "run-token-counter-fallback"
    provider = ProviderConfig(name="openai", model="gpt-4o-mini")
    storage.create_run(run_id=run_id, started_at=0.0, config_json="{}")
    storage.upsert_provider_snapshot(run_id=run_id, provider=provider)

    def _raise_token_counter(**_: object) -> int:
        raise RuntimeError("unsupported model")

    monkeypatch.setattr("litellm.token_counter", _raise_token_counter)
    runner = BenchmarkRunner(
        storage=storage,
        client=FakeClient({"p1": ["A", "B", "C"]}),
        clock=StepClock(start=0.0, step=0.1),
        sleep_fn=lambda _: None,
    )
    runner.run_provider(
        run_id=run_id,
        provider=provider,
        prompts=["p1"],
        target_rps=None,
        slo=SLOConfig(max_ttft_s=1.0, max_e2e_s=2.0),
    )

    row = storage.connection.execute(
        """
        SELECT output_tokens
        FROM requests
        WHERE run_id = ? AND provider_name = ? AND request_id = ?
        """,
        [run_id, provider.name, f"{provider.name}-0"],
    ).fetchone()
    assert row is not None
    assert int(row[0]) == 3


def test_litellm_client_raises_when_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_API_KEY", raising=False)
    client = LiteLLMClient()
    provider = ProviderConfig(name="openai", model="gpt-4o-mini", api_key_env="MISSING_API_KEY")
    with pytest.raises(ValueError):
        list(client.stream_completion(provider=provider, prompt="hello"))


def test_litellm_client_suppresses_debug_info(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm

    captured: dict[str, object] = {}
    litellm.suppress_debug_info = False

    def _fake_completion(**_: object):
        captured["suppress_debug_info"] = litellm.suppress_debug_info
        return [{"choices": [{"delta": {"content": "ok"}}]}]

    monkeypatch.setattr("litellm.completion", _fake_completion)

    client = LiteLLMClient()
    provider = ProviderConfig(name="openai", model="openai/gpt-4o-mini")
    tokens = list(client.stream_completion(provider=provider, prompt="hello"))

    assert tokens == ["ok"]
    assert captured["suppress_debug_info"] is True


def test_runner_supports_prompt_concurrency(tmp_path: Path) -> None:
    storage = BenchmarkStorage(tmp_path / "bench.duckdb")
    run_id = "run-concurrent-prompts"
    provider = ProviderConfig(name="openai", model="gpt-4o-mini")
    storage.create_run(run_id=run_id, started_at=0.0, config_json="{}")
    storage.upsert_provider_snapshot(run_id=run_id, provider=provider)

    tracking_client = ConcurrencyTrackingClient(hold_s=0.05)
    runner = BenchmarkRunner(storage=storage, client=tracking_client)
    result = runner.run_provider(
        run_id=run_id,
        provider=provider,
        prompts=["p1", "p2", "p3", "p4"],
        target_rps=None,
        prompt_concurrency=4,
        slo=SLOConfig(max_ttft_s=2.0, max_e2e_s=2.0),
    )

    assert result.total_requests == 4
    assert result.success_requests == 4
    assert tracking_client.max_active_calls >= 2
