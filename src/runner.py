from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
from threading import Lock
import time
from typing import Callable, Iterable, Protocol

from metrics import LatencyTimeline, compute_latency_metrics
from providers import ProviderConfig
from records import RequestRecord, TokenEvent
from storage import BenchmarkStorage


class LLMClientProtocol(Protocol):
    def stream_completion(self, provider: ProviderConfig, prompt: str) -> Iterable[str]:
        ...


RequestCompleteCallback = Callable[[RequestRecord], None]
RequestExecutionOutput = tuple[int, RequestRecord, list[TokenEvent], bool, bool]


@dataclass(slots=True)
class SLOConfig:
    max_ttft_s: float | None = None
    max_e2e_s: float | None = None


@dataclass(slots=True)
class ProviderRunResult:
    run_id: str
    provider_name: str
    total_requests: int
    success_requests: int
    failed_requests: int
    slo_passed_requests: int


@dataclass(slots=True)
class ProviderRunData:
    result: ProviderRunResult
    request_records: list[RequestRecord]
    token_events: list[TokenEvent]


class LiteLLMClient:
    @staticmethod
    def _configure_litellm() -> None:
        import litellm

        # Keep benchmark output clean by hiding LiteLLM guidance banners in error paths.
        litellm.suppress_debug_info = True

    def stream_completion(self, provider: ProviderConfig, prompt: str) -> Iterable[str]:
        self._configure_litellm()
        from litellm import completion

        request_options: dict[str, object] = {
            "model": provider.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }

        if provider.api_base:
            request_options["api_base"] = provider.api_base
        if provider.api_key_env:
            api_key = os.getenv(provider.api_key_env)
            if not api_key:
                raise ValueError(f"Missing API key from environment variable {provider.api_key_env!r}")
            request_options["api_key"] = api_key
        if provider.extra_headers:
            request_options["extra_headers"] = provider.extra_headers
        if provider.temperature is not None:
            request_options["temperature"] = provider.temperature
        if provider.max_tokens is not None:
            request_options["max_tokens"] = provider.max_tokens
        if provider.timeout_s is not None:
            request_options["timeout"] = provider.timeout_s

        stream = completion(**request_options)
        for chunk in stream:
            token_text = self._extract_text_from_chunk(chunk)
            if token_text:
                yield token_text

    @staticmethod
    def _extract_text_from_chunk(chunk: object) -> str:
        # Supports both dict-style and object-style chunk payloads.
        if isinstance(chunk, dict):
            choices = chunk.get("choices") or []
            if not choices:
                return ""
            first_choice = choices[0] or {}
            delta = first_choice.get("delta", {})
            content = delta.get("content")
            if content is None:
                content = first_choice.get("text")
            return str(content or "")

        choices = getattr(chunk, "choices", None)
        if not choices:
            return ""
        first_choice = choices[0]
        delta = getattr(first_choice, "delta", None)
        content = getattr(delta, "content", None) if delta is not None else None
        if content is None:
            content = getattr(first_choice, "text", None)
        return str(content or "")


class BenchmarkRunner:
    def __init__(
        self,
        storage: BenchmarkStorage,
        client: LLMClientProtocol | None = None,
        clock: callable = time.perf_counter,
        sleep_fn: callable = time.sleep,
    ) -> None:
        self.storage = storage
        self.client = client or LiteLLMClient()
        self.clock = clock
        self.sleep_fn = sleep_fn
        self._schedule_lock = Lock()

    def run_provider(
        self,
        run_id: str,
        provider: ProviderConfig,
        prompts: list[str],
        target_rps: float | None,
        prompt_concurrency: int = 1,
        slo: SLOConfig | None = None,
        on_request_complete: RequestCompleteCallback | None = None,
    ) -> ProviderRunResult:
        run_data = self.run_provider_collect(
            run_id=run_id,
            provider=provider,
            prompts=prompts,
            target_rps=target_rps,
            prompt_concurrency=prompt_concurrency,
            slo=slo,
            on_request_complete=on_request_complete,
        )
        self._persist_provider_data(run_data=run_data)
        return run_data.result

    def run_provider_collect(
        self,
        run_id: str,
        provider: ProviderConfig,
        prompts: list[str],
        target_rps: float | None,
        prompt_concurrency: int = 1,
        slo: SLOConfig | None = None,
        on_request_complete: RequestCompleteCallback | None = None,
    ) -> ProviderRunData:
        if prompt_concurrency < 1:
            raise ValueError("prompt_concurrency must be >= 1")
        if target_rps is not None and target_rps <= 0:
            raise ValueError("target_rps must be > 0")

        request_records: list[RequestRecord] = []
        token_events: list[TokenEvent] = []
        success_requests = 0
        failed_requests = 0
        slo_passed_requests = 0

        interval_s = (1.0 / target_rps) if target_rps is not None else None

        request_outputs = self._run_prompts(
            run_id=run_id,
            provider=provider,
            prompts=prompts,
            interval_s=interval_s,
            prompt_concurrency=prompt_concurrency,
            slo=slo,
            on_request_complete=on_request_complete,
        )
        for _, request_record, request_token_events, success, slo_passed in request_outputs:
            request_records.append(request_record)
            token_events.extend(request_token_events)
            if success:
                success_requests += 1
            else:
                failed_requests += 1
            if slo_passed:
                slo_passed_requests += 1

        return ProviderRunData(
            result=ProviderRunResult(
                run_id=run_id,
                provider_name=provider.name,
                total_requests=len(prompts),
                success_requests=success_requests,
                failed_requests=failed_requests,
                slo_passed_requests=slo_passed_requests,
            ),
            request_records=request_records,
            token_events=token_events,
        )

    def _persist_provider_data(self, run_data: ProviderRunData) -> None:
        provider_name = run_data.result.provider_name
        self.storage.insert_request_records(run_data.request_records)
        self.storage.insert_token_events(run_data.token_events)
        self.storage.refresh_window_metrics(run_id=run_data.result.run_id, provider_name=provider_name)

    def _run_prompts(
        self,
        run_id: str,
        provider: ProviderConfig,
        prompts: list[str],
        interval_s: float | None,
        prompt_concurrency: int,
        slo: SLOConfig | None,
        on_request_complete: RequestCompleteCallback | None,
    ) -> list[RequestExecutionOutput]:
        if prompt_concurrency <= 1 or len(prompts) <= 1:
            outputs: list[RequestExecutionOutput] = []
            next_schedule_at = self.clock() if interval_s is not None else None
            for request_index, prompt in enumerate(prompts):
                output = self._run_single_request(
                    run_id=run_id,
                    provider=provider,
                    request_index=request_index,
                    prompt=prompt,
                    scheduled_at=next_schedule_at,
                    slo=slo,
                )
                self._notify_request_complete(callback=on_request_complete, request_record=output[1])
                outputs.append(output)
                if interval_s is not None and next_schedule_at is not None:
                    next_schedule_at += interval_s
            return outputs

        base_schedule_at = self.clock() if interval_s is not None else None
        max_workers = min(prompt_concurrency, len(prompts))
        outputs = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self._run_single_request,
                    run_id=run_id,
                    provider=provider,
                    request_index=request_index,
                    prompt=prompt,
                    scheduled_at=(
                        base_schedule_at + (request_index * interval_s)
                        if base_schedule_at is not None and interval_s is not None
                        else None
                    ),
                    slo=slo,
                )
                for request_index, prompt in enumerate(prompts)
            ]
            for future in as_completed(futures):
                output = future.result()
                self._notify_request_complete(callback=on_request_complete, request_record=output[1])
                outputs.append(output)

        outputs.sort(key=lambda item: item[0])
        return outputs

    def _run_single_request(
        self,
        run_id: str,
        provider: ProviderConfig,
        request_index: int,
        prompt: str,
        scheduled_at: float | None,
        slo: SLOConfig | None,
    ) -> tuple[int, RequestRecord, list[TokenEvent], bool, bool]:
        request_id = f"{provider.name}-{request_index}"
        request_started_at = self._wait_for_schedule(scheduled_at)

        token_texts: list[str] = []
        token_timestamps: list[float] = []
        first_token_at: float | None = None
        success = True
        error_type: str | None = None
        error_message: str | None = None

        try:
            for token_text in self.client.stream_completion(provider, prompt):
                token_timestamp = self.clock()
                token_timestamps.append(token_timestamp)
                token_texts.append(token_text)
                if first_token_at is None:
                    first_token_at = token_timestamp
        except Exception as exc:  # noqa: BLE001
            success = False
            error_type = type(exc).__name__
            error_message = str(exc)

        response_done_at = self.clock()
        latency = compute_latency_metrics(
            LatencyTimeline(
                request_sent_at=request_started_at,
                first_token_at=first_token_at,
                token_timestamps=token_timestamps,
                response_done_at=response_done_at,
            )
        )
        slo_passed = self._is_slo_passed(latency=latency, success=success, slo=slo)
        output_text = "".join(token_texts)
        output_tokens = self._count_output_tokens(
            provider=provider,
            output_text=output_text,
            fallback_count=len(token_texts),
        )

        request_record = RequestRecord(
            run_id=run_id,
            provider_name=provider.name,
            request_id=request_id,
            prompt=prompt,
            request_started_at=request_started_at,
            first_token_at=first_token_at,
            response_done_at=response_done_at,
            output_tokens=output_tokens,
            success=success,
            error_type=error_type,
            error_message=error_message,
            ttft_s=latency.ttft,
            tbt_s=latency.tbt,
            tpot_s=latency.tpot,
            e2e_s=latency.e2e,
            slo_passed=slo_passed,
        )
        token_events = [
            TokenEvent(
                run_id=run_id,
                provider_name=provider.name,
                request_id=request_id,
                token_index=token_index,
                token_timestamp=token_timestamp,
                token_text=token_text,
            )
            for token_index, (token_timestamp, token_text) in enumerate(zip(token_timestamps, token_texts))
        ]
        return request_index, request_record, token_events, success, slo_passed

    def _wait_for_schedule(self, next_schedule_at: float | None) -> float:
        if next_schedule_at is None:
            return self.clock()

        with self._schedule_lock:
            now = self.clock()
            if now < next_schedule_at:
                self.sleep_fn(next_schedule_at - now)
                return self.clock()
            return now

    @staticmethod
    def _notify_request_complete(
        callback: RequestCompleteCallback | None,
        request_record: RequestRecord,
    ) -> None:
        if callback is None:
            return
        try:
            callback(request_record)
        except Exception:  # noqa: BLE001
            return

    @staticmethod
    def _count_output_tokens(provider: ProviderConfig, output_text: str, fallback_count: int) -> int:
        if not output_text:
            return 0
        try:
            LiteLLMClient._configure_litellm()
            from litellm import token_counter

            token_count = int(
                token_counter(
                    model=provider.model,
                    text=output_text,
                    count_response_tokens=True,
                )
            )
            if token_count >= 0:
                return token_count
        except Exception:  # noqa: BLE001
            pass
        return fallback_count

    @staticmethod
    def _is_slo_passed(latency: object, success: bool, slo: SLOConfig | None) -> bool:
        if not success:
            return False
        if slo is None:
            return True

        ttft = getattr(latency, "ttft", None)
        e2e = getattr(latency, "e2e", None)

        if slo.max_ttft_s is not None:
            if ttft is None or ttft > slo.max_ttft_s:
                return False
        if slo.max_e2e_s is not None and (e2e is None or e2e > slo.max_e2e_s):
            return False
        return True
