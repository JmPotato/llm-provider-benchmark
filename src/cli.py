from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import re
import sys
from threading import Lock
import time
import uuid

import typer

from providers import ProviderConfig, ProviderRegistry
from records import RequestRecord
from runner import BenchmarkRunner, LiteLLMClient, ProviderRunData, SLOConfig
from storage import BenchmarkStorage


logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    """Configure root logger from LLM_BENCH_LOG_LEVEL env var (default: WARNING)."""
    level_name = os.environ.get("LLM_BENCH_LOG_LEVEL", "WARNING").upper()
    level = getattr(logging, level_name, logging.WARNING)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


app = typer.Typer(no_args_is_help=True, help="LLM Provider benchmark CLI")
provider_app = typer.Typer(no_args_is_help=True, help="Provider management commands")
report_app = typer.Typer(no_args_is_help=True, help="Report commands")
prompt_app = typer.Typer(no_args_is_help=True, help="Prompt file utilities")
app.add_typer(provider_app, name="provider")
app.add_typer(report_app, name="report")
app.add_typer(prompt_app, name="prompt")


DEFAULT_PROVIDER_CONFIG = Path("providers.toml")
DEFAULT_BENCH_DB = Path("bench.duckdb")
DEFAULT_PROMPT_FILE = Path("prompts.txt")
DEFAULT_PROMPT_OUTPUT = Path("prompts.txt")
DEFAULT_PROMPT_INSTRUCTION = (
    "Prompts should be practical for benchmarking chatbot-style models."
)
DEFAULT_PROVIDER_TEST_PROMPT = "Reply with one short sentence."
QUANTILE_KEYS = ("p50", "p90", "p95", "p99")


@dataclass(slots=True)
class _RunProgress:
    run_id: str
    total_requests: int
    prompts_per_provider: int
    provider_concurrency: int
    prompt_concurrency: int
    provider_totals: dict[str, int]
    enabled: bool = True
    min_update_interval_s: float = 0.2
    completed_requests: int = field(init=False, default=0)
    success_requests: int = field(init=False, default=0)
    failed_requests: int = field(init=False, default=0)
    provider_completed: dict[str, int] = field(init=False)
    _lock: Lock = field(init=False, repr=False)
    _interactive: bool = field(init=False, repr=False)
    _start_perf: float = field(init=False, repr=False)
    _last_emit_perf: float = field(init=False, default=0.0, repr=False)
    _last_line_len: int = field(init=False, default=0, repr=False)
    _started: bool = field(init=False, default=False, repr=False)
    _finalized: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        self._lock = Lock()
        self._interactive = bool(self.enabled and sys.stderr.isatty())
        self._start_perf = time.perf_counter()
        self.provider_completed = {
            provider_name: 0 for provider_name in self.provider_totals
        }

    def start(self) -> None:
        if not self.enabled or self._started:
            return
        self._started = True
        typer.echo(
            (
                f"[run] run_id={self.run_id} providers={len(self.provider_totals)} "
                f"prompts/provider={self.prompts_per_provider} total_requests={self.total_requests} "
                f"provider_concurrency={self.provider_concurrency} prompt_concurrency={self.prompt_concurrency}"
            ),
            err=True,
        )

    def on_request_complete(self, request_record: RequestRecord) -> None:
        if not self.enabled:
            return

        with self._lock:
            self.completed_requests += 1
            if request_record.success:
                self.success_requests += 1
            else:
                self.failed_requests += 1
            self.provider_completed[request_record.provider_name] = (
                self.provider_completed.get(request_record.provider_name, 0) + 1
            )

            now = time.perf_counter()
            finished = self.completed_requests >= self.total_requests
            if not finished and now - self._last_emit_perf < self.min_update_interval_s:
                return
            self._last_emit_perf = now
            self._emit_progress(now=now, final=finished)

    def finalize(self) -> None:
        if not self.enabled:
            return

        with self._lock:
            if self._finalized:
                return
            self._finalized = True
            if self.completed_requests < self.total_requests:
                self._emit_progress(now=time.perf_counter(), final=True)
            elif self._interactive:
                typer.echo("", err=True)

    def _emit_progress(self, now: float, final: bool) -> None:
        percent = (
            (self.completed_requests / self.total_requests) * 100.0
            if self.total_requests > 0
            else 100.0
        )
        elapsed_s = max(now - self._start_perf, 1e-9)
        done_providers = sum(
            1
            for provider_name, total in self.provider_totals.items()
            if self.provider_completed.get(provider_name, 0) >= total
        )
        rps = self.completed_requests / elapsed_s
        line = (
            f"[progress] {self.completed_requests}/{self.total_requests} ({percent:5.1f}%) "
            f"ok={self.success_requests} fail={self.failed_requests} "
            f"providers_done={done_providers}/{len(self.provider_totals)} "
            f"rps={rps:6.2f}"
        )

        if self._interactive and not final:
            padded_line = line
            if len(line) < self._last_line_len:
                padded_line = line + (" " * (self._last_line_len - len(line)))
            self._last_line_len = len(line)
            typer.echo(f"\r{padded_line}", err=True, nl=False)
            return

        if self._interactive:
            padded_line = line
            if len(line) < self._last_line_len:
                padded_line = line + (" " * (self._last_line_len - len(line)))
            self._last_line_len = len(line)
            typer.echo(f"\r{padded_line}", err=True)
            return

        typer.echo(line, err=True)


def _load_prompts(prompt_file: Path) -> list[str]:
    lines = prompt_file.read_text(encoding="utf-8").splitlines()
    prompts: list[str] = []
    if prompt_file.suffix.lower() == ".jsonl":
        for line_number, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Invalid JSONL at line {line_number}: each row must be an object "
                    "with `prompt` or `content`"
                )

            prompt = payload.get("prompt")
            if prompt is None:
                prompt = payload.get("content")
            if prompt is not None:
                value = str(prompt).strip()
                if value:
                    prompts.append(value)
        return prompts

    for line in lines:
        value = line.strip()
        if value:
            prompts.append(value)
    return prompts


def _build_prompt_generation_task(count: int, instruction: str) -> str:
    return (
        "You are preparing benchmark prompts for LLM latency and quality testing.\n"
        f"Generate exactly {count} diverse user prompts.\n"
        "Constraints:\n"
        "- Cover mixed tasks: reasoning, coding, extraction, summarization, and creative writing.\n"
        "- Keep each prompt concise and realistic.\n"
        "- Return only a JSON array of strings.\n"
        f"Additional requirement: {instruction}\n"
    )


def _extract_generated_prompts(raw_output: str, count: int) -> list[str]:
    content = raw_output.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    prompts: list[str] = []
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, list):
        prompts = [str(item).strip() for item in payload if str(item).strip()]
    elif isinstance(payload, dict):
        values = payload.get("prompts")
        if isinstance(values, list):
            prompts = [str(item).strip() for item in values if str(item).strip()]

    if not prompts:
        for line in content.splitlines():
            candidate = line.strip()
            if not candidate:
                continue
            candidate = re.sub(r"^\s*[-*]\s*", "", candidate)
            candidate = re.sub(r"^\s*\d+[\.\)]\s*", "", candidate)
            candidate = candidate.strip().strip('"').strip("'")
            if candidate:
                prompts.append(candidate)

    if len(prompts) < count:
        raise ValueError(
            f"Model output has only {len(prompts)} prompts, expected at least {count}"
        )
    return prompts[:count]


def _validate_provider_response(
    provider: ProviderConfig, prompt: str = DEFAULT_PROVIDER_TEST_PROMPT
) -> int:
    client = LiteLLMClient()
    tokens = list(client.stream_completion(provider=provider, prompt=prompt))
    if not tokens:
        raise ValueError("Provider returned an empty response")
    return len(tokens)


def _validate_providers(
    providers: list[ProviderConfig],
    prompt: str = DEFAULT_PROVIDER_TEST_PROMPT,
) -> tuple[list[dict[str, object]], int]:
    results: list[dict[str, object]] = []
    passed = 0
    for provider in providers:
        try:
            output_tokens = _validate_provider_response(
                provider=provider, prompt=prompt
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "provider": provider.name,
                    "model": provider.model,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            continue

        results.append(
            {
                "provider": provider.name,
                "model": provider.model,
                "status": "ok",
                "output_tokens": output_tokens,
            }
        )
        passed += 1
    return results, passed


def _render_provider_validation_report(
    scope: str, results: list[dict[str, object]], passed: int, failed: int
) -> str:
    lines = [
        "Provider validation summary",
        f"Scope : {scope}",
        f"Total : {len(results)}",
        f"Passed: {passed}",
        f"Failed: {failed}",
        "",
        "Results:",
    ]
    for result in results:
        provider = str(result.get("provider", "-"))
        model = str(result.get("model", "-"))
        status = str(result.get("status", "failed"))
        if status == "ok":
            output_tokens = result.get("output_tokens", 0)
            lines.append(f"- OK   {provider} ({model}) output_tokens={output_tokens}")
            continue

        error_type = str(result.get("error_type", "Error"))
        error_message = str(result.get("error_message", ""))
        if error_message:
            lines.append(f"- FAIL {provider} ({model}) {error_type}: {error_message}")
        else:
            lines.append(f"- FAIL {provider} ({model}) {error_type}")
    return "\n".join(lines)


def _format_quantile(value: object, *, as_percent: bool = False) -> str:
    if value is None:
        return "-"

    numeric = float(value)
    if as_percent:
        return f"{numeric * 100:.2f}%"
    return f"{numeric:.4f}"


def _render_quantile_line(
    label: str,
    quantiles: dict[str, object],
    *,
    unit: str | None = None,
    as_percent: bool = False,
) -> str:
    label_with_unit = label if not unit else f"{label} ({unit})"
    count = int(quantiles.get("count") or 0)
    values = " ".join(
        f"{key}={_format_quantile(quantiles.get(key), as_percent=as_percent)}"
        for key in QUANTILE_KEYS
    )
    return f"- {label_with_unit:<18} count={count:<5} {values}"


def _render_report_summary(
    run_id: str,
    provider_summaries: dict[str, dict[str, object]],
    db: Path,
) -> str:
    lines = [
        "Benchmark summary",
        f"Run ID   : {run_id}",
        f"Providers: {len(provider_summaries)}",
        f"DB       : {db}",
    ]

    for provider_name in sorted(provider_summaries):
        summary = provider_summaries[provider_name]
        latency = summary.get("latency", {})
        throughput = summary.get("throughput", {})
        quality = summary.get("quality", {})

        lines.extend(
            [
                "",
                f"[{provider_name}]",
                "Latency:",
                _render_quantile_line("ttft", latency.get("ttft", {}), unit="s"),
                _render_quantile_line("tpot", latency.get("tpot", {}), unit="s"),
                _render_quantile_line("tbt", latency.get("tbt", {}), unit="s"),
                _render_quantile_line("e2e", latency.get("e2e", {}), unit="s"),
                _render_quantile_line("itl", latency.get("itl", {}), unit="s"),
                "Throughput:",
                _render_quantile_line("rps", throughput.get("rps", {}), unit="req/s"),
                _render_quantile_line("tps", throughput.get("tps", {}), unit="tok/s"),
                "Quality:",
                _render_quantile_line(
                    "goodput", quality.get("goodput", {}), unit="req/s"
                ),
                _render_quantile_line(
                    "error_rate", quality.get("error_rate", {}), as_percent=True
                ),
            ]
        )

    return "\n".join(lines)


def _parse_config_json(config_json: object) -> dict[str, object]:
    if not isinstance(config_json, str) or not config_json.strip():
        return {}
    try:
        payload = json.loads(config_json)
    except json.JSONDecodeError:
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _format_timestamp(value: object) -> str:
    if value is None:
        return "-"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "-"
    return datetime.fromtimestamp(numeric).astimezone().isoformat(timespec="seconds")


def _format_duration(value: object) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.3f}s"
    except (TypeError, ValueError):
        return "-"


def _render_report_list(runs: list[dict[str, object]], db: Path) -> str:
    lines = [
        "Benchmark runs",
        f"Total : {len(runs)}",
        f"DB    : {db}",
        "",
        "Runs:",
    ]
    for run in runs:
        run_id = str(run.get("run_id", "-"))
        started_at = _format_timestamp(run.get("started_at"))
        finished_at = _format_timestamp(run.get("finished_at"))
        duration_s = _format_duration(run.get("duration_s"))
        provider_count = int(run.get("provider_count") or 0)
        request_count = int(run.get("request_count") or 0)
        success_count = int(run.get("success_count") or 0)
        failed_count = int(run.get("failed_count") or 0)
        lines.append(
            (
                f"- {run_id} started={started_at} finished={finished_at} duration={duration_s} "
                f"providers={provider_count} requests={request_count} ok={success_count} fail={failed_count}"
            )
        )
    return "\n".join(lines)


@prompt_app.command("generate")
def prompt_generate(
    output: Path = typer.Option(
        DEFAULT_PROMPT_OUTPUT,
        "--output",
        help="Output prompt file path (.txt or .jsonl)",
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        help="Provider name used to generate prompts. Defaults to the first configured provider.",
    ),
    count: int = typer.Option(
        10, "--count", "-n", min=1, help="Number of prompts to generate"
    ),
    force: bool = typer.Option(
        False, "--force", help="Overwrite if output file already exists"
    ),
    format: str | None = typer.Option(
        None, "--format", help="Output format: txt or jsonl", hidden=True
    ),
    config: Path = typer.Option(
        DEFAULT_PROVIDER_CONFIG, "--config", help="Provider registry file", hidden=True
    ),
    instruction: str = typer.Option(
        DEFAULT_PROMPT_INSTRUCTION,
        "--instruction",
        help="Additional generation instruction",
        hidden=True,
    ),
) -> None:
    if output.exists() and not force:
        typer.echo(f"Output file already exists: {output}. Use --force to overwrite.")
        raise typer.Exit(1)

    if format is None:
        resolved_format = "jsonl" if output.suffix.lower() == ".jsonl" else "txt"
    else:
        resolved_format = format.lower().strip()

    if resolved_format not in {"txt", "jsonl"}:
        typer.echo("Unsupported format. Use txt or jsonl.")
        raise typer.Exit(1)

    registry = ProviderRegistry(config)
    provider_name = provider
    if not provider_name:
        configured_providers = registry.list_providers()
        if not configured_providers:
            typer.echo(
                "No providers configured. Add one with `llm-bench provider add`."
            )
            raise typer.Exit(1)
        provider_name = configured_providers[0].name

    try:
        provider_config = registry.get_provider(provider_name)
    except KeyError:
        typer.echo(f"Provider not found: {provider_name}")
        raise typer.Exit(1)

    request_prompt = _build_prompt_generation_task(count=count, instruction=instruction)
    client = LiteLLMClient()
    logger.debug("Generating %d prompts using provider %r (model=%s)", count, provider_name, provider_config.model)
    try:
        generated_text = "".join(
            client.stream_completion(provider=provider_config, prompt=request_prompt)
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Prompt generation failed: [%s] %s", type(exc).__name__, exc, exc_info=True)
        typer.echo(f"Prompt generation failed: {type(exc).__name__}: {exc}")
        raise typer.Exit(1)

    try:
        prompts = _extract_generated_prompts(generated_text, count=count)
    except ValueError as exc:
        typer.echo(f"Prompt generation parse failed: {exc}")
        raise typer.Exit(1)

    output.parent.mkdir(parents=True, exist_ok=True)
    if resolved_format == "txt":
        body = "\n".join(prompts)
    else:
        body = "\n".join(
            json.dumps({"prompt": prompt}, ensure_ascii=False) for prompt in prompts
        )
    output.write_text(body + "\n", encoding="utf-8")
    logger.debug("Wrote %d prompts to %s (format=%s)", len(prompts), output, resolved_format)

    typer.echo(
        json.dumps(
            {
                "output": str(output),
                "provider": provider_name,
                "format": resolved_format,
                "count": count,
            },
            ensure_ascii=False,
        )
    )


@provider_app.command("add")
def provider_add(
    name: str = typer.Option(..., "--name", help="Provider name"),
    model: str = typer.Option(..., "--model", help="Model identifier"),
    api_base: str | None = typer.Option(None, "--api-base", help="Provider base URL"),
    api_key_env: str | None = typer.Option(
        None, "--api-key-env", help="Environment variable storing API key"
    ),
    config: Path = typer.Option(
        DEFAULT_PROVIDER_CONFIG, "--config", help="Provider registry file", hidden=True
    ),
) -> None:
    provider = ProviderConfig(
        name=name,
        model=model,
        api_base=api_base,
        api_key_env=api_key_env,
    )

    registry = ProviderRegistry(config)
    registry.save_provider(provider)
    logger.debug("Provider %r added to %s", name, config)
    typer.echo(f"Provider added: {name}")


@provider_app.command("list")
def provider_list(
    config: Path = typer.Option(
        DEFAULT_PROVIDER_CONFIG, "--config", help="Provider registry file", hidden=True
    ),
) -> None:
    registry = ProviderRegistry(config)
    providers = registry.list_providers()
    if not providers:
        typer.echo("No providers configured.")
        return

    for provider in providers:
        api_base = provider.api_base or "-"
        api_key_env = provider.api_key_env or "-"
        typer.echo(f"{provider.name}\t{provider.model}\t{api_base}\t{api_key_env}")


@provider_app.command("remove")
def provider_remove(
    name: str = typer.Option(..., "--name", help="Provider name"),
    config: Path = typer.Option(
        DEFAULT_PROVIDER_CONFIG, "--config", help="Provider registry file", hidden=True
    ),
) -> None:
    registry = ProviderRegistry(config)
    try:
        registry.remove_provider(name)
    except KeyError:
        typer.echo(f"Provider not found: {name}")
        raise typer.Exit(1)
    typer.echo(f"Provider removed: {name}")


@provider_app.command("validate")
def provider_validate(
    name: str | None = typer.Option(
        None, "--name", help="Provider name (validate all when omitted)"
    ),
    prompt: str = typer.Option(
        DEFAULT_PROVIDER_TEST_PROMPT, "--prompt", "-p", help="Validation prompt"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output machine-readable JSON"
    ),
    config: Path = typer.Option(
        DEFAULT_PROVIDER_CONFIG, "--config", help="Provider registry file", hidden=True
    ),
) -> None:
    registry = ProviderRegistry(config)
    if name:
        try:
            providers = [registry.get_provider(name)]
        except KeyError:
            typer.echo(f"Provider not found: {name}")
            raise typer.Exit(1)
        scope = name
    else:
        providers = registry.list_providers()
        if not providers:
            typer.echo("No providers configured.")
            raise typer.Exit(1)
        scope = "all"

    results, passed = _validate_providers(providers=providers, prompt=prompt)
    failed = len(results) - passed
    payload = {
        "scope": scope,
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "results": results,
    }
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False))
    else:
        typer.echo(
            _render_provider_validation_report(
                scope=scope, results=results, passed=passed, failed=failed
            )
        )
    if failed:
        raise typer.Exit(1)


@app.command("run")
def run_benchmark(
    providers: str | None = typer.Option(
        None,
        "--providers",
        help="Comma-separated provider names. Defaults to all configured providers.",
    ),
    prompt_file: Path = typer.Option(
        DEFAULT_PROMPT_FILE, "--prompt-file", help="Prompt file (.txt or .jsonl)"
    ),
    db: Path = typer.Option(DEFAULT_BENCH_DB, "--db", "-d", help="DuckDB output file"),
    target_rps: float | None = typer.Option(
        None, "--rps", "--target-rps", help="Target requests per second"
    ),
    provider_concurrency: int = typer.Option(
        1,
        "--provider-concurrency",
        min=1,
        help="Max concurrent providers",
    ),
    prompt_concurrency: int = typer.Option(
        1,
        "--prompt-concurrency",
        min=1,
        help="Max concurrent prompts per provider",
    ),
    progress: bool = typer.Option(
        True,
        "--progress/--no-progress",
        help="Show live progress on stderr",
    ),
    max_ttft_s: float | None = typer.Option(
        None, "--max-ttft-s", help="SLO threshold for TTFT"
    ),
    max_e2e_s: float | None = typer.Option(
        None, "--max-e2e-s", help="SLO threshold for E2E latency"
    ),
    config: Path = typer.Option(
        DEFAULT_PROVIDER_CONFIG, "--config", help="Provider registry file", hidden=True
    ),
    run_id: str | None = typer.Option(
        None, "--run-id", help="Optional run id", hidden=True
    ),
) -> None:
    if target_rps is not None and target_rps <= 0:
        typer.echo("Target requests per second must be greater than 0.")
        raise typer.Exit(1)

    registry = ProviderRegistry(config)
    selected_providers: list[ProviderConfig] = []
    if providers:
        provider_names = [name.strip() for name in providers.split(",") if name.strip()]
        if not provider_names:
            typer.echo("No providers specified.")
            raise typer.Exit(1)

        seen_provider_names: set[str] = set()
        duplicate_provider_names: list[str] = []
        for provider_name in provider_names:
            if (
                provider_name in seen_provider_names
                and provider_name not in duplicate_provider_names
            ):
                duplicate_provider_names.append(provider_name)
            seen_provider_names.add(provider_name)
        if duplicate_provider_names:
            typer.echo(
                "Duplicate provider names are not allowed: "
                + ", ".join(duplicate_provider_names)
            )
            raise typer.Exit(1)

        for provider_name in provider_names:
            try:
                selected_providers.append(registry.get_provider(provider_name))
            except KeyError:
                typer.echo(f"Provider not found: {provider_name}")
                raise typer.Exit(1)
    else:
        selected_providers = registry.list_providers()
        if not selected_providers:
            typer.echo(
                "No providers configured. Add one with `llm-bench provider add`."
            )
            raise typer.Exit(1)
        provider_names = [provider.name for provider in selected_providers]

    if not prompt_file.exists():
        typer.echo(f"Prompt file not found: {prompt_file}")
        raise typer.Exit(1)
    logger.debug("Loading prompts from %s", prompt_file)
    try:
        prompts = _load_prompts(prompt_file)
    except ValueError as exc:
        typer.echo(f"Failed to load prompts: {exc}")
        raise typer.Exit(1)
    if not prompts:
        typer.echo("Prompt file does not contain usable prompts.")
        raise typer.Exit(1)
    logger.debug("Loaded %d prompts from %s", len(prompts), prompt_file)

    actual_run_id = run_id or f"run-{uuid.uuid4().hex[:8]}"
    run_progress = _RunProgress(
        run_id=actual_run_id,
        total_requests=len(selected_providers) * len(prompts),
        prompts_per_provider=len(prompts),
        provider_concurrency=provider_concurrency,
        prompt_concurrency=prompt_concurrency,
        provider_totals={
            provider.name: len(prompts) for provider in selected_providers
        },
        enabled=progress,
    )
    started_at = time.time()
    storage = BenchmarkStorage(db)
    run_created = False
    run_finished = False
    run_progress.start()
    logger.info(
        "Starting benchmark run %s: providers=%d prompts/provider=%d "
        "provider_concurrency=%d prompt_concurrency=%d target_rps=%s",
        actual_run_id, len(selected_providers), len(prompts),
        provider_concurrency, prompt_concurrency, target_rps,
    )
    try:
        storage.create_run(
            run_id=actual_run_id,
            started_at=started_at,
            config_json=json.dumps(
                {
                    "providers": provider_names,
                    "prompt_file": str(prompt_file),
                    "target_rps": target_rps,
                    "provider_concurrency": provider_concurrency,
                    "prompt_concurrency": prompt_concurrency,
                    "max_ttft_s": max_ttft_s,
                    "max_e2e_s": max_e2e_s,
                },
                ensure_ascii=True,
            ),
        )
        run_created = True
        for provider in selected_providers:
            storage.upsert_provider_snapshot(run_id=actual_run_id, provider=provider)

        provider_results: list[dict[str, object]] = []
        slo = SLOConfig(max_ttft_s=max_ttft_s, max_e2e_s=max_e2e_s)

        def run_provider_collect(provider: ProviderConfig) -> ProviderRunData:
            runner = BenchmarkRunner(storage=storage, client=LiteLLMClient())
            return runner.run_provider_collect(
                run_id=actual_run_id,
                provider=provider,
                prompts=prompts,
                target_rps=target_rps,
                prompt_concurrency=prompt_concurrency,
                slo=slo,
                on_request_complete=run_progress.on_request_complete
                if progress
                else None,
            )

        def persist_provider_data(run_data: ProviderRunData) -> dict[str, object]:
            storage.insert_request_records(run_data.request_records)
            storage.insert_token_events(run_data.token_events)
            storage.refresh_window_metrics(
                run_id=actual_run_id, provider_name=run_data.result.provider_name
            )
            return asdict(run_data.result)

        if provider_concurrency <= 1 or len(selected_providers) <= 1:
            for provider in selected_providers:
                run_data = run_provider_collect(provider)
                provider_results.append(persist_provider_data(run_data))
        else:
            max_workers = min(provider_concurrency, len(selected_providers))
            completed_results: dict[str, dict[str, object]] = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(run_provider_collect, provider): provider
                    for provider in selected_providers
                }
                for future in as_completed(futures):
                    provider = futures[future]
                    run_data = future.result()
                    completed_results[provider.name] = persist_provider_data(run_data)

            provider_results = [
                completed_results[provider.name] for provider in selected_providers
            ]

        storage.finish_run(run_id=actual_run_id, finished_at=time.time())
        run_finished = True
        logger.info("Benchmark run %s finished", actual_run_id)
        typer.echo(
            json.dumps(
                {
                    "run_id": actual_run_id,
                    "providers": provider_results,
                    "db": str(db),
                },
                ensure_ascii=False,
            )
        )
    finally:
        if run_created and not run_finished:
            try:
                storage.finish_run(run_id=actual_run_id, finished_at=time.time())
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Failed to mark run %s as finished during cleanup",
                    actual_run_id, exc_info=True,
                )
        run_progress.finalize()
        storage.close()


@report_app.command("summary")
def report_summary(
    run_id: str | None = typer.Option(
        None, "--run-id", help="Run identifier. Defaults to latest run."
    ),
    provider: str | None = typer.Option(None, "--provider", help="Provider name"),
    json_output: bool = typer.Option(
        False, "--json", help="Output machine-readable JSON"
    ),
    db: Path = typer.Option(DEFAULT_BENCH_DB, "--db", "-d", help="DuckDB output file"),
) -> None:
    storage = BenchmarkStorage(db)
    try:
        actual_run_id = run_id
        if not actual_run_id:
            runs = storage.list_runs()
            if not runs:
                typer.echo("No runs found.")
                raise typer.Exit(1)
            actual_run_id = str(runs[0]["run_id"])

        if provider:
            try:
                summary = storage.get_provider_summary(
                    run_id=actual_run_id, provider_name=provider
                )
            except KeyError:
                typer.echo(f"Provider not found in run: {provider}")
                raise typer.Exit(1)
            if json_output:
                typer.echo(
                    json.dumps(
                        {
                            "run_id": actual_run_id,
                            "provider": provider,
                            "summary": summary,
                        },
                        ensure_ascii=False,
                    )
                )
            else:
                typer.echo(
                    _render_report_summary(
                        run_id=actual_run_id,
                        provider_summaries={provider: summary},
                        db=db,
                    )
                )
            return

        providers = storage.list_run_providers(actual_run_id)
        if not providers:
            typer.echo(f"No providers found for run: {actual_run_id}")
            raise typer.Exit(1)

        summaries = {
            provider_name: storage.get_provider_summary(
                run_id=actual_run_id, provider_name=provider_name
            )
            for provider_name in providers
        }
        if json_output:
            typer.echo(
                json.dumps(
                    {"run_id": actual_run_id, "providers": summaries},
                    ensure_ascii=False,
                )
            )
        else:
            typer.echo(
                _render_report_summary(
                    run_id=actual_run_id, provider_summaries=summaries, db=db
                )
            )
    finally:
        storage.close()


@report_app.command("list")
def report_list(
    limit: int | None = typer.Option(
        None, "--limit", "-n", min=1, help="Maximum number of runs to show"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output machine-readable JSON"
    ),
    db: Path = typer.Option(DEFAULT_BENCH_DB, "--db", "-d", help="DuckDB output file"),
) -> None:
    storage = BenchmarkStorage(db)
    try:
        runs = storage.list_runs_with_stats()
        if not runs:
            typer.echo("No runs found.")
            raise typer.Exit(1)

        if limit is not None:
            runs = runs[:limit]

        output_runs: list[dict[str, object]] = []
        for run in runs:
            config = _parse_config_json(run.get("config_json"))
            output_runs.append(
                {
                    "run_id": run.get("run_id"),
                    "started_at": run.get("started_at"),
                    "finished_at": run.get("finished_at"),
                    "duration_s": run.get("duration_s"),
                    "started_at_iso": _format_timestamp(run.get("started_at")),
                    "finished_at_iso": _format_timestamp(run.get("finished_at")),
                    "provider_count": run.get("provider_count"),
                    "request_count": run.get("request_count"),
                    "success_count": run.get("success_count"),
                    "failed_count": run.get("failed_count"),
                    "prompt_file": config.get("prompt_file"),
                    "target_rps": config.get("target_rps"),
                }
            )

        if json_output:
            typer.echo(
                json.dumps(
                    {
                        "db": str(db),
                        "total": len(output_runs),
                        "runs": output_runs,
                    },
                    ensure_ascii=False,
                )
            )
            return

        typer.echo(_render_report_list(output_runs, db=db))
    finally:
        storage.close()


@report_app.command("remove")
def report_remove(
    run_id: str = typer.Option(..., "--run-id", help="Run identifier to remove"),
    db: Path = typer.Option(DEFAULT_BENCH_DB, "--db", "-d", help="DuckDB output file"),
) -> None:
    storage = BenchmarkStorage(db)
    try:
        deleted = storage.delete_run(run_id=run_id)
        if not deleted:
            typer.echo(f"Run not found: {run_id}")
            raise typer.Exit(1)
        typer.echo(f"Run removed: {run_id}")
    finally:
        storage.close()


def main() -> None:
    _setup_logging()
    app()


if __name__ == "__main__":
    main()
