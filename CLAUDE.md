# CLAUDE.md

## Project Overview

LiteLLM-based multi-provider LLM benchmark CLI. Measures latency (TTFT, ITL, TBT, TPOT, E2E), throughput (RPS, TPS), and quality (Goodput, Error Rate) across configurable providers. All metrics stored in DuckDB and reported as P50/P90/P95/P99 quantiles.

## Commands

```bash
uv sync                  # Install deps and register llm-bench entry point
uv run pytest            # Run all tests
uv run pytest -x -q      # Fail fast, quiet output
uv run llm-bench --help  # CLI help
```

## Architecture

```
src/
  cli.py        # Typer CLI commands (run, provider, prompt, report)
  runner.py     # BenchmarkRunner + LiteLLMClient — executes requests, collects timing
  storage.py    # BenchmarkStorage — DuckDB schema, insert, query, window metrics
  metrics.py    # Pure computation: latency/window metrics, quantile helpers
  providers.py  # ProviderConfig dataclass + ProviderRegistry (TOML read/write)
  records.py    # RequestRecord and TokenEvent dataclasses
main.py         # Entry point: adds src/ to sys.path, calls cli.main()
providers.toml  # Provider registry (git-tracked sample; runtime file is local)
prompts.txt     # Benchmark prompt file (txt or jsonl)
bench.duckdb    # Runtime database (not committed)
```

## Key Conventions

**Package structure**: All source lives in `src/` and is loaded via `sys.path.insert` in `main.py`. Modules import each other by bare name (e.g. `from providers import ProviderConfig`), not as a package.

**Dataclasses**: Use `@dataclass(slots=True)` throughout for memory efficiency.

**Concurrency**: Two-level ThreadPoolExecutor — provider-level (across providers) and prompt-level (within a provider). Rate limiting uses a `Lock`-guarded schedule clock in `BenchmarkRunner._wait_for_schedule`.

**Storage**: DuckDB with five tables: `runs`, `providers`, `requests`, `token_events`, `window_metrics_1s`. Window metrics are materialized after each provider completes via `refresh_window_metrics`. All SQL parameters use `?` placeholders — no string interpolation of user data. Column names in `_quantiles_from_*` helpers are internal constants, not user input.

**TOML serialization**: `ProviderRegistry._write_raw` hand-writes TOML to preserve formatting. `extra_headers` is written as a nested table section.

**Logging**: Standard `logging` module. Default level `WARNING` (silent). Override with `LLM_BENCH_LOG_LEVEL=DEBUG`. Configured in `cli._setup_logging()`, called once from `main()`. Each module has its own `logger = logging.getLogger(__name__)`.

**CLI output**: Normal output to stdout (JSON), progress/status to stderr. `--json` flag available on `validate`, `report summary`, and `report list`.

**SLO**: Configured via `--max-ttft-s` / `--max-e2e-s` on `run`. Evaluated per-request in `BenchmarkRunner._is_slo_passed`.

## Testing

Tests use `typer.testing.CliRunner` and inject fake clients via dependency injection — `BenchmarkRunner` accepts a `client` argument of type `LLMClientProtocol`. No real network calls in tests.

Test files mirror source modules: `test_cli.py`, `test_runner.py`, `test_storage.py`, `test_metrics.py`, `test_providers.py`.

Fixtures use `tmp_path` for filesystem isolation. DuckDB connections use `:memory:` or temp paths.

## Provider Config Fields

| Field | Required | Description |
|-------|----------|-------------|
| `model` | Yes | LiteLLM model string (e.g. `gpt-4o-mini`, `anthropic/claude-opus-4.6`) |
| `api_base` | No | Override base URL |
| `api_key_env` | No | Env var name holding the API key |
| `extra_headers` | No | Additional HTTP headers (TOML subtable) |
| `temperature` | No | Sampling temperature |
| `max_tokens` | No | Max output tokens |
| `timeout_s` | No | Per-request timeout in seconds |
