# llm-provider-benchmark

A LiteLLM-based multi-provider benchmark CLI that measures:

- Custom providers (`api_base`, `api_key_env`, `model`)
- Latency metrics: `TTFT`, `TPOT`, `ITL`, `TBT`, `E2E`
- Throughput metrics: `TPS`, `RPS`
- Quality metrics: `Goodput`, `Error Rate`
- P50/P90/P95/P99 quantiles for all metrics (backed by DuckDB)

## Quick Start

Sync the environment first (installs the `llm-bench` command):

```bash
uv sync
```

### 1. Add a provider

```bash
uv run llm-bench provider add \
  --name openai \
  --model gpt-4o-mini \
  --api-base https://api.openai.com/v1 \
  --api-key-env OPENAI_API_KEY
```

`provider add` makes a live test request before saving. The provider is only written to `providers.toml` if the model returns a valid response.

### 2. List providers

```bash
uv run llm-bench provider list
```

### 3. Validate providers

Validate all configured providers (default) or a single one:

```bash
uv run llm-bench provider validate
uv run llm-bench provider validate --name openai
uv run llm-bench provider validate --json
```

### 4. Generate a prompt file

Defaults to `./prompts.txt` using the first configured provider:

```bash
uv run llm-bench prompt generate --count 50
```

### 5. Run a benchmark

Defaults to `./prompts.txt` and all configured providers:

```bash
uv run llm-bench run \
  --provider-concurrency 2 \
  --prompt-concurrency 8
```

- `--provider-concurrency`: max number of providers running in parallel
- `--prompt-concurrency`: max concurrent prompts per provider
- Live progress is printed to stderr by default; use `--no-progress` to disable

### 6. View a report

Show the latest run (or a specific one):

```bash
uv run llm-bench report summary
uv run llm-bench report summary --json
```

### 7. List historical runs

```bash
uv run llm-bench report list
```

### 8. Delete a run

```bash
uv run llm-bench report remove --run-id run-xxxxxx
```

## Prompt File Format

- **txt**: one prompt per line (blank lines are ignored)
- **jsonl**: one JSON object per line; reads the `prompt` or `content` field

## Metric Definitions

| Metric | Definition |
|--------|------------|
| `TTFT` | `first_token_at − request_started_at` |
| `ITL` | Inter-token latency (gap between consecutive tokens) |
| `TBT` | `last_token_at − first_token_at` |
| `TPOT` | `TBT / max(output_tokens − 1, 1)` |
| `E2E` | `response_done_at − request_started_at` |
| `RPS` | Requests completed per 1-second window |
| `TPS` | Output tokens produced per 1-second window |
| `Goodput` | Requests passing SLO per 1-second window |
| `Error Rate` | Failed requests / total requests per 1-second window |

## Logging

Set `LLM_BENCH_LOG_LEVEL` to enable diagnostic output (default: `WARNING`):

```bash
LLM_BENCH_LOG_LEVEL=DEBUG uv run llm-bench run
```

## Running Tests

```bash
uv run pytest
```
