# llm-provider-benchmark

基于 LiteLLM 的多 Provider Benchmark CLI，支持：

- 自定义 Provider（`api_base`、`api_key_env`、`model`）
- 延迟指标：`TTFT`、`TPOT`、`ITL`、`TBT`、`E2E`
- 吞吐指标：`TPS`、`RPS`
- 综合质量指标：`Goodput`、`Error Rate`
- 全部指标提供 `P50/P90/P95/P99` 分位数统计（DuckDB）

## 快速开始

首次使用先同步环境（会安装 `llm-bench` 命令）：

```bash
uv sync
```

1. 添加 Provider：

```bash
uv run llm-bench provider add \
  --name openai \
  --model gpt-4o-mini \
  --api-base https://api.openai.com/v1 \
  --api-key-env OPENAI_API_KEY
```

`provider add` 会先使用给定 `api_base`、`api_key_env` 和 `model` 做一次请求验证，只有拿到模型响应才会写入 provider 列表。

2. 查看 Provider：

```bash
uv run llm-bench provider list
```

3. 验证 Provider（默认验证 `providers.toml` 里的全部 Provider）：

```bash
uv run llm-bench provider validate
```

只验证单个 Provider：

```bash
uv run llm-bench provider validate
uv run llm-bench provider validate --name openai
uv run llm-bench provider validate --json
```

4. 生成 prompt 文件（默认输出 `./prompts.txt`，默认使用第一个已配置 Provider）：

```bash
uv run llm-bench prompt generate --count 50
```

5. 运行 Benchmark（默认读取 `./prompts.txt`，默认跑全部已配置 Provider）：

```bash
uv run llm-bench run \
  --provider-concurrency 2 \
  --prompt-concurrency 8
```

- `--provider-concurrency`：同时并发执行的 provider 数量上限
- `--prompt-concurrency`：单个 provider 内 prompt 并发数上限
- 默认会在终端实时输出进度（`stderr`），可用 `--no-progress` 关闭

6. 查看报告（默认查看最新一次 run）：

```bash
uv run llm-bench report summary
uv run llm-bench report summary --json
```

7. 查看历史 run 列表（包含 run id、开始/结束时间、运行时长、请求统计）：

```bash
uv run llm-bench report list
```

8. 删除指定 run/report 结果：

```bash
uv run llm-bench report remove --run-id run-xxxxxx
```

## Prompt 文件格式

- `txt`：每行一个 prompt（空行忽略）
- `jsonl`：每行 JSON，读取 `prompt` 或 `content` 字段

## 指标口径

- `TTFT`：`first_token_at - request_started_at`
- `ITL`：相邻 token 到达间隔
- `TBT`：`last_token_at - first_token_at`
- `TPOT`：`TBT / max(output_tokens - 1, 1)`
- `E2E`：`response_done_at - request_started_at`
- `RPS`：每 1 秒窗口完成请求数
- `TPS`：每 1 秒窗口输出 token 数
- `Goodput`：每 1 秒窗口内通过 SLO 的请求数
- `Error Rate`：每 1 秒窗口失败请求占比

## 测试

```bash
uv run pytest
```
