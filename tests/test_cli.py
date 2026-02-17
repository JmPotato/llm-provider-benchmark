from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cli import app


class FakeLiteLLMClient:
    def stream_completion(self, provider, prompt):  # noqa: ANN001
        for token in ["tok-1", "tok-2"]:
            yield token


class FakePromptGeneratorClient:
    def stream_completion(self, provider, prompt):  # noqa: ANN001
        payload = '["Generated prompt 1", "Generated prompt 2", "Generated prompt 3"]'
        for token in [payload[:25], payload[25:45], payload[45:]]:
            yield token


class FailingLiteLLMClient:
    def stream_completion(self, provider, prompt):  # noqa: ANN001
        raise RuntimeError("request rejected")


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def default_fake_litellm_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cli.LiteLLMClient", FakeLiteLLMClient)


def test_provider_add_list_remove(cli_runner: CliRunner, tmp_path: Path) -> None:
    config_path = tmp_path / "providers.toml"

    add_result = cli_runner.invoke(
        app,
        [
            "provider",
            "add",
            "--name",
            "openai",
            "--model",
            "gpt-4o-mini",
            "--api-base",
            "https://api.openai.com/v1",
            "--api-key-env",
            "OPENAI_API_KEY",
            "--config",
            str(config_path),
        ],
    )
    assert add_result.exit_code == 0

    list_result = cli_runner.invoke(
        app, ["provider", "list", "--config", str(config_path)]
    )
    assert list_result.exit_code == 0
    assert "openai" in list_result.stdout
    assert "gpt-4o-mini" in list_result.stdout

    remove_result = cli_runner.invoke(
        app,
        ["provider", "remove", "--name", "openai", "--config", str(config_path)],
    )
    assert remove_result.exit_code == 0

    list_result_after = cli_runner.invoke(
        app, ["provider", "list", "--config", str(config_path)]
    )
    assert list_result_after.exit_code == 0
    assert "No providers configured." in list_result_after.stdout


def test_provider_add_fails_validation_and_not_persisted(
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "providers.toml"
    monkeypatch.setattr("cli.LiteLLMClient", FailingLiteLLMClient)

    add_result = cli_runner.invoke(
        app,
        [
            "provider",
            "add",
            "--name",
            "openai",
            "--model",
            "gpt-4o-mini",
            "--api-base",
            "https://api.openai.com/v1",
            "--api-key-env",
            "OPENAI_API_KEY",
            "--config",
            str(config_path),
        ],
    )
    assert add_result.exit_code != 0
    assert "Provider validation failed" in add_result.stdout

    list_result = cli_runner.invoke(
        app, ["provider", "list", "--config", str(config_path)]
    )
    assert list_result.exit_code == 0
    assert "No providers configured." in list_result.stdout


def test_provider_validate_all(cli_runner: CliRunner, tmp_path: Path) -> None:
    config_path = tmp_path / "providers.toml"
    add_result = cli_runner.invoke(
        app,
        [
            "provider",
            "add",
            "--name",
            "openai",
            "--model",
            "gpt-4o-mini",
            "--config",
            str(config_path),
        ],
    )
    assert add_result.exit_code == 0

    result = cli_runner.invoke(
        app,
        [
            "provider",
            "validate",
            "--json",
            "--config",
            str(config_path),
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["scope"] == "all"
    assert payload["total"] == 1
    assert payload["passed"] == 1
    assert payload["failed"] == 0
    assert payload["results"][0]["provider"] == "openai"
    assert payload["results"][0]["status"] == "ok"


def test_provider_validate_single_provider(
    cli_runner: CliRunner, tmp_path: Path
) -> None:
    config_path = tmp_path / "providers.toml"
    add_result = cli_runner.invoke(
        app,
        [
            "provider",
            "add",
            "--name",
            "openai",
            "--model",
            "gpt-4o-mini",
            "--config",
            str(config_path),
        ],
    )
    assert add_result.exit_code == 0

    result = cli_runner.invoke(
        app,
        [
            "provider",
            "validate",
            "--name",
            "openai",
            "--json",
            "--config",
            str(config_path),
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["scope"] == "openai"
    assert payload["total"] == 1
    assert payload["passed"] == 1
    assert payload["failed"] == 0
    assert payload["results"][0]["provider"] == "openai"


def test_provider_validate_default_output_is_human_readable(
    cli_runner: CliRunner, tmp_path: Path
) -> None:
    config_path = tmp_path / "providers.toml"
    add_result = cli_runner.invoke(
        app,
        [
            "provider",
            "add",
            "--name",
            "openai",
            "--model",
            "gpt-4o-mini",
            "--config",
            str(config_path),
        ],
    )
    assert add_result.exit_code == 0

    result = cli_runner.invoke(
        app,
        [
            "provider",
            "validate",
            "--config",
            str(config_path),
        ],
    )
    assert result.exit_code == 0
    assert "Provider validation summary" in result.stdout
    assert "Scope : all" in result.stdout
    assert "Passed: 1" in result.stdout
    assert "- OK   openai (gpt-4o-mini) output_tokens=2" in result.stdout


def test_provider_validate_fails_when_provider_missing(
    cli_runner: CliRunner, tmp_path: Path
) -> None:
    result = cli_runner.invoke(
        app,
        [
            "provider",
            "validate",
            "--name",
            "missing",
            "--config",
            str(tmp_path / "providers.toml"),
        ],
    )
    assert result.exit_code != 0
    assert "Provider not found" in result.stdout


def test_provider_validate_returns_nonzero_if_any_failed(
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "providers.toml"
    add_result = cli_runner.invoke(
        app,
        [
            "provider",
            "add",
            "--name",
            "openai",
            "--model",
            "gpt-4o-mini",
            "--config",
            str(config_path),
        ],
    )
    assert add_result.exit_code == 0

    monkeypatch.setattr("cli.LiteLLMClient", FailingLiteLLMClient)
    result = cli_runner.invoke(
        app,
        [
            "provider",
            "validate",
            "--json",
            "--config",
            str(config_path),
        ],
    )
    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    assert payload["total"] == 1
    assert payload["passed"] == 0
    assert payload["failed"] == 1
    assert payload["results"][0]["provider"] == "openai"
    assert payload["results"][0]["status"] == "failed"
    assert payload["results"][0]["error_type"] == "RuntimeError"


def test_run_and_summary_report_with_fake_client(
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "providers.toml"
    db_path = tmp_path / "bench.duckdb"
    prompts_path = tmp_path / "prompts.txt"
    prompts_path.write_text("hello\nworld\n", encoding="utf-8")

    add_result = cli_runner.invoke(
        app,
        [
            "provider",
            "add",
            "--name",
            "openai",
            "--model",
            "gpt-4o-mini",
            "--api-key-env",
            "OPENAI_API_KEY",
            "--config",
            str(config_path),
        ],
    )
    assert add_result.exit_code == 0

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("cli.LiteLLMClient", FakeLiteLLMClient)

    run_result = cli_runner.invoke(
        app,
        [
            "run",
            "--providers",
            "openai",
            "--prompt-file",
            str(prompts_path),
            "--config",
            str(config_path),
            "--db",
            str(db_path),
            "--run-id",
            "run-cli",
            "--max-ttft-s",
            "1.0",
            "--max-e2e-s",
            "2.0",
        ],
    )
    assert run_result.exit_code == 0
    run_payload = json.loads(run_result.stdout)
    assert run_payload["run_id"] == "run-cli"
    assert run_payload["providers"][0]["provider_name"] == "openai"
    assert run_payload["providers"][0]["total_requests"] == 2

    report_result = cli_runner.invoke(
        app,
        [
            "report",
            "summary",
            "--run-id",
            "run-cli",
            "--provider",
            "openai",
            "--db",
            str(db_path),
        ],
    )
    assert report_result.exit_code == 0
    assert "Benchmark summary" in report_result.stdout
    assert "Run ID   : run-cli" in report_result.stdout
    assert "Providers: 1" in report_result.stdout
    assert "[openai]" in report_result.stdout
    assert "Latency:" in report_result.stdout

    report_json_result = cli_runner.invoke(
        app,
        [
            "report",
            "summary",
            "--run-id",
            "run-cli",
            "--provider",
            "openai",
            "--json",
            "--db",
            str(db_path),
        ],
    )
    assert report_json_result.exit_code == 0
    summary_payload = json.loads(report_json_result.stdout)
    assert summary_payload["run_id"] == "run-cli"
    assert summary_payload["provider"] == "openai"
    assert "latency" in summary_payload["summary"]
    assert "ttft" in summary_payload["summary"]["latency"]

    report_all_result = cli_runner.invoke(
        app,
        [
            "report",
            "summary",
            "--run-id",
            "run-cli",
            "--db",
            str(db_path),
        ],
    )
    assert report_all_result.exit_code == 0
    assert "Run ID   : run-cli" in report_all_result.stdout
    assert "[openai]" in report_all_result.stdout


def test_run_fails_when_provider_missing(cli_runner: CliRunner, tmp_path: Path) -> None:
    prompts_path = tmp_path / "prompts.txt"
    prompts_path.write_text("hello\n", encoding="utf-8")

    result = cli_runner.invoke(
        app,
        [
            "run",
            "--providers",
            "missing",
            "--prompt-file",
            str(prompts_path),
            "--config",
            str(tmp_path / "providers.toml"),
            "--db",
            str(tmp_path / "bench.duckdb"),
        ],
    )
    assert result.exit_code != 0
    assert "Provider not found" in result.stdout


def test_run_fails_when_target_rps_is_non_positive(
    cli_runner: CliRunner, tmp_path: Path
) -> None:
    config_path = tmp_path / "providers.toml"
    db_path = tmp_path / "bench.duckdb"
    prompts_path = tmp_path / "prompts.txt"
    prompts_path.write_text("hello\n", encoding="utf-8")

    add_result = cli_runner.invoke(
        app,
        [
            "provider",
            "add",
            "--name",
            "openai",
            "--model",
            "gpt-4o-mini",
            "--config",
            str(config_path),
        ],
    )
    assert add_result.exit_code == 0

    result = cli_runner.invoke(
        app,
        [
            "run",
            "--providers",
            "openai",
            "--prompt-file",
            str(prompts_path),
            "--config",
            str(config_path),
            "--db",
            str(db_path),
            "--rps",
            "0",
        ],
    )
    assert result.exit_code != 0
    assert "must be greater than 0" in result.stdout


def test_run_fails_when_duplicate_providers_requested(
    cli_runner: CliRunner, tmp_path: Path
) -> None:
    config_path = tmp_path / "providers.toml"
    db_path = tmp_path / "bench.duckdb"
    prompts_path = tmp_path / "prompts.txt"
    prompts_path.write_text("hello\n", encoding="utf-8")

    add_result = cli_runner.invoke(
        app,
        [
            "provider",
            "add",
            "--name",
            "openai",
            "--model",
            "gpt-4o-mini",
            "--config",
            str(config_path),
        ],
    )
    assert add_result.exit_code == 0

    result = cli_runner.invoke(
        app,
        [
            "run",
            "--providers",
            "openai,openai",
            "--prompt-file",
            str(prompts_path),
            "--config",
            str(config_path),
            "--db",
            str(db_path),
        ],
    )
    assert result.exit_code != 0
    assert "Duplicate provider names are not allowed" in result.stdout


def test_run_fails_when_prompt_jsonl_has_invalid_line(
    cli_runner: CliRunner, tmp_path: Path
) -> None:
    config_path = tmp_path / "providers.toml"
    db_path = tmp_path / "bench.duckdb"
    prompts_path = tmp_path / "prompts.jsonl"
    prompts_path.write_text('{"prompt":"ok"}\nnot-json\n', encoding="utf-8")

    add_result = cli_runner.invoke(
        app,
        [
            "provider",
            "add",
            "--name",
            "openai",
            "--model",
            "gpt-4o-mini",
            "--config",
            str(config_path),
        ],
    )
    assert add_result.exit_code == 0

    result = cli_runner.invoke(
        app,
        [
            "run",
            "--providers",
            "openai",
            "--prompt-file",
            str(prompts_path),
            "--config",
            str(config_path),
            "--db",
            str(db_path),
        ],
    )
    assert result.exit_code != 0
    assert "Invalid JSONL at line 2" in result.stdout


def test_run_supports_jsonl_prompt_file(
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "providers.toml"
    db_path = tmp_path / "bench.duckdb"
    prompts_path = tmp_path / "prompts.jsonl"
    prompts_path.write_text(
        '{"prompt":"say hi"}\n{"content":"say bye"}\n',
        encoding="utf-8",
    )

    add_result = cli_runner.invoke(
        app,
        [
            "provider",
            "add",
            "--name",
            "openai",
            "--model",
            "gpt-4o-mini",
            "--api-key-env",
            "OPENAI_API_KEY",
            "--config",
            str(config_path),
        ],
    )
    assert add_result.exit_code == 0

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("cli.LiteLLMClient", FakeLiteLLMClient)
    result = cli_runner.invoke(
        app,
        [
            "run",
            "--providers",
            "openai",
            "--prompt-file",
            str(prompts_path),
            "--config",
            str(config_path),
            "--db",
            str(db_path),
            "--run-id",
            "run-jsonl",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["providers"][0]["total_requests"] == 2


def test_run_defaults_to_all_providers_and_prompts_txt(
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "prompts.txt").write_text("hello\n", encoding="utf-8")
    add_result = cli_runner.invoke(
        app,
        [
            "provider",
            "add",
            "--name",
            "openai",
            "--model",
            "gpt-4o-mini",
        ],
    )
    assert add_result.exit_code == 0

    monkeypatch.setattr("cli.LiteLLMClient", FakeLiteLLMClient)
    run_result = cli_runner.invoke(app, ["run"])
    assert run_result.exit_code == 0
    payload = json.loads(run_result.stdout)
    assert payload["providers"][0]["provider_name"] == "openai"
    assert payload["providers"][0]["total_requests"] == 1


def test_run_supports_provider_and_prompt_concurrency(
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "providers.toml"
    db_path = tmp_path / "bench.duckdb"
    prompts_path = tmp_path / "prompts.txt"
    prompts_path.write_text("hello\nworld\n", encoding="utf-8")

    for name, model in [
        ("openai", "gpt-4o-mini"),
        ("openrouter", "anthropic/claude-3-7-sonnet"),
    ]:
        add_result = cli_runner.invoke(
            app,
            [
                "provider",
                "add",
                "--name",
                name,
                "--model",
                model,
                "--config",
                str(config_path),
            ],
        )
        assert add_result.exit_code == 0

    monkeypatch.setattr("cli.LiteLLMClient", FakeLiteLLMClient)
    run_result = cli_runner.invoke(
        app,
        [
            "run",
            "--providers",
            "openai,openrouter",
            "--provider-concurrency",
            "2",
            "--prompt-concurrency",
            "2",
            "--prompt-file",
            str(prompts_path),
            "--config",
            str(config_path),
            "--db",
            str(db_path),
            "--run-id",
            "run-concurrency",
        ],
    )
    assert run_result.exit_code == 0
    payload = json.loads(run_result.stdout)
    assert payload["run_id"] == "run-concurrency"
    assert len(payload["providers"]) == 2
    provider_names = sorted(
        provider["provider_name"] for provider in payload["providers"]
    )
    assert provider_names == ["openai", "openrouter"]
    assert all(provider["total_requests"] == 2 for provider in payload["providers"])


def test_run_emits_live_progress_to_stderr(
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "providers.toml"
    db_path = tmp_path / "bench.duckdb"
    prompts_path = tmp_path / "prompts.txt"
    prompts_path.write_text("hello\nworld\n", encoding="utf-8")

    add_result = cli_runner.invoke(
        app,
        [
            "provider",
            "add",
            "--name",
            "openai",
            "--model",
            "gpt-4o-mini",
            "--config",
            str(config_path),
        ],
    )
    assert add_result.exit_code == 0

    monkeypatch.setattr("cli.LiteLLMClient", FakeLiteLLMClient)
    run_result = cli_runner.invoke(
        app,
        [
            "run",
            "--providers",
            "openai",
            "--prompt-file",
            str(prompts_path),
            "--config",
            str(config_path),
            "--db",
            str(db_path),
            "--run-id",
            "run-progress",
            "--progress",
        ],
    )
    assert run_result.exit_code == 0
    payload = json.loads(run_result.stdout)
    assert payload["run_id"] == "run-progress"
    assert "[run] run_id=run-progress" in run_result.stderr
    assert "[progress]" in run_result.stderr


def test_prompt_generate_txt_file(
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "providers.toml"
    output_path = tmp_path / "prompts.txt"
    add_result = cli_runner.invoke(
        app,
        [
            "provider",
            "add",
            "--name",
            "openai",
            "--model",
            "gpt-4o-mini",
            "--api-key-env",
            "OPENAI_API_KEY",
            "--config",
            str(config_path),
        ],
    )
    assert add_result.exit_code == 0

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("cli.LiteLLMClient", FakePromptGeneratorClient)
    result = cli_runner.invoke(
        app,
        [
            "prompt",
            "generate",
            "--output",
            str(output_path),
            "--count",
            "3",
            "--provider",
            "openai",
            "--config",
            str(config_path),
        ],
    )
    assert result.exit_code == 0
    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8").splitlines() == [
        "Generated prompt 1",
        "Generated prompt 2",
        "Generated prompt 3",
    ]


def test_prompt_generate_jsonl_file(
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "providers.toml"
    output_path = tmp_path / "prompts.jsonl"
    add_result = cli_runner.invoke(
        app,
        [
            "provider",
            "add",
            "--name",
            "openai",
            "--model",
            "gpt-4o-mini",
            "--api-key-env",
            "OPENAI_API_KEY",
            "--config",
            str(config_path),
        ],
    )
    assert add_result.exit_code == 0

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("cli.LiteLLMClient", FakePromptGeneratorClient)
    result = cli_runner.invoke(
        app,
        [
            "prompt",
            "generate",
            "--output",
            str(output_path),
            "--count",
            "2",
            "--provider",
            "openai",
            "--config",
            str(config_path),
        ],
    )
    assert result.exit_code == 0
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"prompt": "Generated prompt 1"}
    assert json.loads(lines[1]) == {"prompt": "Generated prompt 2"}


def test_prompt_generate_refuses_overwrite_without_force(
    cli_runner: CliRunner,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "providers.toml"
    add_result = cli_runner.invoke(
        app,
        [
            "provider",
            "add",
            "--name",
            "openai",
            "--model",
            "gpt-4o-mini",
            "--config",
            str(config_path),
        ],
    )
    assert add_result.exit_code == 0

    output_path = tmp_path / "prompts.txt"
    output_path.write_text("existing\n", encoding="utf-8")
    result = cli_runner.invoke(
        app,
        [
            "prompt",
            "generate",
            "--output",
            str(output_path),
            "--count",
            "1",
            "--provider",
            "openai",
            "--config",
            str(config_path),
        ],
    )
    assert result.exit_code != 0
    assert "already exists" in result.stdout


def test_prompt_generate_fails_when_provider_not_found(
    cli_runner: CliRunner, tmp_path: Path
) -> None:
    output_path = tmp_path / "prompts.txt"
    result = cli_runner.invoke(
        app,
        [
            "prompt",
            "generate",
            "--output",
            str(output_path),
            "--provider",
            "missing",
            "--config",
            str(tmp_path / "providers.toml"),
        ],
    )
    assert result.exit_code != 0
    assert "Provider not found" in result.stdout


def test_prompt_generate_defaults_to_first_provider(
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    add_b_result = cli_runner.invoke(
        app,
        [
            "provider",
            "add",
            "--name",
            "z-provider",
            "--model",
            "model-z",
        ],
    )
    assert add_b_result.exit_code == 0

    add_a_result = cli_runner.invoke(
        app,
        [
            "provider",
            "add",
            "--name",
            "a-provider",
            "--model",
            "model-a",
        ],
    )
    assert add_a_result.exit_code == 0

    monkeypatch.setattr("cli.LiteLLMClient", FakePromptGeneratorClient)
    result = cli_runner.invoke(app, ["prompt", "generate", "--count", "2"])
    assert result.exit_code == 0

    payload = json.loads(result.stdout)
    assert payload["provider"] == "a-provider"
    generated = (tmp_path / "prompts.txt").read_text(encoding="utf-8").splitlines()
    assert generated == ["Generated prompt 1", "Generated prompt 2"]


def test_prompt_generate_fails_without_configured_provider(
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    result = cli_runner.invoke(app, ["prompt", "generate"])
    assert result.exit_code != 0
    assert "No providers configured" in result.stdout


def test_report_summary_defaults_to_latest_run(
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "providers.toml"
    db_path = tmp_path / "bench.duckdb"
    prompts_path = tmp_path / "prompts.txt"
    prompts_path.write_text("hello\n", encoding="utf-8")

    add_result = cli_runner.invoke(
        app,
        [
            "provider",
            "add",
            "--name",
            "openai",
            "--model",
            "gpt-4o-mini",
            "--config",
            str(config_path),
        ],
    )
    assert add_result.exit_code == 0

    monkeypatch.setattr("cli.LiteLLMClient", FakeLiteLLMClient)
    run_result = cli_runner.invoke(
        app,
        [
            "run",
            "--providers",
            "openai",
            "--prompt-file",
            str(prompts_path),
            "--config",
            str(config_path),
            "--db",
            str(db_path),
            "--run-id",
            "run-latest",
        ],
    )
    assert run_result.exit_code == 0

    report_result = cli_runner.invoke(
        app,
        [
            "report",
            "summary",
            "--db",
            str(db_path),
        ],
    )
    assert report_result.exit_code == 0
    assert "Benchmark summary" in report_result.stdout
    assert "Run ID   : run-latest" in report_result.stdout
    assert "Providers: 1" in report_result.stdout
    assert "[openai]" in report_result.stdout
    assert "Latency:" in report_result.stdout
    assert "Throughput:" in report_result.stdout
    assert "Quality:" in report_result.stdout
    assert "error_rate" in report_result.stdout


def test_report_summary_supports_json_output(
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "providers.toml"
    db_path = tmp_path / "bench.duckdb"
    prompts_path = tmp_path / "prompts.txt"
    prompts_path.write_text("hello\n", encoding="utf-8")

    add_result = cli_runner.invoke(
        app,
        [
            "provider",
            "add",
            "--name",
            "openai",
            "--model",
            "gpt-4o-mini",
            "--config",
            str(config_path),
        ],
    )
    assert add_result.exit_code == 0

    monkeypatch.setattr("cli.LiteLLMClient", FakeLiteLLMClient)
    run_result = cli_runner.invoke(
        app,
        [
            "run",
            "--providers",
            "openai",
            "--prompt-file",
            str(prompts_path),
            "--config",
            str(config_path),
            "--db",
            str(db_path),
            "--run-id",
            "run-latest",
        ],
    )
    assert run_result.exit_code == 0

    report_result = cli_runner.invoke(
        app,
        [
            "report",
            "summary",
            "--json",
            "--db",
            str(db_path),
        ],
    )
    assert report_result.exit_code == 0
    payload = json.loads(report_result.stdout)
    assert payload["run_id"] == "run-latest"
    assert "openai" in payload["providers"]
    assert "latency" in payload["providers"]["openai"]


def test_report_list_supports_json_output(
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "providers.toml"
    db_path = tmp_path / "bench.duckdb"
    prompts_path = tmp_path / "prompts.txt"
    prompts_path.write_text("hello\n", encoding="utf-8")

    add_result = cli_runner.invoke(
        app,
        [
            "provider",
            "add",
            "--name",
            "openai",
            "--model",
            "gpt-4o-mini",
            "--config",
            str(config_path),
        ],
    )
    assert add_result.exit_code == 0

    monkeypatch.setattr("cli.LiteLLMClient", FakeLiteLLMClient)
    run_result = cli_runner.invoke(
        app,
        [
            "run",
            "--providers",
            "openai",
            "--prompt-file",
            str(prompts_path),
            "--config",
            str(config_path),
            "--db",
            str(db_path),
            "--run-id",
            "run-list-json",
        ],
    )
    assert run_result.exit_code == 0

    list_result = cli_runner.invoke(
        app,
        [
            "report",
            "list",
            "--json",
            "--db",
            str(db_path),
        ],
    )
    assert list_result.exit_code == 0
    payload = json.loads(list_result.stdout)
    assert payload["total"] == 1
    run_payload = payload["runs"][0]
    assert run_payload["run_id"] == "run-list-json"
    assert run_payload["provider_count"] == 1
    assert run_payload["request_count"] == 1
    assert run_payload["success_count"] == 1
    assert run_payload["failed_count"] == 0
    assert run_payload["started_at_iso"] != "-"
    assert run_payload["finished_at_iso"] != "-"


def test_report_list_default_output_is_human_readable(
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "providers.toml"
    db_path = tmp_path / "bench.duckdb"
    prompts_path = tmp_path / "prompts.txt"
    prompts_path.write_text("hello\n", encoding="utf-8")

    add_result = cli_runner.invoke(
        app,
        [
            "provider",
            "add",
            "--name",
            "openai",
            "--model",
            "gpt-4o-mini",
            "--config",
            str(config_path),
        ],
    )
    assert add_result.exit_code == 0

    monkeypatch.setattr("cli.LiteLLMClient", FakeLiteLLMClient)
    run_result = cli_runner.invoke(
        app,
        [
            "run",
            "--providers",
            "openai",
            "--prompt-file",
            str(prompts_path),
            "--config",
            str(config_path),
            "--db",
            str(db_path),
            "--run-id",
            "run-list-human",
        ],
    )
    assert run_result.exit_code == 0

    list_result = cli_runner.invoke(
        app,
        [
            "report",
            "list",
            "--db",
            str(db_path),
        ],
    )
    assert list_result.exit_code == 0
    assert "Benchmark runs" in list_result.stdout
    assert "run-list-human" in list_result.stdout
    assert "duration=" in list_result.stdout
    assert "requests=1" in list_result.stdout


def test_report_list_fails_when_no_runs(cli_runner: CliRunner, tmp_path: Path) -> None:
    result = cli_runner.invoke(
        app,
        [
            "report",
            "list",
            "--db",
            str(tmp_path / "bench.duckdb"),
        ],
    )
    assert result.exit_code != 0
    assert "No runs found." in result.stdout


def test_report_remove_deletes_run(
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "providers.toml"
    db_path = tmp_path / "bench.duckdb"
    prompts_path = tmp_path / "prompts.txt"
    prompts_path.write_text("hello\n", encoding="utf-8")

    add_result = cli_runner.invoke(
        app,
        [
            "provider",
            "add",
            "--name",
            "openai",
            "--model",
            "gpt-4o-mini",
            "--config",
            str(config_path),
        ],
    )
    assert add_result.exit_code == 0

    monkeypatch.setattr("cli.LiteLLMClient", FakeLiteLLMClient)
    run_result = cli_runner.invoke(
        app,
        [
            "run",
            "--providers",
            "openai",
            "--prompt-file",
            str(prompts_path),
            "--config",
            str(config_path),
            "--db",
            str(db_path),
            "--run-id",
            "run-remove",
        ],
    )
    assert run_result.exit_code == 0

    remove_result = cli_runner.invoke(
        app,
        [
            "report",
            "remove",
            "--run-id",
            "run-remove",
            "--db",
            str(db_path),
        ],
    )
    assert remove_result.exit_code == 0
    assert "Run removed: run-remove" in remove_result.stdout

    list_result = cli_runner.invoke(
        app,
        [
            "report",
            "list",
            "--db",
            str(db_path),
        ],
    )
    assert list_result.exit_code != 0
    assert "No runs found." in list_result.stdout


def test_report_remove_fails_when_run_missing(
    cli_runner: CliRunner, tmp_path: Path
) -> None:
    result = cli_runner.invoke(
        app,
        [
            "report",
            "remove",
            "--run-id",
            "missing-run",
            "--db",
            str(tmp_path / "bench.duckdb"),
        ],
    )
    assert result.exit_code != 0
    assert "Run not found: missing-run" in result.stdout


def test_help_output_hides_advanced_options(cli_runner: CliRunner) -> None:
    run_help = cli_runner.invoke(app, ["run", "--help"])
    assert run_help.exit_code == 0
    assert "--providers" in run_help.stdout
    assert "--prompt-file" in run_help.stdout
    assert "--provider-concu" in run_help.stdout
    assert "--prompt-concurr" in run_help.stdout
    assert "--progress" in run_help.stdout
    assert "--config" not in run_help.stdout
    assert "--run-id" not in run_help.stdout

    prompt_help = cli_runner.invoke(app, ["prompt", "generate", "--help"])
    assert prompt_help.exit_code == 0
    assert "--output" in prompt_help.stdout
    assert "--provider" in prompt_help.stdout
    assert "--instruction" not in prompt_help.stdout
    assert "--format" not in prompt_help.stdout
