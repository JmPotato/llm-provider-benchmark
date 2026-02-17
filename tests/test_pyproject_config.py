from __future__ import annotations

from pathlib import Path
import tomllib


def test_pyproject_declares_cli_script() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    scripts = data["project"]["scripts"]
    assert scripts["llm-bench"] == "cli:main"
