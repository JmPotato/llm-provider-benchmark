from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from providers import ProviderConfig, ProviderRegistry


@pytest.fixture
def registry(tmp_path: Path) -> ProviderRegistry:
    return ProviderRegistry(tmp_path / "providers.toml")


def test_save_and_load_round_trip(registry: ProviderRegistry) -> None:
    registry.save_provider(
        ProviderConfig(
            name="openai",
            model="gpt-4o-mini",
            api_base="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
        )
    )
    registry.save_provider(
        ProviderConfig(
            name="custom",
            model="my-model",
            api_base="https://llm.example.com/v1",
            api_key_env="CUSTOM_API_KEY",
            extra_headers={"x-tenant-id": "bench-team"},
        )
    )

    loaded = registry.load()
    assert set(loaded.keys()) == {"openai", "custom"}
    assert loaded["openai"].model == "gpt-4o-mini"
    assert loaded["custom"].extra_headers["x-tenant-id"] == "bench-team"


def test_list_providers_returns_name_sorted(registry: ProviderRegistry) -> None:
    registry.save_provider(ProviderConfig(name="zeta", model="m1"))
    registry.save_provider(ProviderConfig(name="alpha", model="m2"))
    providers = registry.list_providers()
    assert [provider.name for provider in providers] == ["alpha", "zeta"]


def test_remove_provider_deletes_existing_entry(registry: ProviderRegistry) -> None:
    registry.save_provider(ProviderConfig(name="openai", model="gpt-4o-mini"))
    registry.remove_provider("openai")
    assert registry.load() == {}


def test_remove_provider_raises_when_missing(registry: ProviderRegistry) -> None:
    with pytest.raises(KeyError):
        registry.remove_provider("missing")


def test_get_provider_raises_when_missing(registry: ProviderRegistry) -> None:
    with pytest.raises(KeyError):
        registry.get_provider("missing")


def test_get_provider_returns_saved_provider(registry: ProviderRegistry) -> None:
    saved = ProviderConfig(
        name="openrouter",
        model="anthropic/claude-3-7-sonnet",
        api_base="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
    )
    registry.save_provider(saved)
    loaded = registry.get_provider("openrouter")
    assert loaded.name == saved.name
    assert loaded.model == saved.model
    assert loaded.api_base == saved.api_base
    assert loaded.api_key_env == saved.api_key_env


def test_load_raises_when_provider_model_missing(registry: ProviderRegistry) -> None:
    registry.config_path.write_text(
        '[providers."bad-provider"]\napi_base = "https://example.com"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        registry.load()


def test_from_dict_normalizes_optional_string_fields() -> None:
    provider = ProviderConfig.from_dict(
        "openai",
        {
            "model": "gpt-4o-mini",
            "api_base": None,
            "api_key_env": "   ",
        },
    )
    assert provider.api_base is None
    assert provider.api_key_env is None
