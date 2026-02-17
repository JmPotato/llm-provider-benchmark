from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
import tomllib


logger = logging.getLogger(__name__)


def _escape_toml_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _coerce_optional_string(value: object) -> str | None:
    if value is None:
        return None
    parsed = str(value).strip()
    return parsed or None


def _format_toml_kv(key: str, value: object) -> str:
    if isinstance(value, str):
        return f'{key} = "{_escape_toml_string(value)}"'
    if isinstance(value, bool):
        return f"{key} = {'true' if value else 'false'}"
    if isinstance(value, int | float):
        return f"{key} = {value}"
    raise TypeError(f"Unsupported TOML value type: {type(value)!r}")


@dataclass(slots=True)
class ProviderConfig:
    name: str
    model: str
    api_base: str | None = None
    api_key_env: str | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)
    temperature: float | None = None
    max_tokens: int | None = None
    timeout_s: float | None = None

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"model": self.model}
        if self.api_base:
            data["api_base"] = self.api_base
        if self.api_key_env:
            data["api_key_env"] = self.api_key_env
        if self.extra_headers:
            data["extra_headers"] = self.extra_headers
        if self.temperature is not None:
            data["temperature"] = self.temperature
        if self.max_tokens is not None:
            data["max_tokens"] = self.max_tokens
        if self.timeout_s is not None:
            data["timeout_s"] = self.timeout_s
        return data

    @classmethod
    def from_dict(cls, name: str, data: dict[str, object]) -> "ProviderConfig":
        model = _coerce_optional_string(data.get("model"))
        if model is None:
            raise ValueError(f"Provider {name!r} missing required field 'model'")

        extra_headers_raw = data.get("extra_headers", {})
        if extra_headers_raw is None:
            extra_headers_raw = {}
        if not isinstance(extra_headers_raw, dict):
            raise ValueError("extra_headers must be a mapping")

        extra_headers: dict[str, str] = {}
        for header_name, header_value in extra_headers_raw.items():
            extra_headers[str(header_name)] = str(header_value)

        return cls(
            name=name,
            model=model,
            api_base=_coerce_optional_string(data.get("api_base")),
            api_key_env=_coerce_optional_string(data.get("api_key_env")),
            extra_headers=extra_headers,
            temperature=(
                float(data["temperature"])
                if "temperature" in data and data["temperature"] is not None
                else None
            ),
            max_tokens=(
                int(data["max_tokens"])
                if "max_tokens" in data and data["max_tokens"] is not None
                else None
            ),
            timeout_s=(
                float(data["timeout_s"])
                if "timeout_s" in data and data["timeout_s"] is not None
                else None
            ),
        )


class ProviderRegistry:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path

    def load(self) -> dict[str, ProviderConfig]:
        raw = self._read_raw()
        providers_raw = raw.get("providers", {})
        if not isinstance(providers_raw, dict):
            raise ValueError("Top-level 'providers' must be a table")

        loaded: dict[str, ProviderConfig] = {}
        for name, data in providers_raw.items():
            if not isinstance(data, dict):
                raise ValueError(f"Provider {name!r} entry must be a table")
            loaded[str(name)] = ProviderConfig.from_dict(str(name), data)
        logger.debug("Loaded %d provider(s) from %s", len(loaded), self.config_path)
        return loaded

    def list_providers(self) -> list[ProviderConfig]:
        providers = self.load()
        return [providers[name] for name in sorted(providers)]

    def save_provider(self, provider: ProviderConfig) -> None:
        if not provider.name.strip():
            raise ValueError("Provider name cannot be empty")

        raw = self._read_raw()
        providers_raw = raw.setdefault("providers", {})
        if not isinstance(providers_raw, dict):
            raise ValueError("Top-level 'providers' must be a table")

        providers_raw[provider.name] = provider.to_dict()
        self._write_raw(raw)
        logger.debug("Saved provider %r to %s", provider.name, self.config_path)

    def remove_provider(self, name: str) -> None:
        raw = self._read_raw()
        providers_raw = raw.get("providers", {})
        if not isinstance(providers_raw, dict):
            raise ValueError("Top-level 'providers' must be a table")

        if name not in providers_raw:
            raise KeyError(name)
        del providers_raw[name]
        self._write_raw(raw)
        logger.debug("Removed provider %r from %s", name, self.config_path)

    def get_provider(self, name: str) -> ProviderConfig:
        providers = self.load()
        if name not in providers:
            raise KeyError(name)
        return providers[name]

    def _read_raw(self) -> dict[str, object]:
        if not self.config_path.exists():
            return {"providers": {}}

        with self.config_path.open("rb") as handle:
            parsed = tomllib.load(handle)
        if "providers" not in parsed:
            parsed["providers"] = {}
        return parsed

    def _write_raw(self, data: dict[str, object]) -> None:
        providers_raw = data.get("providers", {})
        if not isinstance(providers_raw, dict):
            raise ValueError("Top-level 'providers' must be a table")

        lines: list[str] = []
        for provider_name in sorted(providers_raw):
            provider_data = providers_raw[provider_name]
            if not isinstance(provider_data, dict):
                raise ValueError(f"Provider {provider_name!r} entry must be a table")

            quoted_name = _escape_toml_string(str(provider_name))
            lines.append(f'[providers."{quoted_name}"]')
            for key in sorted(provider_data):
                value = provider_data[key]
                if key == "extra_headers":
                    continue
                lines.append(_format_toml_kv(str(key), value))

            headers = provider_data.get("extra_headers", {})
            if headers:
                if not isinstance(headers, dict):
                    raise ValueError(
                        f"Provider {provider_name!r}.extra_headers must be a table"
                    )
                lines.append(f'[providers."{quoted_name}".extra_headers]')
                for header_name in sorted(headers):
                    lines.append(
                        _format_toml_kv(str(header_name), str(headers[header_name]))
                    )
            lines.append("")

        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(lines).strip()
        self.config_path.write_text(
            content + ("\n" if content else ""), encoding="utf-8"
        )
