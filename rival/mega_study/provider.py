"""Pinned OpenRouter transport for complete Mega-Study survey responses."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from .constants import MODEL_CONFIG
from .utils import MegaStudyError, canonical_hash


class MegaStudyProviderError(MegaStudyError):
    pass


def _redact(value: object, limit: int = 800) -> str:
    detail = " ".join(str(value).split())
    detail = re.sub(r"(?i)\b(bearer\s+)[^\s\"']+", r"\1[REDACTED]", detail)
    detail = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[REDACTED]", detail)
    return detail[:limit]


def _http_error(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read(16_384).decode("utf-8", errors="replace")
    except Exception:  # pragma: no cover - defensive transport path
        raw = ""
    message: object = exc.reason
    code: object | None = None
    if raw:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            message = raw
        else:
            error = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(error, dict):
                message, code = error.get("message", message), error.get("code")
    suffix = f" code={code}" if code not in (None, "") else ""
    return _redact(f"HTTP {exc.code}{suffix}: {message}")


@dataclass(frozen=True, slots=True)
class SurveyCompletion:
    content: str
    response_id: str | None
    attempts: int
    latency_ms: float
    usage: dict[str, float]


class OpenRouterSurveyProvider:
    """A model/provider/configuration-pinned complete-survey client."""

    def __init__(self, api_key: str | None = None, config: dict[str, Any] | None = None):
        self.config = dict(config or MODEL_CONFIG)
        if self.config != MODEL_CONFIG:
            raise MegaStudyProviderError("provider configuration must equal frozen protocol")
        self.api_key = api_key or os.getenv("RIVAL_API_KEY") or os.getenv(
            "OPENROUTER_API_KEY"
        )
        if not self.api_key:
            raise MegaStudyProviderError(
                "missing API key; set RIVAL_API_KEY or OPENROUTER_API_KEY"
            )
        endpoint = urllib.parse.urlsplit(str(self.config["base_url"]))
        if endpoint.scheme != "https" or endpoint.hostname != "openrouter.ai":
            raise MegaStudyProviderError("frozen provider endpoint must be OpenRouter HTTPS")

    @staticmethod
    def identity() -> dict[str, Any]:
        configuration = {
            key: value
            for key, value in MODEL_CONFIG.items()
            if key not in {"input_cost_per_million_usd", "output_cost_per_million_usd"}
        }
        return {
            "provider_name": MODEL_CONFIG["api_provider"],
            "provider_version": "rival-mega-openrouter-v1",
            "model": MODEL_CONFIG["model"],
            "upstream_provider": MODEL_CONFIG["upstream_provider"],
            "configuration_sha256": canonical_hash(configuration),
            "endpoint_sha256": canonical_hash(MODEL_CONFIG["base_url"]),
        }

    @staticmethod
    def request_payload(system: str, user: str) -> dict[str, Any]:
        return {
            "model": MODEL_CONFIG["model"],
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": MODEL_CONFIG["temperature"],
            "seed": MODEL_CONFIG["seed"],
            "max_tokens": MODEL_CONFIG["max_output_tokens"],
            "response_format": {"type": "json_object"},
            "reasoning": {"enabled": False, "exclude": True},
            "provider": {
                "order": [MODEL_CONFIG["upstream_provider"]],
                "allow_fallbacks": MODEL_CONFIG["allow_provider_fallbacks"],
                "require_parameters": True,
            },
        }

    @staticmethod
    def _content(payload: dict[str, Any]) -> str:
        error = payload.get("error")
        if error is not None:
            raise MegaStudyProviderError(_redact(error))
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise MegaStudyProviderError("provider response has no message content") from exc
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            rendered = "".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ).strip()
            if rendered:
                return rendered
        raise MegaStudyProviderError("provider returned empty content")

    @staticmethod
    def _usage(payload: dict[str, Any]) -> dict[str, float]:
        raw = payload.get("usage")
        if not isinstance(raw, dict):
            return {}
        result: dict[str, float] = {}
        candidates = {
            "prompt_tokens": ("prompt_tokens", "input_tokens"),
            "completion_tokens": ("completion_tokens", "output_tokens"),
            "total_tokens": ("total_tokens",),
            "provider_cost_usd": ("cost", "total_cost", "cost_usd"),
        }
        for destination, aliases in candidates.items():
            for alias in aliases:
                value = raw.get(alias)
                if isinstance(value, (int, float)) and float(value) >= 0:
                    result[destination] = float(value)
                    break
        if "total_tokens" not in result and {
            "prompt_tokens",
            "completion_tokens",
        } <= set(result):
            result["total_tokens"] = (
                result["prompt_tokens"] + result["completion_tokens"]
            )
        return result

    def complete(self, system: str, user: str) -> SurveyCompletion:
        payload = self.request_payload(system, user)
        request = urllib.request.Request(
            str(self.config["base_url"]),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://rival.local",
                "X-Title": "Rival Mega-Study",
            },
            method="POST",
        )
        started = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(1, int(self.config["max_attempts"]) + 1):
            retry_after = 0.0
            try:
                with urllib.request.urlopen(
                    request, timeout=int(self.config["timeout_seconds"])
                ) as response:
                    provider_payload = json.loads(response.read().decode("utf-8"))
                return SurveyCompletion(
                    content=self._content(provider_payload),
                    response_id=(
                        str(provider_payload["id"])
                        if provider_payload.get("id") is not None
                        else None
                    ),
                    attempts=attempt,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    usage=self._usage(provider_payload),
                )
            except urllib.error.HTTPError as exc:
                last_error = MegaStudyProviderError(_http_error(exc))
                try:
                    retry_after = min(30.0, max(0.0, float(exc.headers.get("Retry-After", 0))))
                except (AttributeError, TypeError, ValueError):
                    retry_after = 0.0
            except urllib.error.URLError as exc:
                last_error = MegaStudyProviderError(
                    "network error: " + _redact(exc.reason)
                )
            except (TimeoutError, ValueError, MegaStudyProviderError) as exc:
                last_error = exc
            if attempt < int(self.config["max_attempts"]):
                time.sleep(max(retry_after, float(2 ** (attempt - 1))))
        raise MegaStudyProviderError(
            f"survey completion failed after {self.config['max_attempts']} attempts: "
            f"{_redact(last_error or 'unknown error')}"
        ) from last_error
