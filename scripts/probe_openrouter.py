#!/usr/bin/env python3
"""Make one secret-safe OpenRouter compatibility call.

This probe deliberately uses only Python's standard library. It does not load
Twin-2K data, write a result ledger, or run the scientific pilot. Its only job
is to prove that a selected model can return Rival's probability-object shape.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


DEFAULT_MODEL = "dots-studio/dots-3-note-preview:free"
DEFAULT_URL = "https://openrouter.ai/api/v1/chat/completions"


class ProbeError(RuntimeError):
    pass


def _redact(value: object, api_key: str, limit: int = 800) -> str:
    text = " ".join(str(value).split())
    if api_key:
        text = text.replace(api_key, "[REDACTED]")
    text = re.sub(r"(?i)\b(bearer\s+)[^\s\"']+", r"\1[REDACTED]", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[REDACTED]", text)
    return text[:limit]


def _payload(model: str) -> dict[str, Any]:
    properties = {
        "option_a": {"type": "number", "minimum": 0, "maximum": 1},
        "option_b": {"type": "number", "minimum": 0, "maximum": 1},
    }
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return only a calibrated probability distribution over the "
                    "supplied action IDs. Do not explain or role-play."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "person": {
                            "attributes": {"age_band": "35-44"},
                            "history": ["Previously preferred predictable monthly costs."],
                        },
                        "scenario": {
                            "question": "Which subscription would this person choose?",
                            "choices": {
                                "option_a": "Lower fixed fee",
                                "option_b": "Higher fee with extra features",
                            },
                        },
                    },
                    sort_keys=True,
                ),
            },
        ],
        "temperature": 0,
        "max_tokens": 300,
        "reasoning": {"effort": "none", "exclude": True},
        "provider": {"require_parameters": True},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "rival_choice_probabilities",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties),
                    "additionalProperties": False,
                },
            },
        },
    }


def _content(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "".join(parts).strip()
    return ""


def _validate(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        choice = payload["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProbeError("response has no assistant message") from exc
    content = _content(message)
    reasoning = message.get("reasoning")
    finish_reason = choice.get("finish_reason")
    if not content:
        raise ProbeError(
            "assistant content is empty "
            f"(finish_reason={finish_reason!r}, reasoning_chars={len(str(reasoning or ''))})"
        )
    try:
        probabilities = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ProbeError(
            f"assistant content is not JSON: {content[:300]!r}"
        ) from exc
    if set(probabilities) != {"option_a", "option_b"}:
        raise ProbeError(f"wrong output keys: {sorted(probabilities)}")
    try:
        values = {key: float(value) for key, value in probabilities.items()}
    except (TypeError, ValueError) as exc:
        raise ProbeError("probabilities are not numeric") from exc
    total = sum(values.values())
    if total <= 0 or any(value < 0 for value in values.values()):
        raise ProbeError("probabilities are not a valid nonnegative distribution")
    return {
        "status": "PASS",
        "finish_reason": finish_reason,
        "probabilities": {key: value / total for key, value in values.items()},
        "usage": payload.get("usage", {}),
        "response_id": payload.get("id"),
    }


def _http_error(exc: urllib.error.HTTPError, api_key: str) -> str:
    try:
        body = exc.read(16_384).decode("utf-8", errors="replace")
    except Exception:
        body = ""
    return _redact(f"HTTP {exc.code}: {body or exc.reason}", api_key)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args(argv)

    endpoint = urllib.parse.urlsplit(args.url)
    if endpoint.scheme != "https" or not endpoint.hostname:
        print("FAIL: probe URL must be a valid HTTPS endpoint", file=sys.stderr)
        return 2
    api_key = os.getenv("RIVAL_API_KEY") or getpass.getpass(
        "OpenRouter API key (input hidden): "
    )
    if not api_key:
        print("FAIL: no API key supplied", file=sys.stderr)
        return 2

    request = urllib.request.Request(
        args.url,
        data=json.dumps(_payload(args.model)).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://rival.local",
            "X-Title": "Rival One-Call Compatibility Probe",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        summary = _validate(response_payload)
    except urllib.error.HTTPError as exc:
        print("FAIL: " + _http_error(exc, api_key), file=sys.stderr)
        return 2
    except urllib.error.URLError as exc:
        print("FAIL: network error: " + _redact(exc.reason, api_key), file=sys.stderr)
        return 2
    except (json.JSONDecodeError, ProbeError) as exc:
        print("FAIL: " + _redact(exc, api_key), file=sys.stderr)
        return 2
    finally:
        api_key = ""

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
