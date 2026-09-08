"""Shared durable transport for general probability and behavioral providers."""

import json
import math
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from .execution import AttemptJournal, ExecutionError
from .mathx import canonical_hash, validate_probabilities


class ExecutionSession:
    """One persistent spending scope, shared by all providers in a study batch.

    An unchanged study/person/request is recovered from the journal. To request
    an independent model replicate, use an explicitly distinct scenario ID.
    Budgets and the physical attempt limit apply over the journal's lifetime.
    """

    def __init__(self, path: str | Path, *, scope_id: str, budget_usd: float,
                 not_after: datetime, reservation_usd: float, max_total_attempts: int):
        if str(path) == ":memory:" or not scope_id.strip():
            raise ValueError("managed execution requires a durable path and named scope")
        if (not isinstance(max_total_attempts, int) or isinstance(max_total_attempts, bool)
                or max_total_attempts < 1 or not_after.tzinfo is None
                or not math.isfinite(budget_usd) or budget_usd <= 0
                or not math.isfinite(reservation_usd) or reservation_usd <= 0):
            raise ValueError("finite positive limits and an aware expiry are required")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.journal = AttemptJournal(path, canonical_hash({"runtime": "rival.managed-probability.v1", "scope_id": scope_id}))
        self.budget_usd = budget_usd
        self.not_after = not_after
        self.reservation_usd = reservation_usd
        self.max_total_attempts = max_total_attempts

    def _complete(self, provider, person, scenario, *, payload, identity, accept):
        from .providers import OpenAICompatibleProvider

        request_identity = {"provider": identity, "payload": payload}
        work_id = canonical_hash({"study": scenario.scenario_id, "person": person.person_id,
                                  "provider_slot": provider.name})
        request = urllib.request.Request(provider.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-Title": "Rival Managed Research",
                     **({"Authorization": f"Bearer {provider.api_key}"} if provider.api_key else {})}, method="POST")

        def send():
            started_attempt = time.perf_counter()
            def measured(value):
                value["_rival_transport"] = {"latency_ms": (time.perf_counter() - started_attempt) * 1000}
                return value
            try:
                with urllib.request.urlopen(request, timeout=provider.timeout_seconds) as response:
                    raw = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                try:
                    raw = json.loads(exc.read().decode("utf-8"))
                except (ValueError, UnicodeError):
                    raise ExecutionError(f"HTTP {exc.code}; billing unknown") from None
                if not isinstance(raw, dict):
                    raise ExecutionError(f"HTTP {exc.code}; invalid response") from None
                try:
                    retry_after = float(exc.headers.get("Retry-After", 0))
                except (TypeError, ValueError, AttributeError):
                    retry_after = 0
                return measured({"id": raw.get("id"), "usage": raw.get("usage"),
                        "error": {"http_status": exc.code}, "_retry_after": retry_after})
            if not isinstance(raw, dict):
                raise ValueError("provider response must be an object")
            # Error strings may echo credentials; retain only the error signal.
            if "error" in raw:
                return measured({"id": raw.get("id"), "usage": raw.get("usage"), "error": {"present": True}})
            return measured({key: raw[key] for key in
                ("id", "usage", "choices", "model", "provider", "system_fingerprint") if key in raw})

        def validated(raw):
            if hasattr(provider, "validate_response"):
                provider.validate_response(raw)
            return accept(raw)

        reserved_ordinals = []
        started = time.perf_counter()
        raw = self.journal.execute(work_id, request_identity, send, validated,
            budget_usd=self.budget_usd, not_after=self.not_after,
            reservation_usd=self.reservation_usd, max_attempts=provider.max_retries,
            max_total_attempts=self.max_total_attempts, reserved_ordinals=reserved_ordinals)
        validated(raw)  # Recheck cached and operator-reconciled completions too.
        attempts = self.journal.attempts_for(work_id)
        # Attribute only attempts reserved by this invocation, including when
        # another process completed the work just before our cached read.
        fresh = [row for row in attempts if row["ordinal"] in reserved_ordinals]
        diagnostics = {}
        for row in fresh:
            for key, value in OpenAICompatibleProvider._usage_diagnostics(json.loads(row["payload"] or "{}")).items():
                diagnostics[key] = diagnostics.get(key, 0.0) + value
        diagnostics["provider_cost_usd"] = sum(row["cost"] for row in fresh)
        diagnostics["request_cost_usd"] = sum(row["cost"] for row in attempts)
        diagnostics["request_attempts"] = float(len(attempts))
        return raw, diagnostics, len(fresh), not fresh, (time.perf_counter() - started) * 1000

    def predict(self, provider, person, scenario):
        from .providers import OpenAICompatibleProvider, ProviderPrediction

        def accept(raw):
            content = OpenAICompatibleProvider._extract_content(raw).strip()
            if content.startswith("```"):
                content = content.strip("`")
                if content.startswith("json"):
                    content = content[4:].lstrip()

            def unique_pairs(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError("duplicate probability key")
                    result[key] = value
                return result

            parsed = json.loads(content, object_pairs_hook=unique_pairs)
            keys = [choice.choice_id for choice in scenario.choices]
            if not isinstance(parsed, dict) or set(parsed) != set(keys):
                raise ValueError("provider returned the wrong choice_id set")
            values = validate_probabilities(parsed[key] for key in keys)
            return dict(zip(keys, map(float, values), strict=True))

        raw, diagnostics, attempts, cached, latency = self._complete(provider, person, scenario,
            payload=provider._request_payload(person, scenario), identity=provider.identity().model_dump(mode="json"), accept=accept)
        return ProviderPrediction(probabilities=accept(raw), diagnostics=diagnostics,
            provider_request_id=raw.get("id"), attempts=attempts, cache_hit=cached, latency_ms=latency)

    def generate(self, provider, person, scenario):
        from .elicitation import GeneratedText
        from .providers import OpenAICompatibleProvider

        accept = OpenAICompatibleProvider._extract_content
        raw, diagnostics, attempts, cached, latency = self._complete(provider, person, scenario,
            payload=provider._payload(person, scenario), identity=provider.identity, accept=accept)
        return GeneratedText(text=accept(raw), request_id=raw.get("id"), attempts=attempts,
                             latency_ms=latency, diagnostics=diagnostics, cache_hit=cached)

    def close(self):
        self.journal.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
