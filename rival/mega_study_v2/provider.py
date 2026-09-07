"""Pinned survey payload with v2 durable accounting and replay protection."""

import json
import time
import urllib.error
import urllib.request
from datetime import datetime

from ..execution import AttemptJournal, ExecutionError
from ..mega_study.provider import OpenRouterSurveyProvider, SurveyCompletion
from . import RUNTIME_REVISION


class ManagedSurveyProvider(OpenRouterSurveyProvider):
    def complete(self, system: str, user: str) -> SurveyCompletion:
        raise ExecutionError("v2 requires complete_managed with a durable attempt journal")

    def complete_managed(self, system: str, user: str, *, work_id: str,
                         journal: AttemptJournal, budget_usd: float,
                         not_after: datetime, reservation_usd: float,
                         max_new_attempts: int | None = None) -> SurveyCompletion:
        payload = self.request_payload(system, user)
        request = urllib.request.Request(
            str(self.config["base_url"]), data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )

        def send():
            try:
                with urllib.request.urlopen(request, timeout=int(self.config["timeout_seconds"])) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                # Preserve reported usage even on an error. An HTTP status alone
                # establishes neither a charge nor zero cost.
                try:
                    raw = json.loads(exc.read().decode("utf-8"))
                except (ValueError, UnicodeError):
                    raise exc
                # Error details are not required for accounting; do not persist
                # arbitrary echoed authorization headers from an upstream error.
                try:
                    retry_after = float(exc.headers.get("Retry-After", 0))
                except (TypeError, ValueError, AttributeError):
                    retry_after = 0
                return {"id": raw.get("id"), "usage": raw.get("usage"), "_retry_after": retry_after,
                        "error": {"http_status": exc.code}}

        started = time.perf_counter()
        raw = journal.execute(work_id, {"runtime_revision": RUNTIME_REVISION,
                                       "provider": self.identity(), "payload": payload},
                              send, self._content, budget_usd=budget_usd, not_after=not_after,
                              reservation_usd=reservation_usd, max_attempts=int(self.config["max_attempts"]),
                              max_new_attempts=max_new_attempts)
        attempts = journal.attempts_for(work_id)
        usage = {}
        for row in attempts:
            for key, value in self._usage(json.loads(row["payload"] or "{}")).items():
                usage[key] = usage.get(key, 0.0) + value
        usage["provider_cost_usd"] = sum(row["cost"] for row in attempts)
        return SurveyCompletion(content=self._content(raw), response_id=raw.get("id"),
                                attempts=len(attempts), latency_ms=(time.perf_counter() - started) * 1000,
                                usage=usage)
