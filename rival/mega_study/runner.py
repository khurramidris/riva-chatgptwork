"""Resumable, budgeted A/B/C/D Mega-Study execution without outcome access."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .constants import MODEL_CONFIG, SCHEMA_VERSION, SELECTION_SEED, VARIANTS
from .outcomes import (
    extract_outcome_cells,
    parse_model_json,
    validate_complete_response,
)
from .prompts import render_prompt
from .protocol import load_manifest
from .provider import MegaStudyProviderError, OpenRouterSurveyProvider
from .stage import load_prediction_stage
from .utils import (
    MegaStudyError,
    ProtocolError,
    ResponseParseError,
    append_jsonl,
    atomic_json,
    canonical_hash,
    file_hash,
    read_jsonl,
    stable_key,
    text_hash,
)


ProgressCallback = Callable[[dict[str, Any]], None]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_instant(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    instant = datetime.fromisoformat(normalized)
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=timezone.utc)
    return instant.astimezone(timezone.utc)


@dataclass(slots=True)
class RunBudget:
    budget_usd: float
    not_after: datetime
    spent_usd: float = 0.0
    attempted_calls: int = 0

    def authorize(self, estimated_cost: float) -> None:
        if datetime.now(timezone.utc) >= self.not_after:
            raise MegaStudyError("run authorization expired before provider call")
        if estimated_cost < 0:
            raise ValueError("estimated cost must be nonnegative")
        if self.spent_usd + estimated_cost > self.budget_usd + 1e-12:
            raise MegaStudyError("next call could exceed the run budget")

    def charge(self, cost: float) -> None:
        if cost < 0:
            raise ValueError("provider cost must be nonnegative")
        self.spent_usd += cost
        self.attempted_calls += 1


def balanced_variant_order(study_id: str, selection_rank: int) -> tuple[str, ...]:
    """Assign each study exactly 25 cases to each cyclic starting variant."""

    base = int(stable_key(SELECTION_SEED, "variant-order", study_id)[:8], 16) % 4
    offset = (base + int(selection_rank) - 1) % 4
    return VARIANTS[offset:] + VARIANTS[:offset]


def work_schedule(
    cases: list[dict[str, Any]], phase: str
) -> list[tuple[dict[str, Any], str]]:
    if phase not in {"preflight", "pilot"}:
        raise ValueError("phase must be preflight or pilot")
    selected = cases[:1] if phase == "preflight" else cases
    schedule: list[tuple[dict[str, Any], str]] = []
    for case in selected:
        frozen_order = tuple(case.get("variant_order", ()))
        expected_order = balanced_variant_order(
            str(case["study_id"]), int(case["selection_rank"])
        )
        if frozen_order != expected_order:
            raise ProtocolError(f"variant order drifted for {case['case_id']}")
        schedule.extend((case, variant) for variant in frozen_order)
    return schedule


def work_id(case_id: str, variant: str) -> str:
    return f"{case_id}::{variant}"


def _latest_ledger(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path) if path.exists() else []:
        identifier = str(row.get("work_id", ""))
        if not identifier:
            raise ProtocolError("Mega-Study ledger contains a row without work_id")
        latest[identifier] = row
    return latest


def _cost(usage: dict[str, float]) -> float:
    provider_cost = usage.get("provider_cost_usd")
    if provider_cost is not None:
        return float(provider_cost)
    return (
        usage.get("prompt_tokens", 0.0)
        * MODEL_CONFIG["input_cost_per_million_usd"]
        + usage.get("completion_tokens", 0.0)
        * MODEL_CONFIG["output_cost_per_million_usd"]
    ) / 1_000_000.0


def _worst_case_cost(context_chars: int) -> float:
    # One token per three characters is deliberately conservative for the
    # pre-call guard. Provider-reported usage replaces it after completion.
    input_tokens = math.ceil(context_chars / 3.0)
    return (
        input_tokens * MODEL_CONFIG["input_cost_per_million_usd"]
        + MODEL_CONFIG["max_output_tokens"]
        * MODEL_CONFIG["output_cost_per_million_usd"]
    ) / 1_000_000.0


def run_benchmark(
    stage_root: str | Path,
    results_path: str | Path,
    *,
    api_key: str | None = None,
    phase: str = "preflight",
    budget_usd: float,
    not_after: str,
    max_new_calls: int | None = None,
    max_errors: int = 20,
    summary_path: str | Path | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    manifest = load_manifest()
    stage, cases, personas = load_prediction_stage(stage_root)
    provider = OpenRouterSurveyProvider(api_key=api_key)
    identity = provider.identity()
    if identity != manifest["provider_identity"]:
        raise ProtocolError("provider identity does not match frozen manifest")
    results = Path(results_path)
    latest = _latest_ledger(results)
    for row in latest.values():
        if row.get("protocol_sha256") != manifest["manifest_sha256"]:
            raise ProtocolError("existing ledger belongs to a different protocol")
        if row.get("provider") != identity:
            raise ProtocolError("existing ledger used a different model/provider identity")
    prior_spend = sum(float(row.get("cost_usd", 0.0)) for row in latest.values())
    guard = RunBudget(
        budget_usd=float(budget_usd),
        not_after=parse_instant(not_after),
        spent_usd=prior_spend,
    )
    terminal = {"SUCCESS", "PARSE_FAILURE", "API_FAILURE", "CONTEXT_FAILURE"}
    schedule = work_schedule(cases, phase)
    errors = 0
    new_calls = 0
    stop_reason: str | None = None
    for case, variant in schedule:
        identifier = work_id(str(case["case_id"]), variant)
        if latest.get(identifier, {}).get("status") in terminal:
            continue
        if max_new_calls is not None and new_calls >= max_new_calls:
            stop_reason = "maximum new call count reached"
            break
        persona = personas.get(str(case["pid"]))
        if persona is None:
            raise ProtocolError(f"staged persona missing for {case['pid']}")
        prompt = render_prompt(case, persona, variant)
        prompt_template_hash = manifest["prompt_template_sha256"][variant]
        try:
            if prompt.context_chars > int(manifest["model_context_policy"]["max_chars"]):
                raise ProtocolError("rendered prompt exceeds frozen context policy")
            guard.authorize(_worst_case_cost(prompt.context_chars))
        except MegaStudyError as exc:
            stop_reason = str(exc)
            break
        base_row: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "study_id": manifest["study_id"],
            "protocol_sha256": manifest["manifest_sha256"],
            "stage_sha256": stage["stage_sha256"],
            "work_id": identifier,
            "case_id": case["case_id"],
            "participant_study": case["study_id"],
            "pid": case["pid"],
            "variant": variant,
            "provider": identity,
            "prompt_template_sha256": prompt_template_hash,
            "prompt_sha256": prompt.prompt_sha256,
            "context_chars": prompt.context_chars,
            "estimated_input_tokens_upper": math.ceil(prompt.context_chars / 3.0),
            "retrieval_audit": list(prompt.retrieval_audit),
            "recorded_at": utc_now(),
        }
        try:
            completion = provider.complete(prompt.system, prompt.user)
            cost = _cost(completion.usage)
            guard.charge(cost)
            new_calls += 1
            try:
                response = parse_model_json(completion.content)
                validate_complete_response(str(case["survey_text"]), response)
                cells = extract_outcome_cells(
                    str(case["study_id"]), str(case["survey_text"]), response
                )
            except ResponseParseError as exc:
                row = {
                    **base_row,
                    "status": "PARSE_FAILURE",
                    "failure_type": type(exc).__name__,
                    "failure_detail": str(exc)[:500],
                    "raw_response": completion.content,
                    "raw_response_sha256": text_hash(completion.content),
                    "provider_response_id": completion.response_id,
                    "attempts": completion.attempts,
                    "latency_ms": completion.latency_ms,
                    "usage": completion.usage,
                    "cost_usd": cost,
                    "predicted_cells": [],
                }
                errors += 1
            else:
                row = {
                    **base_row,
                    "status": "SUCCESS",
                    "raw_response": completion.content,
                    "raw_response_sha256": text_hash(completion.content),
                    "provider_response_id": completion.response_id,
                    "attempts": completion.attempts,
                    "latency_ms": completion.latency_ms,
                    "usage": completion.usage,
                    "cost_usd": cost,
                    "predicted_cells": [cell.as_dict() for cell in cells],
                }
        except MegaStudyProviderError as exc:
            # Provider errors are already redacted. No prompt or key is persisted.
            guard.charge(0.0)
            new_calls += 1
            errors += 1
            detail = str(exc)[:800]
            status = (
                "CONTEXT_FAILURE"
                if any(
                    marker in detail.casefold()
                    for marker in ("context length", "context window", "too many tokens")
                )
                else "API_FAILURE"
            )
            row = {
                **base_row,
                "status": status,
                "failure_type": type(exc).__name__,
                "failure_detail": detail,
                "cost_usd": 0.0,
                "predicted_cells": [],
            }
        append_jsonl(results, row)
        latest[identifier] = row
        if progress:
            progress(
                {
                    "event": "result",
                    "work_id": identifier,
                    "status": row["status"],
                    "new_calls": new_calls,
                    "terminal_work": len(latest),
                    "spent_usd": guard.spent_usd,
                }
            )
        if errors >= max_errors:
            stop_reason = "maximum error count reached"
            break
    expected = len(schedule)
    terminal_count = sum(
        latest.get(work_id(str(case["case_id"]), variant), {}).get("status") in terminal
        for case, variant in schedule
    )
    successful = sum(
        latest.get(work_id(str(case["case_id"]), variant), {}).get("status") == "SUCCESS"
        for case, variant in schedule
    )
    complete = terminal_count == expected
    summary = {
        "schema_version": SCHEMA_VERSION,
        "study_id": manifest["study_id"],
        "protocol_sha256": manifest["manifest_sha256"],
        "phase": phase,
        "status": "COMPLETE" if complete else "STOPPED",
        "stop_reason": None if complete else stop_reason or "work remains",
        "expected_work_items": expected,
        "terminal_work_items": terminal_count,
        "successful_work_items": successful,
        "new_calls": new_calls,
        "errors_this_run": errors,
        "spent_usd": guard.spent_usd,
        "budget_usd": guard.budget_usd,
        "provider": identity,
        "results_sha256": file_hash(results) if results.exists() else None,
        "finished_at": utc_now(),
    }
    if summary_path:
        atomic_json(summary_path, summary)
    return summary


def freeze_predictions(
    stage_root: str | Path,
    results_path: str | Path,
    marker_path: str | Path,
) -> dict[str, Any]:
    manifest = load_manifest()
    stage, cases, _ = load_prediction_stage(stage_root)
    results = Path(results_path)
    latest = _latest_ledger(results)
    required = {
        work_id(str(case["case_id"]), variant)
        for case in cases
        for variant in VARIANTS
    }
    missing = sorted(required - set(latest))
    if missing:
        raise ProtocolError(f"cannot freeze predictions: {len(missing)} work items missing")
    terminal = {"SUCCESS", "PARSE_FAILURE", "API_FAILURE", "CONTEXT_FAILURE"}
    nonterminal = sorted(
        identifier for identifier in required if latest[identifier].get("status") not in terminal
    )
    if nonterminal:
        raise ProtocolError(
            f"cannot freeze predictions: {len(nonterminal)} work items are nonterminal"
        )
    mismatched = sorted(
        identifier
        for identifier in required
        if latest[identifier].get("protocol_sha256") != manifest["manifest_sha256"]
        or latest[identifier].get("stage_sha256") != stage["stage_sha256"]
        or latest[identifier].get("provider") != manifest["provider_identity"]
    )
    if mismatched:
        raise ProtocolError(
            f"cannot freeze predictions: {len(mismatched)} work items have "
            "protocol, stage, or provider drift"
        )
    status_counts: dict[str, int] = {}
    for identifier in required:
        status = str(latest[identifier]["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    payload = {
        "schema_version": SCHEMA_VERSION,
        "study_id": manifest["study_id"],
        "protocol_sha256": manifest["manifest_sha256"],
        "stage_sha256": stage["stage_sha256"],
        "results_sha256": file_hash(results),
        "required_work_items": len(required),
        "status_counts": status_counts,
        "frozen_at": utc_now(),
        "outcomes_opened": False,
    }
    payload["freeze_sha256"] = canonical_hash(payload)
    atomic_json(marker_path, payload)
    return payload
