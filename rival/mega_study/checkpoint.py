"""Outcome-blind operational audits for partial Mega-Study ledgers."""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .constants import FORBIDDEN_PREDICTION_KEYS, VARIANTS
from .protocol import load_manifest
from .stage import load_prediction_stage
from .utils import LeakageError, ProtocolError, canonical_hash, file_hash, read_jsonl


TERMINAL_STATUSES = {
    "SUCCESS",
    "PARSE_FAILURE",
    "API_FAILURE",
    "CONTEXT_FAILURE",
}
FAILURE_STATUSES = TERMINAL_STATUSES - {"SUCCESS"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _work_id(case_id: str, variant: str) -> str:
    return f"{case_id}::{variant}"


def _forbidden_keys(value: Any) -> set[str]:
    forbidden = {key.casefold() for key in FORBIDDEN_PREDICTION_KEYS}
    collisions: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, nested in item.items():
                normalized = str(key).casefold()
                if normalized in forbidden:
                    collisions.add(normalized)
                visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)

    visit(value)
    return collisions


def _finite_nonnegative(value: Any, *, label: str, row_number: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"ledger row {row_number} has invalid {label}") from exc
    if not math.isfinite(number) or number < 0:
        raise ProtocolError(f"ledger row {row_number} has invalid {label}")
    return number


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _stats(values: Iterable[float]) -> dict[str, float | int | None]:
    numbers = [float(value) for value in values]
    return {
        "n": len(numbers),
        "mean": sum(numbers) / len(numbers) if numbers else None,
        "min": min(numbers) if numbers else None,
        "p50": _quantile(numbers, 0.50),
        "p95": _quantile(numbers, 0.95),
        "max": max(numbers) if numbers else None,
    }


def audit_checkpoint(
    stage_root: str | Path,
    results_path: str | Path,
    *,
    expected_terminal: int | None = None,
    budget_usd: float | None = None,
    max_failures: int = 0,
) -> dict[str, Any]:
    """Verify a ledger prefix without opening or summarizing target outcomes."""

    if expected_terminal is not None and expected_terminal < 0:
        raise ValueError("expected_terminal must be nonnegative")
    if budget_usd is not None and budget_usd <= 0:
        raise ValueError("budget_usd must be positive")
    if max_failures < 0:
        raise ValueError("max_failures must be nonnegative")

    root = Path(stage_root)
    results = Path(results_path)
    if (root / "sealed").exists():
        raise LeakageError(
            "sealed Mega-Study outcomes already exist; partial blind auditing is closed"
        )

    manifest = load_manifest()
    stage, cases, _ = load_prediction_stage(root)
    schedule: list[tuple[dict[str, Any], str]] = []
    for case in cases:
        order = tuple(str(value) for value in case.get("variant_order", ()))
        if len(order) != len(VARIANTS) or set(order) != set(VARIANTS):
            raise ProtocolError("staged case has an invalid frozen variant order")
        schedule.extend((case, variant) for variant in order)
    if len(schedule) != 1200:
        raise ProtocolError("frozen Mega-Study schedule must contain 1,200 work items")

    rows = read_jsonl(results) if results.exists() else []
    if len(rows) > len(schedule):
        raise ProtocolError("ledger contains more rows than the frozen schedule")
    if expected_terminal is not None and expected_terminal > len(schedule):
        raise ValueError("expected_terminal exceeds the frozen schedule")

    seen: set[str] = set()
    status_counts: Counter[str] = Counter()
    study_counts: dict[str, Counter[str]] = {
        str(study): Counter() for study in manifest["development_studies"]
    }
    variant_counts: Counter[str] = Counter()
    costs: list[float] = []
    latencies: list[float] = []
    attempts: list[float] = []
    prompt_tokens: list[float] = []
    completion_tokens: list[float] = []

    for row_number, (row, expected) in enumerate(zip(rows, schedule), 1):
        case, variant = expected
        expected_id = _work_id(str(case["case_id"]), variant)
        identifier = str(row.get("work_id", ""))
        if identifier != expected_id:
            raise ProtocolError(
                f"ledger is not the canonical frozen schedule prefix at row {row_number}"
            )
        if identifier in seen:
            raise ProtocolError(f"ledger duplicates a work item at row {row_number}")
        seen.add(identifier)
        if _forbidden_keys(row):
            raise LeakageError(f"protected outcome key entered ledger row {row_number}")
        if row.get("protocol_sha256") != manifest["manifest_sha256"]:
            raise ProtocolError(f"protocol identity drifted at ledger row {row_number}")
        if row.get("stage_sha256") != stage["stage_sha256"]:
            raise ProtocolError(f"stage identity drifted at ledger row {row_number}")
        if row.get("provider") != manifest["provider_identity"]:
            raise ProtocolError(f"provider identity drifted at ledger row {row_number}")
        if (
            row.get("prompt_template_sha256")
            != manifest["prompt_template_sha256"][variant]
        ):
            raise ProtocolError(f"prompt template drifted at ledger row {row_number}")
        if (
            row.get("case_id") != case["case_id"]
            or row.get("participant_study") != case["study_id"]
            or row.get("pid") != case["pid"]
            or row.get("variant") != variant
        ):
            raise ProtocolError(f"frozen work metadata drifted at ledger row {row_number}")
        status = str(row.get("status", ""))
        if status not in TERMINAL_STATUSES:
            raise ProtocolError(f"ledger row {row_number} is not terminal")

        context_chars = _finite_nonnegative(
            row.get("context_chars"), label="context_chars", row_number=row_number
        )
        if context_chars > int(manifest["model_context_policy"]["max_chars"]):
            raise ProtocolError(f"context policy exceeded at ledger row {row_number}")
        costs.append(
            _finite_nonnegative(
                row.get("cost_usd", 0.0), label="cost_usd", row_number=row_number
            )
        )
        latencies.append(
            _finite_nonnegative(
                row.get("latency_ms", 0.0),
                label="latency_ms",
                row_number=row_number,
            )
        )
        attempts.append(
            _finite_nonnegative(
                row.get("attempts", 0.0), label="attempts", row_number=row_number
            )
        )
        usage = row.get("usage")
        if usage is not None and not isinstance(usage, dict):
            raise ProtocolError(f"ledger row {row_number} has invalid usage metadata")
        usage = usage or {}
        prompt_tokens.append(
            _finite_nonnegative(
                usage.get("prompt_tokens", 0.0),
                label="prompt_tokens",
                row_number=row_number,
            )
        )
        completion_tokens.append(
            _finite_nonnegative(
                usage.get("completion_tokens", 0.0),
                label="completion_tokens",
                row_number=row_number,
            )
        )

        predicted_cells = row.get("predicted_cells")
        if not isinstance(predicted_cells, list):
            raise ProtocolError(f"ledger row {row_number} has invalid predicted_cells")
        if status == "SUCCESS":
            expected_outcomes = {str(value) for value in case["expected_outcome_ids"]}
            observed_outcomes = {
                str(cell.get("outcome_id"))
                for cell in predicted_cells
                if isinstance(cell, dict)
            }
            if (
                observed_outcomes != expected_outcomes
                or len(predicted_cells) != len(expected_outcomes)
            ):
                raise ProtocolError(
                    f"successful ledger row {row_number} has incomplete outcome structure"
                )
        elif predicted_cells:
            raise ProtocolError(
                f"failed ledger row {row_number} contains predicted outcome cells"
            )

        status_counts[status] += 1
        study_counts[str(case["study_id"])][variant] += 1
        variant_counts[variant] += 1

    failure_count = sum(status_counts[name] for name in FAILURE_STATUSES)
    if expected_terminal is not None and len(rows) != expected_terminal:
        raise ProtocolError(
            f"expected {expected_terminal} terminal work items, found {len(rows)}"
        )
    if failure_count > max_failures:
        raise ProtocolError(
            f"checkpoint contains {failure_count} failures; allowed maximum is {max_failures}"
        )
    total_cost = sum(costs)
    if budget_usd is not None and total_cost > budget_usd + 1e-12:
        raise ProtocolError(
            f"checkpoint cost ${total_cost:.6f} exceeds ${budget_usd:.6f} budget"
        )

    terminal_work = len(rows)
    complete = terminal_work == len(schedule)
    verification = (
        "PREDICTIONS_COMPLETE"
        if complete
        else "EMPTY_LEDGER_READY"
        if terminal_work == 0
        else "CHECKPOINT_COMPLETE"
        if expected_terminal is not None
        else "CANONICAL_PREFIX_VALID"
    )
    report: dict[str, Any] = {
        "schema_version": "rival.mega-study.checkpoint-audit.v1",
        "study_id": manifest["study_id"],
        "status": "PASS",
        "verification_status": verification,
        "audit_mode": "OUTCOME_BLIND_OPERATIONAL_ONLY",
        "protocol_sha256": manifest["manifest_sha256"],
        "stage_sha256": stage["stage_sha256"],
        "results_sha256": file_hash(results) if results.exists() else None,
        "ledger": {
            "canonical_prefix_verified": True,
            "terminal_work_items": terminal_work,
            "successful_work_items": status_counts["SUCCESS"],
            "failure_work_items": failure_count,
            "remaining_work_items": len(schedule) - terminal_work,
            "complete": complete,
            "status_counts": {
                name: status_counts[name] for name in sorted(TERMINAL_STATUSES)
            },
        },
        "coverage": {
            "by_variant": {name: variant_counts[name] for name in VARIANTS},
            "by_study_and_variant": {
                study: {name: counts[name] for name in VARIANTS}
                for study, counts in study_counts.items()
            },
        },
        "operations": {
            "spent_usd": total_cost,
            "budget_usd": budget_usd,
            "within_budget": budget_usd is None or total_cost <= budget_usd + 1e-12,
            "latency_ms": _stats(latencies),
            "attempts": _stats(attempts),
            "prompt_tokens": {
                "total": sum(prompt_tokens),
                "per_call": _stats(prompt_tokens),
            },
            "completion_tokens": {
                "total": sum(completion_tokens),
                "per_call": _stats(completion_tokens),
            },
        },
        "provider_identity": manifest["provider_identity"],
        "leakage_firewall": {
            "sealed_outcome_directory_absent": True,
            "protected_outcome_keys_absent": True,
            "human_outcomes_loaded": False,
            "prediction_values_summarized": False,
            "partial_arm_comparisons_performed": False,
        },
        "scientific_accuracy_status": "NOT_EVALUATED_TO_PRESERVE_BLIND",
        "audited_at": _utc_now(),
    }
    report["audit_sha256"] = canonical_hash(report)
    return report
