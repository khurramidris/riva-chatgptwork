#!/usr/bin/env python3
"""Audit a partial live-pilot ledger without reading protected outcomes."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from rival.live_pilot import _sha256_portable_text
from rival.mathx import canonical_hash


DEFAULT_STUDY = REPOSITORY_ROOT / "rival" / "studies" / "twin2k_live_v2"


class CheckpointAuditError(RuntimeError):
    pass


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise CheckpointAuditError(f"missing result ledger: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise CheckpointAuditError(
                f"invalid JSON at {path}:{line_number}"
            ) from exc
    return rows


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
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "n": len(finite),
        "mean": sum(finite) / len(finite) if finite else None,
        "min": min(finite) if finite else None,
        "p50": _quantile(finite, 0.50),
        "p95": _quantile(finite, 0.95),
        "max": max(finite) if finite else None,
    }


def _normalized_entropy(probabilities: list[float]) -> float:
    if len(probabilities) <= 1:
        return 0.0
    entropy = -sum(value * math.log(value) for value in probabilities if value > 0)
    return entropy / math.log(len(probabilities))


def _jsd(left: list[float], right: list[float]) -> float:
    midpoint = [(a + b) / 2 for a, b in zip(left, right, strict=True)]

    def divergence(values: list[float]) -> float:
        return sum(
            value * math.log2(value / middle)
            for value, middle in zip(values, midpoint, strict=True)
            if value > 0 and middle > 0
        )

    return 0.5 * divergence(left) + 0.5 * divergence(right)


def audit_checkpoint(
    protocol_path: Path,
    cases_path: Path,
    results_path: Path,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    unsigned_protocol = dict(protocol)
    expected_protocol_hash = unsigned_protocol.pop("protocol_sha256", None)
    if expected_protocol_hash != canonical_hash(unsigned_protocol):
        raise CheckpointAuditError("protocol hash does not verify")
    if _sha256_portable_text(cases_path) != protocol["cases"]["sha256"]:
        raise CheckpointAuditError("cases hash does not verify")

    cases = _read_jsonl(cases_path)
    case_by_id = {str(case["case_id"]): case for case in cases}
    target_by_column = {
        str(target["column"]): target for target in protocol["targets"]
    }
    rows = _read_jsonl(results_path)
    successes: dict[str, dict[str, Any]] = {}
    error_rows: list[dict[str, Any]] = []
    providers: set[str] = set()

    for row in rows:
        unsigned = dict(row)
        expected_row_hash = unsigned.pop("result_sha256", None)
        if expected_row_hash != canonical_hash(unsigned):
            raise CheckpointAuditError(
                f"result hash does not verify for {row.get('case_id')}"
            )
        if row.get("protocol_sha256") != expected_protocol_hash:
            raise CheckpointAuditError("ledger mixes protocol identities")
        case_id = str(row.get("case_id"))
        if case_id not in case_by_id:
            raise CheckpointAuditError(f"unknown case in ledger: {case_id}")
        providers.add(canonical_hash(row.get("provider")))
        if row.get("status") == "SUCCESS":
            if case_id in successes:
                raise CheckpointAuditError(f"duplicate successful case: {case_id}")
            successes[case_id] = row
        elif row.get("status") == "ERROR":
            error_rows.append(row)
        else:
            raise CheckpointAuditError(f"unknown ledger status for {case_id}")
    if len(providers) != 1:
        raise CheckpointAuditError("ledger mixes provider identities")

    probability_rows: list[dict[str, Any]] = []
    coverage: dict[str, Counter[str]] = defaultdict(Counter)
    participants: set[int] = set()
    anchors = {int(value) for value in protocol["selection"]["anchor_participant_ids"]}
    paired: dict[tuple[int, str], dict[str, dict[str, Any]]] = defaultdict(dict)

    for case_id, row in successes.items():
        case = case_by_id[case_id]
        target_column = str(case["target_column"])
        variant = str(case["variant"])
        participant = int(case["participant_id"])
        expected_ids = [
            str(choice["choice_id"])
            for choice in target_by_column[target_column]["scenario"]["choices"]
        ]
        raw = row.get("probabilities")
        if not isinstance(raw, dict) or set(raw) != set(expected_ids):
            raise CheckpointAuditError(f"wrong probability keys for {case_id}")
        probabilities = [float(raw[choice_id]) for choice_id in expected_ids]
        if any(not math.isfinite(value) or value < 0 for value in probabilities):
            raise CheckpointAuditError(f"invalid probability value for {case_id}")
        total = sum(probabilities)
        if abs(total - 1.0) > 1e-6:
            raise CheckpointAuditError(f"probabilities do not sum to one for {case_id}")
        top = max(probabilities)
        top_indices = [
            index for index, value in enumerate(probabilities) if abs(value - top) <= 1e-12
        ]
        normalized_entropy = _normalized_entropy(probabilities)
        probability_rows.append(
            {
                "case_id": case_id,
                "target_column": target_column,
                "variant": variant,
                "participant_id": participant,
                "probabilities": probabilities,
                "top_choice": expected_ids[top_indices[0]],
                "top_tie": len(top_indices) > 1,
                "top_probability": top,
                "normalized_entropy": normalized_entropy,
                "near_uniform": max(
                    abs(value - 1 / len(probabilities)) for value in probabilities
                )
                <= 0.02,
            }
        )
        coverage[target_column][variant] += 1
        participants.add(participant)
        paired[(participant, target_column)][variant] = probability_rows[-1]

    paired_rows: list[dict[str, Any]] = []
    for key, variants in paired.items():
        if set(variants) != {"generic", "twin"}:
            continue
        generic = variants["generic"]
        twin = variants["twin"]
        paired_rows.append(
            {
                "participant_id": key[0],
                "target_column": key[1],
                "jsd_bits": _jsd(generic["probabilities"], twin["probabilities"]),
                "top_choice_changed": generic["top_choice"] != twin["top_choice"],
                "twin_minus_generic_top_probability": (
                    twin["top_probability"] - generic["top_probability"]
                ),
                "twin_minus_generic_entropy": (
                    twin["normalized_entropy"] - generic["normalized_entropy"]
                ),
            }
        )

    billed_costs = [float(row.get("billed_cost_usd", 0.0)) for row in successes.values()]
    latencies = [float(row.get("latency_ms", 0.0)) for row in successes.values()]
    prompt_tokens = [
        float(row.get("diagnostics", {}).get("prompt_tokens", 0.0))
        for row in successes.values()
    ]
    completion_tokens = [
        float(row.get("diagnostics", {}).get("completion_tokens", 0.0))
        for row in successes.values()
    ]
    attempts = [float(row.get("attempts", 0.0)) for row in successes.values()]
    total_cost = sum(billed_costs)
    success_count = len(successes)
    mean_cost = total_cost / success_count if success_count else 0.0
    mean_latency = sum(latencies) / success_count if success_count else 0.0

    variant_payload: dict[str, Any] = {}
    for variant in ("generic", "twin"):
        subset = [item for item in probability_rows if item["variant"] == variant]
        variant_payload[variant] = {
            "calls": len(subset),
            "top_probability": _stats(item["top_probability"] for item in subset),
            "normalized_entropy": _stats(
                item["normalized_entropy"] for item in subset
            ),
            "near_uniform_fraction": (
                sum(bool(item["near_uniform"]) for item in subset) / len(subset)
                if subset
                else None
            ),
            "top_tie_fraction": (
                sum(bool(item["top_tie"]) for item in subset) / len(subset)
                if subset
                else None
            ),
        }

    target_coverage = {
        target: {
            "generic": counts.get("generic", 0),
            "twin": counts.get("twin", 0),
            "paired_cells": sum(
                1
                for (participant, column), variants in paired.items()
                if column == target and set(variants) == {"generic", "twin"}
            ),
        }
        for target, counts in sorted(coverage.items())
    }
    anchor_calls = sum(
        int(case_by_id[case_id]["participant_id"]) in anchors for case_id in successes
    )
    warnings: list[str] = []
    if len(target_coverage) < int(protocol["selection"]["target_count"]):
        warnings.append(
            "Checkpoint does not yet cover all frozen target questions; behavior "
            "summaries are not a representative final scientific result."
        )
    warnings.append(
        "Protected human outcomes were not read. Accuracy, Brier, NLL, TVD, "
        "calibration improvement and PASS/FAIL gates remain blinded until 1,500/1,500."
    )

    return {
        "status": "OUTCOME_FREE_CHECKPOINT_AUDIT",
        "study_id": protocol["study_id"],
        "protocol_sha256": expected_protocol_hash,
        "portable_results_sha256": _sha256_portable_text(results_path),
        "provider": next(iter(successes.values())).get("provider") if successes else None,
        "ledger": {
            "rows": len(rows),
            "successful_cases": success_count,
            "error_rows": len(error_rows),
            "integrity": "PASS",
        },
        "coverage": {
            "frozen_total_cases": int(protocol["cases"]["total"]),
            "frozen_target_questions": int(protocol["selection"]["target_count"]),
            "covered_target_questions": len(target_coverage),
            "participants": len(participants),
            "generic_calls": sum(item["variant"] == "generic" for item in probability_rows),
            "twin_calls": sum(item["variant"] == "twin" for item in probability_rows),
            "paired_cells": len(paired_rows),
            "unpaired_cells": len(paired) - len(paired_rows),
            "anchor_calls": anchor_calls,
            "by_target": target_coverage,
        },
        "operations": {
            "billed_cost_usd": total_cost,
            "cost_per_call_usd": _stats(billed_costs),
            "latency_ms": _stats(latencies),
            "prompt_tokens": _stats(prompt_tokens),
            "completion_tokens": _stats(completion_tokens),
            "attempts": _stats(attempts),
            "active_api_time_seconds": sum(latencies) / 1000,
            "projected_1500_cost_usd_at_checkpoint_mean": mean_cost * 1500,
            "projected_remaining_cost_usd_at_checkpoint_mean": (
                mean_cost * max(0, 1500 - success_count)
            ),
            "projected_1500_active_api_hours_at_checkpoint_mean": (
                mean_latency * 1500 / 3_600_000
            ),
        },
        "prediction_behavior": {
            "by_variant": variant_payload,
            "paired_generic_vs_twin": {
                "pairs": len(paired_rows),
                "jsd_bits": _stats(item["jsd_bits"] for item in paired_rows),
                "top_choice_change_fraction": (
                    sum(bool(item["top_choice_changed"]) for item in paired_rows)
                    / len(paired_rows)
                    if paired_rows
                    else None
                ),
                "twin_minus_generic_top_probability": _stats(
                    item["twin_minus_generic_top_probability"] for item in paired_rows
                ),
                "twin_minus_generic_entropy": _stats(
                    item["twin_minus_generic_entropy"] for item in paired_rows
                ),
            },
            "unique_probability_vectors_rounded_6dp": len(
                {
                    tuple(round(value, 6) for value in item["probabilities"])
                    for item in probability_rows
                }
            ),
            "overconfident_fraction_max_probability_ge_0_90": (
                sum(item["top_probability"] >= 0.90 for item in probability_rows)
                / len(probability_rows)
                if probability_rows
                else None
            ),
        },
        "scientific_accuracy_status": "NOT_EVALUATED_TO_PRESERVE_BLIND",
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_STUDY / "protocol.json")
    parser.add_argument("--cases", type=Path, default=DEFAULT_STUDY / "cases.jsonl")
    parser.add_argument(
        "--results", type=Path, default=Path("reports/live_pilot_v2_results.jsonl")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/live_pilot_v2_checkpoint_audit.json"),
    )
    args = parser.parse_args(argv)
    try:
        report = audit_checkpoint(args.protocol, args.cases, args.results)
    except (OSError, ValueError, CheckpointAuditError) as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Audit saved to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
