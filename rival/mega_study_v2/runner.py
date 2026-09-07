"""Explicit v2 runs into a separate ledger, with durable per-attempt accounting."""

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from ..execution import AttemptJournal, ExecutionError, AuthorizationStopped, ReconciliationRequired
from ..mega_study.constants import MODEL_CONFIG, SCHEMA_VERSION
from ..mega_study.prompts import render_prompt
from .resources import load_manifest
from ..mega_study.runner import work_schedule, parse_instant
from .resources import load_prediction_stage
from ..mega_study.utils import ProtocolError, ResponseParseError, append_jsonl, canonical_hash, read_jsonl, text_hash
from . import RUNTIME_REVISION
from .outcomes import parse_model_json, extract_outcome_cells
from .provider import ManagedSurveyProvider


@contextmanager
def exclusive_run_lock(path: Path):
    """OS lock, released on process death; never steal a live writer's ledger."""
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ExecutionError("another process owns this run ledger") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run_benchmark(stage_root, results_path, *, api_key=None, phase="preflight",
                  budget_usd: float, not_after: str, max_new_calls: int | None = None):
    """Run new v2 work. v1 result files must never be passed to this function."""
    results = Path(results_path)
    journal_path = results.with_suffix(results.suffix + ".attempts.sqlite3")
    with exclusive_run_lock(results.with_suffix(results.suffix + ".lock")):
        manifest = load_manifest()
        stage, cases, personas = load_prediction_stage(stage_root)
        schedule = work_schedule(cases, phase)
        if (Path(stage_root) / "sealed" / "outcome_manifest.json").exists():
            raise ProtocolError("outcomes are already materialized; only reanalysis is allowed")
        rows = read_jsonl(results) if results.exists() else []
        scope = {"runtime_revision": RUNTIME_REVISION, "protocol_sha256": manifest["manifest_sha256"],
                 "stage_sha256": stage["stage_sha256"]}
        for row in rows:
            if any(row.get(key) != value for key, value in scope.items()):
                raise ProtocolError("existing results are not from this v2 execution scope")
        if rows and not journal_path.exists():
            raise ProtocolError("attempt journal is missing; cannot safely reconstruct billing from results")
        latest = {row["work_id"]: row for row in rows}
        provider = ManagedSurveyProvider(api_key=api_key)
        if provider.identity() != manifest["provider_identity"]:
            raise ProtocolError("model/provider payload differs from the frozen study")
        journal = AttemptJournal(journal_path, canonical_hash(scope))
        new_work = 0
        before = journal.summary()["attempts"]
        stop_reason = None
        try:
            for case, variant in schedule:
                work_id = f"{case['case_id']}::{variant}"
                if latest.get(work_id, {}).get("status") in {"SUCCESS", "PARSE_FAILURE", "API_FAILURE"}:
                    continue
                if max_new_calls is not None and journal.summary()["attempts"] - before >= max_new_calls:
                    stop_reason = "maximum new attempt count reached"
                    break
                prompt = render_prompt(case, personas[str(case["pid"])], variant)
                if prompt.context_chars > manifest["model_context_policy"]["max_chars"]:
                    raise ProtocolError("rendered prompt exceeds frozen context policy")
                # UTF-8 byte count avoids the old chars/3 under-reservation for
                # token-dense text. This is still a price estimate, not billing.
                input_bound = len((prompt.system + prompt.user).encode("utf-8")) + 1024
                reservation = (input_bound * MODEL_CONFIG["input_cost_per_million_usd"]
                               + MODEL_CONFIG["max_output_tokens"] * MODEL_CONFIG["output_cost_per_million_usd"]) / 1e6
                row = {**scope, "schema_version": SCHEMA_VERSION, "study_id": manifest["study_id"],
                       "work_id": work_id, "case_id": case["case_id"], "participant_study": case["study_id"],
                       "pid": case["pid"], "variant": variant, "provider": provider.identity(),
                       "prompt_sha256": prompt.prompt_sha256,
                       "prompt_template_sha256": manifest["prompt_template_sha256"][variant],
                       "recorded_at": datetime.now(timezone.utc).isoformat(), "predicted_cells": []}
                try:
                    completion = provider.complete_managed(
                        prompt.system, prompt.user, work_id=work_id, journal=journal,
                        budget_usd=budget_usd, not_after=parse_instant(not_after), reservation_usd=reservation,
                        max_new_attempts=(None if max_new_calls is None else max_new_calls - (journal.summary()["attempts"] - before)),
                    )
                    row.update(raw_response=completion.content, raw_response_sha256=text_hash(completion.content),
                               provider_response_id=completion.response_id, usage=completion.usage,
                               cost_usd=completion.usage["provider_cost_usd"], attempts=completion.attempts)
                    try:
                        cells = extract_outcome_cells(str(case["study_id"]), str(case["survey_text"]), parse_model_json(completion.content))
                        row.update(status="SUCCESS", predicted_cells=[cell.as_dict() for cell in cells])
                    except ResponseParseError as exc:
                        row.update(status="PARSE_FAILURE", failure_detail=str(exc))
                except AuthorizationStopped as exc:
                    stop_reason = str(exc)
                    break
                except ReconciliationRequired as exc:
                    row.update(status="RECONCILIATION_REQUIRED", cost_usd=None, failure_detail=str(exc))
                    stop_reason = str(exc)
                except ExecutionError as exc:
                    attempts = journal.attempts_for(work_id)
                    row.update(status="API_FAILURE", cost_usd=sum(item["cost"] for item in attempts),
                               failure_detail=str(exc), attempts=len(attempts))
                append_jsonl(results, row)
                latest[work_id] = row
                new_work += 1
                if stop_reason or row["status"] != "SUCCESS":
                    stop_reason = stop_reason or "stopped after failed work item; review before continuing"
                    break
            terminal = sum(latest.get(f"{case['case_id']}::{variant}", {}).get("status")
                           in {"SUCCESS", "PARSE_FAILURE", "API_FAILURE"} for case, variant in schedule)
            return {**scope, **journal.summary(), "status": "COMPLETE" if terminal == len(schedule) else "STOPPED",
                    "new_attempts": journal.summary()["attempts"] - before, "new_work_items": new_work,
                    "terminal_work_items": terminal, "expected_work_items": len(schedule), "stop_reason": stop_reason}
        finally:
            journal.close()
