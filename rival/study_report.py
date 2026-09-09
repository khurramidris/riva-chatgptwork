"""Aggregate study exports; raw personas and workspace keys stay in the ledger."""

import hashlib
import html
import json
import os
from pathlib import Path
import tempfile

from .mathx import canonical_hash
from .readiness import release_claims


def build_study_report(request, prepared, simulation, sealed, accounting, evaluation=None, *, model_execution=None,
                       calibration=None, calibration_comparison=None, uncertainty=None, uncertainty_comparison=None):
    claims = release_claims()
    claims["release"] = prepared["release"]
    reference = request.brief.choices[0].choice_id
    choices = [{"choice_id": choice.choice_id, "label": choice.label,
                "simulated_share": simulation.distribution[choice.choice_id],
                "difference_from_reference_pp": 100 * (simulation.distribution[choice.choice_id] - simulation.distribution[reference])}
               for choice in request.brief.choices]
    sources = [{key: source.model_dump(mode="json")[key] for key in
                ("source_id", "name", "source_type", "collected_at", "geography",
                 "rights_reference", "permitted_uses", "sha256")}
               for source in request.audience.sources]
    warnings = [
        "Choice shares describe model outputs for the supplied audience; customer-domain accuracy is not established.",
        "Resampled draws do not add independent people or human evidence.",
        "Differences between choice shares are not causal effects of an intervention.",
        "Source rights, dates, geography and independence are operator declarations, not independently verified facts.",
        "Confidence remains unqualified; no decision-ready interval or recommendation is issued.",
    ]
    if any(source.source_type == "synthetic" for source in request.audience.sources):
        warnings.insert(0, "This study includes generated audience records; it is an engineering demonstration.")
    if request.execution.mode == "offline":
        warnings.insert(0, "Offline heuristic baseline: no language model was called.")
    if request.audience.targets is None:
        warnings.append("No population controls were supplied; results refer to the weighted seed sample.")
    report = {
        "schema_version": "rival.study-report.v1", "release_claims": claims,
        "study_id": request.brief.study_id, "title": request.brief.title,
        "decision": request.brief.decision, "question": request.brief.question,
        "request_sha256": prepared["request_sha256"], "run_id": simulation.run_id,
        "created_at": simulation.created_at.isoformat(), "runtime_release": prepared["release"],
        "audience": {"description": request.audience.description,
                     "geography": request.audience.geography, "input_seed_records": len(request.audience.records),
                     "eligible_seed_records": prepared["eligible_seed_records"],
                     "effective_seed_records": prepared["effective_seed_records"],
                     "unique_sampled_seed_records": prepared["planned_unique_seeds"],
                     "simulation_draws": prepared["planned_draws"],
                     "population_diagnostics": prepared["population_diagnostics"]},
        "choices": choices, "reference_choice_id": reference,
        "confidence": {"label": "unqualified", "abstain": True, "lower_tvd": 0.0, "upper_tvd": 1.0},
        "execution": {"mode": request.execution.mode, "provider": prepared["prediction_context"]["provider"],
                      "accounting": accounting, "sample_plan_sha256": prepared["sample_plan_sha256"]},
        "evidence": {"assignment": request.evidence.model_dump(mode="json"), "sources": sources,
                     "information_cutoff": request.brief.information_cutoff.isoformat(),
                     "excluded_history_entries": sum(entry["excluded_history_count"] for entry in prepared["retrieval_audit"]["entries"]),
                     "retrieval_audit_sha256": prepared["retrieval_audit"]["audit_sha256"],
                     "manifest_sha256": canonical_hash(sealed),
                     "local_manifest_verified": True,
                     "protected_comparison_verified": evaluation is not None,
                     "verification_scope": "workspace-local signature and ledger; key custody and host clock remain trusted"},
        "evaluation": evaluation, "limitations": warnings,
    }
    if "support_audit" in prepared:
        report["schema_version"] = "rival.study-report.v2"
        report["support"] = prepared["support_audit"]
        report["evidence"]["import_verification"] = prepared["import_verification"]
        warnings.extend(prepared["support_audit"]["warnings"])
        warnings.append("Declared support checks do not establish population representativeness or validate the scenario's meaning.")
    if "model_execution" in prepared:
        if model_execution is None:
            raise ValueError("model execution evidence is missing")
        report["schema_version"] = "rival.study-report.v3"
        report["execution"]["model_specification"] = prepared["model_execution"]
        report["execution"]["measurements"] = model_execution
        warnings.extend([prepared["model_execution"]["identity_scope"],
                         prepared["model_execution"]["replication_scope"]])
        if model_execution.get("elicitation", {}).get("ssr_degenerate", 0):
            warnings.append("Some SSR responses had no distinguishing embedding signal and received uniform mass; inspect the execution audit.")
    if "calibration" in prepared:
        if calibration is None:
            raise ValueError("calibration prediction evidence is missing")
        report["schema_version"] = "rival.study-report.v4"
        report["calibration"] = {"prediction": calibration, "comparison": calibration_comparison}
        warnings.append("Calibration was fitted only to its declared reference training groups. New-question support and improvement are unqualified.")
        if not calibration["fit_diagnostics"]["converged"]:
            warnings.append("Calibration reached its iteration limit before the declared simplex-gap tolerance; inspect fit diagnostics.")
    if "uncertainty" in prepared:
        if uncertainty is None:
            raise ValueError("uncertainty prediction evidence is missing")
        report["schema_version"] = "rival.study-report.v5"
        report["uncertainty"] = {"prediction": uncertainty, "comparison": uncertainty_comparison}
        report["confidence"] = uncertainty["confidence"]
    return report


def _text(value):
    text = html.escape(str(value)).replace("\n", " ").replace("\r", " ")
    for character in ("\\", "|", "*", "_", "`", "[", "]", "#"):
        text = text.replace(character, "\\" + character)
    return text


def study_markdown(report):
    population = report["audience"]
    lines = [f"# {_text(report['title'])}", "", "**Research simulation — confidence unqualified.**", "",
             f"Decision being explored: {_text(report['decision'])}", "",
             f"Question: {_text(report['question'])}", "", "## Simulated choices", "",
             "| Choice | Share | Difference from reference |", "|---|---:|---:|"]
    for choice in report["choices"]:
        lines.append(f"| {_text(choice['label'])} | {choice['simulated_share']:.1%} | {choice['difference_from_reference_pp']:+.1f} percentage points |")
    lines.extend(["", f"Reference: {_text(report['reference_choice_id'])}. These differences are not causal effects.",
        "", "## Audience and execution", "",
        f"- Audience: {_text(population['description'])}",
        f"- Declared geography: {_text(', '.join(population['geography']))}",
        f"- Eligible seed records: {population['eligible_seed_records']}",
        f"- Effective seed records under the weights: {population['effective_seed_records']:.1f}",
        f"- Unique seed records sampled: {population['unique_sampled_seed_records']}",
        f"- Simulation draws: {population['simulation_draws']}",
        f"- Execution mode: {_text(report['execution']['mode'])}",
        f"- Accounted request attempts: {report['execution']['accounting']['attempts']}",
        f"- Total recorded request cost: ${report['execution']['accounting']['total_cost_usd']:.6f}",
        f"- Information cutoff: {_text(report['evidence']['information_cutoff'])}",
        "", "## Evidence", ""])
    for source in report["evidence"]["sources"]:
        lines.append(f"- {_text(source['name'])} ({_text(source['source_type'])}); rights reference: {_text(source['rights_reference'])}")
    if "support" in report:
        support = report["support"]
        lines.extend(["", "## Declared population support", "",
            f"- Support checks passed: {support['passed']}",
            f"- Seed records excluded by geography and audience filters: {support['excluded_seed_records']}",
            "- Source files and conversions were verified at preparation.",
            "- Condition IDs: " + _text(", ".join(support["conditions"]))])
        for cell in support["cells"]:
            lines.append(f"- {_text(cell['label'])}: {cell['seed_records']} seeds; effective count {cell['effective_seed_records']:.1f}.")
        lines.append("- Checks describe supplied evidence; they do not establish representative sampling or predictive accuracy.")
    if report["evaluation"]:
        lines.append(f"- Protected outcome comparison TVD: {report['evaluation']['metrics']['tvd']:.4f}; verified against the local sealed ledger.")
    else:
        lines.append("- No observed-outcome comparison is available for this study.")
    if "measurements" in report["execution"]:
        measured = report["execution"]["measurements"]
        settings = report["execution"]["model_specification"]["settings"]
        lines.extend(["", "## Model execution", "",
            f"- Requested model: {_text(settings['model'])}; elicitation: {_text(settings['elicitation']['method'])}.",
            f"- Accepted seed requests: {measured['accepted_seed_requests']}/{measured['planned_seed_requests']}.",
            f"- Measured HTTP time: {measured['transport_latency_ms']['total'] / 1000:.2f} seconds across {measured['transport_latency_ms']['measured_attempts']} attempts.",
            "- Saved model outputs are reused on resume. A fresh repeat is a separate study and may differ.",
            "- This measures model execution; accuracy against real people remains unqualified."])
    if "calibration" in report:
        prediction = report["calibration"]["prediction"]
        lines.extend(["", "## Calibration", "",
            f"Fitted to {prediction['training_groups']} training groups using a fixed panel of {prediction['seed_records']} seeds. Both predictions were saved before outcome reveal.", "",
            "| Choice | Raw simulation | Calibrated | Training mean baseline |", "|---|---:|---:|---:|"])
        for choice in report["choices"]:
            identifier = choice["choice_id"]
            lines.append(f"| {_text(choice['label'])} | {prediction['raw_distribution'][identifier]:.1%} | {prediction['calibrated_distribution'][identifier]:.1%} | {prediction['historical_mean_distribution'][identifier]:.1%} |")
        comparison = report["calibration"]["comparison"]
        if comparison:
            lines.extend(["", "| Compared with protected outcomes | TVD (lower is better) |", "|---|---:|"])
            for method, metrics in comparison["metrics"].items():
                lines.append(f"| {_text(method)} | {metrics['tvd']:.4f} |")
            lines.append(f"\nTVD reduction versus raw: {comparison['tvd_reduction_vs_raw']:+.4f}. A negative value means calibration made this study worse.")
        else:
            lines.append("\nNo protected outcome comparison yet; these adjustments are not evidence of better accuracy.")
    if "uncertainty" in report:
        prediction = report["uncertainty"]["prediction"]
        research = prediction["research_assessment"]
        counts = prediction["evidence_counts"]
        lines.extend(["", "## Reliability research", "",
            f"Estimated error: {research['expected_tvd']:.4f} TVD. Research error range: [0, {research['upper_tvd']:.4f}].",
            f"This assesses the {prediction['prediction_kind']} distribution using {counts['training']} training, {counts['calibration']} calibration and {counts['evaluation']} evaluation study groups.",
            f"Candidate policy would accept: {research['candidate_accept']}. Held-out statistical gates passed: {prediction['statistical_gates_passed']}.",
            "Customer decision support: abstain. Untouched domain qualification remains pending.",
            "The research bound assumes comparable independent studies. It does not guarantee accuracy for a specific person, subgroup or changed market."])
        for reason in prediction["confidence"]["reason_codes"]:
            lines.append("- " + _text(reason.replace("_", " ")))
        comparison = report["uncertainty"]["comparison"]
        if comparison:
            lines.append(f"Observed error: {comparison['observed_tvd']:.4f}. Research bound covered it: {comparison['research_bound_covered']}. No refitting occurred.")
    lines.extend(["", "## Limitations", "", *["- " + _text(item) for item in report["limitations"]],
        "", f"Study: {_text(report['study_id'])}. Input fingerprint: `{report['request_sha256']}`.", ""])
    return "\n".join(lines)


def export_report(report, output, workspace_root):
    output = Path(output).resolve()
    if output.is_relative_to(workspace_root) or workspace_root.is_relative_to(output):
        raise ValueError("report destination must be separate from the study workspace")
    files = {"report.json": (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"),
             "report.md": study_markdown(report).encode("utf-8")}
    manifest = {"schema_version": "rival.study-export.v1", "study_id": report["study_id"],
                "request_sha256": report["request_sha256"],
                "files": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}}
    files["manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if output.exists():
        if (not output.is_dir() or {path.name for path in output.iterdir()} != set(files)
                or any((output / name).is_symlink() or not (output / name).is_file()
                       or (output / name).read_bytes() != content for name, content in files.items())):
            raise ValueError("report destination already contains different content; choose a new destination")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="rival-export-", dir=output.parent) as directory:
            staged = Path(directory) / "report"
            staged.mkdir()
            for name, content in files.items():
                with (staged / name).open("xb") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
            staged.rename(output)
    return {"output": str(output), "study_id": report["study_id"], "files": manifest["files"]}
