"""Conservative all-roster qualification statistics and public aggregate export."""

import math

from .mathx import canonical_hash
from .qualification_contract import QualificationProtocol
from .runtime_uncertainty import binomial_bounds


BASELINES = ("synthetic_only", "weighted_history", "classical_multinomial", "human_only")


def design_feasibility(protocol):
    """Best possible gates with this roster, not a power or accuracy forecast."""
    protocol = QualificationProtocol.model_validate(protocol)
    n = len(protocol.members)
    tail = protocol.family_error_rate / (len(BASELINES) + 5)
    zero_error_upper = min(1.0, math.sqrt(math.log(1 / tail) / (2 * n)))
    best_improvement_lower = max(-1.0, 1 - math.sqrt(2 * math.log(1 / tail) / n))
    all_success_lower = binomial_bounds(n, n, tail)["lower"]
    zero_failure_upper = binomial_bounds(0, n, tail)["upper"]
    possible = {"absolute_error": zero_error_upper <= protocol.max_tvd,
        "baseline_improvement": best_improvement_lower >= protocol.minimum_baseline_improvement,
        "coverage": all_success_lower >= protocol.required_coverage,
        "acceptance_rate": all_success_lower >= protocol.minimum_acceptance_rate,
        "accepted_error_risk": zero_failure_upper <= protocol.max_bad_acceptance_rate,
        "failure_rate": zero_failure_upper <= protocol.max_failure_rate}
    return {"statistical_gates_possible": all(possible.values()), "possible": possible,
        "zero_error_mean_upper": zero_error_upper, "maximum_improvement_lower": best_improvement_lower,
        "perfect_coverage_lower": all_success_lower, "zero_failure_rate_upper": zero_failure_upper,
        "interpretation": "Optimistic mathematical feasibility only. Passing requires measured evidence; this is not a power calculation."}


def tvd(a, b):
    return sum(abs(x - y) for x, y in zip(a, b, strict=True)) / 2


def build_assessment(plan, baseline, rows):
    protocol = QualificationProtocol.model_validate(plan["protocol"])
    specifications = {m.study_id: m for m in protocol.members}
    if len(rows) != len(specifications) or {r["study_id"] for r in rows} != set(specifications):
        raise ValueError("assessment must retain the entire frozen study roster")
    n = len(rows)
    # One mean error, four paired improvements, four binomial gates. This is
    # simultaneous control for ONE predeclared experiment, not model search.
    tail = protocol.family_error_rate / (len(BASELINES) + 5)
    mean_radius = math.sqrt(math.log(1 / tail) / (2 * n))
    difference_radius = math.sqrt(2 * math.log(1 / tail) / n)
    errors, deltas = [], {name: [] for name in BASELINES}
    public_rows = []
    covered = accepted = bad = failed = 0
    operational, missingness = [], []
    historical = prospective = 0
    known_cost = 0.0
    accounting_unknown = 0
    for row in rows:
        spec = specifications[row["study_id"]]
        item = {"study_id": row["study_id"], "source_reference": spec.source_reference,
                "evidence_origin": spec.evidence_origin, "outcome_access": spec.outcome_access,
                "timing": spec.timing}
        accounting = row.get("accounting", {})
        known_cost += accounting.get("known_cost_usd", 0.0)
        accounting_unknown += int(not accounting or accounting.get("total_cost_usd") is None)
        failure = row.get("failure")
        if not failure and row["prediction"]["baselines"]["human_only"] is None:
            failure = "no_observed_human_anchor_responses"
        if failure:
            failed += 1
            if row.get("prediction", {}).get("uncertainty", {}).get("candidate_accept"):
                accepted += 1
                bad += 1
            errors.append(1.0)
            for name in BASELINES:
                deltas[name].append(-1.0)
            public_rows.append({**item, "status": "FAILURE", "failure": failure,
                "candidate_tvd": None, "all_roster_error_penalty": 1.0})
            operational.append(False)
            missingness.append(False)
            continue
        prediction, observed = row["prediction"], row["observed"]
        error = tvd(prediction["candidate"], observed)
        missing_mass = row["human_sample"]["missing_weight_fraction"]
        # Unknown missing human responses may be anywhere in the simplex.
        # TVD triangle inequality gives conservative finite-roster sensitivity.
        conservative_error = min(1.0, error + missing_mass)
        errors.append(conservative_error)
        baseline_errors = {name: tvd(prediction["baselines"][name], observed) for name in BASELINES}
        for name in BASELINES:
            deltas[name].append(max(-1.0, baseline_errors[name] - error - 2 * missing_mass))
        decision = prediction["uncertainty"]
        covered += int(error <= decision["upper_tvd"])
        accepted += int(decision["candidate_accept"])
        bad += int(decision["candidate_accept"] and conservative_error > protocol.max_tvd)
        missing_ok = max(row["human_sample"]["missing_rate"], missing_mass,
            prediction["anchor_summary"]["missing_rate"], prediction["anchor_summary"]["missing_weight_fraction"]) <= protocol.max_missing_rate
        missingness.append(missing_ok)
        operation = row.get("operation")
        api_cost = accounting.get("total_cost_usd")
        candidate_cost = None if operation is None or api_cost is None else api_cost + operation["candidate_other_cost_usd"]
        operating_ok = (None if candidate_cost is None else
            candidate_cost <= protocol.budget_per_study_usd + 1e-9
            and operation["human_only_total_cost_usd"] <= protocol.budget_per_study_usd + 1e-9
            and operation["human_only_total_cost_usd"] + 1e-9 >= len(spec.anchor_units) * protocol.human_unit_cost_usd
            and row["candidate_turnaround_seconds"] <= protocol.max_turnaround_seconds
            and operation["human_only_turnaround_seconds"] <= protocol.max_turnaround_seconds)
        operational.append(operating_ok)
        fresh_human = spec.evidence_origin == "human" and spec.outcome_access == "uninspected"
        historical += int(fresh_human and spec.timing == "historical")
        prospective += int(fresh_human and row["prospective_timing_verified"])
        public_rows.append({**item, "status": "SCORED", "candidate_tvd": error,
            "missingness_tvd_upper": conservative_error, "baseline_tvd": baseline_errors,
            "baseline_improvement": {name: baseline_errors[name] - error for name in BASELINES},
            "candidate_accept": decision["candidate_accept"], "research_upper_tvd": decision["upper_tvd"],
            "bound_covers_observed_distribution": error <= decision["upper_tvd"],
            "human_sample": row["human_sample"], "anchor_sample": prediction["anchor_summary"],
            "candidate_total_cost_usd": candidate_cost,
            "human_only_total_cost_usd": operation["human_only_total_cost_usd"] if operation else None,
            "candidate_turnaround_seconds": row["candidate_turnaround_seconds"],
            "missingness_gate": missing_ok, "operating_gate": operating_ok,
            "prediction_sha256": canonical_hash(prediction),
            "outcome_evidence_sha256": row["outcome_evidence_sha256"]})
    comparisons = {name: {"planned_groups": n, "conservative_mean_improvement": sum(values) / n,
        "improvement_lower": max(-1.0, sum(values) / n - difference_radius)} for name, values in deltas.items()}
    coverage = binomial_bounds(covered, n, tail)
    acceptance = binomial_bounds(accepted, n, tail)
    risk = binomial_bounds(bad, accepted, tail)
    failure = binomial_bounds(failed, n, tail)
    upper_error = min(1.0, sum(errors) / n + mean_radius)
    gates = {"absolute_error": upper_error <= protocol.max_tvd,
        **{"improvement_over_" + name: value["improvement_lower"] >= protocol.minimum_baseline_improvement
           for name, value in comparisons.items()},
        "coverage": coverage["lower"] >= protocol.required_coverage,
        "acceptance_rate": acceptance["lower"] >= protocol.minimum_acceptance_rate,
        "accepted_error_risk": bool(accepted) and risk["upper"] <= protocol.max_bad_acceptance_rate,
        "failure_rate": failure["upper"] <= protocol.max_failure_rate,
        "missingness": all(missingness),
        "classical_fit_converged": baseline["model"]["diagnostics"]["converged"],
        "operating_cost_and_time": False if any(v is False for v in operational) else None if any(v is None for v in operational) else True,
        "untouched_historical_evidence": historical >= protocol.minimum_historical_groups,
        "prospective_human_evidence": prospective >= protocol.minimum_prospective_groups,
        "priority_subgroups": None if protocol.priority_subgroups else True}
    status = "PASS" if all(v is True for v in gates.values()) else "FAIL" if any(v is False for v in gates.values()) else "UNEVALUABLE"
    return {"schema_version": "rival.qualification-report.v1", "status": status,
        "plan_sha256": canonical_hash(plan), "baselines_sha256": plan["baselines_sha256"],
        "design_feasibility": design_feasibility(protocol), "prediction_pins": plan["pins"],
        "status_scope": "one frozen aggregate study-family evaluation under declared custody and independence assumptions",
        "customer_qualified": False, "release_status": "research; delivery authorization is separate",
        "study_family": protocol.study_family, "intended_decision": protocol.intended_decision,
        "audience": protocol.audience, "geography": protocol.geography,
        "planned_groups": n, "scored_groups": n - failed, "failed_groups": failed,
        "fresh_historical_groups": historical, "prospective_human_groups": prospective,
        "mean_missingness_adjusted_tvd": sum(errors) / n, "mean_tvd_upper": upper_error,
        "baseline_comparisons": comparisons, "covered_groups": covered,
        "accepted_groups": accepted, "bad_accepted_groups": bad,
        "coverage_lower": coverage["lower"], "acceptance_rate_lower": acceptance["lower"],
        "bad_acceptance_rate_upper": risk["upper"], "failure_rate_upper": failure["upper"],
        "known_api_cost_usd": known_cost, "groups_with_unknown_api_total": accounting_unknown,
        "per_gate_tail_error": tail, "gates": gates, "rows": public_rows,
        "limits": ["Public historical evidence can still be in model pretraining; local freezing does not prove contamination freedom.",
            "Human roster, source truth, weights, semantic independence and non-journal costs are custodian declarations.",
            "Each declared study group counts once. More synthetic draws do not increase independent evidence.",
            "Coverage refers to the observed human distribution; it is not a latent population or subgroup interval.",
            "Missingness bounds concern the frozen eligible sample, not selection bias outside that sample.",
            "Human-only uses a separate full-quota sample under the same budget cap; realized spending need not be equal.",
            "Mean-error and improvement bounds use Hoeffding; binomial gates assume independent exchangeable study groups.",
            "Priority subgroup claims require separate protected evidence; this increment cannot qualify them.",
            "No model, threshold, or feature search may reuse this final evaluation set."]}


def qualification_markdown(report):
    from .study_report import _text
    origins = sorted({row["evidence_origin"] for row in report["rows"]})
    lines = ["# Rival frozen qualification", "", f"Result: **{report['status']}**", "",
        _text(report["status_scope"]), "", "Customer qualification: **not granted**.", "",
        "Study family: " + _text(report["study_family"]), "",
        "Audience: " + _text(report["audience"]), "",
        "Evidence origins: " + ", ".join(_text(origin) for origin in origins) + ".", "",
        f"Planned groups: {report['planned_groups']}; scored: {report['scored_groups']}; failed: {report['failed_groups']}.",
        "", "| Gate | Result |", "|---|---|"]
    for name, value in report["gates"].items():
        lines.append(f"| {_text(name)} | {'UNEVALUABLE' if value is None else 'PASS' if value else 'FAIL'} |")
    lines += ["", "| Baseline | Conservative mean improvement | Simultaneous lower bound |",
              "|---|---:|---:|"]
    for name, value in report["baseline_comparisons"].items():
        lines.append(f"| {_text(name)} | {value['conservative_mean_improvement']:.4f} | {value['improvement_lower']:.4f} |")
    lines += ["", "Positive improvement favors Rival. Failures receive maximum error and minimum paired improvement.",
              "", "| Study | Result | Rival TVD |", "|---|---|---:|"]
    for row in report["rows"]:
        value = row.get("candidate_tvd")
        lines.append(f"| {_text(row['study_id'])} | {_text(row.get('failure', row['status']))} | {'—' if value is None else f'{value:.4f}'} |")
    lines += ["", *["- " + _text(limit) for limit in report["limits"]], ""]
    return "\n".join(lines)
