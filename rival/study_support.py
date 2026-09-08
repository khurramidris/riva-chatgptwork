"""Declared support checks; seed counts and weights are not confidence evidence."""

import math

from pydantic import Field, StrictInt, model_validator

from .mathx import canonical_hash, effective_sample_size
from .population import PopulationCompiler
from .schemas import StrictModel


class SupportCell(StrictModel):
    label: str = Field(min_length=1)
    filters: dict = Field(min_length=1)
    min_seed_records: StrictInt = Field(default=1, ge=1)
    min_effective_seed_records: float = Field(default=1., ge=1, allow_inf_nan=False)


class StudySupportPolicy(StrictModel):
    geography_attribute: str = Field(min_length=1)
    conditions: list[str] = Field(min_length=1)
    required_attributes: list[str] = Field(default_factory=list)
    min_seed_records: StrictInt = Field(default=1, ge=1)
    min_effective_seed_records: float = Field(default=1., ge=1, allow_inf_nan=False)
    max_source_age_days: StrictInt | None = Field(default=None, ge=0)
    cells: list[SupportCell] = Field(default_factory=list)

    @model_validator(mode="after")
    def explicit_scope(self):
        names = [self.geography_attribute, *self.conditions, *self.required_attributes, *[cell.label for cell in self.cells]]
        if any(not name.strip() for name in names):
            raise ValueError("support declarations must be nonblank")
        for values in (self.conditions, self.required_attributes, [cell.label for cell in self.cells]):
            if len(values) != len(set(values)):
                raise ValueError("support conditions, attributes and cell labels must be unique")
        for cell in self.cells:
            validate_filters(cell.filters)
        return self


class UnsupportedStudy(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__("study support checks failed: " + "; ".join(item["message"] for item in report["issues"]))


def validate_filters(filters, records=()):
    for feature, expected in filters.items():
        if not feature.strip():
            raise ValueError("filter names cannot be blank")
        if isinstance(expected, dict):
            if not expected or not set(expected) <= {"min", "max"}:
                raise ValueError("range filters accept min and max only")
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in expected.values()):
                raise ValueError("range bounds must be finite numbers")
            if expected.get("min", -math.inf) > expected.get("max", math.inf):
                raise ValueError("range minimum exceeds maximum")
            for record in records:
                value = record.attributes.get(feature)
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
                    raise ValueError("numeric range filter encountered a nonnumeric attribute")
        elif isinstance(expected, list):
            if not expected or any(value is None or isinstance(value, (dict, list)) for value in expected):
                raise ValueError("category filters require nonempty scalar values")
        elif expected is None:
            raise ValueError("null is not a supported audience filter")


def assess_support(request, *, prepared_records=None, population_diagnostics=None):
    policy, audience = request.support, request.audience
    issues, warnings = [], []

    def issue(code, message):
        issues.append({"code": code, "message": message})

    filters = request.scenario().population_filter
    validate_filters(filters, audience.records)
    filtered = PopulationCompiler.filter_records(audience.records, filters)
    active_sources = {identifier for record in filtered for identifier in record.evidence_ids}
    source_rows = []
    for bundle in request.imports:
        spec = bundle.spec
        used = spec.source.source_id in active_sources
        source_rows.append({"source_id": spec.source.source_id, "bundle_sha256": bundle.bundle_sha256,
            "artifact_sha256": spec.artifact_sha256, "revision": spec.revision,
            "origin": spec.origin, "collected_at": spec.source.collected_at.isoformat(),
            "records_sha256": bundle.records_sha256, "conversion_spec_sha256": canonical_hash(spec),
            "importer_version": bundle.importer_version,
            "released_at": spec.released_at.isoformat(), "retrieved_at": spec.retrieved_at.isoformat(),
            "selected_for_audience": used, "retrieved_after_cutoff": spec.retrieved_at > request.brief.information_cutoff})
        # All imported inputs must have existed by the declared cutoff, including
        # rows later excluded by population filters.
        if spec.released_at > request.brief.information_cutoff:
            issue("source_after_cutoff", f"source {spec.source.source_id} was released after the information cutoff")
        if spec.geography_attribute != policy.geography_attribute:
            issue("geography_mapping_mismatch", f"source {spec.source.source_id} uses a different geography attribute")
        if used and not set(policy.conditions) <= set(spec.conditions):
            issue("condition_not_supported", f"source {spec.source.source_id} does not declare every requested condition")
        if used and policy.max_source_age_days is not None:
            age = (request.brief.information_cutoff - spec.source.collected_at).total_seconds() / 86400
            if age > policy.max_source_age_days:
                issue("source_too_old", f"source {spec.source.source_id} exceeds the declared age limit")
        if used and spec.retrieved_at > request.brief.information_cutoff:
            warnings.append("Some source bytes were retrieved after the information cutoff; historical availability is declared, not independently witnessed.")

    compiler = PopulationCompiler()
    compiled = filtered if prepared_records is None else prepared_records
    diagnostics = population_diagnostics
    if not filtered:
        issue("empty_audience", "no seed records match the requested audience")
    elif audience.targets and prepared_records is None:
        try:
            compiled, diagnostics = compiler.calibrate(filtered, audience.targets)
        except ValueError as exc:
            issue("population_controls_unsupported", str(exc))
    if diagnostics is not None and not diagnostics.converged:
        issue("population_controls_not_converged", "population controls did not converge")
    positive = [record for record in compiled if record.weight > 0]
    ess = effective_sample_size(record.weight for record in positive) if positive else 0.
    if len(positive) < policy.min_seed_records:
        issue("insufficient_seed_records", "eligible positive-weight seed count is below the declared minimum")
    if ess + 1e-9 < policy.min_effective_seed_records:
        issue("insufficient_effective_records", "effective seed count is below the declared minimum")
    geography = {place: sum(record.attributes.get(policy.geography_attribute) == place for record in positive) for place in audience.geography}
    for place, count in geography.items():
        if not count:
            issue("geography_not_supported", f"requested geography {place} has no positive-weight seed records")
    missing = {attribute: sum(record.attributes.get(attribute) is None or record.attributes.get(attribute) == "" for record in positive) for attribute in policy.required_attributes}
    if any(missing.values()):
        issue("missing_required_attributes", "some selected records lack required attributes")
    for feature, expected in audience.filters.items():
        if isinstance(expected, list):
            for value in expected:
                if not any(record.attributes.get(feature) == value for record in positive):
                    issue("filter_category_not_supported", f"requested category {feature}={value} has no positive-weight seed records")
    cells = []
    for cell in policy.cells:
        validate_filters(cell.filters, positive)
        selected = compiler.filter_records(positive, cell.filters)
        cell_ess = effective_sample_size(record.weight for record in selected) if selected else 0.
        passed = len(selected) >= cell.min_seed_records and cell_ess + 1e-9 >= cell.min_effective_seed_records
        cells.append({"label": cell.label, "seed_records": len(selected), "effective_seed_records": cell_ess,
                      "min_seed_records": cell.min_seed_records, "min_effective_seed_records": cell.min_effective_seed_records, "passed": passed})
        if not passed:
            issue("cell_not_supported", f"declared cell {cell.label} lacks the required support")
    report = {"schema_version": "rival.study-support.v1", "passed": not issues,
        "policy_sha256": canonical_hash(policy), "input_seed_records": len(audience.records),
        "filtered_seed_records": len(filtered), "positive_weight_seed_records": len(positive),
        "excluded_seed_records": len(audience.records) - len(filtered), "effective_seed_records": ess,
        "geography_counts": geography, "missing_attributes": missing, "cells": cells,
        "conditions": policy.conditions, "sources": source_rows,
        "population_diagnostics": diagnostics.model_dump(mode="json") if diagnostics else None,
        "issues": issues, "warnings": sorted(set(warnings)),
        "scope": "declared source and seed support; not representativeness, semantic condition validation or predictive qualification"}
    report["audit_sha256"] = canonical_hash(report)
    return report


def require_support(request, **kwargs):
    report = assess_support(request, **kwargs)
    if not report["passed"]:
        raise UnsupportedStudy(report)
    return report
