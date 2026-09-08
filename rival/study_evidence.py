"""Bind catalog evidence into portable study requests before preparation."""

from .evidence_catalog import EvidenceCatalog
from .mathx import canonical_hash
from .study_contract import StudyRequestV2, parse_study_request
from .study_support import StudySupportPolicy


def bind_evidence(payload, catalog_root, bundle_ids, policy):
    if not isinstance(payload, dict) or not isinstance(payload.get("audience"), dict):
        raise ValueError("study brief must be an object with an audience declaration")
    catalog = EvidenceCatalog(catalog_root)
    manifests, records, sources = [], [], []
    for bundle_id in bundle_ids:
        manifest, population = catalog.load(bundle_id)
        manifests.append(manifest.model_dump(mode="json"))
        records.extend(record.model_dump(mode="json") for record in population)
        sources.append(manifest.source().model_dump(mode="json"))
    version = payload.get("schema_version", "rival.study-request.v2")
    if version not in {"rival.study-request.v1", "rival.study-request.v2", "rival.study-request.v3", "rival.study-request.v4"}:
        raise ValueError("unsupported study request version")
    if version == "rival.study-request.v1":
        version = "rival.study-request.v2"
    payload = {**payload, "schema_version": version, "imports": manifests,
               "support": StudySupportPolicy.model_validate(policy).model_dump(mode="json"),
               "audience": {**payload["audience"], "records": records, "sources": sources}}
    return parse_study_request(payload)


def verify_imports(request, catalog_root):
    if not isinstance(request, StudyRequestV2):
        return None
    if catalog_root is None:
        raise ValueError("new v2 studies require the evidence catalog to verify original bytes and conversion")
    catalog = EvidenceCatalog(catalog_root)
    verified = []
    for expected in request.imports:
        actual, _ = catalog.load(expected.bundle_sha256)
        if canonical_hash(expected) != canonical_hash(actual):
            raise ValueError("study import differs from the verified catalog bundle")
        verified.append(expected.bundle_sha256)
    return {"bundle_sha256": verified, "raw_bytes_and_conversion_verified": True,
            "scope": "local source-byte and normalization verification; origin, rights and dates remain declarations"}
