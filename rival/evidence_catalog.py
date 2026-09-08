"""Local, content-addressed imports with replayable, explicit conversions.

Hashes establish which bytes and transformation were used. Source availability,
rights and sampling frames remain declarations; imports do not qualify a model.
"""

import csv
from datetime import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Literal

from pydantic import Field, StrictInt, field_validator, model_validator

from .integrity import _forbidden_paths
from .mathx import canonical_hash
from .mega_study_v2.runner import exclusive_run_lock
from .schemas import EvidenceSource, PopulationRecord, StrictModel


IMPORTER_VERSION = "rival.evidence-importer.v1"
MAX_BYTES = 50 * 1024 * 1024
MAX_RECORDS = 100_000


def strict_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError(f"non-finite JSON value: {value}")

    return json.loads(text, object_pairs_hook=unique, parse_constant=nonfinite)


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_bounded(path):
    with Path(path).open("rb") as handle:
        data = handle.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("evidence file exceeds the 50 MiB import limit")
    return data


class ImportSource(EvidenceSource):
    source_id: str = Field(min_length=1)
    collected_at: datetime

    @model_validator(mode="after")
    def declared_source(self):
        if any(not value.strip() for value in (self.source_id, self.name, self.rights_reference)):
            raise ValueError("source ID, name and rights reference must be explicit")
        if self.source_type == "outcome":
            raise ValueError("outcomes cannot be imported as prediction evidence")
        if "simulation" not in self.permitted_uses or "simulation" in self.prohibited_uses:
            raise ValueError("the source must explicitly permit simulation")
        if not self.geography or any(not place.strip() for place in self.geography):
            raise ValueError("source geography must be declared")
        if self.collected_at.tzinfo is None:
            raise ValueError("source collection date must include a timezone")
        if "rival_import" in self.metadata:
            raise ValueError("rival_import is reserved for verified import metadata")
        return self


class ColumnMapping(StrictModel):
    column: str = Field(min_length=1)
    kind: Literal["string", "integer", "number", "boolean"] = "string"
    nullable: bool = False
    categories: dict[str, str] | None = None

    @model_validator(mode="after")
    def category_mapping(self):
        if self.categories is not None and (self.kind != "string" or not self.categories):
            raise ValueError("category recoding requires a nonempty mapping and string output")
        return self


class CsvMapping(StrictModel):
    id_column: str = Field(min_length=1)
    weight_column: str | None = None
    attributes: dict[str, ColumnMapping]
    preferences: dict[str, str] = Field(default_factory=dict)


class EvidenceImportSpec(StrictModel):
    schema_version: Literal["rival.evidence-import.v1"] = "rival.evidence-import.v1"
    source: ImportSource
    origin: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    released_at: datetime
    retrieved_at: datetime
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    format: Literal["csv", "population-jsonl"]
    csv_mapping: CsvMapping | None = None
    geography_attribute: str = Field(min_length=1)
    conditions: list[str] = Field(min_length=1)
    expected_records: StrictInt = Field(ge=1, le=MAX_RECORDS)

    @model_validator(mode="after")
    def consistent_declarations(self):
        json.dumps(self.model_dump(mode="python"), default=str, allow_nan=False)
        if any(not item.strip() for item in [self.origin, self.revision, self.geography_attribute, *self.conditions]):
            raise ValueError("import origin, revision, geography field and conditions must be nonblank")
        if len(set(self.conditions)) != len(self.conditions):
            raise ValueError("condition IDs must be unique")
        if self.released_at.tzinfo is None or self.retrieved_at.tzinfo is None:
            raise ValueError("release and retrieval dates must include a timezone")
        if not self.source.collected_at <= self.released_at <= self.retrieved_at:
            raise ValueError("dates must follow collection <= release <= retrieval")
        if self.source.sha256 and self.source.sha256 != self.artifact_sha256:
            raise ValueError("source and artifact hashes disagree")
        if (self.format == "csv") != (self.csv_mapping is not None):
            raise ValueError("CSV imports require a mapping; JSONL imports use native population records")
        if self.csv_mapping and self.geography_attribute not in self.csv_mapping.attributes:
            raise ValueError("the geography attribute must be mapped explicitly")
        for value in (self.source.period_start, self.source.period_end):
            if value is not None:
                instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if instant.tzinfo is None or instant > self.source.collected_at:
                    raise ValueError("source periods require timezone-aware dates no later than collection")
        if self.source.period_start and self.source.period_end:
            if datetime.fromisoformat(self.source.period_start.replace("Z", "+00:00")) > datetime.fromisoformat(self.source.period_end.replace("Z", "+00:00")):
                raise ValueError("source period starts after it ends")
        return self


class EvidenceImportManifest(StrictModel):
    schema_version: Literal["rival.evidence-bundle.v1"] = "rival.evidence-bundle.v1"
    importer_version: Literal["rival.evidence-importer.v1"] = IMPORTER_VERSION
    spec: EvidenceImportSpec
    records_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def verify_digest(self):
        payload = self.model_dump(mode="json", exclude={"bundle_sha256"})
        if canonical_hash(payload) != self.bundle_sha256:
            raise ValueError("evidence bundle manifest hash does not verify")
        return self

    def source(self):
        source = self.spec.source.model_dump(mode="json")
        source["sha256"] = self.spec.artifact_sha256
        source["metadata"]["rival_import"] = {
            "bundle_sha256": self.bundle_sha256, "records_sha256": self.records_sha256,
            "importer_version": self.importer_version, "origin": self.spec.origin,
            "revision": self.spec.revision, "released_at": self.spec.released_at.isoformat(),
            "retrieved_at": self.spec.retrieved_at.isoformat(),
            "geography_attribute": self.spec.geography_attribute, "conditions": self.spec.conditions,
        }
        return EvidenceSource.model_validate(source)


def _number(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("numeric evidence must be finite")
    return result


def _mapped(value, mapping):
    if value == "":
        if mapping.nullable:
            return None
        raise ValueError("required CSV field is blank")
    if mapping.categories is not None:
        if value not in mapping.categories:
            raise ValueError("CSV category is absent from the explicit recoding map")
        return mapping.categories[value]
    if mapping.kind == "number":
        return _number(value)
    if mapping.kind == "integer":
        if not re.fullmatch(r"[+-]?\d+", value):
            raise ValueError("integer CSV field is not an integer")
        return int(value)
    if mapping.kind == "boolean":
        if value not in {"true", "false", "1", "0"}:
            raise ValueError("boolean CSV fields accept true, false, 1 or 0")
        return value in {"true", "1"}
    return value


def convert_records(data, spec):
    if len(data) > MAX_BYTES or sha256(data) != spec.artifact_sha256:
        raise ValueError("source bytes differ from the pinned artifact hash or exceed the size limit")
    text = data.decode("utf-8-sig")
    records, ids = [], set()
    if spec.format == "csv":
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        header = next(reader, [])
        if not header or len(header) != len(set(header)) or any(not name for name in header):
            raise ValueError("CSV headers must be nonempty and unique")
        mapping = spec.csv_mapping
        required = {mapping.id_column, *[field.column for field in mapping.attributes.values()], *mapping.preferences.values()}
        if mapping.weight_column:
            required.add(mapping.weight_column)
        if not required <= set(header):
            raise ValueError("mapped CSV columns are missing")

        def rows():
            for cells in reader:
                if len(cells) != len(header):
                    raise ValueError("CSV row width differs from its header")
                row = dict(zip(header, cells, strict=True))
                attributes = {key: _mapped(row[field.column], field) for key, field in mapping.attributes.items()}
                yield {"person_id": row[mapping.id_column],
                       "weight": _number(row[mapping.weight_column]) if mapping.weight_column else 1.0,
                       "attributes": {key: value for key, value in attributes.items() if value is not None},
                       "preferences": {key: _number(row[column]) for key, column in mapping.preferences.items()}}
        raw_records = rows()
    else:
        raw_records = (strict_json(line) for line in text.splitlines() if line.strip())
    for raw in raw_records:
        if len(records) >= MAX_RECORDS:
            raise ValueError("evidence exceeds the record limit")
        json.dumps(raw, allow_nan=False)
        record = PopulationRecord.model_validate(raw)
        if not record.person_id.strip() or record.person_id in ids:
            raise ValueError("source person IDs must be nonblank and unique")
        if record.evidence_ids:
            raise ValueError("imported records cannot supply their own evidence references")
        if _forbidden_paths(record.attributes, "attributes") or _forbidden_paths(record.preferences, "preferences") or _forbidden_paths(record.history, "history"):
            raise ValueError("protected outcome fields cannot be imported as prediction evidence")
        if record.attributes.get(spec.geography_attribute) not in spec.source.geography:
            raise ValueError("record geography is absent or outside the source declaration")
        for entry in record.history:
            timestamps = [entry[key] for key in ("occurred_at", "recorded_at", "collected_at", "timestamp", "date") if key in entry]
            if not timestamps:
                raise ValueError("imported history requires a timestamp")
            for raw_instant in timestamps:
                instant = datetime.fromisoformat(str(raw_instant).replace("Z", "+00:00"))
                if instant.tzinfo is None or instant > spec.source.collected_at:
                    raise ValueError("history dates must include a timezone and precede source collection")
        ids.add(record.person_id)
        # Stable source-qualified identifiers prevent accidental collisions when
        # combining releases. Independence across sources is not inferred.
        record.person_id = spec.source.source_id + ":" + record.person_id
        record.evidence_ids = [spec.source.source_id]
        records.append(record)
    if len(records) != spec.expected_records:
        raise ValueError("source row count differs from the declared expected_records")
    total = sum(record.weight for record in records)
    squared = sum(record.weight * record.weight for record in records)
    if not math.isfinite(total * total) or not math.isfinite(squared) or squared == 0:
        raise ValueError("source weights exceed the supported numerical range; rescale their units")
    return sorted(records, key=lambda record: record.person_id)


def records_digest(records):
    return canonical_hash([record.model_dump(mode="json") for record in sorted(records, key=lambda item: item.person_id)])


def _bundle(data, spec):
    records = convert_records(data, spec)
    payload = {"schema_version": "rival.evidence-bundle.v1", "importer_version": IMPORTER_VERSION,
               "spec": spec.model_dump(mode="json"), "records_sha256": records_digest(records)}
    manifest = EvidenceImportManifest.model_validate({**payload, "bundle_sha256": canonical_hash(payload)})
    normalized = b"".join((json.dumps(row.model_dump(mode="json"), sort_keys=True, allow_nan=False) + "\n").encode("utf-8") for row in records)
    return manifest, records, {"source.bin": data, "manifest.json": json_bytes(manifest.model_dump(mode="json")), "records.jsonl": normalized}


class EvidenceCatalog:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def import_file(self, path, spec):
        json.dumps(spec.model_dump(mode="python"), default=str, allow_nan=False)
        spec = EvidenceImportSpec.model_validate(spec.model_dump(mode="json"))
        manifest, records, files = _bundle(read_bounded(path), spec)
        if any(len(content) > MAX_BYTES for content in files.values()):
            raise ValueError("normalized evidence exceeds the 50 MiB file limit")
        with exclusive_run_lock(self.root.with_name(self.root.name + ".lock")):
            destination = self.root / manifest.bundle_sha256
            if destination.exists():
                existing, _ = self.load(manifest.bundle_sha256)
                if existing != manifest:
                    raise ValueError("catalog import conflicts with an existing bundle")
            else:
                self.root.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix="import-", dir=self.root) as directory:
                    staged = Path(directory) / "bundle"
                    staged.mkdir()
                    for name, content in files.items():
                        with (staged / name).open("xb") as handle:
                            handle.write(content)
                            handle.flush()
                            os.fsync(handle.fileno())
                    staged.rename(destination)
        return manifest

    def load(self, bundle_id):
        if not re.fullmatch(r"[0-9a-f]{64}", bundle_id):
            raise ValueError("invalid evidence bundle ID")
        root = self.root / bundle_id
        if root.is_symlink() or not root.is_dir() or {item.name for item in root.iterdir()} != {"manifest.json", "source.bin", "records.jsonl"}:
            raise ValueError("evidence bundle is missing or has unexpected files")
        if any(path.is_symlink() for path in root.iterdir()):
            raise ValueError("evidence bundles cannot contain symbolic links")
        manifest = EvidenceImportManifest.model_validate(strict_json(read_bounded(root / "manifest.json")))
        if manifest.bundle_sha256 != bundle_id:
            raise ValueError("catalog identity differs from the bundle manifest")
        actual, records, files = _bundle(read_bounded(root / "source.bin"), manifest.spec)
        if actual != manifest or read_bounded(root / "records.jsonl") != files["records.jsonl"]:
            raise ValueError("normalized evidence does not reproduce from its pinned source and conversion")
        return manifest, records
