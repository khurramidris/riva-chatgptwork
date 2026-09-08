# Evidence imports and declared population support (L08)

Version `0.6.0.dev6` connects a local evidence catalog to the saved study workflow.
It accepts versioned CSV or native population JSONL snapshots. No source is fetched
automatically and importing or preparing a study makes no model requests.

## What is checked

The importer requires a source ID, origin, revision, declared rights, collection,
release and retrieval dates, exact file hash and expected record count. It retains
the source bytes, import specification and normalized records in a content-addressed
bundle. Loading the bundle replays the conversion from those bytes and verifies
the result. Changed files, mappings, row counts, duplicate IDs/headers/JSON keys,
non-finite values, malformed rows and declared outcome fields fail explicitly.
New source versions create separate bundles; old versions are not overwritten.

CSV mappings select columns explicitly and support strings, numbers, integers,
booleans and exact category recoding. Required blanks or unknown category codes
fail. Unmapped columns never enter population records. Native JSONL contains one
`PopulationRecord` per line, without evidence IDs: the importer assigns its own
source references and source-qualified person IDs. Native history requires
timezone-aware timestamps no later than source collection. Both formats require
an explicit geography attribute. Files are limited to 50 MiB and 100,000 records;
this increment does not implement streaming ingestion for larger sources.

Preparation requires access to the original catalog and checks its files against
the study's embedded manifests and records. It saves the verified inputs and signed
audit in the study workspace. Subsequent execution and recovery need that workspace,
not the original catalog. The catalog is local integrity evidence, not an external
signature or independent certification of source provenance.

## Complete generated example

Run from the checkout or an installed package:

```sh
python -m rival evidence example --output-dir .rival-data/evidence-example
python -m rival evidence import --input .rival-data/evidence-example/people.csv --spec .rival-data/evidence-example/import.json --catalog .rival-data/evidence-catalog
```

Copy `bundle_sha256` from the import result into `BUNDLE_ID` below. The commands
also work in Windows PowerShell; replace `python` with `.venv\Scripts\python.exe`
where needed. The example contains generated people, not consumer evidence.

```sh
python -m rival evidence inspect --catalog .rival-data/evidence-catalog --bundle BUNDLE_ID
python -m rival study bind-evidence --input .rival-data/evidence-example/brief.json --support .rival-data/evidence-example/support.json --catalog .rival-data/evidence-catalog --bundle BUNDLE_ID --output .rival-data/imported-study.json
python -m rival study check --input .rival-data/imported-study.json --catalog .rival-data/evidence-catalog
python -m rival study prepare --input .rival-data/imported-study.json --catalog .rival-data/evidence-catalog --workspace .rival-data/imported-study
python -m rival study run --workspace .rival-data/imported-study
python -m rival study export --workspace .rival-data/imported-study --output reports/imported-study-report
```

`evidence list --catalog PATH` lists verified versions. `evidence fingerprint
--input FILE` computes a local hash; compare it with the upstream publication or
your retained acquisition record before pinning it. A local hash alone proves no
publisher identity. `study schema --version v2 --output FILE` writes the contract.
Repeat `--bundle` to combine different sources. Only one release per source ID can
enter a study. Select subsets with audience filters; do not edit imported records.

## Population and condition support

The support policy specifies the geography attribute, condition IDs, required
attributes, minimum seed/effective counts and any explicitly requested joint cells.
The policy's thresholds are operator choices, not statistically validated defaults.

- Requested geography becomes an actual population filter. Every named region
  needs positive-weight records after filtering and population weighting.
- Every selected source must declare the requested condition IDs. This is an
  exact tag check; it does not interpret the meaning of the question or establish
  that a source supports a causal intervention.
- Every selected record must contain the required attributes. Every requested
  category and explicitly defined joint cell needs the specified support.
- Population controls must be supported and converge. Effective counts use the
  final seed weights; simulation draws do not inflate these counts.
- Collection and release precede the study cutoff. A later retrieval is permitted
  for an explicitly disclosed historical reconstruction. Optional source-age
  limits apply to the declared collection date, not to independently verified
  dates for every individual observation.

Unsupported studies return a nonzero exit and an audit of the missing support.
`check` performs preparation's no-call checks without creating a study workspace.
Passing checks does not establish national representativeness, independence,
coverage of unrequested joint cells or model accuracy. Full semantic leakage
prevention, source permission review and external custody remain human obligations.

## Public-data rehearsal

[The pinned import specification](evidence/twin2k-demographics/import.json),
[support policy](evidence/twin2k-demographics/support.json) and
[brief](evidence/twin2k-demographics/brief.json) use the author-published Twin-2K
wave-1–3 CSV at revision `f883165a3026fde855dfd448e0cd16443ab257b6`.
The exact file hash is
`fc227b1682e2654fd4b2a8152294b68118c0e3be4d83f4a9af5f3e96df56d3f0`.
Download the `origin` URL in the specification to your own evidence directory,
then use the same import/bind/check/prepare commands with these three files.
For a new acquisition, record its own retrieval timestamp in a copy of the spec.

The rehearsal imported all 2,058 rows and selected only participant ID, region,
sex at birth and age band. The recoding follows the preserved official question
catalog. The other survey columns remain outside provider-visible records. No
behavioral simulation, outcome evaluation or paid model request was performed.
See [the actual import and preparation receipt](examples/evidence-support/public-import-check.json).
The example is development evidence; it is not a fresh held-out qualification set.

Collection and publication timestamps in this recipe are explicitly documented
upper bounds, not invented individual timestamps: the paper describes wave-1–3
completion before the February 25, 2025 wave-4 invitation, and the pinned revision
is dated March 28, 2026. Retrieval records the actual acquisition in this session.
The [author dataset card](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500)
declares CC BY 4.0 and provides attribution; the [paper](https://arxiv.org/abs/2505.17479)
describes collection. The specification retains those references and the explicit
selection. Neither these new files nor the rehearsal alter Rival's frozen
Wave-4/Mega datasets, manifests, ledgers or designs.

## Compatibility and remaining work

New studies with public, licensed, participant or customer sources require v2
imports and support checks. The generated v1 demo remains usable. Existing v1
completed workspaces remain readable and keep their original release identity
when exported. Incomplete old workspaces retain their pinned runtime version;
use that version to resume rather than rewriting their signed context.

Reports carry aggregate support counts and source/version/hash references. Raw
person records, source bytes, workspace signing keys and model credentials are
not included. Preserve the private catalog/workspace for your own audit and
recovery. Broader source adapters, automated retrieval and a browser evidence
manager remain extensions; L09 real-model execution and L10 runtime calibration
are the next phase-two milestones.
