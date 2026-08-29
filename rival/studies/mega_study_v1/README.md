# Mega-Study v1

This directory is the frozen scientific specification for Rival's parallel
Twin-2K-500 Mega-Study development benchmark.

- `MEGA_STUDY_PROTOCOL.md` — complete operational protocol
- `MEGA_STUDY_PRE_REGISTRATION.md` — hypotheses, estimands, and interpretation
- `MEGA_STUDY_MANIFEST.json` — machine-verifiable sources and configuration
- `cohort.jsonl` — deterministic 300-case allocation, without human outcomes
- `OUTCOME_MAPPING.json` — official dependent-variable mapping
- `LEAKAGE_FIREWALL.md` — prediction/evaluation capability separation
- `OFFICIAL_RESOURCE_AUDIT.md` — paper, repository, data, and replication audit

No data file containing selected participants' target answers is committed.
The existing `twin2k_live_v2` benchmark remains a separate experiment.
