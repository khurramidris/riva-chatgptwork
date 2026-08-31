# Mega-Study SYN-DIGITS E/F Supplement

Status: **design specified; execution blocked**

This directory specifies a separate calibration supplement for Rival's frozen
Mega-Study A/B/C/D development benchmark. It does not alter, replace, or
reinterpret any raw A-D prediction. It also does not resume or modify the
paused 1,500-case Wave-4 experiment.

The proposed additional variants are:

- E — Full Persona + SYN-DIGITS
- F — Rival Retrieval + SYN-DIGITS

E/F may run only after a same-model historical reference ledger,
reference-outcome capability separation, prediction-only SYN-DIGITS adapter,
cost manifest, and their tests are implemented and frozen. Until then, raw
A/B/C/D is the only authorized Mega-Study run.

See SYN_DIGITS_EF_DESIGN.md for the scientific design and
CALIBRATION_DESIGN.json for machine-checkable invariants.
