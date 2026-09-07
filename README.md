# Rival

Rival is building a supervised research service for comparing concepts, messages
and proposed scenarios for a declared audience. The first output is simulated
population choice shares. Individual synthetic responses are intermediate model
outputs; they are not verified predictions of real people.

`0.6.0.dev4` is a research development release. See the
[delivery board](docs/CUSTOMER_READINESS.md),
[supported offering](docs/SUPPORTED_OFFERING.md), and
[phase-one acceptance evidence](docs/verification/phase_one.json).
Customer-domain accuracy, confidence coverage and launch readiness remain
unqualified. No Aaru/Simile parity claim is made.

## What goes in and comes out

- Input: a question, alternatives, audience records, optional population controls,
  evidence provenance and a relevant information cutoff.
- Processing: prepare the audience, obtain a model probability distribution for
  each seed person, aggregate simulated draws, and optionally correct against
  observed human anchors.
- Output: choice shares, diagnostics, research intervals, evidence and execution
  identities, limitations, and observed comparison metrics when available.
- Interface: a local browser demo and APIs exist. The demo uses generated people,
  anchors and outcomes. The complete customer study interface is later work.

## Install and run offline

Python 3.11 or newer is required. No API key is needed for these commands.

```sh
python -m pip install .
python -m rival status
python -m rival demo --sample-size 100 --human-anchor-size 20
python -m rival serve --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080` for the research demonstration. It does not call a
model API. The interface labels generated data and withholds decision confidence.

## Managed model execution

Current general probability, behavioral-model and text/SSR adapters require an
`ExecutionSession`: a persistent journal, explicit budget, expiry, conservative
per-attempt reservation and total physical-attempt limit. Reported charges from
failed attempts count. Missing billing or an interrupted request stops further
spending until reconciled. Repeated draws of one seed person reuse its response.

[Managed execution guide](docs/MANAGED_EXECUTION.md) describes the Python API,
`python -m rival simulate-managed`, recovery and the limits of local cost estimates.
Credentials belong in environment variables, never command arguments.

## Scientific evidence and frozen experiments

The original Twin-2K Wave-4 experiment, Mega A–D manifest, implementation witnesses
and SYN-DIGITS E/F design remain preserved. Historical results and their negative
findings remain available; they do not qualify this release or new audiences.
The [historical README](docs/HISTORICAL_README_DEV2.md) retains the earlier numbers
and their original context.

Corrected Mega execution and reanalysis use the explicit v2 namespace and new
artifact paths. Installed wheels include an archive of the original verification
witnesses; the archive is not represented as the current runtime.

```sh
python -m rival mega-v2 verify-resources
python -m rival mega-v2 --help
python -m rival mega-v2 reanalyze --help
```

Do not append v2 rows to the existing v1 experiment. Keep an existing historical
execution environment pinned to its original commit. The current CLI exposes
managed execution; the frozen v1 CLI remains an archival witness.

## Verification and provenance

```sh
python -m unittest discover -s tests -v
python -m rival qualify-integrity
python -m rival qualify-research-components
python -m rival qualify-all --output-dir reports/current --compact
python -m pip wheel . --no-deps --wheel-dir dist
python scripts/verify_installed_wheel.py --wheel dist/rival_sim-0.6.0.dev4-py3-none-any.whl --output reports/installed_wheel.json
```

`qualify-all` fails when a reproduction or engineering check fails. A PASS means
those checks passed; it never enables a customer or confidence claim. Training,
calibration and evaluation studies are assigned immutable roles before prediction
lock. Only verified training-role studies enter the research confidence fit;
even a fitted model continues to abstain pending independent qualification.

See [third-party notices](THIRD_PARTY_NOTICES.md) and `upstreams.lock.json` for
incorporated code, papers, data and pinned revisions. Complete supplemental MIT
notices are bundled separately so original audited files stay byte-identical.
