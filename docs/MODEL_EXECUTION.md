# Pinned model studies and elicitation (L09)

The v3 study contract connects a declared model to the saved evidence workflow.
An operator chooses direct probability elicitation or semantic similarity rating
(SSR). Both use durable attempt accounting, fail on unsupported output, preserve
the full planned denominator, and export aggregate execution evidence. This is
research infrastructure; model execution does not establish human accuracy.

## Input and operator flow

Start with the evidence brief, support policy and imported bundles described in
[EVIDENCE_IMPORTS.md](EVIDENCE_IMPORTS.md). Set its `schema_version` to
`rival.study-request.v3` and add explicit managed execution settings. Generate
the full JSON schema with `python -m rival study schema --version v3 --output schema.json`.
An example settings object is below. Replace the model, revision, reference,
endpoint and expiry with the reviewed deployment's actual values before use.

```json
{
  "mode": "managed",
  "model": "your-model-deployment",
  "base_url": "https://your-provider.example/v1/chat/completions",
  "model_pin": {
    "kind": "hosted_endpoint",
    "revision": "your-reviewed-deployment-revision",
    "reference": "your-deployment-record",
    "expected_response_model": "actual-returned-model-id"
  },
  "generation_seed": 42,
  "temperature": 0,
  "max_retries": 2,
  "timeout_seconds": 120,
  "history_limit": 16,
  "max_output_tokens": 300,
  "budget_usd": 1.0,
  "reservation_usd": 0.05,
  "max_attempts": 20,
  "not_after": "2026-09-08T23:59:00Z",
  "elicitation": {"method": "direct"}
}
```

The budget values are illustrative local limits, not a current provider price
quote. Missing billing remains unresolved until reconciled with actual billing
evidence. Reservations are estimates; provider spending controls are the remote
ceiling. Expiry, budgets and attempt limits apply across all resumes.

```sh
python -m rival study bind-evidence --input brief.json --support support.json --catalog CATALOG --bundle BUNDLE_SHA --output study.json
python -m rival study check --input study.json --catalog CATALOG
python -m rival study prepare --input study.json --catalog CATALOG --workspace WORKSPACE
python -m rival study run --workspace WORKSPACE
python -m rival study execution-audit --workspace WORKSPACE
python -m rival study export --workspace WORKSPACE --output REPORT
```

Check/prepare make no model requests or embedding downloads. Set `RIVAL_API_KEY`
or `OPENROUTER_API_KEY` only for execution. Credentials are not part of the saved
request or aggregate report. A new v3 study always binds evidence imports and
an explicit population support policy, including generated rehearsals.

## What is pinned and checked

The preparation binds model/endpoint identity, revision declaration, expected
response model, prompt adapter, generation seed, token/history limits, elicitation
settings and numerical-library runtime identity. All completions must include
the expected model, a request ID, exactly one normally finished answer, and any
declared provider or runtime fingerprint. A changed model/provider/fingerprint,
refusal or truncated answer is terminal; a known charge is still counted.
Invalid probability responses can retry within the saved limits. A valid direct
response must contain the exact choice keys and finite probabilities summing to
one. The adapter does not normalize an invalid answer to make it pass.

OpenRouter requests require `provider_route` and `expected_response_provider` in
the model pin. The route is sent in `provider.only`, with `allow_fallbacks=false`
and `require_parameters=true`. The returned provider name must match the saved
expectation. The route slug and returned display name may differ; record both.

A `checkpoint` pin requires a full 40/64-character commit/artifact hash plus an
expected runtime fingerprint. A hosted revision is an operator declaration.
Checking response metadata does not independently prove which weights a remote
operator loaded. If that endpoint supplies no system fingerprint, the report
does not imply one was checked. Pinning a seed also does not guarantee that a
hosted backend is deterministic. See the official [routing reference](https://openrouter.ai/docs/guides/routing/provider-selection)
and [response reference](https://openrouter.ai/docs/api_reference/overview).

## SSR configuration and numerical corrections

To elicit a natural-language answer, use:

```json
{
  "method": "ssr",
  "embedding": {
    "kind": "sentence_transformer",
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "device": "cpu"
  },
  "temperature": 1.0,
  "epsilon": 0.00000001
}
```

Install the `semantic` extra. The generator sees the actual alternatives and
uses the declared history/token limits. JSON, code blocks and purely numerical
ratings are rejected by the text adapter. This is a structural check; relevance,
naturalness and fidelity to a person still require empirical assessment. Choice
labels/descriptions become the semantic anchors. This general categorical use
is an adaptation; it is not a replication of a published five-point purchase
intent experiment or evidence that semantic mass is calibrated probability.

The embedding model needs a Hub ID and full immutable commit; local arbitrary
paths/mutable revisions cannot be passed through v3. The encoder is loaded
lazily with its pinned revision. The anchor matrix is validated before model
generation. Anchor embeddings and repeated response embeddings are reused.
`hashing` embeddings are permitted only for synthetic development rehearsals.

The original SSR computation remains byte-for-byte in the vendor archive. Rival's
`rival.ssr.v2` adapter uses the upstream equation for unique extrema, with explicit
extensions at ambiguous/numerically unstable cases:

- Equal similarities yield uniform mass and a degeneracy diagnostic.
- Tied minima share epsilon equally, eliminating choice-order preference.
- Zero-temperature tied maxima share probability equally.
- Positive-temperature scaling operates in log space to avoid underflow.
- Nonfinite embeddings fail; zero/flat embedding signals are exposed as degenerate.

Tests cover all permutations of a tied case, unique-extrema equality with the
preserved upstream implementation, extreme temperatures and embedding reuse.
These changes remove numerical artifacts; they do not demonstrate higher human
accuracy. Sources: [upstream SSR repository](https://github.com/pymc-labs/semantic-similarity-rating),
[SSR paper](https://arxiv.org/html/2510.08338v1), and
[SentenceTransformer revision documentation](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html).

## Measurements and reproducibility

`execution-audit` reports planned/accepted seed requests, physical attempts,
failure categories, known/unresolved charges, returned model/provider/fingerprint
sets, reported token counts and measured HTTP durations. Unexpected/unverified
response metadata is hashed in exports; exact values stay in the private journal. Latency includes failed
HTTP responses but excludes backoff, preparation and local embedding computation;
unreturned/ambiguous attempts are counted as unmeasured. Missing token reports
are shown through coverage counts, not interpreted as zero tokens. A blocked
study has no completed population distribution.

For a completed study, these measurements are stored with a local signature and
the aggregate export. Resuming returns the saved output without spending again.
To measure a fresh repeat, create another workspace with an otherwise identical
request and a different `brief.study_id`, then run it separately:

```sh
python -m rival study compare-runs --workspace FIRST --replicate SECOND
```

This read-only command checks matching inputs and disjoint returned request IDs,
then reports per-seed and aggregate total-variation differences. It does not
confuse journal replay with a new independent completion. Matching outputs in
one small experiment do not establish cross-runtime determinism or human accuracy.

Completed v1/v2 studies retain their saved request/export identities. Their schema
defaults are unchanged. Incomplete older studies still need their pinned runtime.

## Repeating the real local model rehearsal

The optional scripts use [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
at `7ae557604adf67be50417f59c2c2f167def9a775` and the pinned MiniLM above.
These are small public checkpoints selected for an engineering rehearsal, not
recommended production simulation models. Model weights remain outside git.

Install CPU PyTorch for your platform, then install Rival and the pinned extras
in a separate environment. The recorded rehearsal uses `torch==2.8.0+cpu`,
`transformers==4.51.3` and `sentence-transformers==4.1.0` on Linux.

```sh
python -m pip install -e ".[semantic]" transformers==4.51.3 sentence-transformers==4.1.0
python scripts/fetch_local_model_assets.py --output .rival-data/l09-models
python scripts/run_local_model_rehearsal.py --models .rival-data/l09-models --workspace .rival-data/l09-rehearsal --output reports/l09-rehearsal
```

The downloader uses fixed revisions. The runner verifies weight bytes against
their content-addressed Hub blobs and records all actual model/tokenizer/config
hashes, CPU settings and runtime fingerprint. It disables model downloads during
inference, runs a loopback HTTP server with real CPU generation, and records
unmodified model outputs privately. A new directory is required for another run.
Model failures remain in the receipt and produce exit code 2; successful studies
export normally. Local API fees are zero; compute/electricity costs and hosted
latency/cost are not measured. Neither script touches the frozen human benchmarks.

The recorded rehearsal made 14 calls: two direct studies stopped on their first
answer because the probabilities summed to 1.1. Both SSR studies completed 6/6
seed requests and 40 draws. Their six seed distributions matched exactly on the
fresh repeat; neither resume issued any new model calls. This verifies the SSR
execution path and demonstrates a direct-response failure of the small model.
The diagnostic script correctly exited 2 because not every route succeeded.
See [the actual receipt](examples/model-execution/rehearsal.json) and
[verification record](verification/phase_two_model_execution.json). Production model selection, domain qualification and calibration
must use their own evidence; completing this engineering workflow does not
authorize decision-ready customer claims.
