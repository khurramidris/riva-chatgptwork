# First Live-Model Test

## Purpose

This study asks whether a pinned live language model can predict held-out
Twin-2K Wave-4 answers from outcome-free Wave-1-to-3 behavioral histories. It
compares the same model in two conditions:

- `generic`: panel identity and the new question, without personal history;
- `twin`: the same question plus up to 16 target-family-excluded historical
  answers from that participant.

It is a retrospective public-data qualification, not an independently blinded
prospective customer study. The repository binds and retains that limitation.

## Frozen design

The committed `rival/studies/twin2k_live_v1/protocol.json` fixes:

- 50 participants selected by a deterministic seed;
- all 15 eligible independent categorical target-question families;
- 10 participants reserved as human calibration anchors;
- target, same-QuestionID, normalized-block and semantic-family exclusion;
- paired generic/twin cases;
- a 5-person by 3-question preflight subset;
- individual accuracy, multiclass Brier score, negative log likelihood and
  population TVD;
- generic-model, population-mode, human test-retest, released-model and
  equal-size human-anchor comparators.

`cases.jsonl` contains identifiers, donor-column selections and provider-input
hashes. It contains no Wave-4 outcomes. The preparation path hashes the
protected outcome file without parsing it. Outcome values are loaded only by
the separate evaluation command after predictions exist.

## Security and spending controls

Never put an API key in a command argument, file, result, Git commit or chat
message. Set a newly created restricted key in `RIVAL_API_KEY` or
`OPENROUTER_API_KEY`. Give the key a provider-side hard spending cap and expiry.

The runner independently requires:

- an exact model ID;
- input and output prices per million tokens;
- a local USD ceiling;
- a maximum successful-call count;
- an ISO-8601 run-expiry timestamp.

Before every request it reserves a conservative token-cost estimate. It writes
each successful response to an append-only JSONL ledger and can resume without
paying for completed cases again. Provider-reported token usage and cost are
retained when available. The provider-side cap remains the final protection
against retries, pricing mistakes or missing usage metadata.

The frozen price-independent envelope is:

| Phase | Calls | Conservative input-token estimate | Maximum configured output tokens |
|---|---:|---:|---:|
| Preflight | 30 | 55,208 | 9,000 |
| Complete paired pilot | 1,500 | 2,764,328 | 450,000 |

The output figure is a ceiling, not expected usage; a small JSON probability
object should normally use far fewer tokens. Once the model is named, calculate
the local USD cap from its current provider prices and keep the separately
configured API-key cap above that estimate but below the account-wide balance.

## Tomorrow's sequence

1. Record the provider, exact model ID, current input/output prices, key cap and
   key expiry.
2. Set the key as an environment variable.
3. Run `preflight` only. This is normally 30 paired generic/twin calls.
4. Inspect schema validity, error rate, latency, provider-reported usage and
   projected full-pilot cost.
5. Authorize the full call count only if the preflight is clean and its cost
   projection fits the remaining provider-side cap.
6. Resume with `--phase pilot` using the same model identity and result file.
7. Run `evaluate-live-pilot` only after all eligible calls finish.
8. Commit the outcome-free protocol and the redacted scientific report. Never
   commit the API key or raw provider headers.

Example command shape (placeholder values only):

```bash
rival run-live-pilot \
  --phase preflight \
  --model PROVIDER_MODEL_ID \
  --budget-usd LOCAL_HARD_CAP \
  --input-cost-per-million INPUT_PRICE \
  --output-cost-per-million OUTPUT_PRICE \
  --max-calls 30 \
  --not-after 2026-08-29T18:00:00Z
```

The command deliberately has no `--api-key` option.

### GitHub execution fallback

`.github/workflows/live-pilot.yml` exposes the same operation as a manually
dispatched workflow. It never runs on push or pull request. Configure a
restricted repository secret named `RIVAL_API_KEY`, enter the exact model,
prices, local cap, call cap and expiry in the workflow form, and run `preflight`.
The JSONL ledger and summary are returned as a time-limited workflow artifact.

The full `pilot` should not be launched as a fresh independent workflow after a
preflight because GitHub artifacts are not automatically restored between
runs. Download the clean preflight ledger and resume it locally, or explicitly
provide a reviewed artifact-restoration step before authorizing the remaining
calls. This prevents silently paying for the preflight twice.

## Interpretation

The live result is useful if it tells us whether personal history improves the
same underlying model, whether human-anchor correction reduces distribution
error and how the system compares with strong classical/human-history
baselines. A failed gate is retained as a failed result; it is not repaired by
changing targets, participants, prompts or metrics after outcomes are read.
