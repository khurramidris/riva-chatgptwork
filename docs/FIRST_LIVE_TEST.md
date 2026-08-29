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

The committed `rival/studies/twin2k_live_v2/protocol.json` fixes:

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

V2 supersedes the V1 provider contract. V1 made compatibility calls but
produced zero parseable study predictions and was never evaluated against
outcomes. Before any successful prediction, V2 replaced generic JSON mode with
a strict per-choice JSON Schema, disabled reasoning for the probability-only
response and required a provider endpoint supporting those parameters. The
cohort, targets, outcome firewall, comparators and evaluation gates were not
changed.

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
| Preflight | 30 | 60,642 | 9,000 |
| Complete paired pilot | 1,500 | 3,015,526 | 450,000 |

The output figure is a ceiling, not expected usage; a small JSON probability
object should normally use far fewer tokens. Once the model is named, calculate
the local USD cap from its current provider prices and keep the separately
configured API-key cap above that estimate but below the account-wide balance.

## Live sequence

1. Record the provider, exact model ID, current input/output prices, key cap and
   key expiry.
2. Run `scripts/probe_openrouter.py` once. It prompts for the key without echoing
   or storing it and validates the exact strict-output contract with one call.
3. After the probe passes, run one actual frozen study case. The secure launcher
   again prompts without echoing the key and does not retain it:

   ```bash
   python scripts/run_live_preflight.py --max-calls 1
   ```

4. If it reports `ONE_FROZEN_CASE_PASS`, resume the same ledger and complete
   the 30-call preflight:

   ```bash
   python scripts/run_live_preflight.py --max-calls 30
   ```

   The second command skips the successful first case, so it makes 29 new calls.
5. Inspect schema validity, error rate, latency, provider-reported usage and
   projected full-pilot cost.
6. Authorize the full call count only if the preflight is clean and its cost
   projection fits the remaining provider-side cap.
7. Resume the same ledger in cumulative checkpoints. For example, the first
   full-pilot checkpoint is:

   ```bash
   python scripts/run_live_pilot_secure.py \
     --target-total 300 \
     --model PROVIDER_MODEL_ID \
     --budget-usd LOCAL_HARD_CAP \
     --input-cost-per-million INPUT_PRICE \
     --output-cost-per-million OUTPUT_PRICE
   ```

   Continue with targets 600, 900, 1200 and 1500. Each invocation skips all
   successful rows already in the append-only ledger. Calls are sequential,
   and progress is printed every 10 new successes.
8. Run `evaluate-live-pilot` only after all eligible calls finish.
9. Commit the outcome-free protocol and the redacted scientific report. Never
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

The commands deliberately have no `--api-key` option. The secure launcher is
preconfigured for `dots-studio/dots-3-note-preview:free`, zero per-token price,
a $0.01 local ceiling and a two-hour authorization window. When using a paid
model, pass its current prices with `--input-cost-per-million` and
`--output-cost-per-million`; never leave them at zero.

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
