Rival could accept invalid survey answers, mishandle fractional outcomes, advance
prospective studies with unverified reveal receipts and inflate confidence by
repeatedly evaluating the same unprotected result. Retried model requests lost
attempt costs, and installed packages could not verify their frozen resources.

This change completes phase-one engineering (L01–L06) for the research development
release `0.6.0.dev4`:

- Validate probabilities and survey primitives, score composites correctly, retain
  missing/failing cases, use tied ranks and serialize undefined variance as null.
- Bind vault-mediated reveals and recomputed evaluations to exact sealed inputs,
  predictions, outcome payloads and signed phase evidence.
- Assign confidence evidence roles before lock, deduplicate independent-group
  declarations and exclude held-out roles from fitting. Research fits always
  abstain; no confidence or customer qualification is inferred.
- Require durable execution sessions for probability, behavioral and text/SSR
  adapters; retain every attempt charge, enforce budget/expiry/attempt limits,
  quarantine unknown billing and recover cached output. Repeated population draws
  share one seed-person model response. Mega v2 remains a separate namespace.
- Package exact archived frozen witnesses, complete supplemental notices, current
  CLI commands, offline installation checks and hash-bound release inventories.
  Qualification failure returns a nonzero exit code; historical benchmarks stay
  separate from current claims. The GUI labels its generated demonstration data.

Validation: 136 local tests passed on Linux/Python 3.12.13; engineering integrity
and research reproduction checks passed. An offline wheel installation outside
the checkout exercised imports, notices, archived witnesses, stage loading,
freeze, CLI entry points, demo execution and the local HTTP API. All 68 protected
files matched the starting commit. See `docs/verification/phase_one.json`.
No paid inference, customer-domain accuracy improvement, deployment or launch is
claimed. Windows CI is configured and pending branch publication.

Compatibility: current network adapters require ExecutionSession; use the new
`simulate-managed` or `mega-v2` interface. Frozen v1 commands remain archival
witnesses and are not current CLI options. Existing experiments and historical
reports are unchanged. Old receipt-only chains are not upgraded by inventing
missing evidence. L07–L14 remain on the delivery board.

Publication: committed/prepared locally. The earlier automatic approval review
rejected the branch push because explicit authorization to upload code to the
existing GitHub destination was not established. No remote branch, PR, CI run,
merge or deployment has been completed for this change.
