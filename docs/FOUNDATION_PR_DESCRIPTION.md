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
claimed. GitHub CI covers Linux and Windows/Python 3.11; see the PR checks for
the latest result. The installation checker resolves temporary-directory aliases
and retains subprocess diagnostics; an aliased-directory rehearsal reproduced
the old path-check failure and passed with the correction.

Compatibility: current network adapters require ExecutionSession; use the new
`simulate-managed` or `mega-v2` interface. Frozen v1 commands remain archival
witnesses and are not current CLI options. Existing experiments and historical
reports are unchanged. Old receipt-only chains are not upgraded by inventing
missing evidence. L07–L14 remain on the delivery board.

Publication: [draft PR #5](https://github.com/khurramidris/riva-chatgptwork/pull/5)
is open on `codex/customer-readiness-foundation`. The initial remote commit
`4da97a0becf16d8914b9db236b45becd8491801b` exactly matches local checkpoint
`be4221781676932bc5be27ac41a9a8ffa6ab013b` (tree
`d2fa86508829dcb5f12ef5335cf7395b91d7cdcd`). Commit IDs in the existing local
verification records identify those pre-publication checkpoints; subsequent
publication fixes are recorded in the PR history. No merge or deployment has
been performed.
