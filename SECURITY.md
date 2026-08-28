# Security Policy

## Supported version

Rival is pre-1.0 research software. Only the latest tagged release receives security fixes.

## Reporting

Do not open a public issue for a suspected vulnerability or disclose customer data, provider credentials, manifest keys, outcome-vault keys, or protected outcomes. Use the repository's private GitHub security-reporting channel and include only the minimum reproduction data required.

## Secret boundaries

- Provider credentials belong in environment variables or a deployment secret manager.
- `RIVAL_MANIFEST_KEY` must be at least 32 bytes and must not share custody with the outcome-vault key in a prospective study.
- Outcome keys and plaintext outcomes must never enter the simulation service, Git history, CI logs, qualification artifacts, or support tickets.
- The SQLite vault is encrypted at the application layer, but production custody still requires encrypted disks, restricted service accounts, backups, rotation, and a KMS/HSM.

## Release checks

Every release must pass the unit/integration suite, `qualify-integrity`, the real-data qualification tracks, a clean wheel import, source secret scan, and release-manifest hash verification. A passing integrity qualification is an engineering result, not evidence of predictive validity.
