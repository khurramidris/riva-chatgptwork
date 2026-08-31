# Mega-Study Windows Operator Runbook

This checklist operates Rival's frozen Twin-2K-500 Mega-Study development
benchmark. It changes no cohort, prompt, model route, retrieval rule, parser,
outcome map, or metric. The existing 1,500-case Wave-4 experiment remains
paused and separate.

## Non-negotiable boundaries

- Run from D:\Experiment\allegory\rival with Python 3.11.
- Never put an API key in a command, file, screenshot, report, or chat message.
- Never open or edit reports\mega_study\results.jsonl during prediction.
- Never inspect survey_json_with_human_response or a Mega-Study sealed
  directory before the prediction and calibration gates are complete.
- Stop after any nonzero exit code, API/parse/context failure, provider drift,
  budget stop, hash failure, or unexpected count. Preserve all output.
- Partial checkpoints are operationally audited only. They are never scored,
  compared, or used to tune Rival.

## 1. Update and verify the checkout

Open Command Prompt and run:

~~~bat
cd /d D:\Experiment\allegory\rival
git status --short
git pull --ff-only
git log -1 --oneline
.venv\Scripts\python.exe --version
~~~

Git status must show no tracked changes. Python must be 3.11.x. Stop if the pull
is not a fast-forward or the checkout is dirty.

Run the Windows regression and Mega-Study tests:

~~~bat
.venv\Scripts\python.exe -m unittest tests.test_mega_study_utils tests.test_mega_study_protocol tests.test_mega_study_runtime tests.test_mega_study_checkpoint -v
~~~

All tests must pass.

## 2. Rebuild and audit the outcome-free stage

The prior attempt downloaded and checksum-verified the required sources, so
require the existing cache:

~~~bat
.venv\Scripts\python.exe scripts\prepare_mega_study.py --no-download
~~~

The final JSON must show PASS, zero real API calls, 300 cases, 1,200 planned
calls, official replication PASS, leakage firewall PASS, and protected outcomes
not materialized. Do not improvise around a missing or failed checksum.

Confirm that the live result ledger is still empty:

~~~bat
.venv\Scripts\python.exe scripts\audit_mega_checkpoint.py --expect-terminal 0 --budget-usd 0.10 --report reports\mega_study\checkpoint_0000.json
~~~

Expected verification: EMPTY_LEDGER_READY, zero failures, zero spending, and
NOT_EVALUATED_TO_PRESERVE_BLIND.

## 3. Run exactly four preflight calls

~~~bat
.venv\Scripts\python.exe scripts\run_mega_study_secure.py --phase preflight --budget-usd 0.10 --expiry-minutes 30
~~~

Paste the OpenRouter key only into the hidden prompt. The frozen route must be:

~~~text
OpenRouter / DeepInfra / deepseek/deepseek-v4-flash-0731
~~~

The run must finish with PREFLIGHT_PASS, four successful work items, and zero
errors. Immediately audit it:

~~~bat
.venv\Scripts\python.exe scripts\audit_mega_checkpoint.py --expect-terminal 4 --budget-usd 0.10 --report reports\mega_study\checkpoint_0004.json
~~~

Stop here. Review only the secure-run and checkpoint-audit summaries. Do not
open the result ledger or begin the pilot until the preflight is accepted.

## 4. Run resumable 100-call checkpoints

Each authorized batch uses:

~~~bat
.venv\Scripts\python.exe scripts\run_mega_study_secure.py --phase pilot --max-new-calls 100 --budget-usd 7.00 --expiry-minutes 120
~~~

The $7 ceiling is cumulative across the ledger. After every clean batch, audit
the cumulative total. Expected totals are:

~~~text
104, 204, 304, 404, 504, 604,
704, 804, 904, 1004, 1104, 1200
~~~

For example, after the first 100-call pilot batch:

~~~bat
.venv\Scripts\python.exe scripts\audit_mega_checkpoint.py --expect-terminal 104 --budget-usd 7.00 --report reports\mega_study\checkpoint_0104.json
~~~

Change the expected total and report suffix after each batch. The final batch
contains only the remaining 96 calls and should report
PILOT_PREDICTIONS_COMPLETE. Every audit must show PASS, the exact count, zero
failures, spending within budget, and NOT_EVALUATED_TO_PRESERVE_BLIND.

If a command stops, rerunning is safe only after its cause is understood. The
ledger is append-only and terminal work is skipped.

## 5. Freeze raw A-D, then hold outcomes closed

After all 1,200 work items pass the final blind audit, preserve the summaries
and obtain final review. Then freeze the raw ledger:

~~~bat
.venv\Scripts\python.exe scripts\freeze_mega_study.py
~~~

It must report outcomes_opened false. Do not run evaluate_mega_study.py yet if
the separate SYN-DIGITS E/F supplement will be used. E/F must be derived and
frozen without Mega target outcomes first; otherwise it is exploratory only.

## Stop-condition record

If anything fails, retain these files unchanged:

~~~text
reports\mega_study\results.jsonl
reports\mega_study\run_summary.json
reports\mega_study\checkpoint_*.json
reports\mega_study\preparation_audit.json
~~~

Record the command, local time, exit code, last safe terminal count, displayed
failure, and whether any provider call occurred. Never delete or rewrite a
failed scientific run to make a later run look clean.
