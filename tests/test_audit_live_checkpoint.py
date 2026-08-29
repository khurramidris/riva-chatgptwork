import json
import tempfile
import unittest
from pathlib import Path

from rival.mathx import canonical_hash
from scripts.audit_live_checkpoint import audit_checkpoint


class CheckpointAuditTests(unittest.TestCase):
    def test_audit_is_outcome_free_and_detects_a_complete_pair(self):
        study = Path(__file__).parents[1] / "rival" / "studies" / "twin2k_live_v2"
        protocol = json.loads((study / "protocol.json").read_text(encoding="utf-8"))
        cases = [
            json.loads(line)
            for line in (study / "cases.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        first = cases[0]
        second = next(
            case
            for case in cases
            if case["participant_id"] == first["participant_id"]
            and case["target_column"] == first["target_column"]
            and case["variant"] != first["variant"]
        )
        target = next(
            item for item in protocol["targets"] if item["column"] == first["target_column"]
        )
        choice_ids = [choice["choice_id"] for choice in target["scenario"]["choices"]]
        probabilities = {choice_id: 1 / len(choice_ids) for choice_id in choice_ids}
        provider = {"provider_name": "test", "provider_version": "1", "model": "test"}
        rows = []
        for case in (first, second):
            row = {
                "status": "SUCCESS",
                "protocol_sha256": protocol["protocol_sha256"],
                "case_id": case["case_id"],
                "provider": provider,
                "probabilities": probabilities,
                "diagnostics": {"prompt_tokens": 10, "completion_tokens": 5},
                "attempts": 1,
                "latency_ms": 100,
                "billed_cost_usd": 0.00001,
            }
            row["result_sha256"] = canonical_hash(row)
            rows.append(row)

        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "results.jsonl"
            results.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            report = audit_checkpoint(
                study / "protocol.json", study / "cases.jsonl", results
            )

        self.assertEqual(report["ledger"]["integrity"], "PASS")
        self.assertEqual(report["coverage"]["paired_cells"], 1)
        self.assertEqual(
            report["scientific_accuracy_status"], "NOT_EVALUATED_TO_PRESERVE_BLIND"
        )
        self.assertNotIn("accuracy", report["prediction_behavior"])


if __name__ == "__main__":
    unittest.main()
