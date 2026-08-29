import inspect
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from rival.mega_study.constants import MODEL_CONFIG, VARIANTS
from rival.mega_study.outcomes import (
    ResponseParseError,
    extract_outcome_cells,
    parse_model_json,
    validate_complete_response,
)
from rival.mega_study.prompts import render_prompt, template_hashes
from rival.mega_study.protocol import load_cohort, load_manifest
from rival.mega_study.provider import OpenRouterSurveyProvider
from rival.mega_study.retrieval import (
    demographics_text,
    retrieve_evidence,
)
from rival.mega_study.runner import balanced_variant_order
from rival.mega_study.stage import _audit_prediction_package
from rival.mega_study.utils import LeakageError, write_jsonl


def synthetic_persona_json() -> str:
    demographics = [
        {
            "QuestionID": f"D{index}",
            "QuestionText": f"Demographic field {index}?",
            "Answers": {"SelectedText": [f"Value {index}"]},
        }
        for index in range(1, 15)
    ]
    history = [
        {
            "QuestionID": f"H{index}",
            "QuestionText": (
                f"Earlier attitude {index} about privacy trust price fairness and risk?"
            ),
            "Answers": {"SelectedText": [f"Historical answer {index}"]},
        }
        for index in range(1, 101)
    ]
    return json.dumps(
        [
            {"BlockName": "Demographics", "Questions": demographics},
            {"BlockName": "Earlier Surveys", "Questions": history},
        ]
    )


class FrozenMegaProtocolTests(unittest.TestCase):
    def test_manifest_cohort_and_legacy_witness_verify(self):
        manifest = load_manifest()
        cohort = load_cohort(manifest)
        self.assertEqual(len(cohort), 300)
        self.assertEqual(len({row["pid"] for row in cohort}), 273)
        self.assertTrue(all("protected_response_sha256" not in row for row in cohort))
        self.assertEqual(manifest["model_config"], MODEL_CONFIG)
        self.assertEqual(manifest["prompt_template_sha256"], template_hashes())
        self.assertFalse(manifest["replication_check"]["target_study_overlap"])
        reference_names = {
            source["local_name"]
            for source in manifest["sources"]
            if source["kind"] == "official_reference_output"
        }
        self.assertTrue(
            all("digital_certification" in name for name in reference_names)
        )
        for study in manifest["development_studies"]:
            rows = [row for row in cohort if row["study_id"] == study]
            self.assertEqual(len(rows), 100)
            starts = Counter(row["variant_order"][0] for row in rows)
            self.assertEqual(starts, Counter({variant: 25 for variant in VARIANTS}))
            for row in rows:
                self.assertEqual(
                    tuple(row["variant_order"]),
                    balanced_variant_order(study, row["selection_rank"]),
                )

    def test_provider_payload_is_fully_pinned(self):
        payload = OpenRouterSurveyProvider.request_payload("system", "user")
        self.assertEqual(payload["model"], MODEL_CONFIG["model"])
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["seed"], 20260829)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["reasoning"], {"enabled": False, "exclude": True})
        self.assertEqual(
            payload["provider"],
            {
                "order": ["DeepInfra"],
                "allow_fallbacks": False,
                "require_parameters": True,
            },
        )

    def test_retrieval_is_deterministic_and_has_no_outcome_argument(self):
        persona_json = synthetic_persona_json()
        query = "A new privacy and technology trust decision with a price risk."
        first, first_audit = retrieve_evidence(persona_json, query)
        second, second_audit = retrieve_evidence(persona_json, query)
        self.assertEqual([item.evidence_id for item in first], [item.evidence_id for item in second])
        self.assertEqual(first_audit, second_audit)
        self.assertGreaterEqual(len(first), 1)
        self.assertLessEqual(len(first), 24)
        self.assertLessEqual(len("\n\n".join(item.text for item in first)), 16000)
        self.assertEqual(
            tuple(inspect.signature(retrieve_evidence).parameters),
            ("persona_json", "query", "top_k", "max_chars"),
        )

    def test_four_prompts_change_only_person_representation(self):
        persona_json = synthetic_persona_json()
        persona = {
            "persona_text": "Complete historical persona",
            "persona_json": persona_json,
            "demographics": demographics_text(persona_json),
        }
        case = {
            "study_id": "privacy",
            "pid": "pid_1",
            "survey_text": "Q1:\nChoose a privacy protection rating.\n### Format Instructions:",
        }
        rendered = {variant: render_prompt(case, persona, variant) for variant in VARIANTS}
        self.assertEqual(len({prompt.system for prompt in rendered.values()}), 1)
        self.assertNotIn("Complete historical persona", rendered["generic"].user)
        self.assertNotIn("Demographic field", rendered["generic"].user)
        self.assertIn("Demographic field", rendered["demographics"].user)
        self.assertIn("Complete historical persona", rendered["full_persona"].user)
        self.assertTrue(rendered["rival_retrieval"].retrieval_audit)

    def test_prediction_package_fails_closed_on_protected_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = root / "prediction" / "cases.jsonl"
            personas = root / "prediction" / "personas.jsonl"
            case = {
                "case_id": "x",
                "study_id": "privacy",
                "pid": "pid_1",
                "selection_rank": 1,
                "survey_text": "Q1:\nSafe",
                "survey_text_sha256": "x",
                "expected_outcome_ids": ["PPV"],
                "answerable_question_count": 1,
                "variant_order": list(VARIANTS),
            }
            persona = {
                "pid": "pid_1",
                "persona_text": "history",
                "persona_json": "[]",
                "demographics": "demo",
                "demographics_sha256": "x",
                "historical_evidence_count": 100,
            }
            write_jsonl(cases, [case])
            write_jsonl(personas, [persona])
            audit = _audit_prediction_package(cases, personas, expected_cases=1)
            self.assertEqual(audit["status"], "PASS")
            case["ground_truth"] = 4
            write_jsonl(cases, [case])
            with self.assertRaises(LeakageError):
                _audit_prediction_package(cases, personas, expected_cases=1)


class MegaOutcomeMapTests(unittest.TestCase):
    @staticmethod
    def _response(positions):
        return {
            key: {"Answers": {"SelectedByPosition": value}}
            for key, value in positions.items()
        }

    def test_privacy_mapping_and_strict_parser(self):
        response = parse_model_json('{"Q1":{"Answers":{"SelectedByPosition":6}}}')
        cells = extract_outcome_cells("privacy", "Q1:\nPrivacy\n", response)
        self.assertEqual(cells[0].value, 6.0)
        with self.assertRaises(ResponseParseError):
            validate_complete_response(
                "Q1:\nFirst\nQ2:\nSecond\n### Format Instructions:\nJSON",
                response,
            )
        with self.assertRaises(ResponseParseError):
            extract_outcome_cells(
                "privacy",
                "Q1:\nPrivacy\n",
                self._response({"Q1": 8}),
            )

    def test_hiring_mapping_and_recodes(self):
        positions = {
            "Q2": 5,
            "Q3": 4,
            "Q4": 1,
            "Q5": 3,
            "Q6": 2,
            "Q7": 2,
            "Q8": 5,
            "Q9": 1,
            "Q19": [1, 2, 3, 4, 5, 6, 7, 1],
            "Q20": [2, 3, 4, 5, 6, 7, 1, 2],
            "Q21": [3, 4, 5, 6, 7, 1, 2, 3],
            "Q22": [4, 5, 6, 7, 1, 2, 3, 4],
        }
        cells = extract_outcome_cells(
            "hiring_algorithms", "Q1:\nPlaceholder\n", self._response(positions)
        )
        values = {cell.outcome_id: cell.value for cell in cells}
        self.assertEqual(len(cells), 40)
        self.assertEqual(values["Q9"], 4.0)
        self.assertEqual(values["Q14"], 4.0)
        self.assertEqual(values["job4_item8"], 4.0)

    def test_junk_fee_derived_outcomes(self):
        chunks = []
        positions = {}
        for index in range(1, 7):
            chunks.append(
                f"Q{index}:\nWhich of the following do you think best represents what "
                f"fee {index} is assessed for?"
            )
            positions[f"Q{index}"] = 1 if index < 6 else 2
        for index in range(7, 13):
            chunks.append(
                f"Q{index}:\nHow fair do you think it is to charge for service {index}?"
            )
            positions[f"Q{index}"] = index - 5
        chunks.append("Q13:\nShould pricing practices be regulated by the government?")
        chunks.append(
            "Q14:\nDo you support government regulation that bans firms from "
            "separating out mandatory fees?"
        )
        positions.update({"Q13": 4, "Q14": 6})
        survey = "\n".join(chunks) + "\n### Format Instructions:\nJSON"
        cells = extract_outcome_cells("junk_fees", survey, self._response(positions))
        values = {cell.outcome_id: cell.value for cell in cells}
        self.assertAlmostEqual(values["percent_correct"], 100 * 5 / 6)
        self.assertAlmostEqual(values["fairness_average"], 4.5)
        self.assertEqual(values["reg_support"], 5.0)


if __name__ == "__main__":
    unittest.main()
