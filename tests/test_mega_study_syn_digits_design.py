import json
import unittest
from pathlib import Path

from rival.mega_study.constants import (
    CONFIRMATION_STUDIES,
    DEVELOPMENT_STUDIES,
    VARIANTS,
)
from rival.research.synthetic_control import SYN_DIGITS_COMMIT


class MegaSynDigitsDesignTests(unittest.TestCase):
    def test_design_preserves_raw_protocol_and_blocks_target_access(self):
        root = Path(__file__).parents[1]
        design = json.loads(
            (
                root
                / "rival"
                / "studies"
                / "mega_study_syn_digits_v1"
                / "CALIBRATION_DESIGN.json"
            ).read_text(encoding="utf-8")
        )
        mega = json.loads(
            (
                root
                / "rival"
                / "studies"
                / "mega_study_v1"
                / "MEGA_STUDY_MANIFEST.json"
            ).read_text(encoding="utf-8")
        )
        wave = json.loads(
            (
                root
                / "rival"
                / "studies"
                / "twin2k_live_v2"
                / "protocol.json"
            ).read_text(encoding="utf-8")
        )

        self.assertEqual(
            design["status"], "DESIGN_SPECIFIED_EXECUTION_BLOCKED"
        )
        self.assertEqual(
            design["raw_baseline"]["manifest_sha256"],
            mega["manifest_sha256"],
        )
        self.assertEqual(
            design["raw_baseline"]["cohort_sha256"],
            mega["cohort"]["sha256"],
        )
        self.assertEqual(
            tuple(design["raw_baseline"]["raw_variants"]),
            VARIANTS,
        )
        self.assertFalse(design["raw_baseline"]["mutation_allowed"])
        self.assertEqual(
            design["reference_bank"]["legacy_protocol_sha256"],
            wave["protocol_sha256"],
        )
        self.assertFalse(
            design["reference_bank"]["legacy_ledger_reused_or_modified"]
        )
        self.assertEqual(
            design["reference_bank"]["reference_columns"],
            [target["column"] for target in wave["targets"]],
        )
        self.assertEqual(
            design["reference_bank"]["planned_reference_calls"],
            273 * 15 * 2,
        )
        self.assertEqual(
            design["method"]["upstream_commit"],
            SYN_DIGITS_COMMIT,
        )
        self.assertFalse(
            design["method"]["target_human_values_used_for_fit_or_selection"]
        )
        self.assertEqual(
            tuple(design["protected_mega_studies"]["development"]),
            DEVELOPMENT_STUDIES,
        )
        self.assertEqual(
            tuple(design["protected_mega_studies"]["confirmation"]),
            CONFIRMATION_STUDIES,
        )
        self.assertTrue(design["execution_blockers"])


if __name__ == "__main__":
    unittest.main()
