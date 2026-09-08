import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from datetime import timedelta

from rival.commands import main
from rival.evidence_catalog import (EvidenceCatalog, EvidenceImportSpec, json_bytes,
                                    records_digest, sha256, strict_json)
from rival.evidence_commands import example_files
from rival.study_contract import StudyRequest, StudyRequestV2, parse_study_request
from rival.study_evidence import bind_evidence
from rival.study_support import UnsupportedStudy, assess_support
from rival.study_workflow import check_study, export_study, prepare_study, run_study, study_status
from rival.schemas import utc_now
from test_execution_journal import Response, completion


class EvidenceSupportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.catalog = EvidenceCatalog(self.root / "catalog")
        self.files = example_files()
        self.spec = strict_json(self.files["import.json"])
        self.policy = strict_json(self.files["support.json"])
        self.brief = strict_json(self.files["brief.json"])
        self.brief["brief"]["sample_size"] = 40
        self.workspace = self.root / "study"

    def imported(self, data=None, spec=None):
        data = self.files["people.csv"] if data is None else data
        spec = copy.deepcopy(self.spec if spec is None else spec)
        spec["artifact_sha256"] = sha256(data)
        path = self.root / "source"
        path.write_bytes(data)
        manifest = self.catalog.import_file(path, EvidenceImportSpec.model_validate(spec))
        return manifest

    def request(self, manifest=None):
        manifest = manifest or self.imported()
        return bind_evidence(self.brief, self.catalog.root, [manifest.bundle_sha256], self.policy)

    def native(self):
        records = copy.deepcopy(self.brief["audience"]["records"])
        for record in records:
            record.pop("evidence_ids")
            record["attributes"]["geography"] = "example-only"
        spec = copy.deepcopy(self.spec)
        spec["format"] = "population-jsonl"
        spec.pop("csv_mapping")
        return records, spec

    @staticmethod
    def lines(records):
        return b"".join(json_bytes(row).replace(b"\n", b" ") + b"\n" for row in records)

    def test_import_replays_mapping_and_is_immutable_and_repeatable(self):
        with patch("urllib.request.urlopen") as network:
            first = self.imported()
            second = self.imported()
            manifest, records = self.catalog.load(first.bundle_sha256)
        network.assert_not_called()
        self.assertEqual(first, second)
        self.assertEqual(manifest.records_sha256, records_digest(records))
        self.assertEqual(len(records), 6)
        self.assertEqual(records[0].attributes, {"geography": "example-only", "segment": "value"})
        self.assertEqual(records[0].preferences, {"value": .9, "convenience": .2})
        self.assertEqual(records[0].evidence_ids, ["generated-example"])
        self.assertEqual(manifest.source().sha256, sha256(self.files["people.csv"]))
        changed = copy.deepcopy(self.spec)
        changed["revision"] = "second-release"
        another = self.imported(spec=changed)
        self.assertNotEqual(first.bundle_sha256, another.bundle_sha256)
        self.assertTrue((self.catalog.root / first.bundle_sha256).is_dir())

    def test_wrong_hash_and_row_count_fail_without_creating_catalog(self):
        path = self.root / "source"
        path.write_bytes(self.files["people.csv"] + b"unexpected bytes")
        with self.assertRaisesRegex(ValueError, "pinned artifact"):
            self.catalog.import_file(path, EvidenceImportSpec.model_validate(self.spec))
        self.assertFalse(self.catalog.root.exists())
        wrong = copy.deepcopy(self.spec)
        wrong["expected_records"] = 7
        with self.assertRaisesRegex(ValueError, "row count"):
            self.imported(spec=wrong)
        self.assertFalse(self.catalog.root.exists())

    def test_category_recoding_is_explicit_and_unmapped_codes_fail(self):
        spec = copy.deepcopy(self.spec)
        spec["csv_mapping"]["attributes"]["segment"]["categories"] = {"value": "Value seekers", "convenience": "Convenience seekers"}
        manifest = self.imported(spec=spec)
        _, records = self.catalog.load(manifest.bundle_sha256)
        self.assertEqual(records[0].attributes["segment"], "Value seekers")
        del spec["csv_mapping"]["attributes"]["segment"]["categories"]["convenience"]
        with self.assertRaisesRegex(ValueError, "recoding map"):
            self.imported(spec=spec)

    def test_raw_or_normalized_tampering_and_missing_files_are_detected(self):
        manifest = self.imported()
        root = self.catalog.root / manifest.bundle_sha256
        for filename in ("source.bin", "records.jsonl", "manifest.json"):
            original = (root / filename).read_bytes()
            with self.subTest(filename=filename):
                (root / filename).write_bytes(original.replace(b"value", b"other"))
                with self.assertRaises(ValueError):
                    self.catalog.load(manifest.bundle_sha256)
                (root / filename).write_bytes(original)
        (root / "records.jsonl").unlink()
        with self.assertRaisesRegex(ValueError, "missing|unexpected"):
            self.catalog.load(manifest.bundle_sha256)
        with self.assertRaisesRegex(ValueError, "invalid"):
            self.catalog.load("../outside")

    def test_malformed_csv_and_duplicate_ids_cannot_silently_drop_rows(self):
        cases = [self.files["people.csv"].replace(b"id,geography", b"id,id"),
                 self.files["people.csv"].replace(b"1,example-only", b"2,example-only"),
                 self.files["people.csv"].replace(b"1,example-only,value,0.9,0.2", b"1,example-only,value,0.9"),
                 self.files["people.csv"].replace(b"0.9", b"NaN"),
                 self.files["people.csv"].replace(b"1,example-only", b"1,undeclared-region")]
        for data in cases:
            with self.subTest(data=data), self.assertRaises(ValueError):
                self.imported(data)
        self.assertFalse(self.catalog.root.exists())

    def test_unmapped_columns_stay_outside_prediction_inputs(self):
        text = self.files["people.csv"].decode().splitlines()
        data = (text[0] + ",observed_choice\n" + "\n".join(row + ",secret-answer" for row in text[1:]) + "\n").encode()
        manifest = self.imported(data)
        request = self.request(manifest)
        self.assertNotIn("secret-answer", request.model_dump_json())
        mapped = copy.deepcopy(self.spec)
        mapped["csv_mapping"]["attributes"]["observed_choice"] = {"column": "observed_choice"}
        with self.assertRaisesRegex(ValueError, "protected outcome"):
            self.imported(data, mapped)

    def test_native_history_requires_aware_dated_precollection_evidence(self):
        records, spec = self.native()
        records[0]["history"] = [{"recorded_at": "2026-07-01T00:00:00Z", "text": "prior preference"}]
        manifest = self.imported(self.lines(records), spec)
        self.assertEqual(len(self.catalog.load(manifest.bundle_sha256)[1][0].history), 1)
        bad_entries = [{"text": "undated"}, {"date": "2026-07-01"}, {"date": "2030-01-01T00:00:00Z"},
                       {"date": "2026-07-01T00:00:00Z", "observed_outcome": "answer"},
                       {"recorded_at": "2026-07-01T00:00:00Z", "timestamp": "2030-01-01T00:00:00Z"}]
        for entry in bad_entries:
            records[0]["history"] = [entry]
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                self.imported(self.lines(records), spec)

    def test_native_nonfinite_weights_duplicate_json_and_forged_source_ids_fail(self):
        records, spec = self.native()
        for weight in (float("inf"), float("nan"), 1e300, 1e-300):
            for record in records:
                record["weight"] = weight
            data = b"".join((json.dumps(row) + "\n").encode() for row in records)
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                self.imported(data, spec)
        records, spec = self.native()
        records[0]["evidence_ids"] = ["injected-source"]
        with self.assertRaisesRegex(ValueError, "own evidence"):
            self.imported(self.lines(records), spec)
        with self.assertRaisesRegex(ValueError, "duplicate JSON"):
            self.imported(b'{"person_id":"a","person_id":"b"}\n', spec)

    def test_rights_and_snapshot_dates_are_explicit(self):
        cases = []
        for key in ("source_id", "collected_at"):
            spec = copy.deepcopy(self.spec)
            del spec["source"][key]
            cases.append(spec)
        for patch_source in ({"permitted_uses": []}, {"prohibited_uses": ["simulation"]}, {"source_type": "outcome"}):
            spec = copy.deepcopy(self.spec)
            spec["source"].update(patch_source)
            cases.append(spec)
        for field, value in (("released_at", "2026-08-02"), ("retrieved_at", "2020-01-01T00:00:00Z"), ("revision", " ")):
            spec = copy.deepcopy(self.spec)
            spec[field] = value
            cases.append(spec)
        for spec in cases:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                EvidenceImportSpec.model_validate(spec)

    def test_geography_is_applied_to_actual_simulation_inputs(self):
        records, spec = self.native()
        for i in range(2):
            extra = copy.deepcopy(records[i])
            extra["person_id"] = "outside-" + str(i)
            extra["attributes"]["geography"] = "elsewhere"
            records.append(extra)
        spec["expected_records"] = 8
        spec["source"]["geography"].append("elsewhere")
        request = self.request(self.imported(self.lines(records), spec))
        checked = check_study(request, catalog_root=self.catalog.root)
        self.assertEqual(checked["support_audit"]["excluded_seed_records"], 2)
        with patch("urllib.request.urlopen") as network:
            prepared = prepare_study(self.workspace, request, catalog_root=self.catalog.root)
            completed = run_study(self.workspace)
            export_study(self.workspace, self.root / "report")
        network.assert_not_called()
        self.assertEqual(prepared["planned_unique_seeds"], 6)
        self.assertTrue(completed["complete"])
        report = strict_json((self.root / "report/report.json").read_bytes())
        self.assertEqual(report["audience"]["eligible_seed_records"], 6)
        self.assertTrue(report["evidence"]["import_verification"]["raw_bytes_and_conversion_verified"])
        self.assertEqual(report["support"]["geography_counts"], {"example-only": 6})
        self.assertNotIn("generated-example:generated-0", json.dumps(report))

    def test_missing_geographies_conditions_and_stale_sources_block_before_calls(self):
        manifest = self.imported()
        mutations = [("geography", ["example-only", "missing"]), ("conditions", ["undeclared-condition"]), ("max_source_age_days", 1)]
        for field, value in mutations:
            request = self.request(manifest)
            if field == "geography":
                request.audience.geography = value
            else:
                setattr(request.support, field, value)
            with self.subTest(field=field), patch("urllib.request.urlopen") as network:
                with self.assertRaises(UnsupportedStudy) as error:
                    prepare_study(self.workspace, request, catalog_root=self.catalog.root)
                self.assertFalse(error.exception.report["passed"])
                self.assertFalse(self.workspace.exists())
                network.assert_not_called()

    def test_missing_joint_cells_and_effective_counts_are_exposed(self):
        request = self.request()
        request.support.cells[0].filters = {"segment": "value", "missing_attribute": "yes"}
        report = assess_support(request)
        self.assertIn("cell_not_supported", [row["code"] for row in report["issues"]])
        records, spec = self.native()
        records[-1]["weight"] = 100.
        request = self.request(self.imported(self.lines(records), spec))
        report = assess_support(request)
        self.assertEqual(report["positive_weight_seed_records"], 6)
        self.assertLess(report["effective_seed_records"], 5)
        self.assertIn("insufficient_effective_records", [row["code"] for row in report["issues"]])

    def test_release_date_and_retrieval_date_have_distinct_meanings(self):
        spec = copy.deepcopy(self.spec)
        spec["retrieved_at"] = "2026-09-05T00:00:00Z"
        request = self.request(self.imported(spec=spec))
        report = assess_support(request)
        self.assertTrue(report["passed"])
        self.assertTrue(report["warnings"])
        spec["released_at"] = "2026-09-04T00:00:00Z"
        request = self.request(self.imported(spec=spec))
        report = assess_support(request)
        self.assertFalse(report["passed"])
        self.assertIn("source_after_cutoff", [row["code"] for row in report["issues"]])

    def test_altered_request_or_substituted_catalog_cannot_prepare(self):
        request = self.request()
        changed = request.model_dump(mode="json")
        changed["audience"]["records"][0]["weight"] = 10.
        with self.assertRaisesRegex(ValueError, "differ from the normalized"):
            parse_study_request(changed)
        with self.assertRaisesRegex(ValueError, "catalog"):
            prepare_study(self.workspace, request)
        root = self.catalog.root / request.imports[0].bundle_sha256
        (root / "source.bin").write_bytes(b"altered source")
        with patch("urllib.request.urlopen") as network, self.assertRaises(ValueError):
            prepare_study(self.workspace, request, catalog_root=self.catalog.root)
        network.assert_not_called()
        self.assertFalse(self.workspace.exists())

    def test_prepared_request_survives_catalog_removal_and_exports_remain_repeatable(self):
        request = self.request()
        prepare_study(self.workspace, request, catalog_root=self.catalog.root)
        shutil.rmtree(self.catalog.root)
        prepared = prepare_study(self.workspace, request)
        self.assertFalse(prepared["complete"])
        with patch("urllib.request.urlopen") as network:
            result = run_study(self.workspace)
            self.assertEqual(run_study(self.workspace), result)
            output = self.root / "report"
            export_study(self.workspace, output)
            original = {path.name: path.read_bytes() for path in output.iterdir()}
            export_study(self.workspace, output)
        network.assert_not_called()
        self.assertEqual(original, {path.name: path.read_bytes() for path in output.iterdir()})
        self.assertEqual(study_status(self.workspace), result)

    def test_new_legacy_human_inputs_cannot_bypass_imports(self):
        self.brief["audience"]["sources"][0]["source_type"] = "public"
        request = StudyRequest.model_validate(self.brief)
        with self.assertRaisesRegex(ValueError, "v2 request"):
            check_study(request)
        with self.assertRaisesRegex(ValueError, "v2 request"):
            prepare_study(self.workspace, request)
        self.assertFalse(self.workspace.exists())

    def test_completed_legacy_export_keeps_its_original_release_identity(self):
        with patch("rival.integrity.__version__", "0.6.0.dev5"), patch("rival.study_workflow.__version__", "0.6.0.dev5"):
            prepare_study(self.workspace, StudyRequest.model_validate(self.brief))
            run_study(self.workspace)
            output = self.root / "legacy-report"
            export_study(self.workspace, output)
        original = {path.name: path.read_bytes() for path in output.iterdir()}
        self.assertEqual(strict_json(original["report.json"])["release_claims"]["release"], "0.6.0.dev5")
        run_study(self.workspace)
        export_study(self.workspace, output)
        self.assertEqual(original, {path.name: path.read_bytes() for path in output.iterdir()})

    def test_missing_attributes_and_bad_range_filters_are_explicit(self):
        request = self.request()
        request.support.required_attributes.append("uncollected_attribute")
        report = assess_support(request)
        self.assertEqual(report["missing_attributes"]["uncollected_attribute"], 6)
        self.assertFalse(report["passed"])
        for rule in ({"min": 3, "max": 1}, {"min": "non-numeric"}, []):
            payload = request.model_dump(mode="json")
            payload["audience"]["filters"] = {"segment": rule}
            with self.subTest(rule=rule), self.assertRaises(ValueError):
                parse_study_request(payload)

    def test_managed_v2_recovers_original_seed_calls_and_accounting(self):
        self.brief["execution"] = {"mode": "managed", "model": "test-model-r1", "base_url": "https://provider.example/v1/chat/completions",
            "budget_usd": 1., "reservation_usd": .01, "max_attempts": 10,
            "not_after": (utc_now() + timedelta(hours=1)).isoformat()}
        request = self.request()
        prepare_study(self.workspace, request, catalog_root=self.catalog.root)
        response = completion('{"standard": 0.5, "flexible": 0.3, "neither": 0.2}', .01)
        with patch("urllib.request.urlopen", return_value=Response(response)) as network:
            result = run_study(self.workspace, api_key="test-only-key")
        self.assertEqual(network.call_count, 6)
        self.assertAlmostEqual(result["accounting"]["total_cost_usd"], .06)
        with patch("urllib.request.urlopen") as network:
            self.assertEqual(run_study(self.workspace), result)
            export_study(self.workspace, self.root / "managed-report")
        network.assert_not_called()

    def test_cli_import_bind_check_execute_and_report(self):
        example = self.root / "example"
        catalog = self.root / "cli-catalog"
        def call(*args):
            output = io.StringIO()
            with patch("sys.stdout", output):
                code = main(list(map(str, args)))
            self.assertEqual(code, 0, output.getvalue())
            return strict_json(output.getvalue())
        with patch("urllib.request.urlopen") as network:
            call("evidence", "example", "--output-dir", example)
            manifest = call("evidence", "import", "--input", example / "people.csv", "--spec", example / "import.json", "--catalog", catalog)
            self.assertTrue(call("evidence", "inspect", "--catalog", catalog, "--bundle", manifest["bundle_sha256"])["verified"])
            self.assertEqual(len(call("evidence", "list", "--catalog", catalog)), 1)
            path = self.root / "request.json"
            call("study", "bind-evidence", "--input", example / "brief.json", "--support", example / "support.json", "--catalog", catalog, "--bundle", manifest["bundle_sha256"], "--output", path)
            self.assertTrue(call("study", "check", "--input", path, "--catalog", catalog)["passed"])
            call("study", "prepare", "--input", path, "--catalog", catalog, "--workspace", self.workspace)
            self.assertTrue(call("study", "run", "--workspace", self.workspace)["complete"])
            call("study", "export", "--workspace", self.workspace, "--output", self.root / "export")
        network.assert_not_called()
