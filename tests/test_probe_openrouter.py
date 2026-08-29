import importlib.util
import unittest
from pathlib import Path


_PATH = Path(__file__).parents[1] / "scripts" / "probe_openrouter.py"
_SPEC = importlib.util.spec_from_file_location("probe_openrouter", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe)


class ProbeTests(unittest.TestCase):
    def test_valid_structured_response(self):
        result = probe._validate(
            {
                "id": "request-1",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"option_a": 2, "option_b": 1}'},
                    }
                ],
                "usage": {"total_tokens": 20, "cost": 0},
            }
        )
        self.assertEqual(result["status"], "PASS")
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0)

    def test_redaction_removes_explicit_and_bearer_keys(self):
        key = "sk-or-v1-example-secret-value"
        detail = probe._redact(f"bad {key} and Bearer {key}", key)
        self.assertNotIn(key, detail)
        self.assertIn("[REDACTED]", detail)


if __name__ == "__main__":
    unittest.main()
