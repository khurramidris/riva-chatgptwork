import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rival.mega_study.utils import atomic_json


class MegaStudyAtomicJsonTests(unittest.TestCase):
    def test_fsync_uses_the_open_writable_descriptor(self):
        def reject_read_only_descriptor(file_descriptor: int) -> None:
            # A zero-byte write is non-mutating but still verifies that the
            # descriptor was opened for writing.  Windows raises EBADF when
            # fsync is attempted on the former read-only implementation.
            os.write(file_descriptor, b"")

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "nested" / "report.json"
            with patch(
                "rival.mega_study.utils.os.fsync",
                side_effect=reject_read_only_descriptor,
            ):
                atomic_json(destination, {"status": "PASS", "count": 3})

            self.assertEqual(
                json.loads(destination.read_text(encoding="utf-8")),
                {"status": "PASS", "count": 3},
            )
            self.assertFalse(destination.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
