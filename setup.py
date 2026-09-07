"""Build frozen verification resources without editing historical witnesses."""

import hashlib
import json
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildWithWitnesses(build_py):
    def run(self):
        super().run()
        root = Path(__file__).parent
        relative_manifest = Path("rival/studies/mega_study_v1/MEGA_STUDY_MANIFEST.json")
        manifest = json.loads((root / relative_manifest).read_text(encoding="utf-8"))
        witnesses = {**manifest["implementation_witness"], **manifest["preserved_wave4_witness"]}
        witnesses[str(relative_manifest.parent / manifest["cohort"]["path"])] = manifest["cohort"]["sha256"]
        for relative, expected in witnesses.items():
            source = root / relative
            actual = hashlib.sha256(source.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
            if actual != expected:
                raise RuntimeError(f"frozen witness changed: {relative}")
        for relative in [*witnesses, str(relative_manifest)]:
            destination = Path(self.build_lib) / "rival/frozen/mega_v1" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            self.copy_file(str(root / relative), str(destination))
        for relative in ("THIRD_PARTY_NOTICES.md", "upstreams.lock.json", "docs/SUPPORTED_OFFERING.md"):
            destination = Path(self.build_lib) / "rival/notices" / Path(relative).name
            destination.parent.mkdir(parents=True, exist_ok=True)
            self.copy_file(str(root / relative), str(destination))


setup(cmdclass={"build_py": BuildWithWitnesses})
