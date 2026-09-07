"""Bind a built wheel and verification evidence without overwriting historical releases."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rival.release import build_release_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--evidence", action="append", default=[])
    args = parser.parse_args()
    print(build_release_manifest(args.wheel, args.output_dir, args.evidence))


if __name__ == "__main__":
    main()
