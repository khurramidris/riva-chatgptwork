"""Download the explicitly pinned public weights for the local L09 rehearsal.

No inference API is used. Requires huggingface-hub. Downloads stay outside git.
"""
import argparse
import json
from pathlib import Path
from huggingface_hub import snapshot_download

MODELS = {
    "Qwen/Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775",
    "sentence-transformers/all-MiniLM-L6-v2": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for model, revision in MODELS.items():
        path = snapshot_download(repo_id=model, revision=revision,
            allow_patterns=["*.json", "*.txt", "*.safetensors", "LICENSE", "README.md"],
            cache_dir=args.output / "cache", max_workers=2)
        meta = {"model": model, "revision": revision, "local_snapshot": str(Path(path).resolve())}
        destination = args.output / (model.split("/")[-1] + ".json")
        encoded = json.dumps(meta, indent=2, sort_keys=True) + "\n"
        if destination.exists() and destination.read_text() != encoded:
            raise ValueError("existing model metadata differs; choose another directory")
        destination.write_text(encoded)
        print(model, revision, "ready", flush=True)


if __name__ == "__main__":
    main()
