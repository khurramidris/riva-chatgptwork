"""Rehearse v3 studies with real local Qwen weights and zero inference API fees.

Run after fetch_local_model_assets.py. Uses generated people and one question;
this is an engineering rehearsal, not a population accuracy test. CPU inference
is served only on loopback. Original model responses are never repaired.
"""

import argparse
from datetime import timedelta
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
from threading import Thread
import time
from uuid import uuid4


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.workspace.exists() or args.output.exists():
        raise ValueError("use new workspace and output directories; earlier evidence is preserved")
    # Set the cache before importing Hub/Transformers. No downloads during runs.
    os.environ["HF_HUB_CACHE"] = str((args.models / "cache").resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rival.evidence_catalog import EvidenceCatalog, EvidenceImportSpec, strict_json
    from rival.evidence_commands import example_files
    from rival.mathx import canonical_hash
    from rival.model_providers import runtime_identity
    from rival.schemas import utc_now
    from rival.study_evidence import bind_evidence
    from rival.study_execution_audit import audit_study_execution, compare_studies
    from rival.study_workflow import prepare_study, run_study, export_study

    model_meta = json.loads((args.models / "Qwen2.5-0.5B-Instruct.json").read_text())
    embedding_meta = json.loads((args.models / "all-MiniLM-L6-v2.json").read_text())
    model_path = Path(model_meta["local_snapshot"])
    pinned = []
    for meta in (model_meta, embedding_meta):
        path = Path(meta["local_snapshot"])
        if path.name != meta["revision"]:
            raise ValueError("snapshot directory differs from its declared revision")
        files = {str(item.relative_to(path)): sha256(item) for item in sorted(path.rglob("*")) if item.is_file()}
        for item in path.rglob("*.safetensors"):
            if item.resolve().name != files[str(item.relative_to(path))]:
                raise ValueError("weight hash differs from its content-addressed Hub blob")
        pinned.append({"model": meta["model"], "revision": meta["revision"], "files_sha256": files})
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    runtime = {**runtime_identity(True), "torch_threads": 4, "dtype": "float32",
        "device": "cpu", "attention": "eager", "deterministic_algorithms": True,
        "server_script_sha256": sha256(Path(__file__)), "models": pinned}
    fingerprint = canonical_hash(runtime)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.float32, attn_implementation="eager")
    model.eval()
    args.workspace.mkdir(parents=True)
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *values):
            pass

        def do_POST(self):
            started = time.perf_counter()
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if payload["model"] != model_meta["model"]:
                raise ValueError("server received a different model request")
            torch.manual_seed(payload["seed"])
            text = tokenizer.apply_chat_template(payload["messages"], tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors="pt")
            if inputs["input_ids"].shape[1] > 4096:
                raise ValueError("prompt exceeds the explicit rehearsal input limit")
            kwargs = {"max_new_tokens": payload["max_tokens"], "do_sample": payload["temperature"] > 0,
                      "pad_token_id": tokenizer.eos_token_id}
            if kwargs["do_sample"]:
                kwargs["temperature"] = payload["temperature"]
            with torch.inference_mode():
                generated = model.generate(**inputs, **kwargs)
            tokens = generated[0, inputs["input_ids"].shape[1]:]
            content = tokenizer.decode(tokens, skip_special_tokens=True)
            eos = model.generation_config.eos_token_id
            eos = [eos] if isinstance(eos, int) else eos
            response = {"id": "local-" + uuid4().hex, "model": model_meta["model"],
                "system_fingerprint": fingerprint,
                "choices": [{"finish_reason": "stop" if int(tokens[-1]) in eos else "length",
                             "message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": int(inputs["input_ids"].shape[1]),
                          "completion_tokens": len(tokens), "cost": 0.0}}
            calls.append({"request_sha256": canonical_hash(payload), "response": response,
                          "inference_seconds": time.perf_counter() - started})
            with (args.workspace / "local-model-calls.jsonl").open("a") as handle:
                handle.write(json.dumps(calls[-1], sort_keys=True) + "\n")
            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    examples = example_files()
    (args.workspace / "people.csv").write_bytes(examples["people.csv"])
    catalog = EvidenceCatalog(args.workspace / "catalog")
    bundle = catalog.import_file(args.workspace / "people.csv", EvidenceImportSpec.model_validate(strict_json(examples["import.json"])))
    args.output.mkdir(parents=True)
    results = {}
    expires = (utc_now() + timedelta(hours=1)).isoformat()
    try:
        for method in ("direct", "ssr"):
            studies = []
            for replicate in (1, 2):
                brief = strict_json(examples["brief.json"])
                brief["schema_version"] = "rival.study-request.v3"
                brief["brief"].update(study_id=f"l09-local-{method}-{replicate}", sample_size=40,
                    title=f"Local {method} model engineering rehearsal")
                brief["execution"] = {"mode": "managed", "model": model_meta["model"],
                    "base_url": f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions",
                    "model_pin": {"kind": "checkpoint", "revision": model_meta["revision"],
                        "reference": "https://huggingface.co/" + model_meta["model"] + "/tree/" + model_meta["revision"],
                        "expected_response_model": model_meta["model"], "expected_system_fingerprint": fingerprint},
                    "temperature": 0, "generation_seed": 42, "max_retries": 1,
                    "history_limit": 16, "max_output_tokens": 160, "timeout_seconds": 600,
                    "budget_usd": 0.1, "reservation_usd": 0.001, "max_attempts": 6,
                    "not_after": expires, "elicitation": {"method": method}}
                if method == "ssr":
                    brief["execution"]["elicitation"]["embedding"] = {
                        "kind": "sentence_transformer", "model": embedding_meta["model"],
                        "revision": embedding_meta["revision"], "device": "cpu"}
                request = bind_evidence(brief, catalog.root, [bundle.bundle_sha256], strict_json(examples["support.json"]))
                workspace = args.workspace / request.brief.study_id
                prepare_study(workspace, request, catalog_root=catalog.root)
                started = time.perf_counter()
                try:
                    status = run_study(workspace, api_key="local-no-secret")
                except (RuntimeError, ValueError) as exc:
                    results[request.brief.study_id] = {"complete": False, "failure_type": type(exc).__name__,
                        "execution": audit_study_execution(workspace),
                        "run_seconds_including_embedding": time.perf_counter() - started}
                    print(request.brief.study_id, "blocked", type(exc).__name__, flush=True)
                    continue
                elapsed = time.perf_counter() - started
                count = len(calls)
                assert run_study(workspace)["complete"] and len(calls) == count
                export_study(workspace, args.output / request.brief.study_id)
                results[request.brief.study_id] = {"complete": status["complete"],
                    "execution": audit_study_execution(workspace), "run_seconds_including_embedding": elapsed,
                    "resume_added_model_calls": len(calls) - count}
                studies.append(workspace)
                print(request.brief.study_id, "complete", round(elapsed, 2), "seconds", flush=True)
            if len(studies) == 2:
                results[method + "_repeat"] = compare_studies(*studies)
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
        receipt = {"schema_version": "rival.local-model-rehearsal.v1", "runtime": runtime,
            "system_fingerprint": fingerprint, "model_calls": len(calls), "api_cost_usd": 0.0,
            "results": results, "call_ledger_sha256": sha256(args.workspace / "local-model-calls.jsonl") if calls else None,
            "limitations": ["Generated audience, one question, small CPU model: no human accuracy evidence.",
                "Local API fee is zero; compute/electricity and production hosted costs are not measured.",
                "JSON output is requested by prompt; the local server does not constrain or repair model output.",
                "Fresh identical-run comparison describes this runtime, not cross-platform determinism."]}
        (args.output / "rehearsal.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    # A diagnostic run must not advertise a full success if a model study failed.
    return 0 if all(row.get("complete", True) for row in results.values()) and len(results) == 6 else 2


if __name__ == "__main__":
    raise SystemExit(main())
