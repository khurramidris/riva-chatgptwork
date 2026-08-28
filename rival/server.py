from __future__ import annotations

import json
import mimetypes
import os
import traceback
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import ValidationError

from .demo import demo_scenario, run_demo
from .engine import RivalEngine
from .integrity import IntegrityError, ManifestSigner, ProspectiveStudyManager
from .mathx import canonical_hash
from .schemas import (
    HumanObservation,
    PopulationRecord,
    PopulationTargets,
    PredictionContext,
    PreregistrationSpec,
    ScenarioSpec,
    SimulationResult,
)
from .research.qualification import load_bundled_summary
from .version import __version__


STATIC_ROOT = Path(__file__).parent / "static"


class RivalApplication:
    def __init__(
        self,
        database_path: str = "rival.sqlite3",
        manifest_secret: str | None = None,
    ):
        self.engine = RivalEngine(store_path=database_path)
        secret = manifest_secret or os.getenv("RIVAL_MANIFEST_KEY")
        self.integrity = (
            ProspectiveStudyManager(self.engine.store, ManifestSigner(secret))
            if secret
            else None
        )

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "rival-sim",
            "version": __version__,
            "providers": self.engine.router.available,
            "mode": "offline-ready",
            "prospective_locking_configured": self.integrity is not None,
            "outcome_reveal_exposed": False,
            "research_components": [
                "semantic-similarity-rating",
                "uq-survey-intervals",
                "syn-digits-synthetic-control",
                "paired-srct",
                "persona-demand-pricing",
                "interview-grounded-personas",
                "centauri-adapter",
                "socrates-adapter",
                "twin2k-mad",
            ],
        }

    def qualification(self) -> dict[str, Any]:
        return load_bundled_summary()

    def demo_config(self) -> dict[str, Any]:
        return demo_scenario().model_dump(mode="json")

    def demo_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        sample_size = int(payload.get("sample_size", 1200))
        anchor_size = int(payload.get("human_anchor_size", 80))
        if sample_size < 20 or sample_size > 10_000:
            raise ValueError("demo sample_size must be between 20 and 10,000")
        if anchor_size < 0 or anchor_size > min(sample_size, 2_000):
            raise ValueError("demo human_anchor_size is outside the allowed range")
        overrides = payload.get("scenario", {})
        if not isinstance(overrides, dict):
            raise ValueError("scenario overrides must be an object")
        return run_demo(
            self.engine,
            sample_size=sample_size,
            human_anchor_size=anchor_size,
            scenario_overrides=overrides,
        )

    def simulate(self, payload: dict[str, Any]) -> dict[str, Any]:
        records = [PopulationRecord.model_validate(row) for row in payload["records"]]
        scenario = ScenarioSpec.model_validate(payload["scenario"])
        targets_payload = payload.get("targets")
        targets = (
            PopulationTargets.model_validate(targets_payload)
            if targets_payload is not None
            else None
        )
        locked_payload = payload.get("locked_context")
        locked_context = (
            PredictionContext.model_validate(locked_payload)
            if locked_payload is not None
            else None
        )
        result = self.engine.simulate(
            records, scenario, targets, locked_context=locked_context
        )
        return result.model_dump(mode="json")

    def prediction_context(self, payload: dict[str, Any]) -> dict[str, Any]:
        records = [PopulationRecord.model_validate(row) for row in payload["records"]]
        scenario = ScenarioSpec.model_validate(payload["scenario"])
        targets_payload = payload.get("targets")
        targets = (
            PopulationTargets.model_validate(targets_payload)
            if targets_payload is not None
            else None
        )
        context, audit = self.engine.prepare_prediction_context(
            records, scenario, targets
        )
        return {
            "prediction_context": context.model_dump(mode="json"),
            "retrieval_audit": audit.model_dump(mode="json"),
        }

    def lock_study(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.integrity is None:
            raise ValueError("prospective locking requires RIVAL_MANIFEST_KEY")
        simulation = SimulationResult.model_validate(payload["simulation"])
        stored = self.engine.store.get("runs", simulation.run_id)
        if stored is None or canonical_hash(stored) != canonical_hash(simulation):
            raise ValueError("only an unchanged run in this ledger can be locked")
        preregistration = PreregistrationSpec.model_validate(
            payload.get("preregistration", {})
        )
        sealed = self.integrity.lock_prediction(simulation, preregistration)
        return sealed.model_dump(mode="json")

    def correct(self, payload: dict[str, Any]) -> dict[str, Any]:
        simulation = SimulationResult.model_validate(payload["simulation"])
        observations = [
            HumanObservation.model_validate(row) for row in payload["observations"]
        ]
        return self.engine.correct(simulation, observations).model_dump(mode="json")

    def research_ssr(self, payload: dict[str, Any]) -> dict[str, Any]:
        from .elicitation import HashingTextEmbedder, SSRScale, SemanticSimilarityRater

        choices = payload["choices"]
        if not isinstance(choices, list):
            raise ValueError("choices must be a list")
        scale = SSRScale(
            choice_ids=tuple(str(item["choice_id"]) for item in choices),
            anchors=tuple(str(item["anchor"]) for item in choices),
        )
        rater = SemanticSimilarityRater(
            scale,
            HashingTextEmbedder(int(payload.get("dimensions", 512))),
            temperature=float(payload.get("temperature", 1.0)),
            epsilon=float(payload.get("epsilon", 1e-8)),
        )
        response = str(payload["response"])
        return {
            "response": response,
            "probabilities": rater.rate(response),
            "identity": rater.identity,
            "warning": "hashing embeddings are an offline fallback, not a qualified production encoder",
        }

    def research_uncertainty(self, payload: dict[str, Any]) -> dict[str, Any]:
        from .uncertainty import synthetic_mean_interval

        interval = synthetic_mean_interval(
            payload["values"],
            k=payload.get("k"),
            confidence=float(payload.get("confidence", 0.95)),
            scale=float(payload.get("scale", 2.0)),
            method=str(payload.get("method", "clt")),
            minimum_k=int(payload.get("minimum_k", 2)),
            parameter_bounds=tuple(payload.get("parameter_bounds", [0.0, 1.0])),
        )
        return asdict(interval)

    def research_srct(self, payload: dict[str, Any]) -> dict[str, Any]:
        from .experiments.srct import PairedPrediction, PrePeriodAnchor, estimate_paired_srct

        predictions = [PairedPrediction(**item) for item in payload["pairs"]]
        pre_period_payload = payload.get("pre_period")
        pre_period = PrePeriodAnchor(**pre_period_payload) if pre_period_payload else None
        return asdict(
            estimate_paired_srct(
                predictions,
                confidence=float(payload.get("confidence", 0.95)),
                pre_period=pre_period,
            )
        )

    def research_pricing(self, payload: dict[str, Any]) -> dict[str, Any]:
        from .pricing import fit_persona_demand, optimize_price

        action = str(payload.get("action", "fit"))
        if action == "fit":
            model = fit_persona_demand(
                payload["persona_probabilities"],
                payload["observed_demand"],
                int(payload["exposure_n"]),
                objective=str(payload.get("objective", "truncated")),
                calibration_iterations=int(payload.get("calibration_iterations", 6)),
            )
            prediction_matrix = payload.get("prediction_persona_probabilities")
            result = {
                "model": {
                    "exposure_n": model.exposure_n,
                    "persona_weights": model.persona_weights.tolist(),
                    "no_buy_weight": model.no_buy_weight,
                    "intercept": model.intercept,
                    "slope": model.slope,
                    "objective": model.objective,
                    "objective_value": model.objective_value,
                    "converged": model.converged,
                    "status": model.status,
                    "source": model.source,
                }
            }
            if prediction_matrix is not None:
                result["purchase_probabilities"] = model.purchase_probability(
                    prediction_matrix
                ).tolist()
                result["mean_demand"] = model.mean_demand(prediction_matrix).tolist()
            return result
        if action == "optimize":
            decision = optimize_price(
                payload["prices"],
                payload["purchase_probabilities"],
                int(payload["market_size"]),
                unit_cost=float(payload.get("unit_cost", 0.0)),
                risk_weight=float(payload.get("risk_weight", 0.0)),
                tail_probability=float(payload.get("tail_probability", 0.1)),
                draws=int(payload.get("draws", 4000)),
                seed=int(payload.get("seed", 20260828)),
            )
            return asdict(decision)
        raise ValueError("pricing action must be fit or optimize")

    def research_personas(self, payload: dict[str, Any]) -> dict[str, Any]:
        from .personas import InterviewPersonaBuilder, InterviewTranscript

        transcripts = [
            InterviewTranscript.model_validate(item) for item in payload["transcripts"]
        ]
        builder = InterviewPersonaBuilder(
            maximum_turns=int(payload.get("maximum_turns", 80)),
            participant_only=bool(payload.get("participant_only", False)),
        )
        return {
            "records": [
                record.model_dump(mode="json")
                for record in builder.build_many(transcripts)
            ]
        }

    def research_synthetic_control_complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        import numpy as np

        from .research.synthetic_control import ResearchSyntheticControl, SYN_DIGITS_COMMIT

        matrix = np.asarray(
            [
                [np.nan if value is None else float(value) for value in row]
                for row in payload["matrix"]
            ],
            dtype=float,
        )
        completed = ResearchSyntheticControl.complete(
            matrix,
            method=str(payload.get("method", "hard_svd")),
            rank=int(payload.get("rank", 2)),
            max_iter=int(payload.get("max_iter", 100)),
            tolerance=float(payload.get("tolerance", 1e-4)),
            regularization=float(payload.get("regularization", 0.1)),
            random_state=int(payload.get("random_state", 42)),
        )
        return {
            "matrix": completed.tolist(),
            "method": str(payload.get("method", "hard_svd")),
            "upstream_commit": SYN_DIGITS_COMMIT,
        }


def make_handler(application: RivalApplication):
    class RivalHandler(BaseHTTPRequestHandler):
        server_version = "Rival/0.5"

        def log_message(self, format: str, *args):
            print(f"[rival] {self.address_string()} - {format % args}")

        def _json(self, value: Any, status: int = 200) -> None:
            body = json.dumps(value, separators=(",", ":"), default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, Any]:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 5_000_000:
                raise ValueError("request body must be between 1 byte and 5 MB")
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            return payload

        def _serve_static(self, path: str) -> None:
            relative = "index.html" if path in {"", "/"} else path.lstrip("/")
            candidate = (STATIC_ROOT / relative).resolve()
            if STATIC_ROOT.resolve() not in candidate.parents and candidate != STATIC_ROOT.resolve():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if not candidate.is_file():
                candidate = STATIC_ROOT / "index.html"
            body = candidate.read_bytes()
            mime_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", f"{mime_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            try:
                if path == "/api/health":
                    self._json(application.health())
                elif path == "/api/demo/config":
                    self._json(application.demo_config())
                elif path == "/api/runs":
                    self._json({"runs": application.engine.store.list_rows("runs", 30)})
                elif path == "/api/qualification":
                    self._json(application.qualification())
                else:
                    self._serve_static(path)
            except Exception as exc:
                self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                payload = self._read_json()
                if path == "/api/demo/run":
                    self._json(application.demo_run(payload))
                elif path == "/api/simulate":
                    self._json(application.simulate(payload))
                elif path == "/api/prediction-context":
                    self._json(application.prediction_context(payload))
                elif path == "/api/studies/lock":
                    self._json(application.lock_study(payload))
                elif path == "/api/hybrid":
                    self._json(application.correct(payload))
                elif path == "/api/research/ssr":
                    self._json(application.research_ssr(payload))
                elif path == "/api/research/uncertainty":
                    self._json(application.research_uncertainty(payload))
                elif path == "/api/research/srct":
                    self._json(application.research_srct(payload))
                elif path == "/api/research/pricing":
                    self._json(application.research_pricing(payload))
                elif path == "/api/research/personas":
                    self._json(application.research_personas(payload))
                elif path == "/api/research/synthetic-control/complete":
                    self._json(application.research_synthetic_control_complete(payload))
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except (
                IntegrityError,
                KeyError,
                ValueError,
                ValidationError,
                json.JSONDecodeError,
            ) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                traceback.print_exc()
                self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    return RivalHandler


def serve(host: str = "127.0.0.1", port: int = 8080, database_path: str = "rival.sqlite3"):
    application = RivalApplication(database_path)
    server = ThreadingHTTPServer((host, port), make_handler(application))
    print(f"Rival is running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Rival")
    finally:
        server.server_close()
        application.engine.store.close()
