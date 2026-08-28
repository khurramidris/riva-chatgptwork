from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .mathx import canonical_hash
from .providers import PredictionProvider, ProviderError, ProviderPrediction
from .schemas import PopulationRecord, ProviderIdentity, ScenarioSpec
from .vendor.semantic_similarity_rating.compute import (
    response_embeddings_to_pmf,
    scale_pmf,
)


class TextEmbedder(Protocol):
    """Minimal embedding interface used by semantic response rating."""

    @property
    def identity(self) -> str: ...

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class HashingTextEmbedder:
    """Deterministic, download-free lexical embedder for tests and offline demos.

    It is intentionally a development fallback. Production studies should bind a
    validated sentence-embedding model through :class:`SentenceTransformerEmbedder`.
    """

    def __init__(self, dimensions: int = 512):
        if dimensions < 32:
            raise ValueError("dimensions must be at least 32")
        self.dimensions = int(dimensions)

    @property
    def identity(self) -> str:
        return f"rival-hashing-text-v1:{self.dimensions}"

    @staticmethod
    def _features(text: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9]+", text.casefold())
        features = [f"w:{token}" for token in tokens]
        features.extend(
            f"b:{left}_{right}" for left, right in zip(tokens, tokens[1:])
        )
        compact = " ".join(tokens)
        features.extend(
            f"c:{compact[index:index + 3]}"
            for index in range(max(0, len(compact) - 2))
        )
        return features or ["<empty>"]

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float64)
        for row, text in enumerate(texts):
            for feature in self._features(str(text)):
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=16).digest()
                index = int.from_bytes(digest[:8], "big") % self.dimensions
                sign = 1.0 if digest[8] & 1 else -1.0
                matrix[row, index] += sign
            norm = float(np.linalg.norm(matrix[row]))
            if norm > 0:
                matrix[row] /= norm
        return matrix


class SentenceTransformerEmbedder:
    """Lazy adapter for an explicitly versioned sentence-transformers model."""

    def __init__(self, model: str, revision: str | None = None, device: str | None = None):
        self.model = model
        self.revision = revision
        self.device = device
        self._encoder = None

    @property
    def identity(self) -> str:
        return f"sentence-transformers:{self.model}@{self.revision or 'unresolved'}"

    def _load(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "SentenceTransformerEmbedder requires the 'semantic' extra"
                ) from exc
            kwargs = {}
            if self.revision is not None:
                kwargs["revision"] = self.revision
            if self.device is not None:
                kwargs["device"] = self.device
            self._encoder = SentenceTransformer(self.model, **kwargs)
        return self._encoder

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        values = self._load().encode(
            list(texts), convert_to_numpy=True, normalize_embeddings=True
        )
        return np.asarray(values, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class SSRScale:
    choice_ids: tuple[str, ...]
    anchors: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.choice_ids) < 2:
            raise ValueError("SSR requires at least two response choices")
        if len(self.choice_ids) != len(self.anchors):
            raise ValueError("choice_ids and anchors must have the same length")
        if len(set(self.choice_ids)) != len(self.choice_ids):
            raise ValueError("SSR choice_ids must be unique")
        if any(not value.strip() for value in self.anchors):
            raise ValueError("SSR anchors must be non-empty")

    @classmethod
    def from_scenario(cls, scenario: ScenarioSpec) -> "SSRScale":
        anchors = tuple(
            ". ".join(part for part in (choice.label, choice.description) if part).strip()
            for choice in scenario.choices
        )
        return cls(
            choice_ids=tuple(choice.choice_id for choice in scenario.choices),
            anchors=anchors,
        )


class SemanticSimilarityRater:
    """Paper-exact SSR computation with fail-safe numerical handling."""

    def __init__(
        self,
        scale: SSRScale,
        embedder: TextEmbedder | None = None,
        temperature: float = 1.0,
        epsilon: float = 1e-8,
    ):
        if temperature < 0 or not math.isfinite(temperature):
            raise ValueError("temperature must be finite and nonnegative")
        if epsilon < 0 or not math.isfinite(epsilon):
            raise ValueError("epsilon must be finite and nonnegative")
        self.scale = scale
        self.embedder = embedder or HashingTextEmbedder()
        self.temperature = float(temperature)
        self.epsilon = float(epsilon)
        self._anchor_embeddings: np.ndarray | None = None

    @property
    def identity(self) -> dict[str, object]:
        return {
            "algorithm": "pymc-labs.semantic-similarity-rating",
            "upstream_commit": "86dcd2597c7824e4fd6546b884c5500c43a4b022",
            "embedder": self.embedder.identity,
            "temperature": self.temperature,
            "epsilon": self.epsilon,
            "scale_sha256": canonical_hash(
                {"choice_ids": self.scale.choice_ids, "anchors": self.scale.anchors}
            ),
        }

    def _anchors(self) -> np.ndarray:
        if self._anchor_embeddings is None:
            values = np.asarray(self.embedder.encode(self.scale.anchors), dtype=float)
            if values.ndim != 2 or values.shape[0] != len(self.scale.anchors):
                raise ValueError("embedder returned an invalid anchor matrix")
            self._anchor_embeddings = values
        return self._anchor_embeddings

    def rate_array(self, response: str) -> np.ndarray:
        response_matrix = np.asarray(self.embedder.encode([response]), dtype=float)
        anchors = self._anchors()
        if response_matrix.ndim != 2 or response_matrix.shape[0] != 1:
            raise ValueError("embedder returned an invalid response matrix")
        if response_matrix.shape[1] != anchors.shape[1]:
            raise ValueError("response and anchor embedding dimensions differ")
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            pmf = response_embeddings_to_pmf(
                response_matrix, anchors.T, epsilon=self.epsilon
            )[0]
        if (
            pmf.shape != (len(self.scale.choice_ids),)
            or not np.all(np.isfinite(pmf))
            or np.any(pmf < 0)
            or float(pmf.sum()) <= 0
        ):
            pmf = np.full(len(self.scale.choice_ids), 1.0 / len(self.scale.choice_ids))
        else:
            pmf = pmf / pmf.sum()
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            scaled = np.asarray(scale_pmf(pmf, self.temperature), dtype=float)
        if not np.all(np.isfinite(scaled)) or float(scaled.sum()) <= 0:
            scaled = np.full(len(pmf), 1.0 / len(pmf))
        return scaled / scaled.sum()

    def rate(self, response: str) -> dict[str, float]:
        values = self.rate_array(response)
        return {
            choice_id: float(value)
            for choice_id, value in zip(self.scale.choice_ids, values, strict=True)
        }


@dataclass(frozen=True, slots=True)
class GeneratedText:
    text: str
    request_id: str | None = None
    attempts: int = 1
    latency_ms: float = 0.0


class TextResponseGenerator(Protocol):
    @property
    def identity(self) -> dict[str, object]: ...

    def generate(self, person: PopulationRecord, scenario: ScenarioSpec) -> GeneratedText: ...


class OpenAICompatibleTextGenerator:
    """Generate unconstrained natural-language intent for subsequent SSR rating."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: int = 60,
        temperature: float = 0.7,
        max_retries: int = 3,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("RIVAL_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url or os.getenv("RIVAL_BASE_URL") or "https://openrouter.ai/api/v1/chat/completions"
        self.timeout_seconds = int(timeout_seconds)
        self.temperature = float(temperature)
        self.max_retries = int(max_retries)
        if not self.api_key:
            raise ProviderError("missing API key; set RIVAL_API_KEY or OPENROUTER_API_KEY")

    @property
    def identity(self) -> dict[str, object]:
        endpoint = urllib.parse.urlsplit(self.base_url)
        endpoint_identity = urllib.parse.urlunsplit(
            (endpoint.scheme, endpoint.netloc, endpoint.path, "", "")
        )
        return {
            "provider": "openai-compatible-text",
            "model": self.model,
            "endpoint_sha256": canonical_hash(endpoint_identity),
            "temperature": self.temperature,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
        }

    def _payload(self, person: PopulationRecord, scenario: ScenarioSpec) -> dict:
        prompt = {
            "person": {
                "attributes": person.attributes,
                "preferences": person.preferences,
                "relevant_history": person.history[-8:],
            },
            "scenario": {
                "question": scenario.question,
                "context": scenario.context,
                "horizon": scenario.horizon,
            },
        }
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer the question as the described person in one short, natural "
                        "sentence. Express intent in words; do not output a scale number, "
                        "choice id, probability, JSON, or explanation."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, sort_keys=True)},
            ],
            "temperature": self.temperature,
            "max_tokens": 160,
        }

    def generate(self, person: PopulationRecord, scenario: ScenarioSpec) -> GeneratedText:
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(self._payload(person, scenario)).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://rival.local",
                "X-Title": "Rival SSR Elicitation",
            },
            method="POST",
        )
        started = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                text = str(payload["choices"][0]["message"]["content"]).strip()
                if not text:
                    raise ProviderError("provider returned an empty text response")
                return GeneratedText(
                    text=text,
                    request_id=str(payload["id"]) if payload.get("id") is not None else None,
                    attempts=attempt,
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
            except (urllib.error.URLError, TimeoutError, KeyError, IndexError, TypeError, ValueError, ProviderError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(2 ** (attempt - 1))
        raise ProviderError(f"text generation failed after {self.max_retries} attempts") from last_error


class SSRElicitationProvider(PredictionProvider):
    """Rival provider: free-text behavioral response -> SSR probability mass."""

    name = "ssr-elicitation"

    def __init__(
        self,
        generator: TextResponseGenerator,
        embedder: TextEmbedder | None = None,
        temperature: float = 1.0,
        epsilon: float = 1e-8,
    ):
        self.generator = generator
        self.embedder = embedder or HashingTextEmbedder()
        self.temperature = temperature
        self.epsilon = epsilon

    def _rater(self, scenario: ScenarioSpec) -> SemanticSimilarityRater:
        return SemanticSimilarityRater(
            SSRScale.from_scenario(scenario),
            self.embedder,
            temperature=self.temperature,
            epsilon=self.epsilon,
        )

    def predict(self, person: PopulationRecord, scenario: ScenarioSpec) -> ProviderPrediction:
        generated = self.generator.generate(person, scenario)
        probabilities = self._rater(scenario).rate(generated.text)
        return ProviderPrediction(
            probabilities=probabilities,
            diagnostics={"ssr_response_characters": float(len(generated.text))},
            provider_request_id=generated.request_id,
            attempts=generated.attempts,
            latency_ms=generated.latency_ms,
        )

    def identity(self) -> ProviderIdentity:
        configuration = {
            "generator": self.generator.identity,
            "embedder": self.embedder.identity,
            "temperature": self.temperature,
            "epsilon": self.epsilon,
            "ssr_upstream_commit": "86dcd2597c7824e4fd6546b884c5500c43a4b022",
        }
        return ProviderIdentity(
            provider_name=self.name,
            provider_version="1",
            model=str(self.generator.identity.get("model", "text-generator+SSR")),
            configuration_sha256=canonical_hash(configuration),
        )

