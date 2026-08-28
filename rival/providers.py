from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from .mathx import canonical_hash, normalize, stable_softmax, stable_unit_interval
from .schemas import (
    PopulationRecord,
    ProviderCallIdentity,
    ProviderIdentity,
    ScenarioSpec,
)


class ProviderError(RuntimeError):
    pass


@dataclass(slots=True)
class ProviderPrediction:
    probabilities: dict[str, float]
    diagnostics: dict[str, float] = field(default_factory=dict)
    provider_request_id: str | None = None
    attempts: int = 1
    latency_ms: float = 0.0
    cache_hit: bool = False


class PredictionProvider(ABC):
    name = "base"

    @abstractmethod
    def predict(
        self, person: PopulationRecord, scenario: ScenarioSpec
    ) -> ProviderPrediction:
        raise NotImplementedError

    def identity(self) -> ProviderIdentity:
        configuration = {
            "provider_name": self.name,
            "implementation": f"{type(self).__module__}.{type(self).__qualname__}",
        }
        return ProviderIdentity(
            provider_name=self.name,
            provider_version="1",
            model=self.name,
            configuration_sha256=canonical_hash(configuration),
        )

    def request_sha256(
        self, person: PopulationRecord, scenario: ScenarioSpec
    ) -> str:
        return canonical_hash(
            {
                "person": person.model_dump(mode="json"),
                "scenario": scenario.model_dump(mode="json"),
            }
        )

    def call_identity(
        self,
        person: PopulationRecord,
        scenario: ScenarioSpec,
        output: ProviderPrediction,
    ) -> ProviderCallIdentity:
        provider = self.identity()
        request_sha256 = self.request_sha256(person, scenario)
        return ProviderCallIdentity(
            provider=provider,
            request_sha256=request_sha256,
            cache_key=canonical_hash(
                {
                    "provider": provider.model_dump(mode="json"),
                    "request_sha256": request_sha256,
                }
            ),
            provider_request_id=output.provider_request_id,
            attempts=output.attempts,
            latency_ms=output.latency_ms,
            cache_hit=output.cache_hit,
        )


class HeuristicChoiceProvider(PredictionProvider):
    """Deterministic offline behavior baseline.

    Choice features are matched to same-named person preferences. ``price`` is
    treated as a cost and multiplied by ``price_sensitivity``. This is a
    transparent baseline and demo provider, not a claim of human validity.
    """

    name = "heuristic-v1"

    def __init__(self, temperature: float = 0.75, idiosyncratic_scale: float = 0.28):
        self.temperature = temperature
        self.idiosyncratic_scale = idiosyncratic_scale

    def predict(
        self, person: PopulationRecord, scenario: ScenarioSpec
    ) -> ProviderPrediction:
        utilities: list[float] = []
        preference_norm = 0.0
        for choice in scenario.choices:
            utility = 0.0
            for feature, value in choice.features.items():
                if feature == "price":
                    sensitivity = float(person.preferences.get("price_sensitivity", 0.5))
                    utility -= float(value) * sensitivity
                    preference_norm += abs(sensitivity)
                else:
                    preference = float(person.preferences.get(feature, 0.0))
                    utility += float(value) * preference
                    preference_norm += abs(preference)

            age = str(person.attributes.get("age_band", ""))
            if age == "18-34" and choice.features.get("novelty", 0) > 0:
                utility += 0.18 * choice.features["novelty"]
            if person.attributes.get("household") == "family":
                utility += 0.12 * choice.features.get("convenience", 0)

            idiosyncratic = stable_unit_interval(
                person.person_id, scenario.scenario_id, choice.choice_id
            )
            utility += (idiosyncratic - 0.5) * 2 * self.idiosyncratic_scale
            utilities.append(utility)

        probs = stable_softmax(utilities, temperature=self.temperature)
        return ProviderPrediction(
            probabilities={
                choice.choice_id: float(probability)
                for choice, probability in zip(scenario.choices, probs, strict=True)
            },
            diagnostics={
                "utility_range": float(max(utilities) - min(utilities)),
                "preference_signal": float(preference_norm / max(len(scenario.choices), 1)),
            },
        )

    def identity(self) -> ProviderIdentity:
        configuration = {
            "temperature": self.temperature,
            "idiosyncratic_scale": self.idiosyncratic_scale,
        }
        return ProviderIdentity(
            provider_name=self.name,
            provider_version="1",
            model="transparent-utility-baseline",
            configuration_sha256=canonical_hash(configuration),
        )


class OpenAICompatibleProvider(PredictionProvider):
    """OpenAI-compatible probability provider using Python's standard library."""

    name = "openai-compatible"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: int = 60,
        temperature: float = 0.2,
        max_retries: int = 3,
        history_limit: int = 16,
        max_output_tokens: int = 300,
        use_response_format: bool = True,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("RIVAL_API_KEY") or os.getenv(
            "OPENROUTER_API_KEY"
        )
        self.base_url = (
            base_url
            or os.getenv("RIVAL_BASE_URL")
            or "https://openrouter.ai/api/v1/chat/completions"
        )
        endpoint = urllib.parse.urlsplit(self.base_url)
        hostname = (endpoint.hostname or "").casefold()
        local_endpoint = hostname in {"localhost", "127.0.0.1", "::1"}
        if endpoint.username or endpoint.password:
            raise ProviderError("provider URL must not contain credentials")
        if endpoint.scheme != "https" and not (
            endpoint.scheme == "http" and local_endpoint
        ):
            raise ProviderError("remote provider URL must use HTTPS")
        if not endpoint.hostname:
            raise ProviderError("provider URL must contain a hostname")
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.max_retries = max_retries
        self.history_limit = int(history_limit)
        self.max_output_tokens = int(max_output_tokens)
        self.use_response_format = bool(use_response_format)
        if self.history_limit < 0:
            raise ValueError("history_limit must be nonnegative")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if not self.api_key:
            raise ProviderError(
                "missing API key; set RIVAL_API_KEY or OPENROUTER_API_KEY"
            )

    def _request_payload(
        self, person: PopulationRecord, scenario: ScenarioSpec
    ) -> dict:
        choice_payload = [choice.model_dump(mode="json") for choice in scenario.choices]
        system = (
            "You are a calibrated behavioral prediction component. Predict a "
            "probability distribution over the supplied valid actions. Do not "
            "role-play or explain. Return one JSON object mapping every choice_id "
            "to a numeric probability. The probabilities must sum to one."
        )
        user = {
            "person": {
                "attributes": person.attributes,
                "preferences": person.preferences,
                "relevant_history": (
                    person.history[-self.history_limit :]
                    if self.history_limit
                    else []
                ),
            },
            "scenario": {
                "question": scenario.question,
                "context": scenario.context,
                "horizon": scenario.horizon,
                "choices": choice_payload,
            },
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, sort_keys=True)},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
        }
        if self.use_response_format:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def identity(self) -> ProviderIdentity:
        endpoint = urllib.parse.urlsplit(self.base_url)
        endpoint_identity = urllib.parse.urlunsplit(
            (endpoint.scheme, endpoint.netloc, endpoint.path, "", "")
        )
        configuration = {
            "model": self.model,
            "temperature": self.temperature,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "history_limit": self.history_limit,
            "max_output_tokens": self.max_output_tokens,
            "use_response_format": self.use_response_format,
        }
        return ProviderIdentity(
            provider_name=self.name,
            provider_version="1",
            model=self.model,
            endpoint_sha256=canonical_hash(endpoint_identity),
            configuration_sha256=canonical_hash(configuration),
        )

    def request_sha256(
        self, person: PopulationRecord, scenario: ScenarioSpec
    ) -> str:
        return canonical_hash(self._request_payload(person, scenario))

    @staticmethod
    def _extract_content(payload: dict) -> str:
        try:
            return str(payload["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("provider response does not contain message content") from exc

    @staticmethod
    def _usage_diagnostics(payload: dict) -> dict[str, float]:
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return {}
        diagnostics: dict[str, float] = {}
        aliases = {
            "prompt_tokens": ("prompt_tokens", "input_tokens"),
            "completion_tokens": ("completion_tokens", "output_tokens"),
            "total_tokens": ("total_tokens",),
            "provider_cost_usd": ("cost", "total_cost", "cost_usd"),
        }
        for destination, candidates in aliases.items():
            for candidate in candidates:
                value = usage.get(candidate)
                if isinstance(value, (int, float)) and float(value) >= 0:
                    diagnostics[destination] = float(value)
                    break
        if "total_tokens" not in diagnostics:
            prompt = diagnostics.get("prompt_tokens")
            completion = diagnostics.get("completion_tokens")
            if prompt is not None and completion is not None:
                diagnostics["total_tokens"] = prompt + completion
        return diagnostics

    def predict(
        self, person: PopulationRecord, scenario: ScenarioSpec
    ) -> ProviderPrediction:
        body = json.dumps(self._request_payload(person, scenario)).encode("utf-8")
        request = urllib.request.Request(
            self.base_url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://rival.local",
                "X-Title": "Rival Simulation",
            },
            method="POST",
        )
        last_error: Exception | None = None
        started = time.perf_counter()
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout_seconds
                ) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                content = self._extract_content(response_payload).strip()
                if content.startswith("```"):
                    content = content.strip("`")
                    if content.startswith("json"):
                        content = content[4:].lstrip()
                parsed = json.loads(content)
                choice_ids = [choice.choice_id for choice in scenario.choices]
                if set(parsed) != set(choice_ids):
                    raise ProviderError("provider returned the wrong choice_id set")
                probabilities = normalize(float(parsed[key]) for key in choice_ids)
                diagnostics = {"attempts": float(attempt + 1)}
                diagnostics.update(self._usage_diagnostics(response_payload))
                return ProviderPrediction(
                    probabilities={
                        key: float(value)
                        for key, value in zip(choice_ids, probabilities, strict=True)
                    },
                    diagnostics=diagnostics,
                    provider_request_id=(
                        str(response_payload.get("id"))
                        if response_payload.get("id") is not None
                        else None
                    ),
                    attempts=attempt + 1,
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
            except (urllib.error.URLError, TimeoutError, ValueError, ProviderError) as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(2**attempt)
        raise ProviderError(f"prediction failed after {self.max_retries} attempts") from last_error


class BehavioralModelProvider(PredictionProvider):
    """OpenAI-compatible adapter for behavior-specialized model servers.

    Rival does not redistribute model weights. The adapter records the exact model
    revision, training-corpus declaration, and model license supplied by the
    deployment operator in every prediction context.
    """

    name = "behavioral-model"

    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        model_revision: str,
        training_corpus: str,
        model_license: str,
        api_key: str | None = None,
        timeout_seconds: int = 60,
        temperature: float = 0.2,
        max_retries: int = 3,
        provider_name: str | None = None,
        behavioral_instruction: str | None = None,
    ):
        if not model_revision.strip():
            raise ValueError("model_revision is required for reproducible inference")
        if not training_corpus.strip() or not model_license.strip():
            raise ValueError("training_corpus and model_license declarations are required")
        self.model = model
        self.base_url = base_url
        self.model_revision = model_revision
        self.training_corpus = training_corpus
        self.model_license = model_license
        self.api_key = api_key or os.getenv("RIVAL_BEHAVIORAL_API_KEY")
        self.timeout_seconds = int(timeout_seconds)
        self.temperature = float(temperature)
        self.max_retries = int(max_retries)
        self.name = provider_name or type(self).name
        self.behavioral_instruction = behavioral_instruction or (
            "Use the described person's pre-outcome evidence to predict behavior. "
            "Return a calibrated probability distribution, not a role-play response."
        )
        hostname = (urllib.parse.urlsplit(base_url).hostname or "").casefold()
        self._local_endpoint = hostname in {"localhost", "127.0.0.1", "::1"}
        if not self.api_key and not self._local_endpoint:
            raise ProviderError(
                "remote behavioral endpoints require RIVAL_BEHAVIORAL_API_KEY or api_key"
            )

    def _request_payload(
        self, person: PopulationRecord, scenario: ScenarioSpec
    ) -> dict:
        choice_payload = [choice.model_dump(mode="json") for choice in scenario.choices]
        user = {
            "person": {
                "attributes": person.attributes,
                "preferences": person.preferences,
                "relevant_history": person.history[-16:],
            },
            "scenario": {
                "question": scenario.question,
                "context": scenario.context,
                "horizon": scenario.horizon,
                "choices": choice_payload,
            },
        }
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"{self.behavioral_instruction} Return one JSON object mapping "
                        "every supplied choice_id to a numeric probability. Include no "
                        "other keys or prose; probabilities must sum to one."
                    ),
                },
                {"role": "user", "content": json.dumps(user, sort_keys=True)},
            ],
            "temperature": self.temperature,
            "max_tokens": 300,
            "response_format": {"type": "json_object"},
        }

    def identity(self) -> ProviderIdentity:
        endpoint = urllib.parse.urlsplit(self.base_url)
        endpoint_identity = urllib.parse.urlunsplit(
            (endpoint.scheme, endpoint.netloc, endpoint.path, "", "")
        )
        configuration = {
            "model": self.model,
            "model_revision": self.model_revision,
            "training_corpus": self.training_corpus,
            "model_license": self.model_license,
            "temperature": self.temperature,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "behavioral_instruction_sha256": canonical_hash(self.behavioral_instruction),
        }
        return ProviderIdentity(
            provider_name=self.name,
            provider_version="1",
            model=f"{self.model}@{self.model_revision}",
            endpoint_sha256=canonical_hash(endpoint_identity),
            configuration_sha256=canonical_hash(configuration),
        )

    def request_sha256(
        self, person: PopulationRecord, scenario: ScenarioSpec
    ) -> str:
        return canonical_hash(self._request_payload(person, scenario))

    def predict(
        self, person: PopulationRecord, scenario: ScenarioSpec
    ) -> ProviderPrediction:
        headers = {"Content-Type": "application/json", "X-Title": "Rival Behavioral Model"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(self._request_payload(person, scenario)).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        started = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                content = OpenAICompatibleProvider._extract_content(response_payload).strip()
                if content.startswith("```"):
                    content = content.strip("`")
                    if content.startswith("json"):
                        content = content[4:].lstrip()
                parsed = json.loads(content)
                choice_ids = [choice.choice_id for choice in scenario.choices]
                if set(parsed) != set(choice_ids):
                    raise ProviderError("behavioral model returned the wrong choice_id set")
                probabilities = normalize(float(parsed[key]) for key in choice_ids)
                return ProviderPrediction(
                    probabilities={
                        key: float(value)
                        for key, value in zip(choice_ids, probabilities, strict=True)
                    },
                    diagnostics={"attempts": float(attempt)},
                    provider_request_id=(
                        str(response_payload.get("id"))
                        if response_payload.get("id") is not None
                        else None
                    ),
                    attempts=attempt,
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
            except (urllib.error.URLError, TimeoutError, ValueError, ProviderError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(2 ** (attempt - 1))
        raise ProviderError(
            f"behavioral prediction failed after {self.max_retries} attempts"
        ) from last_error


class CentauriProvider(BehavioralModelProvider):
    """Inference adapter for an operator-hosted Centauri checkpoint."""

    name = "centauri"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1/chat/completions",
        *,
        model: str = "socius-org/Centauri",
        model_revision: str = "operator-must-pin",
        model_license: str = "operator-must-declare",
        api_key: str | None = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            base_url=base_url,
            model_revision=model_revision,
            training_corpus="Centauri behavior-specialization corpus",
            model_license=model_license,
            api_key=api_key,
            provider_name=self.name,
            behavioral_instruction=(
                "Apply the behavior-specialized Centauri checkpoint to infer the "
                "described person's response from pre-outcome evidence."
            ),
            **kwargs,
        )


class SocratesProvider(BehavioralModelProvider):
    """Inference adapter for an operator-hosted Socrates/SocSci210 checkpoint."""

    name = "socrates"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1/chat/completions",
        *,
        model: str = "Socrates",
        model_revision: str = "operator-must-pin",
        model_license: str = "operator-must-declare",
        api_key: str | None = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            base_url=base_url,
            model_revision=model_revision,
            training_corpus="SocSci210 social-science experiments",
            model_license=model_license,
            api_key=api_key,
            provider_name=self.name,
            behavioral_instruction=(
                "Apply the Socrates social-science behavior checkpoint to infer the "
                "described person's response from pre-outcome evidence."
            ),
            **kwargs,
        )


class EnsembleProvider(PredictionProvider):
    name = "ensemble"

    def __init__(self, providers: list[PredictionProvider]):
        if len(providers) < 2:
            raise ValueError("an ensemble requires at least two providers")
        self.providers = providers

    def predict(
        self, person: PopulationRecord, scenario: ScenarioSpec
    ) -> ProviderPrediction:
        outputs = [provider.predict(person, scenario) for provider in self.providers]
        choice_ids = [choice.choice_id for choice in scenario.choices]
        matrix = np.asarray(
            [[output.probabilities[key] for key in choice_ids] for output in outputs]
        )
        mean = matrix.mean(axis=0)
        disagreement = float(np.mean(np.var(matrix, axis=0)))
        return ProviderPrediction(
            probabilities={
                key: float(value) for key, value in zip(choice_ids, mean, strict=True)
            },
            diagnostics={
                "provider_disagreement": disagreement,
                "ensemble_size": float(len(outputs)),
            },
            attempts=max(output.attempts for output in outputs),
            latency_ms=sum(output.latency_ms for output in outputs),
            cache_hit=all(output.cache_hit for output in outputs),
        )

    def identity(self) -> ProviderIdentity:
        members = [provider.identity().model_dump(mode="json") for provider in self.providers]
        return ProviderIdentity(
            provider_name=self.name,
            provider_version="1",
            model="mean-probability-ensemble",
            configuration_sha256=canonical_hash({"members": members}),
        )
