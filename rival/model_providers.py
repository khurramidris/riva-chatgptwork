"""Pinned v3 adapters. Response metadata cannot independently attest remote weights."""

import importlib.metadata
import json
import platform

from .elicitation import HashingTextEmbedder, SentenceTransformerEmbedder, SSRElicitationProvider
from .execution import TerminalResponseError
from .mathx import canonical_hash
from .providers import OpenAICompatibleProvider, ProviderError
from .schemas import ProviderIdentity


def runtime_identity(semantic=False):
    packages = ["numpy", "pydantic"]
    if semantic:
        packages += ["sentence-transformers", "transformers", "torch", "tokenizers"]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "platform": platform.system(),
            "machine": platform.machine(), "packages": versions}


class PinnedDirectProvider(OpenAICompatibleProvider):
    name = "pinned-direct"

    def __init__(self, options, execution=None, api_key=None):
        self.options = options
        self.pin = options.model_pin
        super().__init__(model=options.model, base_url=options.base_url, api_key=api_key,
            execution=execution, temperature=options.temperature,
            max_retries=options.max_retries, timeout_seconds=options.timeout_seconds,
            history_limit=options.history_limit, max_output_tokens=options.max_output_tokens)
        if self.is_openrouter and self.pin.provider_route is None:
            raise ValueError("OpenRouter studies require an explicit provider route and returned provider identity")
        if not self.is_openrouter and self.pin.provider_route is not None:
            raise ValueError("provider_route is an OpenRouter setting")

    def request_policy(self):
        return {**super().request_policy(), "adapter": "rival.pinned-model.v1",
                "generation_seed": self.options.generation_seed,
                "model_pin": self.pin.model_dump(mode="json"),
                "elicitation": self.options.elicitation.model_dump(mode="json"),
                "runtime": runtime_identity()}

    def identity(self):
        parent = super().identity()
        return parent.model_copy(update={"provider_name": self.name, "provider_version": "1"})

    def _request_payload(self, person, scenario):
        payload = super()._request_payload(person, scenario)
        payload["messages"][0]["content"] = (
            "Estimate the described person's choice probabilities using only the supplied evidence. "
            "Return one JSON object mapping every supplied choice_id to a numeric probability. "
            "Include all choices, make probabilities sum to one, and include no other text."
        )
        payload["seed"] = self.options.generation_seed
        if self.is_openrouter:
            payload["provider"] = {"only": [self.pin.provider_route],
                "allow_fallbacks": False, "require_parameters": True}
        return payload

    def validate_response(self, payload):
        if "error" in payload:
            raise ProviderError("provider returned an error response")
        if payload.get("model") != self.pin.expected_response_model:
            raise TerminalResponseError("returned model does not match its pin")
        if (self.pin.expected_system_fingerprint is not None
                and payload.get("system_fingerprint") != self.pin.expected_system_fingerprint):
            raise TerminalResponseError("returned runtime fingerprint does not match its pin")
        if (self.pin.expected_response_provider is not None
                and payload.get("provider") != self.pin.expected_response_provider):
            raise TerminalResponseError("returned provider does not match its pin")
        if not isinstance(payload.get("id"), str) or not payload["id"].strip():
            raise TerminalResponseError("provider response needs a request ID")
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise ProviderError("expected exactly one model completion")
        choice = choices[0]
        message = choice.get("message")
        if choice.get("finish_reason") != "stop" or not isinstance(message, dict) or message.get("refusal"):
            raise TerminalResponseError("completion was truncated, refused or did not finish normally")


class PinnedTextGenerator(PinnedDirectProvider):
    # The journal slot equals the enclosing prediction provider's slot.
    name = "pinned-ssr"

    def _payload(self, person, scenario):
        payload = super()._request_payload(person, scenario)
        payload.pop("response_format", None)
        payload["messages"][0]["content"] = (
            "Answer the question as the described person using only the supplied evidence. "
            "The scenario includes the alternatives being considered. Express your preference "
            "in one short natural sentence. Do not return probabilities, JSON, or a numbered rating."
        )
        return payload

    @property
    def identity(self):
        return PinnedDirectProvider.identity(self).model_dump(mode="json")

    def validate_response(self, payload):
        super().validate_response(payload)
        text = self._extract_content(payload).strip()
        try:
            json.loads(text)
        except ValueError:
            if text.startswith("```") or not any(character.isalpha() for character in text):
                raise ProviderError("SSR needs natural language, not a code block or numeric rating")
        else:
            raise ProviderError("SSR needs natural language, not JSON or a numeric rating")

    def generate(self, person, scenario):
        if self.execution is None:
            raise ProviderError("text generation requires managed execution")
        return self.execution.generate(self, person, scenario)


class PinnedSSRProvider(SSRElicitationProvider):
    name = "pinned-ssr"

    def __init__(self, options, execution=None, api_key=None):
        self.options = options
        settings = options.elicitation
        pin = settings.embedding
        embedder = (HashingTextEmbedder(pin.dimensions) if pin.kind == "hashing" else
                    SentenceTransformerEmbedder(pin.model, revision=pin.revision, device=pin.device))
        super().__init__(PinnedTextGenerator(options, execution, api_key), embedder,
                         temperature=settings.temperature, epsilon=settings.epsilon)

    def identity(self):
        config = {"generator": self.generator.identity,
                  "elicitation": self.options.elicitation.model_dump(mode="json"),
                  "ssr_adapter": "rival.ssr.v2", "upstream": "86dcd2597c7824e4fd6546b884c5500c43a4b022",
                  "runtime": runtime_identity(self.options.elicitation.embedding.kind == "sentence_transformer")}
        return ProviderIdentity(provider_name=self.name, provider_version="1",
            model=self.options.model, endpoint_sha256=self.generator.identity["endpoint_sha256"],
            configuration_sha256=canonical_hash(config))

    def request_sha256(self, person, scenario):
        return canonical_hash({"payload": self.generator._payload(person, scenario),
                               "rating": self._rater(scenario).identity})

    def prepare_local(self, scenario):
        # Bad packages/checkpoints/anchors fail before paid generation. Saving
        # the request itself is a no-call operation and does not load weights.
        self._rater(scenario)._anchors()


def model_provider(options, execution=None, api_key=None):
    provider = PinnedSSRProvider if options.elicitation.method == "ssr" else PinnedDirectProvider
    return provider(options, execution, api_key)
