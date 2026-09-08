"""Explicit model identities and elicitation settings for prospective studies."""

import re
from typing import Literal

from pydantic import Field, StrictInt, field_validator, model_validator
from .schemas import StrictModel


class ModelPin(StrictModel):
    kind: Literal["hosted_endpoint", "checkpoint"]
    revision: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    expected_response_model: str = Field(min_length=1)
    expected_system_fingerprint: str | None = None
    provider_route: str | None = None
    expected_response_provider: str | None = None

    @field_validator("revision", "reference", "expected_response_model",
                     "expected_system_fingerprint", "provider_route", "expected_response_provider")
    @classmethod
    def explicit_text(cls, value):
        if value is not None and (not value.strip() or value != value.strip()):
            raise ValueError("model identities must be nonblank with no surrounding whitespace")
        return value

    @model_validator(mode="after")
    def immutable_checkpoint(self):
        if self.revision.casefold() in {"main", "master", "latest", "unresolved", "operator-must-pin"}:
            raise ValueError("pin a specific model revision, not a mutable alias")
        if self.kind == "checkpoint":
            if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", self.revision):
                raise ValueError("checkpoint revision needs a full commit or artifact SHA")
            if self.expected_system_fingerprint is None:
                raise ValueError("checkpoint execution needs a checked runtime fingerprint")
        if (self.provider_route is None) != (self.expected_response_provider is None):
            raise ValueError("a provider route needs the expected returned provider identity")
        return self


class EmbeddingPin(StrictModel):
    kind: Literal["sentence_transformer", "hashing"] = "sentence_transformer"
    model: str | None = None
    revision: str | None = None
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    dimensions: StrictInt = Field(default=512, ge=32, le=65536)

    @model_validator(mode="after")
    def pinned_model(self):
        if self.kind == "sentence_transformer":
            if (not self.model or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", self.model)
                    or not re.fullmatch(r"[0-9a-f]{40}", self.revision or "")):
                raise ValueError("sentence embeddings need a Hub model ID and full immutable revision")
        elif self.model is not None or self.revision is not None or self.device != "cpu":
            raise ValueError("the generated-data hashing embedder has no model, revision or accelerator")
        return self


class ElicitationSpec(StrictModel):
    method: Literal["direct", "ssr"] = "direct"
    embedding: EmbeddingPin | None = None
    temperature: float = Field(default=1.0, ge=0, le=100, allow_inf_nan=False)
    epsilon: float = Field(default=1e-8, ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def matching_method(self):
        if (self.method == "ssr") != (self.embedding is not None):
            raise ValueError("SSR requires an explicit embedding pin; direct elicitation has none")
        if self.method == "direct" and (self.temperature != 1.0 or self.epsilon != 1e-8):
            raise ValueError("SSR scaling settings do not apply to direct elicitation")
        return self
