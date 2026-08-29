"""Frozen A/B/C/D prompt construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .constants import (
    PERSONA_HEADER,
    RETRIEVAL_HEADER,
    SURVEY_HEADER,
    SYSTEM_INSTRUCTION,
    VARIANTS,
)
from .retrieval import retrieve_evidence
from .utils import ProtocolError, text_hash


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    variant: str
    system: str
    user: str
    prompt_sha256: str
    context_chars: int
    retrieval_audit: tuple[dict[str, Any], ...] = ()


def template_payloads() -> dict[str, dict[str, str]]:
    return {
        "generic": {
            "system": SYSTEM_INSTRUCTION,
            "user": f"{SURVEY_HEADER}\n{{survey_text}}",
        },
        "demographics": {
            "system": SYSTEM_INSTRUCTION,
            "user": f"{PERSONA_HEADER}\n{{demographics}}\n\n{SURVEY_HEADER}\n{{survey_text}}",
        },
        "full_persona": {
            "system": SYSTEM_INSTRUCTION,
            "user": f"{PERSONA_HEADER}\n{{persona_text}}\n\n{SURVEY_HEADER}\n{{survey_text}}",
        },
        "rival_retrieval": {
            "system": SYSTEM_INSTRUCTION,
            "user": (
                f"{PERSONA_HEADER}\n{{demographics}}\n\n{RETRIEVAL_HEADER}\n"
                f"{{retrieved_evidence}}\n\n{SURVEY_HEADER}\n{{survey_text}}"
            ),
        },
    }


def template_hashes() -> dict[str, str]:
    return {name: text_hash(value["system"] + "\n\n" + value["user"]) for name, value in template_payloads().items()}


def render_prompt(case: dict[str, Any], persona: dict[str, Any], variant: str) -> RenderedPrompt:
    if variant not in VARIANTS:
        raise ProtocolError(f"unknown Mega-Study variant {variant!r}")
    survey_text = str(case.get("survey_text", ""))
    if not survey_text:
        raise ProtocolError("prediction case has empty survey_text")
    templates = template_payloads()
    audit: tuple[dict[str, Any], ...] = ()
    if variant == "generic":
        user = templates[variant]["user"].format(survey_text=survey_text)
    elif variant == "demographics":
        user = templates[variant]["user"].format(
            demographics=str(persona["demographics"]), survey_text=survey_text
        )
    elif variant == "full_persona":
        user = templates[variant]["user"].format(
            persona_text=str(persona["persona_text"]), survey_text=survey_text
        )
    else:
        items, retrieval_audit = retrieve_evidence(
            str(persona["persona_json"]), survey_text
        )
        evidence = "\n\n".join(item.text for item in items)
        user = templates[variant]["user"].format(
            demographics=str(persona["demographics"]),
            retrieved_evidence=evidence,
            survey_text=survey_text,
        )
        audit = tuple(retrieval_audit)
    digest = text_hash(SYSTEM_INSTRUCTION + "\n\n" + user)
    return RenderedPrompt(
        variant=variant,
        system=SYSTEM_INSTRUCTION,
        user=user,
        prompt_sha256=digest,
        context_chars=len(SYSTEM_INSTRUCTION) + len(user),
        retrieval_audit=audit,
    )
