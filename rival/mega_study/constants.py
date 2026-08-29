"""Frozen scientific constants for the Mega-Study development benchmark."""

from __future__ import annotations

from typing import Final


SCHEMA_VERSION: Final = "rival.mega-study.v1"
STUDY_ID: Final = "rival-twin2k-500-mega-development-v1"
SELECTION_SEED: Final = 20260829

DEVELOPMENT_STUDIES: Final = (
    "junk_fees",
    "hiring_algorithms",
    "privacy",
)

# Entire studies are held back before any development result is inspected.
CONFIRMATION_STUDIES: Final = (
    "accuracy_nudges",
    "context_effects",
    "preference_redistribution",
    "quantitative_intuition",
)

VARIANTS: Final = ("generic", "demographics", "full_persona", "rival_retrieval")

VARIANT_LABELS: Final = {
    "generic": "A — Generic",
    "demographics": "B — Demographics",
    "full_persona": "C — Full Persona",
    "rival_retrieval": "D — Rival Retrieval",
}

SYSTEM_INSTRUCTION: Final = (
    "You are an AI assistant. Your task is to answer the 'New Survey Question' "
    "as if you are the person described in the 'Persona Profile' (which consists "
    "of their past survey responses). Remain consistent with the persona's "
    "previous answers and stated traits. Simulate their responses to new "
    "questions while accounting for human cognitive limitations, uncertainty, "
    "and biases. Follow all instructions provided for the new question carefully "
    "regarding the format of your answer."
)

PERSONA_HEADER: Final = "## Persona Profile (This individual's past survey responses):"
SURVEY_HEADER: Final = (
    "## New Survey Question & Instructions "
    "(Please respond as the persona described above):"
)
RETRIEVAL_HEADER: Final = (
    "## Target-relevant evidence retrieved from this individual's earlier surveys:"
)

# OpenRouter's public endpoint metadata on 2026-08-29 reports DeepInfra as a
# parameter-compatible endpoint with a context window far above the audited
# maximum prompt. Pinning an upstream route avoids provider drift across A/B/C/D.
MODEL_CONFIG: Final = {
    "api_provider": "OpenRouter",
    "base_url": "https://openrouter.ai/api/v1/chat/completions",
    "model": "deepseek/deepseek-v4-flash-0731",
    "upstream_provider": "DeepInfra",
    "allow_provider_fallbacks": False,
    "temperature": 0.0,
    "seed": SELECTION_SEED,
    "max_output_tokens": 16384,
    "reasoning_enabled": False,
    "response_format": "json_object",
    "input_cost_per_million_usd": 0.08,
    "output_cost_per_million_usd": 0.18,
    "max_attempts": 3,
    "timeout_seconds": 180,
}

RETRIEVAL_CONFIG: Final = {
    "algorithm": "bm25-mmr-v1",
    "token_pattern": r"[a-z0-9]+",
    "bm25_k1": 1.5,
    "bm25_b": 0.75,
    "mmr_lambda": 0.82,
    "top_k": 24,
    "max_evidence_chars": 16000,
    "always_include_demographics": True,
    "tie_break": "sha256-evidence-id",
}

EXPECTED_OUTCOMES: Final = {
    "junk_fees": ("percent_correct", "fairness_average", "reg_support"),
    "hiring_algorithms": (
        "Q6",
        "Q8",
        "Q9",
        "Q10",
        "Q13",
        "Q14",
        "Q15",
        "Q16",
        *(f"job{job}_item{item}" for job in range(1, 5) for item in range(1, 9)),
    ),
    "privacy": ("PPV",),
}

OUTCOME_RANGES: Final = {
    "junk_fees": {
        "percent_correct": (0.0, 100.0),
        "fairness_average": (1.0, 7.0),
        "reg_support": (1.0, 7.0),
    },
    "hiring_algorithms": {
        "Q6": (1.0, 5.0),
        "Q8": (1.0, 5.0),
        "Q9": (1.0, 4.0),
        "Q10": (1.0, 5.0),
        "Q13": (1.0, 5.0),
        "Q14": (1.0, 4.0),
        "Q15": (1.0, 5.0),
        "Q16": (1.0, 5.0),
        **{
            f"job{job}_item{item}": (1.0, 7.0)
            for job in range(1, 5)
            for item in range(1, 9)
        },
    },
    "privacy": {"PPV": (1.0, 7.0)},
}

FORBIDDEN_PREDICTION_KEYS: Final = {
    "survey_json_with_human_response",
    "ground_truth",
    "human_response",
    "human_outcome",
    "outcome",
    "outcomes",
    "target_response",
    "target_value",
}
