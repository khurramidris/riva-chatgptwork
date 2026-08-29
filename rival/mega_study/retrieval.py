"""Target-blind, deterministic retrieval from one person's old survey history."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .constants import RETRIEVAL_CONFIG
from .utils import ProtocolError, text_hash


_TOKEN_RE = re.compile(RETRIEVAL_CONFIG["token_pattern"])


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    evidence_id: str
    block: str
    question_id: str
    question: str
    answer: str
    text: str

    def audit_dict(self, score: float, rank: int) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "rank": rank,
            "score": round(float(score), 12),
            "evidence_sha256": text_hash(self.text),
            "block": self.block,
            "question_id": self.question_id,
        }


def _questions(value: Any, block: str = "") -> list[tuple[str, dict[str, Any]]]:
    found: list[tuple[str, dict[str, Any]]] = []
    if isinstance(value, dict):
        next_block = str(value.get("BlockName", block)).strip() or block
        if "QuestionID" in value and isinstance(value.get("Answers"), dict):
            found.append((next_block, value))
        else:
            for child in value.values():
                found.extend(_questions(child, next_block))
    elif isinstance(value, list):
        for child in value:
            found.extend(_questions(child, block))
    return found


def _answer_text(answers: dict[str, Any]) -> str:
    selected = answers.get("SelectedText")
    if isinstance(selected, list):
        values = [str(item).strip() for item in selected if str(item).strip()]
        if values:
            return "; ".join(values)
    elif selected not in (None, ""):
        return str(selected).strip()
    positioned = answers.get("SelectedByPosition")
    if isinstance(positioned, list):
        values = [str(item) for item in positioned]
        if values:
            return "; ".join(values)
    elif positioned not in (None, ""):
        return str(positioned)
    text = answers.get("Text") or answers.get("text")
    if text not in (None, ""):
        return str(text).strip()
    return ""


def evidence_items(persona_json: str) -> list[EvidenceItem]:
    try:
        persona = json.loads(persona_json)
    except json.JSONDecodeError as exc:
        raise ProtocolError("persona_json is not valid JSON") from exc
    items: list[EvidenceItem] = []
    seen: Counter[str] = Counter()
    for block, question in _questions(persona):
        answer = _answer_text(question.get("Answers", {}))
        prompt = " ".join(str(question.get("QuestionText", "")).split())
        if not answer or not prompt:
            continue
        question_id = str(question.get("QuestionID", "unknown"))
        base = f"{block}|{question_id}"
        seen[base] += 1
        evidence_id = f"{question_id}:{seen[base]}:{hashlib.sha256(base.encode()).hexdigest()[:8]}"
        text = f"[{block}] {prompt}\nAnswer: {answer}"
        items.append(EvidenceItem(evidence_id, block, question_id, prompt, answer, text))
    if len(items) < 100:
        raise ProtocolError(
            f"persona exposes only {len(items)} usable historical answers; expected at least 100"
        )
    return items


def demographics_text(persona_json: str) -> str:
    try:
        persona = json.loads(persona_json)
    except json.JSONDecodeError as exc:
        raise ProtocolError("persona_json is not valid JSON") from exc
    blocks = [
        item
        for item in persona
        if isinstance(item, dict)
        and str(item.get("BlockName", "")).strip().casefold() == "demographics"
    ]
    if not blocks:
        raise ProtocolError("persona has no Demographics block")
    questions = [
        item for _, item in _questions(blocks[0], "Demographics") if _answer_text(item["Answers"])
    ]
    if len(questions) != 14:
        raise ProtocolError(
            f"demographics block contains {len(questions)} answered fields; expected 14"
        )
    return "\n\n".join(
        f"{str(question.get('QuestionText', '')).strip()}\n"
        f"Answer: {_answer_text(question['Answers'])}"
        for question in questions
    )


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.casefold())


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def retrieve_evidence(
    persona_json: str,
    query: str,
    *,
    top_k: int | None = None,
    max_chars: int | None = None,
) -> tuple[list[EvidenceItem], list[dict[str, Any]]]:
    """Retrieve with per-person BM25 relevance and MMR diversity.

    Only the answer-free new survey is accepted as a query.  No outcome object
    is accepted by this API, which is part of the leakage boundary.
    """

    items = evidence_items(persona_json)
    top_k = int(top_k if top_k is not None else RETRIEVAL_CONFIG["top_k"])
    max_chars = int(
        max_chars if max_chars is not None else RETRIEVAL_CONFIG["max_evidence_chars"]
    )
    if top_k < 1 or max_chars < 1:
        raise ValueError("retrieval limits must be positive")
    documents = [_tokens(item.text) for item in items]
    query_terms = Counter(_tokens(query))
    n_docs = len(documents)
    avg_length = sum(map(len, documents)) / max(n_docs, 1)
    doc_freq = Counter(term for doc in documents for term in set(doc))
    k1 = float(RETRIEVAL_CONFIG["bm25_k1"])
    b = float(RETRIEVAL_CONFIG["bm25_b"])
    relevance: list[float] = []
    token_sets: list[set[str]] = []
    for document in documents:
        frequencies = Counter(document)
        token_sets.append(set(document))
        score = 0.0
        for term, frequency in frequencies.items():
            query_count = query_terms.get(term, 0)
            if not query_count:
                continue
            inverse = math.log((n_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5) + 1.0)
            denominator = frequency + k1 * (
                1.0 - b + b * len(document) / max(avg_length, 1.0)
            )
            score += query_count * inverse * frequency * (k1 + 1.0) / denominator
        relevance.append(score)
    maximum = max(relevance) if relevance else 0.0
    normalized = [score / maximum if maximum > 0 else 0.0 for score in relevance]
    selected: list[int] = []
    used_chars = 0
    mmr_lambda = float(RETRIEVAL_CONFIG["mmr_lambda"])
    while len(selected) < top_k:
        candidates: list[tuple[float, str, int]] = []
        for index, item in enumerate(items):
            if index in selected:
                continue
            redundancy = max(
                (_jaccard(token_sets[index], token_sets[prior]) for prior in selected),
                default=0.0,
            )
            score = mmr_lambda * normalized[index] - (1.0 - mmr_lambda) * redundancy
            candidates.append((score, hashlib.sha256(item.evidence_id.encode()).hexdigest(), index))
        if not candidates:
            break
        _, _, best = max(candidates, key=lambda value: (value[0], -int(value[1], 16)))
        addition = len(items[best].text) + (2 if selected else 0)
        if selected and used_chars + addition > max_chars:
            break
        if not selected and addition > max_chars:
            break
        selected.append(best)
        used_chars += addition
    chosen = [items[index] for index in selected]
    audit = [
        items[index].audit_dict(normalized[index], rank)
        for rank, index in enumerate(selected, 1)
    ]
    return chosen, audit
