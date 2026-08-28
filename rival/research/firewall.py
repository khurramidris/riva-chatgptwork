from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .datasets import TwinQuestion
from .provenance import stable_hash


_BRACKETED_CHOICES = re.compile(r"\[(?:[^\[\]]|\[[^\]]*\])*\]")
_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_NON_WORD = re.compile(r"[^a-z0-9<>]+")
_SPACES = re.compile(r"\s+")
_CONDITION_WORDS = re.compile(
    r"\b(?:low|high|gain|loss|success|failure|yes|no|certainty|noncertainty|"
    r"absolute|relative|self|other|form|version|condition|group|wave)\b"
)


def canonical_question(text: str) -> str:
    value = _BRACKETED_CHOICES.sub(" ", text.lower())
    value = _NUMBER.sub(" <number> ", value)
    value = _NON_WORD.sub(" ", value)
    return _SPACES.sub(" ", value).strip()


def canonical_block(text: str) -> str:
    value = text.lower().replace("wta/wtp", "valuation")
    value = _CONDITION_WORDS.sub(" ", value)
    value = _NUMBER.sub(" ", value)
    value = re.sub(r"\b[abc]\b", " ", value)
    value = _NON_WORD.sub(" ", value)
    return _SPACES.sub(" ", value).strip()


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


@dataclass(frozen=True)
class FamilySplit:
    item_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    folds: np.ndarray
    manifest_hash: str
    family_count: int

    def assert_no_leakage(self) -> None:
        observed: dict[str, int] = {}
        for family, fold in zip(self.family_ids, self.folds, strict=True):
            if family in observed and observed[family] != int(fold):
                raise AssertionError(f"family {family} crosses folds")
            observed[family] = int(fold)


def _semantic_unions(
    union_find: _UnionFind,
    documents: Sequence[str],
    threshold: float,
) -> None:
    if len(documents) < 2:
        return
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        stop_words="english",
    )
    matrix = vectorizer.fit_transform(documents)
    similarities = cosine_similarity(matrix, dense_output=True)
    for left in range(len(documents)):
        matches = np.flatnonzero(similarities[left, left + 1 :] >= threshold)
        for offset in matches:
            union_find.union(left, left + 1 + int(offset))


def _build_split(
    item_ids: Sequence[str],
    union_find: _UnionFind,
    folds: int,
    seed: str,
) -> FamilySplit:
    if folds < 2:
        raise ValueError("at least two folds are required")
    members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(item_ids)):
        members[union_find.find(index)].append(index)

    def group_key(indices: list[int]) -> tuple[int, str]:
        names = "|".join(sorted(item_ids[index] for index in indices))
        digest = hashlib.sha256(f"{seed}|{names}".encode()).hexdigest()
        return -len(indices), digest

    ordered_groups = sorted(members.values(), key=group_key)
    fold_sizes = [0] * folds
    assigned = np.zeros(len(item_ids), dtype=np.int16)
    family_ids = [""] * len(item_ids)
    for indices in ordered_groups:
        names = sorted(item_ids[index] for index in indices)
        family_id = "fam_" + hashlib.sha256("|".join(names).encode()).hexdigest()[:12]
        minimum = min(fold_sizes)
        candidates = [index for index, size in enumerate(fold_sizes) if size == minimum]
        chooser = int(
            hashlib.sha256(f"{seed}|{family_id}".encode()).hexdigest()[:8], 16
        )
        fold = candidates[chooser % len(candidates)]
        for index in indices:
            assigned[index] = fold
            family_ids[index] = family_id
        fold_sizes[fold] += len(indices)

    manifest = [
        {"item_id": item, "family_id": family, "fold": int(fold)}
        for item, family, fold in sorted(
            zip(item_ids, family_ids, assigned, strict=True), key=lambda row: row[0]
        )
    ]
    result = FamilySplit(
        item_ids=tuple(item_ids),
        family_ids=tuple(family_ids),
        folds=assigned,
        manifest_hash=stable_hash(manifest),
        family_count=len(members),
    )
    result.assert_no_leakage()
    return result


def opinionqa_family_split(
    question_ids: Sequence[str],
    question_texts: Sequence[str],
    folds: int = 5,
    seed: str = "rival-opinionqa-v1",
    similarity_threshold: float = 0.84,
) -> FamilySplit:
    if len(question_ids) != len(question_texts):
        raise ValueError("question identifiers and texts must have equal length")
    canonical = [canonical_question(text) for text in question_texts]
    union_find = _UnionFind(len(question_ids))
    exact: dict[str, int] = {}
    for index, text in enumerate(canonical):
        if text in exact:
            union_find.union(index, exact[text])
        else:
            exact[text] = index
    _semantic_unions(union_find, canonical, similarity_threshold)
    return _build_split(question_ids, union_find, folds, seed)


def twin_family_split(
    questions: Sequence[TwinQuestion],
    folds: int = 5,
    seed: str = "rival-twin2k-v1",
    similarity_threshold: float = 0.74,
) -> FamilySplit:
    item_ids = [question.column for question in questions]
    union_find = _UnionFind(len(questions))
    by_qid: dict[str, int] = {}
    by_block: dict[str, int] = {}
    for index, question in enumerate(questions):
        qid = re.sub(r"_\d+$", "", question.question_id.upper())
        if qid in by_qid:
            union_find.union(index, by_qid[qid])
        else:
            by_qid[qid] = index
        block = canonical_block(question.block)
        if block and block in by_block:
            union_find.union(index, by_block[block])
        elif block:
            by_block[block] = index
    # Known counterbalanced siblings in the released instrument. Keeping this
    # explicit makes the conservative exclusion auditable instead of relying
    # only on a similarity threshold.
    for sibling_set in (
        {"QID183", "QID184"},
        {"QID189", "QID190", "QID191"},
        {"QID194", "QID195"},
    ):
        sibling_indices = [
            index
            for index, question in enumerate(questions)
            if question.question_id.upper() in sibling_set
        ]
        for index in sibling_indices[1:]:
            union_find.union(sibling_indices[0], index)
    documents = [
        f"{canonical_block(question.block)} {canonical_question(question.text)}"
        for question in questions
    ]
    _semantic_unions(union_find, documents, similarity_threshold)
    return _build_split(item_ids, union_find, folds, seed)
