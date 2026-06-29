"""Text normalization and similarity helpers.

Provides statement cleanup, SHA-256 hashing, tokenization, token n-grams, and Jaccard similarity
used mostly by deduplication and stable identity generation."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

_SPACE_RE = re.compile(r"\s+")
_PUNCT_SPACE_RE = re.compile(r"\s*([,.;:!?()\[\]{}<>+\-*/=])\s*")


def normalize_statement(text: str | None) -> str:
    if not text:
        return ""
    value = text.lower().replace("\r\n", "\n").replace("\r", "\n")
    boilerplate = [
        "this problem is from codeforces.",
        "for each test case,",
    ]
    for snippet in boilerplate:
        value = value.replace(snippet, " ")
    value = _PUNCT_SPACE_RE.sub(r" \1 ", value)
    value = _SPACE_RE.sub(" ", value)
    return value.strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+|[^\w\s]", text.lower())


def token_ngrams(text: str, n: int = 5) -> set[tuple[str, ...]]:
    seq = tokens(text)
    if not seq:
        return set()
    if len(seq) < n:
        return {tuple(seq)}
    return {tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)}


def jaccard(left: Iterable[object], right: Iterable[object]) -> float:
    a = set(left)
    b = set(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
