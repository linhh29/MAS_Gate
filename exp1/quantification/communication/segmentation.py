"""Deterministic segmentation of node outputs into information units Z_i."""

from __future__ import annotations

import re

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？.!?])\s*|\n+")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.+)$", re.MULTILINE)
_NUMERIC_CONCLUSION_RE = re.compile(
    r"^.*\d+.*(?:结论|结果|因此|所以|表明|显示|约为|达到|是|为).*$",
    re.MULTILINE | re.IGNORECASE,
)


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def segment_output(text: str) -> list[str]:
    """
    Split node output o_i into information units Z_i using deterministic rules:
    1. Sentence boundaries (Chinese/English punctuation + newlines)
    2. List items (-, *, numbered)
    3. Numeric conclusion lines
    """
    if not text or not text.strip():
        return []

    units: list[str] = []

    for sentence in _SENTENCE_SPLIT_RE.split(text):
        sentence = sentence.strip()
        if sentence:
            units.append(sentence)

    for match in _LIST_ITEM_RE.finditer(text):
        units.append(match.group(1).strip())

    for match in _NUMERIC_CONCLUSION_RE.finditer(text):
        units.append(match.group(0).strip())

    return _dedupe_preserve_order(units)
