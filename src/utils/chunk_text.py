from __future__ import annotations

import re
from typing import Iterable, List


WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\x00", " ")
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def chunk_text_by_chars(text: str, size: int, overlap: int) -> List[str]:
    if size <= 0:
        raise ValueError("size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    text = clean_text(text)
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        chunk = text[start:end]
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = end - overlap if overlap > 0 else end
        if start < 0:
            start = 0
    return chunks


def merge_lines(lines: Iterable[str]) -> str:
    return clean_text("\n".join([line for line in lines if line is not None]))
