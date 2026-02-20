from __future__ import annotations

from typing import List
import tiktoken


def chunk_text(text: str, chunk_size: int = 400, chunk_overlap: int = 50) -> List[str]:
    """
    Token-based chunking using tiktoken (cl100k_base) for more consistent retrieval.

    - chunk_size: number of tokens per chunk
    - chunk_overlap: overlapping tokens between consecutive chunks
    """
    if not text or not text.strip():
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    chunks: List[str] = []
    start = 0

    step = chunk_size - chunk_overlap
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))
        start += step

    return chunks