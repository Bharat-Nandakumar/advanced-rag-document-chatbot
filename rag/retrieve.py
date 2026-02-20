from __future__ import annotations

import re
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from openai import OpenAI

from rag.config import Settings, get_openai_client


def retrieve_docs(
    query: str,
    vectorstore: FAISS,
    settings: Settings,
    k: Optional[int] = None,
    score_threshold: Optional[float] = None,
    rerank: bool = False,
    client: Optional[OpenAI] = None,
) -> List[Document]:
    """
    Retrieves top-k relevant chunks from FAISS.
    Applies score filtering and optional LLM-based re-ranking.

    Note: FAISS similarity_search_with_score returns (Document, score)
    where *lower score* means *more similar* for typical distance metrics.
    """
    if not query or not query.strip():
        return []

    k = k if k is not None else settings.top_k
    score_threshold = score_threshold if score_threshold is not None else settings.score_threshold

    docs_and_scores = vectorstore.similarity_search_with_score(query, k=k)

    filtered_docs: List[Document] = []
    for doc, score in docs_and_scores:
        if score <= score_threshold:
            filtered_docs.append(doc)

    if not rerank or len(filtered_docs) <= 1:
        return filtered_docs

    client = client or get_openai_client(settings)
    return _llm_rerank(query=query, docs=filtered_docs, client=client, model=settings.chat_model)


def _llm_rerank(query: str, docs: List[Document], client: OpenAI, model: str) -> List[Document]:
    """
    Uses an LLM to re-rank already-filtered chunks.
    Returns docs in descending relevance order.
    """
    ranking_prompt = (
        "Rank the following chunks by relevance to the query.\n"
        "Return ONLY a list of chunk numbers in descending relevance order.\n\n"
        f"Query: {query}\n\n"
    )

    for i, doc in enumerate(docs):
        ranking_prompt += f"[Chunk {i}]: {doc.page_content}\n\n"

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a ranking engine."},
            {"role": "user", "content": ranking_prompt},
        ],
        temperature=0.0,
    )

    text = resp.choices[0].message.content or ""
    order = [int(x) for x in re.findall(r"\d+", text)]

    # Keep only valid indices; preserve order returned by LLM
    reranked = [docs[i] for i in order if 0 <= i < len(docs)]

    # Fallback: if LLM returns nothing usable, keep original order
    return reranked if reranked else docs
