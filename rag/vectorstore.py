from __future__ import annotations

from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from rag.config import Settings, get_embedding_model


def create_faiss_index(
    chunks: List[str],
    settings: Settings,
    embedding_model: Optional[OpenAIEmbeddings] = None,
) -> FAISS:
    """
    Embed chunks and store them in a FAISS vectorstore.

    Parameters
    ----------
    chunks: list[str]
        Text chunks to embed and index.
    settings: Settings
        App settings (embedding model name + API key).
    embedding_model: Optional[OpenAIEmbeddings]
        If provided, reuse an existing embeddings object.

    Returns
    -------
    FAISS vectorstore
    """
    if not chunks:
        raise ValueError("No chunks provided to create_faiss_index().")

    embeddings = embedding_model or get_embedding_model(settings)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore
