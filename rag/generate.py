# rag/generate.py
from __future__ import annotations

from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from openai import OpenAI

from rag.config import Settings, get_openai_client
from rag.retrieve import retrieve_docs


ChatHistory = List[Tuple[str, str]]  # [(user_msg, assistant_msg), ...]


def format_context_with_citations(docs: List[Document]) -> str:
    """
    Adds citation numbers to each retrieved chunk.
    Output format:
      [0] chunk text
      [1] chunk text
    """
    parts: List[str] = []
    for i, doc in enumerate(docs):
        parts.append(f"[{i}] {doc.page_content}")
    return "\n\n".join(parts)


def generate_answer(
    query: str,
    vectorstore: FAISS,
    chat_history: ChatHistory,
    settings: Settings,
    rerank: bool = False,
    client: Optional[OpenAI] = None,
) -> str:
    """
    Complete RAG pipeline:
    - retrieves docs
    - formats context with citations
    - injects chat history
    - queries OpenAI chat model
    - appends to chat_history
    """
    if not query or not query.strip():
        return "Please enter a question."

    docs = retrieve_docs(
        query=query,
        vectorstore=vectorstore,
        settings=settings,
        rerank=rerank,
        client=client,
    )

    if len(docs) == 0:
        return "No relevant information found in the uploaded documents."

    context = format_context_with_citations(docs)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a RAG chatbot. Answer ONLY using the provided document context. "
                "If the answer is not in the context, say you don't know. "
                "Always include citations like [0], [1], etc."
            ),
        }
    ]

    # Add prior conversation
    for user_msg, bot_msg in chat_history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})

    # Current question + context
    messages.append(
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}",
        }
    )

    client = client or get_openai_client(settings)

    resp = client.chat.completions.create(
        model=settings.chat_model,
        messages=messages,
        temperature=settings.temperature,
    )

    answer = (resp.choices[0].message.content or "").strip()
    if not answer:
        answer = "I don't know."

    chat_history.append((query, answer))
    return answer
