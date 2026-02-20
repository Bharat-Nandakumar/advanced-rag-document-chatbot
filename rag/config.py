from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings


@dataclass(frozen=True)
class Settings:
    """
    Centralized app settings.
    Loads OPENAI_API_KEY from environment (.env supported).
    """
    openai_api_key: str
    embedding_model: str = "text-embedding-3-large"
    chat_model: str = "gpt-4o-mini"

    # Retrieval defaults (you can tune later)
    top_k: int = 5
    score_threshold: float = 2.0  # FAISS distance threshold (lower = more similar)
    temperature: float = 0.2


def get_settings() -> Settings:
    """
    Loads environment variables and returns Settings.
    """
    load_dotenv()  # reads .env if present

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Create a .env file with OPENAI_API_KEY=... "
            "or set it in your environment."
        )

    return Settings(openai_api_key=api_key)


def get_openai_client(settings: Settings) -> OpenAI:
    """
    Creates an OpenAI client instance.
    """
    # openai SDK reads OPENAI_API_KEY automatically, but we pass explicitly for clarity.
    return OpenAI(api_key=settings.openai_api_key)


def get_embedding_model(settings: Settings) -> OpenAIEmbeddings:
    """
    Creates the LangChain embeddings wrapper.
    """
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
