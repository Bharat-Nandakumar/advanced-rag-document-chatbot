# app/ui.py
from __future__ import annotations

from typing import List, Tuple, Optional

import gradio as gr
from langchain_community.vectorstores import FAISS

from rag.config import Settings, get_settings
from rag.extract import extract_text_from_file
from rag.chunking import chunk_text
from rag.vectorstore import create_faiss_index
from rag.generate import generate_answer, ChatHistory

VECTORSTORE = None
CHAT_HISTORY: ChatHistory = []


def build_app(settings: Optional[Settings] = None) -> gr.Blocks:
    """
    Builds and returns the Gradio Blocks app.
    """
    settings = settings or get_settings()

    def process_documents(files) -> str:
        global VECTORSTORE, CHAT_HISTORY
        CHAT_HISTORY = []

        if not files or len(files) == 0:
            return "⚠️ Please upload at least one document."

        all_text_parts: List[str] = []
        for f in files:
            text = extract_text_from_file(f)
            if text and text.strip():
                all_text_parts.append(text)

        all_text = "\n".join(all_text_parts).strip()
        if not all_text:
            return "⚠️ Could not extract any text from the uploaded documents."

        chunks = chunk_text(all_text, chunk_size=400, chunk_overlap=50)
        if not chunks:
            return "⚠️ Chunking produced no chunks (empty text)."

        VECTORSTORE = create_faiss_index(chunks, settings=settings)
        return "✅ Documents processed and knowledge base created successfully!"


    def chat_with_rag(query: str, rerank: bool) -> str:
        global VECTORSTORE, CHAT_HISTORY

        if VECTORSTORE is None:
            return "⚠️ Please upload and process documents first."

        return generate_answer(
            query=query,
            vectorstore=VECTORSTORE,
            chat_history=CHAT_HISTORY,
            settings=settings,
            rerank=rerank,
        )


    with gr.Blocks() as app:
        gr.Markdown("# 📘 **Advanced RAG Chatbot**")
        gr.Markdown("Upload documents → Build knowledge base → Ask questions!")

        with gr.Accordion("Upload Documents to Build Knowledge Base", open=True):
            files = gr.File(file_count="multiple", file_types=[".pdf", ".txt", ".docx"])
            process_btn = gr.Button("Process Documents")
            status = gr.Textbox(label="Status")

            process_btn.click(
                fn=process_documents,
                inputs=[files],
                outputs=[status],
            )

        gr.Markdown("### 💬 Chat With Your Documents")
        rerank_toggle = gr.Checkbox(label="Enable Reranking (Better but slower)", value=False)

        user_query = gr.Textbox(label="Ask a question")
        answer_box = gr.Markdown(label="Answer")
        send_btn = gr.Button("Send")

        send_btn.click(
            fn=chat_with_rag,
            inputs=[user_query, rerank_toggle],
            outputs=[answer_box],
        )

    return app
