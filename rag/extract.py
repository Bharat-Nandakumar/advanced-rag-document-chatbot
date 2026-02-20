from __future__ import annotations

import io
import os
from typing import Optional, Union

from pypdf import PdfReader
from docx import Document


def _read_file_bytes(file_obj: Union[str, bytes, object], filename: Optional[str] = None) -> tuple[bytes, str]:
    """
    Normalize inputs into (file_bytes, resolved_filename).

    Supports:
    - str path (Gradio NamedString path)
    - object with .name and maybe .read() (Gradio upload temp object)
    - raw bytes + filename (debug/testing)
    """
    # CASE 1 → file is a path string
    if isinstance(file_obj, str):
        resolved_filename = file_obj
        with open(file_obj, "rb") as f:
            return f.read(), resolved_filename

    # CASE 2/3 → file is an uploaded object with .name
    if hasattr(file_obj, "name"):
        resolved_filename = getattr(file_obj, "name")
        # If it has .read(), prefer that (works for many file-like wrappers)
        if hasattr(file_obj, "read"):
            file_bytes = file_obj.read()
            # Sometimes subsequent reads return empty; handle by falling back to disk
            if file_bytes:
                return file_bytes, resolved_filename

        # Fall back: read from disk path in .name
        with open(resolved_filename, "rb") as f:
            return f.read(), resolved_filename

    # CASE 4 → raw bytes with explicit filename
    if isinstance(file_obj, (bytes, bytearray)):
        if not filename:
            raise ValueError("When passing raw bytes, you must also pass filename=...")
        return bytes(file_obj), filename

    raise TypeError(f"Unsupported file input type: {type(file_obj)}")


def extract_text_from_file(file_obj: Union[str, bytes, object], filename: Optional[str] = None) -> str:
    """
    Extract text from PDF / DOCX / TXT.

    Parameters
    ----------
    file_obj:
        - Path string
        - Gradio upload wrapper (has .name, optional .read)
        - Raw bytes
    filename:
        Required only when file_obj is raw bytes.

    Returns
    -------
    str: extracted text (may be empty if extraction fails)
    """
    file_bytes, resolved_filename = _read_file_bytes(file_obj, filename=filename)
    ext = os.path.splitext(resolved_filename)[1].lower()
    stream = io.BytesIO(file_bytes)

    if ext == ".pdf":
        return _extract_pdf_text(stream)

    if ext == ".docx":
        return _extract_docx_text(stream)

    if ext == ".txt":
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            # fallback for messy text files
            return file_bytes.decode("latin-1", errors="ignore")

    return ""


def _extract_pdf_text(stream: io.BytesIO) -> str:
    reader = PdfReader(stream)
    parts: list[str] = []
    for page in reader.pages:
        try:
            extracted = page.extract_text()
            if extracted:
                parts.append(extracted)
        except Exception:
            # Keep going even if a page fails
            continue
    return "\n".join(parts)


def _extract_docx_text(stream: io.BytesIO) -> str:
    doc = Document(stream)
    return "\n".join(p.text for p in doc.paragraphs if p.text)
