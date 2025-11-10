"""PDF ingestion helpers."""
from __future__ import annotations
from typing import BinaryIO
from io import BytesIO
try:
    from PyPDF2 import PdfReader  # type: ignore
except Exception as e:  # pragma: no cover
    PdfReader = None  # type: ignore

def extract_text_from_pdf(data: bytes) -> str:
    if PdfReader is None:
        raise RuntimeError("PyPDF2 not installed. Add it to requirements.txt and install.")
    reader = PdfReader(BytesIO(data))
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(texts).strip()
