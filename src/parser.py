import io
import re
from typing import List, Dict, Any
from pypdf import PdfReader
from docx import Document as DocxDocument
from bs4 import BeautifulSoup

def parse_document(file_bytes: bytes, filename: str) -> str:
    """Extract clean text content from PDF, DOCX, TXT, HTML, or Markdown files."""
    ext = filename.lower().split(".")[-1] if "." in filename else ""

    if ext == "pdf":
        reader = PdfReader(io.BytesIO(file_bytes))
        pages_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
        return "\n\n".join(pages_text)

    elif ext in ["docx", "doc"]:
        doc = DocxDocument(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)

    elif ext in ["html", "htm"]:
        soup = BeautifulSoup(file_bytes, "html.parser")
        # Remove scripts and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        return soup.get_text(separator="\n\n").strip()

    else:
        # Default text/markdown/csv/etc.
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return file_bytes.decode("latin-1", errors="replace")


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    separators: List[str] = None
) -> List[str]:
    """
    Semantically-aware recursive text chunker.
    Splits text by natural boundaries (paragraphs, sentences, words)
    ensuring each chunk is within chunk_size with chunk_overlap.
    """
    if not separators:
        separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    text = text.strip()
    if not text:
        return []

    def _split_text(current_text: str, current_separators: List[str]) -> List[str]:
        if len(current_text) <= chunk_size or not current_separators:
            return [current_text] if current_text else []

        sep = current_separators[0]
        remaining_separators = current_separators[1:]

        if sep == "":
            # Hard character split
            return [current_text[i:i + chunk_size] for i in range(0, len(current_text), chunk_size - chunk_overlap)]

        splits = current_text.split(sep)
        chunks = []
        current_chunk = ""

        for part in splits:
            if not part.strip() and sep != " ":
                continue

            candidate = f"{current_chunk}{sep}{part}" if current_chunk else part

            if len(candidate) <= chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                if len(part) > chunk_size:
                    # Recurse with finer separator
                    sub_chunks = _split_text(part, remaining_separators)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = part

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    raw_chunks = _split_text(text, separators)

    # Re-apply overlap if necessary to preserve semantic continuity
    merged_chunks = []
    for i, c in enumerate(raw_chunks):
        c_clean = c.strip()
        if not c_clean:
            continue
        merged_chunks.append(c_clean)

    return merged_chunks
