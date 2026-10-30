import io
import pytest
from src.parser import parse_document, chunk_text

def test_parse_text_document():
    content = b"Enterprise RAG System architecture overview.\nSection 1: Ingestion."
    parsed = parse_document(content, "test.txt")
    assert "Enterprise RAG System" in parsed
    assert "Section 1: Ingestion." in parsed

def test_parse_html_document():
    html_content = b"<html><head><title>Test</title><style>.a{color:red}</style></head><body><h1>Main Title</h1><p>Paragraph content.</p></body></html>"
    parsed = parse_document(html_content, "doc.html")
    assert "Main Title" in parsed
    assert "Paragraph content." in parsed
    assert "color:red" not in parsed  # Style stripped

def test_chunking_length_and_boundary():
    sample_text = (
        "Paragraph one discusses retrieval-augmented generation and vector databases. "
        "It focuses on embeddings and semantic search.\n\n"
        "Paragraph two discusses BM25 keyword matching and hybrid ranking. "
        "Reciprocal Rank Fusion merges dense and sparse results.\n\n"
        "Paragraph three focuses on cross-encoder rerankers and LLM context synthesis."
    )
    chunks = chunk_text(sample_text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 150  # respects size with small variance around words
