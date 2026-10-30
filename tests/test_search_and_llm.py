import pytest
from unittest.mock import MagicMock, patch
from src.schemas import QueryRequest
from src.search import tokenize

def test_tokenize():
    tokens = tokenize("Enterprise RAG System: Vector & BM25!")
    assert "enterprise" in tokens
    assert "rag" in tokens
    assert "vector" in tokens
    assert "bm25" in tokens

def test_query_schema_validation():
    req = QueryRequest(query="What is RAG?", top_k=3, hybrid_alpha=0.7)
    assert req.query == "What is RAG?"
    assert req.top_k == 3
    assert req.hybrid_alpha == 0.7

def test_llm_source_citation_format():
    from src.llm import LLMService
    service = LLMService()
    contexts = [
        {
            "chunk_id": "c123",
            "document_id": "d456",
            "filename": "policy.pdf",
            "chunk_index": 0,
            "content": "All enterprise confidential documents must be encrypted at rest.",
            "relevance_score": 0.95
        }
    ]
    answer, sources = service.generate_answer("How should documents be encrypted?", contexts)
    assert "[Source ID: c123]" in answer
    assert len(sources) == 1
    assert sources[0]["source_id"] == "c123"
    assert sources[0]["filename"] == "policy.pdf"
