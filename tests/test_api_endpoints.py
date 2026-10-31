import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from src.main import app
from src.database import get_db
from src.models import Document, DocumentStatus

client = TestClient(app)

def test_health_endpoint_mocked():
    # Mock DB session and vector_db health
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    
    with patch("src.main.vector_db_service.check_health", return_value=True):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["database"] == "healthy"
        assert data["vector_db"] == "healthy"
    
    app.dependency_overrides.clear()

def test_upload_document_endpoint():
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db

    with patch("src.main.process_document_task") as mock_task:
        file_content = b"Enterprise search document for RAG ingestion test."
        response = client.post(
            "/api/documents",
            files={"file": ("test_doc.txt", file_content, "text/plain")}
        )
        assert response.status_code == 202
        data = response.json()
        assert "document_id" in data
        assert data["filename"] == "test_doc.txt"
        assert data["status"] == "PENDING"
        assert mock_db.add.called
        assert mock_db.commit.called

    app.dependency_overrides.clear()

def test_get_document_status_endpoint():
    mock_db = MagicMock()
    mock_doc = MagicMock()
    mock_doc.id = "doc-123"
    mock_doc.filename = "report.pdf"
    mock_doc.file_type = "pdf"
    mock_doc.file_size = 1024
    mock_doc.status = DocumentStatus.COMPLETED
    mock_doc.error_message = None
    mock_doc.chunk_count = 5
    mock_doc.created_at = None
    mock_doc.updated_at = None
    mock_db.query.return_value.filter.return_value.first.return_value = mock_doc

    app.dependency_overrides[get_db] = lambda: mock_db

    response = client.get("/api/documents/doc-123")
    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == "doc-123"
    assert data["status"] == "COMPLETED"
    assert data["chunk_count"] == 5

    app.dependency_overrides.clear()

def test_delete_document_endpoint():
    mock_db = MagicMock()
    mock_doc = MagicMock()
    mock_doc.id = "doc-to-delete"
    mock_db.query.return_value.filter.return_value.first.return_value = mock_doc

    app.dependency_overrides[get_db] = lambda: mock_db

    with patch("src.main.vector_db_service.delete_document_chunks") as mock_del_vec:
        response = client.delete("/api/documents/doc-to-delete")
        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == "doc-to-delete"
        assert data["status"] == "DELETED"
        assert mock_del_vec.called
        assert mock_db.delete.called
        assert mock_db.commit.called

    app.dependency_overrides.clear()

def test_query_rag_endpoint():
    sample_context = [
        {
            "chunk_id": "chunk-101",
            "document_id": "doc-abc",
            "filename": "security.pdf",
            "chunk_index": 0,
            "content": "All databases must enforce encryption at rest and in transit.",
            "relevance_score": 0.98,
            "metadata": {"section": "policy"}
        }
    ]

    with patch("src.main.hybrid_search_engine.search", return_value=sample_context):
        response = client.post(
            "/api/query",
            json={"query": "How is database security enforced?", "top_k": 3, "hybrid_alpha": 0.5}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "How is database security enforced?"
        assert len(data["sources"]) >= 1
        assert data["sources"][0]["source_id"] == "chunk-101"
        assert data["sources"][0]["document_id"] == "doc-abc"
        assert "[Source ID: chunk-101]" in data["answer"]
