# Enterprise Document Search System with Retrieval-Augmented Generation (RAG)

A production-grade, containerized Enterprise Document Search and Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **PostgreSQL**, **Qdrant Vector Database**, **Sentence Transformers**, **BM25**, and **Cross-Encoder / Cohere Re-rankers**.

---

## 🌟 Architectural Overview

```
                      +---------------------------------------+
                      |               Client                  |
                      +-------------------+-------------------+
                                          |
                        POST /api/query   |   POST /api/documents
                                          v
                      +---------------------------------------+
                      |              FastAPI                  |
                      |            Application                |
                      +---------+-------------------+---------+
                                |                   |
                 +--------------+                   +---------------+
                 v                                                  v
     [Hybrid Search Engine]                            [Document Ingestion Pipeline]
     +--------------------------+                      +--------------------------+
     | 1. Dense Vector Search   |                      | 1. Unstructured Parsing  |
     |    (Qdrant Cosine)       |                      |    (PDF, DOCX, HTML, TXT)|
     | 2. Sparse Keyword Search |                      | 2. Recursive Chunking    |
     |    (BM25 Okapi)          |                      | 3. Dense Embedding       |
     | 3. Reciprocal Rank       |                      |    (all-MiniLM-L6-v2)    |
     |    Fusion (RRF)          |                      | 4. Vector DB Indexing    |
     | 4. Re-Ranking            |                      | 5. PostgreSQL Metadata   |
     |    (Cross-Encoder/Cohere)|                      +--------------------------+
     +------------+-------------+                                   |
                  |                                                 v
                  v                                    +--------------------------+
     [LLM Citation Synthesis]                          |       PostgreSQL         |
     +--------------------------+                      |    (Documents Table)     |
     | Prompt Context Injection |                      +--------------------------+
     | [Source ID: ...] mapping |                                   |
     +------------+-------------+                                   v
                  |                                    +--------------------------+
                  v                                    |   Qdrant Vector DB       |
          Structured Response                          | (document_chunks payload)|
                                                       +--------------------------+
```

---

## 🚀 Key Features

1. **Multi-Format Ingestion**: Ingests `.pdf`, `.docx`, `.html`, and plain text documents with metadata and error handling.
2. **Recursive Semantic Chunking**: Splits large documents into coherent sections maintaining semantic boundaries and configurable overlap.
3. **Dense Vector Embeddings**: Uses CPU-optimized `sentence-transformers/all-MiniLM-L6-v2` generating 384-dimensional dense vectors.
4. **Relational Metadata Store**: Stores document processing states (`PENDING`, `PROCESSING`, `COMPLETED`, `FAILED`), chunk counts, file sizes, and timestamps in PostgreSQL.
5. **High-Performance Vector Database**: Qdrant stores chunks and vector indices with payload filtering by `document_id`.
6. **Hybrid Retrieval**: Combines semantic cosine similarity search with BM25 Okapi keyword matching using Reciprocal Rank Fusion (RRF).
7. **Re-Ranking Step**: Re-ranks top retrieved contexts using a Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) or Cohere Rerank API.
8. **Factual LLM Answer Generation with Strict Citations**: Synthesizes verified answers with references in the exact format `[Source ID: <chunk_id>]` mapped back to document metadata.
9. **Fully Containerized**: Ready to start with `docker-compose up` including automatic health checks on all services.

---

## 📁 Repository Structure

```
.
├── .env.example              # Environment variables template
├── .env                      # Local environment configuration
├── docker-compose.yml        # Multi-container orchestration (api, db, vector_db)
├── Dockerfile                # Production container for API service
├── requirements.txt          # Python dependencies
├── submission.json           # Project metadata & specifications
├── README.md                 # System documentation & API guide
├── src/
│   ├── config.py             # Settings via Pydantic
│   ├── database.py           # PostgreSQL engine and session management
│   ├── models.py             # SQLAlchemy models (Document, DocumentStatus)
│   ├── schemas.py            # Pydantic request & response schemas
│   ├── parser.py             # Document parsers & recursive chunker
│   ├── embeddings.py         # Sentence Transformers embedding service
│   ├── vector_db.py          # Qdrant client & collection management
│   ├── reranker.py           # Cross-Encoder & Cohere re-ranking
│   ├── search.py             # Hybrid Search (Dense Vector + BM25 + RRF)
│   ├── pipeline.py           # Background ingestion worker
│   ├── llm.py                # LLM answer synthesis & citation extraction
│   └── main.py               # FastAPI application & REST endpoints
└── tests/
    ├── test_parser.py        # Parser & chunker tests
    └── test_search_and_llm.py # Hybrid search & citation tests
```

---

## ⚙️ Setup and Installation

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) (v20.10+)
- [Docker Compose](https://docs.docker.com/compose/) (v2.0+)

### Quick Start (Single Command)

1. Clone repository and navigate to root:
   ```bash
   cd document_search_rag
   ```

2. Copy environment file:
   ```bash
   cp .env.example .env
   ```

3. Launch all services:
   ```bash
   docker-compose up --build
   ```

4. Verify health:
   ```bash
   curl http://localhost:8000/health
   ```
   Interactive Swagger docs are accessible at: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📡 API Specification & Usage

### 1. Health Check
- **Endpoint**: `GET /health`
- **Response**:
```json
{
  "status": "healthy",
  "database": "healthy",
  "vector_db": "healthy"
}
```

### 2. Upload Document
- **Endpoint**: `POST /api/documents`
- **Content-Type**: `multipart/form-data`
- **Request**:
```bash
curl -X POST "http://localhost:8000/api/documents" \
  -F "file=@example_policy.pdf"
```
- **Response (202 Accepted)**:
```json
{
  "document_id": "c71e2858-a720-4131-b856-11f845778809",
  "filename": "example_policy.pdf",
  "status": "PENDING",
  "message": "Document uploaded successfully. Background ingestion pipeline started."
}
```

### 3. Check Document Processing Status
- **Endpoint**: `GET /api/documents/{document_id}`
- **Response**:
```json
{
  "document_id": "c71e2858-a720-4131-b856-11f845778809",
  "filename": "example_policy.pdf",
  "file_type": "pdf",
  "file_size": 15420,
  "status": "COMPLETED",
  "error_message": null,
  "chunk_count": 8,
  "created_at": "2026-10-30T13:00:00",
  "updated_at": "2026-10-30T13:00:03"
}
```

### 4. Delete Document
- **Endpoint**: `DELETE /api/documents/{document_id}`
- **Response**:
```json
{
  "document_id": "c71e2858-a720-4131-b856-11f845778809",
  "status": "DELETED",
  "message": "Document c71e2858-a720-4131-b856-11f845778809 and associated vector embeddings successfully deleted."
}
```

### 5. Hybrid RAG Query
- **Endpoint**: `POST /api/query`
- **Request Body**:
```json
{
  "query": "What are the data encryption requirements for production servers?",
  "top_k": 3,
  "hybrid_alpha": 0.5
}
```
- **Response (200 OK)**:
```json
{
  "query": "What are the data encryption requirements for production servers?",
  "answer": "All production data must be encrypted both at rest using AES-256 and in transit using TLS 1.3. [Source ID: 4976722c-aef6-4191-be71-a75d50325fa1]",
  "sources": [
    {
      "source_id": "4976722c-aef6-4191-be71-a75d50325fa1",
      "document_id": "c71e2858-a720-4131-b856-11f845778809",
      "filename": "example_policy.pdf",
      "chunk_index": 2,
      "content": "All production database storage volumes must enforce AES-256 encryption at rest. Network communications across services must use TLS 1.3.",
      "relevance_score": 0.942,
      "metadata": {
        "filename": "example_policy.pdf",
        "document_id": "c71e2858-a720-4131-b856-11f845778809",
        "chunk_index": 2,
        "total_chunks": 8
      }
    }
  ],
  "total_retrieved": 1
}
```

---

## 🧪 Running Automated Tests

To run the automated test suite locally:
```bash
pytest -v tests/
```
