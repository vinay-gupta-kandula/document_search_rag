# Enterprise Document Search System with Retrieval-Augmented Generation (RAG)

A production-ready enterprise document search system featuring hybrid retrieval (Dense Vector + BM25 Okapi), cross-encoder re-ranking, and grounded LLM answer synthesis with source attribution.

Built with **FastAPI**, **PostgreSQL**, **Qdrant Vector Database**, **Sentence Transformers**, and containerized with **Docker Compose**.

---

## Architecture

```
                                 Client / Browser
                   [Web UI at http://localhost:8000 / Swagger at /docs]
                                       |
                +----------------------+----------------------+
                |                                             |
         Upload Document                               Ask Question
      (POST /api/documents)                         (POST /api/query)
                |                                             |
                v                                             v
     +--------------------+                       +-----------------------+
     | Ingestion Pipeline |                       |  Hybrid Search Engine |
     +---------+----------+                       +-----------+-----------+
               |                                              |
      [1. Parse Text]                            [1. Dense Vector Search]
      (pypdf / docx / bs4)                       (Cosine similarity in Qdrant)
               |                                              |
     [2. Recursive Chunk]                        [2. Sparse Keyword Search]
     (500 chars / 80 overlap)                    (BM25 Okapi algorithm)
               |                                              |
      [3. Embed Vectors]                         [3. Reciprocal Rank Fusion]
      (all-MiniLM-L6-v2)                         (Score normalization & merge)
               |                                              |
      [4. Upsert Storage]                        [4. Re-Ranking Layer]
         /           \                           (Cross-Encoder ms-marco)
        /             \                                       |
       v               v                                      v
+-------------+ +-------------+                  +-------------------------+
| PostgreSQL  | |   Qdrant    |                  |  LLM Citation Generator |
| (Metadata & | | (Embeddings |                  | (Strict [Source ID: ...] |
|  Statuses)  | |  & Payload) |                  |   source attribution)   |
+-------------+ +-------------+                  +-------------------------+
```

---

## Step-by-Step Implementation Guide

Here is a breakdown of how each layer of the pipeline was designed and built:

### Step 1: Containerization & Service Orchestration
- Configured multi-container orchestration in `docker-compose.yml` with 3 core services:
  - `api`: FastAPI backend running Uvicorn on port `8000`.
  - `db`: PostgreSQL 15 storing metadata in a dedicated volume with a `pg_isready` health check.
  - `vector_db`: Qdrant vector database running on ports `6333` and `6334` with an automated health probe.
- Created `Dockerfile` with multi-stage layer caching that pre-downloads the embedding model (`sentence-transformers/all-MiniLM-L6-v2`) and cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) directly during container build, preventing latency on the first query.

### Step 2: Relational Metadata Schema (`src/models.py`)
- Created a `documents` table in PostgreSQL using SQLAlchemy:
  - `id`: String UUID primary key.
  - `filename`: Original name of the uploaded document.
  - `file_type`: Extension (`pdf`, `docx`, `html`, `txt`).
  - `file_size`: Size in bytes.
  - `status`: Enum (`PENDING`, `PROCESSING`, `COMPLETED`, `FAILED`).
  - `chunk_count`: Total chunks indexed.
  - `error_message`: Captured stack trace if ingestion fails.
  - `created_at` / `updated_at`: Timestamp tracking.

### Step 3: Multi-Format Parsing & Semantic Chunking (`src/parser.py`)
- **Extraction**: Handles binary PDF streams (`pypdf`), Word documents (`python-docx`), web pages (`beautifulsoup4` stripping script and style tags), and UTF-8 / latin-1 fallback text.
- **Recursive Chunking**: Implemented semantic character splitting with configurable hierarchy (`\n\n` -> `\n` -> sentences -> words) using 500-character windows and 80-character overlap to retain context across boundaries.

### Step 4: Vector Store & Dense Embeddings (`src/embeddings.py`, `src/vector_db.py`)
- Uses `sentence-transformers/all-MiniLM-L6-v2` running on CPU to generate 384-dimensional normalized dense vectors.
- Manages collections in Qdrant with cosine distance metrics. Each point stores the vector along with its payload: `chunk_id`, `document_id`, `filename`, `chunk_index`, and raw text chunk.
- Added payload indexing on `document_id` for fast deletions and document filtering.

### Step 5: Asynchronous Ingestion Worker (`src/pipeline.py`)
- Implemented background task handling via FastAPI `BackgroundTasks`.
- When a file is uploaded, the API responds immediately with `202 Accepted` and a unique `document_id`.
- The background thread parses the file, chunks the text, creates dense embeddings, batch upserts to Qdrant, updates chunk counts, and transitions status to `COMPLETED`.

### Step 6: Hybrid Search & Reciprocal Rank Fusion (`src/search.py`)
- **Dense Vector Search**: Queries Qdrant using the query embedding to return top cosine matches.
- **BM25 Keyword Search**: Evaluates exact term matches over raw text chunks using `rank_bm25` (Okapi formula).
- **Reciprocal Rank Fusion (RRF)**: Merges ranked candidate lists using:
  $$\text{RRF Score} = \alpha \cdot \frac{1}{60 + \text{rank}_{\text{dense}}} + (1 - \alpha) \cdot \frac{1}{60 + \text{rank}_{\text{bm25}}}$$
  Adjustable `hybrid_alpha` parameter allows balancing keyword vs semantic weight.

### Step 7: Re-Ranking Layer (`src/reranker.py`)
- Takes top merged candidates and passes `(query, document_chunk)` pairs to a cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) or Cohere Rerank API.
- Generates deep cross-attention relevance scores and sorts results descending.

### Step 8: Answer Generation & Source Attribution (`src/llm.py`)
- Formats retrieved chunks with explicit identifiers: `[Source ID: <chunk_id>]`.
- Sends instructions to the LLM (OpenAI GPT-4o-mini or the built-in intelligent extractive engine) demanding factual answers grounded solely in the provided context.
- Extracts cited source IDs from the generated text and maps them back into a structured `sources` list adhering to the `SourceAttribution` contract.

### Step 9: Web User Interface (`src/static/index.html`)
- Built a standalone browser dashboard served directly at `http://localhost:8000/`.
- Features drag-and-drop document upload, live status tracking, search controls (Top-K and Hybrid Balance sliders), and grounded answer rendering with source metadata cards.

---

## Essential Commands & Setup

### 1. Clone & Prepare Environment
Create your `.env` configuration file from the template:

```bash
cp .env.example .env
```

*(On Windows PowerShell)*:
```powershell
Copy-Item .env.example .env
```

### 2. Build and Start the Entire System
Run everything using a single Docker Compose command:

```bash
docker compose up --build -d
```

### 3. Check Service Health
Verify that the API, PostgreSQL database, and Qdrant are healthy:

```bash
curl http://localhost:8000/health
```

*(On Windows PowerShell, use `curl.exe`)*:
```powershell
curl.exe http://localhost:8000/health
```

Expected output:
```json
{"status":"healthy","database":"healthy","vector_db":"healthy"}
```

### 4. Ingest a Document via CLI
Upload a document (`.pdf`, `.docx`, `.txt`, `.html`, or `.md`):

```bash
curl -X POST "http://localhost:8000/api/documents" -F "file=@README.md"
```

Response:
```json
{
  "document_id": "0074b64e-0f31-41f1-9418-5ca223fd6d55",
  "filename": "README.md",
  "status": "PENDING",
  "message": "Document uploaded successfully. Background ingestion pipeline started."
}
```

### 5. Check Document Ingestion Status
Track the document processing status using its `document_id`:

```bash
curl http://localhost:8000/api/documents/0074b64e-0f31-41f1-9418-5ca223fd6d55
```

Response:
```json
{
  "document_id": "0074b64e-0f31-41f1-9418-5ca223fd6d55",
  "filename": "README.md",
  "file_type": "md",
  "file_size": 9118,
  "status": "COMPLETED",
  "error_message": null,
  "chunk_count": 24,
  "created_at": "2026-10-31T08:20:05.237275",
  "updated_at": "2026-10-31T08:20:07.371954"
}
```

### 6. Perform a Hybrid RAG Query
Send a question to the system:

```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "How does hybrid search work?", "top_k": 3, "hybrid_alpha": 0.5}'
```

Response:
```json
{
  "query": "How does hybrid search work?",
  "answer": "Hybrid search combines dense vector similarity search from Qdrant with sparse keyword matching from BM25. [Source ID: 9c9dc4aa-0b41-4a09-b1a8-b487f6532b48]",
  "sources": [
    {
      "source_id": "9c9dc4aa-0b41-4a09-b1a8-b487f6532b48",
      "document_id": "0074b64e-0f31-41f1-9418-5ca223fd6d55",
      "filename": "README.md",
      "chunk_index": 5,
      "content": "Hybrid search combines dense vector cosine similarity with BM25 Okapi keyword scoring...",
      "relevance_score": 0.942,
      "metadata": {
        "filename": "README.md",
        "document_id": "0074b64e-0f31-41f1-9418-5ca223fd6d55",
        "chunk_index": 5,
        "total_chunks": 24
      }
    }
  ],
  "total_retrieved": 3
}
```

### 7. Delete a Document
Remove metadata from PostgreSQL and all vector chunks from Qdrant:

```bash
curl -X DELETE "http://localhost:8000/api/documents/0074b64e-0f31-41f1-9418-5ca223fd6d55"
```

### 8. View Logs & Container Status
Check container health and logs:

```bash
# Check running containers
docker compose ps

# View API logs
docker compose logs -f api

# View database logs
docker compose logs -f db

# View vector database logs
docker compose logs -f vector_db
```

### 9. Stop Containers
```bash
docker compose down
```

To stop and wipe database volumes:
```bash
docker compose down -v
```

---

## User Interfaces

- **Web Dashboard**: Open `http://localhost:8000/` in any browser to upload files and ask questions visually.
- **Swagger Documentation**: Open `http://localhost:8000/docs` for interactive API documentation and manual endpoint testing.
- **ReDoc**: Open `http://localhost:8000/redoc` for complete schema reference.

---

## Running Automated Tests

A comprehensive test suite is included in `tests/` covering document parsing, chunking boundaries, BM25 tokenization, source attribution formatting, and all FastAPI endpoints:

```bash
# Run all tests
python -m pytest tests/ -v
```

Expected test result:
```text
collected 11 items

tests/test_api_endpoints.py .....                                        [ 45%]
tests/test_parser.py ...                                                 [ 72%]
tests/test_search_and_llm.py ...                                         [100%]

======================= 11 passed in 16.44s =======================
```

---

## Configuration Reference (`.env`)

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `DB_USER` | `rag_user` | PostgreSQL username |
| `DB_PASSWORD` | `rag_password` | PostgreSQL password |
| `DB_NAME` | `rag_metadata_db` | PostgreSQL database name |
| `DB_HOST` | `db` | Database service host |
| `DB_PORT` | `5432` | Database service port |
| `VECTOR_DB_HOST` | `vector_db` | Qdrant host |
| `VECTOR_DB_PORT` | `6333` | Qdrant HTTP port |
| `QDRANT_COLLECTION_NAME` | `document_chunks` | Name of vector collection |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model on HuggingFace |
| `EMBEDDING_DIM` | `384` | Embedding vector dimensions |
| `RERANKER_MODEL_NAME` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model name |
| `COHERE_API_KEY` | *(empty)* | Optional Cohere API key for cloud reranker |
| `LLM_PROVIDER` | `openai` | LLM engine (`openai` or `mock`) |
| `OPENAI_API_KEY` | *(your key)* | OpenAI API key (falls back cleanly if empty) |
| `OPENAI_MODEL_NAME` | `gpt-4o-mini` | Target OpenAI model |
