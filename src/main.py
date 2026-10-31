import uuid
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.config import settings
from src.database import get_db, init_db, SessionLocal
from src.models import Document, DocumentStatus
from src.schemas import (
    DocumentUploadResponse,
    DocumentStatusResponse,
    DocumentDeleteResponse,
    QueryRequest,
    QueryResponse,
    HealthResponse,
    SourceAttribution
)
from src.pipeline import process_document_task
from src.vector_db import vector_db_service
from src.search import hybrid_search_engine
from src.llm import llm_service

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure tables exist and Qdrant collection is ready
    logger.info("Initializing Enterprise RAG Services...")
    try:
        init_db(max_retries=15, delay_seconds=2)
    except Exception as e:
        logger.warning(f"Database init warning: {e}")

    try:
        vector_db_service.init_collection()
    except Exception as e:
        logger.warning(f"Vector DB init warning: {e}")

    yield
    logger.info("Shutting down Enterprise RAG Services...")

import os
from fastapi.responses import FileResponse

app = FastAPI(
    title="Enterprise Document Search System with RAG",
    description="Multi-component RAG pipeline featuring document ingestion, chunking, embedding generation, hybrid search (dense vector + BM25), re-ranking, and cited LLM answer generation.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/", include_in_schema=False)
def serve_ui():
    """Serve interactive web UI for document ingestion and hybrid RAG search."""
    ui_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(ui_path):
        return FileResponse(ui_path, media_type="text/html")
    return {"message": "Enterprise Document Search API is running. Visit /docs for the Swagger UI."}

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check(db: Session = Depends(get_db)):
    db_status = "healthy"
    try:
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    vdb_status = "healthy" if vector_db_service.check_health() else "unhealthy"

    return HealthResponse(
        status="healthy" if (db_status == "healthy" and vdb_status == "healthy") else "degraded",
        database=db_status,
        vector_db=vdb_status
    )

@app.post("/api/documents", response_model=DocumentUploadResponse, status_code=status.HTTP_202_ACCEPTED, tags=["Documents"])
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload a document (PDF, DOCX, TXT, HTML).
    Triggers an asynchronous background task to parse, chunk, embed, and index.
    Returns the unique document_id immediately.
    """
    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    doc_id = str(uuid.uuid4())
    filename = file.filename or "unknown"
    ext = filename.split(".")[-1] if "." in filename else ""

    # Store initial record in PostgreSQL
    new_doc = Document(
        id=doc_id,
        filename=filename,
        file_type=ext,
        file_size=len(file_bytes),
        status=DocumentStatus.PENDING,
        chunk_count=0
    )
    db.add(new_doc)
    db.commit()

    # Trigger background ingestion pipeline
    background_tasks.add_task(process_document_task, doc_id, file_bytes, filename)

    return DocumentUploadResponse(
        document_id=doc_id,
        filename=filename,
        status=DocumentStatus.PENDING.value,
        message="Document uploaded successfully. Background ingestion pipeline started."
    )

@app.get("/api/documents/{document_id}", response_model=DocumentStatusResponse, tags=["Documents"])
def get_document_status(document_id: str, db: Session = Depends(get_db)):
    """Check document ingestion status: PENDING, PROCESSING, COMPLETED, or FAILED."""
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document with ID {document_id} not found.")

    return DocumentStatusResponse(
        document_id=doc.id,
        filename=doc.filename,
        file_type=doc.file_type,
        file_size=doc.file_size,
        status=doc.status.value if isinstance(doc.status, DocumentStatus) else doc.status,
        error_message=doc.error_message,
        chunk_count=doc.chunk_count,
        created_at=doc.created_at.isoformat() if doc.created_at else None,
        updated_at=doc.updated_at.isoformat() if doc.updated_at else None,
    )

@app.delete("/api/documents/{document_id}", response_model=DocumentDeleteResponse, tags=["Documents"])
def delete_document(document_id: str, db: Session = Depends(get_db)):
    """Delete document metadata from PostgreSQL and vectors/chunks from Vector DB."""
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document with ID {document_id} not found.")

    # Delete from Vector DB
    try:
        vector_db_service.delete_document_chunks(document_id)
    except Exception as e:
        logger.warning(f"Error removing vector chunks for {document_id}: {e}")

    # Delete from PostgreSQL
    db.delete(doc)
    db.commit()

    return DocumentDeleteResponse(
        document_id=document_id,
        status="DELETED",
        message=f"Document {document_id} and associated vector embeddings successfully deleted."
    )

@app.post("/api/query", response_model=QueryResponse, tags=["RAG Search"])
def rag_query(request: QueryRequest):
    """
    Execute Hybrid Search (Dense Vector + BM25 Keyword) followed by Cross-Encoder Re-Ranking
    and Cited LLM Answer Generation.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Step 1 & 2: Hybrid Search & Re-ranking
    contexts = hybrid_search_engine.search(
        query=request.query,
        top_k=request.top_k,
        hybrid_alpha=request.hybrid_alpha,
        apply_rerank=True
    )

    # Step 3: LLM Answer Generation with Source Attribution
    answer, sources = llm_service.generate_answer(request.query, contexts)

    source_attributions = [
        SourceAttribution(
            source_id=s["source_id"],
            document_id=s["document_id"],
            filename=s["filename"],
            chunk_index=s["chunk_index"],
            content=s["content"],
            relevance_score=s["relevance_score"],
            metadata=s.get("metadata", {})
        )
        for s in sources
    ]

    return QueryResponse(
        query=request.query,
        answer=answer,
        sources=source_attributions,
        total_retrieved=len(source_attributions)
    )
