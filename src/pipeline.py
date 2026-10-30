import logging
from src.database import SessionLocal
from src.models import Document, DocumentStatus
from src.parser import parse_document, chunk_text
from src.embeddings import embedding_service
from src.vector_db import vector_db_service

logger = logging.getLogger(__name__)

def process_document_task(document_id: str, file_bytes: bytes, filename: str):
    """
    Background asynchronous pipeline:
    1. Update status to PROCESSING
    2. Parse document text
    3. Semantically chunk text
    4. Generate embeddings
    5. Upsert chunks into Vector Database
    6. Update status to COMPLETED with chunk count (or FAILED on exception)
    """
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            logger.error(f"Document {document_id} not found in database.")
            return

        doc.status = DocumentStatus.PROCESSING
        db.commit()

        # Step 2: Parse document
        raw_text = parse_document(file_bytes, filename)
        if not raw_text.strip():
            raise ValueError("Document appears to be empty or text could not be extracted.")

        # Step 3: Chunking
        chunks = chunk_text(raw_text, chunk_size=500, chunk_overlap=80)
        if not chunks:
            raise ValueError("No text chunks generated.")

        # Step 4: Embedding generation
        embeddings = embedding_service.encode(chunks)

        # Step 5: Store in Vector DB
        metadatas = [
            {
                "document_id": document_id,
                "filename": filename,
                "chunk_index": idx,
                "total_chunks": len(chunks)
            }
            for idx in range(len(chunks))
        ]
        vector_db_service.insert_chunks(chunks, embeddings, metadatas)

        # Step 6: Mark as completed
        doc.status = DocumentStatus.COMPLETED
        doc.chunk_count = len(chunks)
        doc.error_message = None
        db.commit()
        logger.info(f"Document {document_id} processed successfully with {len(chunks)} chunks.")

    except Exception as e:
        logger.exception(f"Error processing document {document_id}: {e}")
        db.rollback()
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.status = DocumentStatus.FAILED
            doc.error_message = str(e)
            db.commit()
    finally:
        db.close()
