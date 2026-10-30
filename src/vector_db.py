import logging
import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from src.config import settings

logger = logging.getLogger(__name__)

class VectorDBService:
    def __init__(self):
        self.host = settings.VECTOR_DB_HOST
        self.port = settings.VECTOR_DB_PORT
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self.client = QdrantClient(host=self.host, port=self.port, timeout=10)

    def init_collection(self):
        """Ensure collection exists in Qdrant with correct vector dimension."""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            if not exists:
                logger.info(f"Creating Qdrant collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=settings.EMBEDDING_DIM,
                        distance=qmodels.Distance.COSINE
                    )
                )
                # Create payload index on document_id for fast filtering/deletion
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="document_id",
                    field_schema=qmodels.PayloadSchemaType.KEYWORD
                )
                logger.info(f"Collection {self.collection_name} created.")
            else:
                logger.info(f"Qdrant collection {self.collection_name} already exists.")
        except Exception as e:
            logger.error(f"Error initializing Qdrant collection: {e}")
            raise

    def insert_chunks(self, chunks: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]]) -> List[str]:
        """Insert chunk vectors with payload into Qdrant."""
        points = []
        chunk_ids = []
        for chunk_text, emb, meta in zip(chunks, embeddings, metadatas):
            point_id = str(uuid.uuid4())
            chunk_ids.append(point_id)
            payload = {
                "chunk_id": point_id,
                "content": chunk_text,
                **meta
            }
            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector=emb,
                    payload=payload
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        return chunk_ids

    def search_vector(self, query_vector: List[float], limit: int = 10, filter_doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Vector similarity search in Qdrant."""
        query_filter = None
        if filter_doc_id:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="document_id",
                        match=qmodels.MatchValue(value=filter_doc_id)
                    )
                ]
            )

        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter
        )

        results = []
        for hit in search_result:
            results.append({
                "chunk_id": hit.payload.get("chunk_id", str(hit.id)),
                "document_id": hit.payload.get("document_id"),
                "filename": hit.payload.get("filename", ""),
                "chunk_index": hit.payload.get("chunk_index", 0),
                "content": hit.payload.get("content", ""),
                "score": float(hit.score),
                "metadata": {k: v for k, v in hit.payload.items() if k not in ["content", "chunk_id"]}
            })
        return results

    def delete_document_chunks(self, document_id: str):
        """Remove all chunks associated with a document_id."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="document_id",
                            match=qmodels.MatchValue(value=document_id)
                        )
                    ]
                )
            )
        )
        logger.info(f"Deleted vector chunks for document_id: {document_id}")

    def get_all_chunks(self, limit: int = 2000) -> List[Dict[str, Any]]:
        """Retrieve stored chunks for keyword/BM25 indexing."""
        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        results = []
        for r in records:
            results.append({
                "chunk_id": r.payload.get("chunk_id", str(r.id)),
                "document_id": r.payload.get("document_id"),
                "filename": r.payload.get("filename", ""),
                "chunk_index": r.payload.get("chunk_index", 0),
                "content": r.payload.get("content", ""),
                "metadata": {k: v for k, v in r.payload.items() if k not in ["content", "chunk_id"]}
            })
        return results

    def check_health(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

vector_db_service = VectorDBService()
