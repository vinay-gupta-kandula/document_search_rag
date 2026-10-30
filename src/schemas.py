from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

class DocumentUploadResponse(BaseModel):
    document_id: str = Field(..., description="Unique ID for the uploaded document")
    filename: str
    status: str
    message: str

class DocumentStatusResponse(BaseModel):
    document_id: str
    filename: str
    file_type: Optional[str] = None
    file_size: Optional[int] = None
    status: str
    error_message: Optional[str] = None
    chunk_count: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class DocumentDeleteResponse(BaseModel):
    document_id: str
    status: str
    message: str

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User search query")
    top_k: int = Field(5, ge=1, le=50, description="Number of results to retrieve")
    hybrid_alpha: float = Field(0.5, ge=0.0, le=1.0, description="Weight between vector search (1.0) and BM25 (0.0)")

class SourceAttribution(BaseModel):
    source_id: str = Field(..., description="Unique ID of chunk / source")
    document_id: str = Field(..., description="ID of source document")
    filename: str = Field(..., description="Source document filename")
    chunk_index: int = Field(..., description="Index of the chunk in document")
    content: str = Field(..., description="Text content of the retrieved chunk")
    relevance_score: float = Field(..., description="Re-ranked or retrieval relevance score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional chunk metadata")

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceAttribution]
    total_retrieved: int

class HealthResponse(BaseModel):
    status: str
    database: str
    vector_db: str
