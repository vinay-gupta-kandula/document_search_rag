import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Database
    DB_USER: str = "rag_user"
    DB_PASSWORD: str = "rag_password"
    DB_NAME: str = "rag_metadata_db"
    DB_HOST: str = "db"
    DB_PORT: int = 5432

    # Vector DB
    VECTOR_DB_HOST: str = "vector_db"
    VECTOR_DB_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "document_chunks"

    # Embedding
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    # Reranker
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    COHERE_API_KEY: str = ""

    # LLM
    LLM_PROVIDER: str = "mock"  # or "openai"
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL_NAME: str = "gpt-4o-mini"

    @property
    def database_url(self) -> str:
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
