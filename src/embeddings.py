import logging
from typing import List
from sentence_transformers import SentenceTransformer
from src.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}")
            try:
                cls._model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME, device="cpu")
                logger.info("Embedding model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                cls._model = None
        return cls._instance

    def encode(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self._model is None:
            # Fallback zero/deterministic embedding if model failed download
            return [[0.0] * settings.EMBEDDING_DIM for _ in texts]
        embeddings = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.tolist()

    def encode_query(self, query: str) -> List[float]:
        return self.encode([query])[0]

embedding_service = EmbeddingService()
