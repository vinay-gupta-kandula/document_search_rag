import logging
from typing import List, Tuple
from sentence_transformers import CrossEncoder
import requests
from src.config import settings

logger = logging.getLogger(__name__)

class RerankerService:
    def __init__(self):
        self.cohere_api_key = settings.COHERE_API_KEY
        self.model_name = settings.RERANKER_MODEL_NAME
        self._local_model = None

    def _get_local_model(self):
        if self._local_model is None:
            try:
                logger.info(f"Loading local cross-encoder model: {self.model_name}")
                self._local_model = CrossEncoder(self.model_name, device="cpu")
                logger.info("Local cross-encoder loaded.")
            except Exception as e:
                logger.warning(f"Could not load cross-encoder ({e}), will use fallback scoring.")
                self._local_model = None
        return self._local_model

    def rerank(self, query: str, documents: List[str], top_n: int = 5) -> List[Tuple[int, float]]:
        """
        Rerank a list of documents given a query.
        Returns list of tuples: (original_index, relevance_score) sorted descending by score.
        """
        if not documents:
            return []

        # If Cohere API Key is provided, use Cohere Rerank API
        if self.cohere_api_key:
            try:
                headers = {
                    "Authorization": f"Bearer {self.cohere_api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "rerank-english-v3.0",
                    "query": query,
                    "documents": documents,
                    "top_n": top_n
                }
                resp = requests.post("https://api.cohere.com/v1/rerank", json=payload, headers=headers, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    return [(item["index"], float(item["relevance_score"])) for item in data["results"]]
                else:
                    logger.warning(f"Cohere API rerank failed ({resp.status_code}): {resp.text}. Falling back to local.")
            except Exception as e:
                logger.warning(f"Cohere API call exception: {e}. Falling back to local.")

        # Local CrossEncoder fallback
        model = self._get_local_model()
        if model is not None:
            pairs = [[query, doc] for doc in documents]
            scores = model.predict(pairs)
            indexed_scores = [(i, float(s)) for i, s in enumerate(scores)]
            indexed_scores.sort(key=lambda x: x[1], reverse=True)
            return indexed_scores[:top_n]

        # Basic fallback: return original indices with descending default scores
        return [(i, 1.0 / (i + 1)) for i in range(min(top_n, len(documents)))]

reranker_service = RerankerService()
