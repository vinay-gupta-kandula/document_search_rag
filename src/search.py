import re
import logging
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from src.vector_db import vector_db_service
from src.embeddings import embedding_service
from src.reranker import reranker_service

logger = logging.getLogger(__name__)

def tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())

class HybridSearchEngine:
    def __init__(self):
        pass

    def search(
        self,
        query: str,
        top_k: int = 5,
        hybrid_alpha: float = 0.5,
        apply_rerank: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining:
        1. Dense Vector Similarity (from Qdrant)
        2. Sparse Keyword Matching (BM25)
        Followed by Reciprocal Rank Fusion / normalized weighted score merge
        and final Cross-Encoder / Cohere Reranking.
        """
        # 1. Vector Search
        query_emb = embedding_service.encode_query(query)
        # Fetch candidate pool (e.g., top 2 * top_k)
        vector_candidates = vector_db_service.search_vector(query_emb, limit=max(top_k * 2, 10))

        # 2. BM25 Search
        all_chunks = vector_db_service.get_all_chunks(limit=5000)
        bm25_candidates = []
        if all_chunks:
            corpus = [tokenize(c["content"]) for c in all_chunks]
            bm25 = BM25Okapi(corpus)
            tokenized_query = tokenize(query)
            if tokenized_query:
                scores = bm25.get_scores(tokenized_query)
                scored_chunks = list(zip(all_chunks, scores))
                scored_chunks.sort(key=lambda x: x[1], reverse=True)
                for chunk, score in scored_chunks[:max(top_k * 2, 10)]:
                    if score > 0:
                        bm25_candidates.append({
                            **chunk,
                            "bm25_score": float(score)
                        })

        # 3. Reciprocal Rank Fusion (RRF) / Score Normalization
        # RRF formula: RRF_score = sum(1.0 / (k + rank))
        k_rrf = 60
        merged_scores: Dict[str, Dict[str, Any]] = {}

        # Process vector ranks
        for rank, item in enumerate(vector_candidates):
            cid = item["chunk_id"]
            if cid not in merged_scores:
                merged_scores[cid] = {
                    "item": item,
                    "rrf_score": 0.0,
                    "vector_rank": rank,
                    "bm25_rank": None
                }
            merged_scores[cid]["rrf_score"] += hybrid_alpha * (1.0 / (k_rrf + rank + 1))

        # Process BM25 ranks
        for rank, item in enumerate(bm25_candidates):
            cid = item["chunk_id"]
            if cid not in merged_scores:
                merged_scores[cid] = {
                    "item": item,
                    "rrf_score": 0.0,
                    "vector_rank": None,
                    "bm25_rank": rank
                }
            else:
                merged_scores[cid]["bm25_rank"] = rank
            merged_scores[cid]["rrf_score"] += (1.0 - hybrid_alpha) * (1.0 / (k_rrf + rank + 1))

        # Sort candidate items by RRF score
        sorted_candidates = sorted(
            merged_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )

        initial_results = [entry["item"] for entry in sorted_candidates[:max(top_k * 2, 10)]]

        if not initial_results:
            return []

        # 4. Re-ranking step
        if apply_rerank:
            candidate_texts = [res["content"] for res in initial_results]
            reranked_tuples = reranker_service.rerank(query, candidate_texts, top_n=top_k)
            
            final_results = []
            for orig_idx, score in reranked_tuples:
                res = dict(initial_results[orig_idx])
                res["relevance_score"] = float(score)
                final_results.append(res)
            return final_results
        else:
            for res in initial_results:
                res["relevance_score"] = res.get("score", 1.0)
            return initial_results[:top_k]

hybrid_search_engine = HybridSearchEngine()
