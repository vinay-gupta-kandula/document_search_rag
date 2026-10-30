import re
import logging
from typing import List, Dict, Any, Tuple
from src.config import settings

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self.api_key = settings.OPENAI_API_KEY
        self.model_name = settings.OPENAI_MODEL_NAME

    def generate_answer(self, query: str, contexts: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generate answer with source citations based on retrieved context chunks.
        Returns: (answer_string, cited_sources)
        """
        if not contexts:
            return (
                "No relevant document context was found to answer this question. Please upload relevant documents first.",
                []
            )

        # Build numbered context block
        context_blocks = []
        for idx, ctx in enumerate(contexts):
            source_id = ctx.get("chunk_id", f"chunk_{idx}")
            doc_id = ctx.get("document_id", "unknown")
            filename = ctx.get("filename", "unknown")
            text = ctx.get("content", "")
            context_blocks.append(
                f"[Source ID: {source_id}] (Document: {filename}, ID: {doc_id}):\n{text}"
            )
        formatted_context = "\n\n".join(context_blocks)

        system_prompt = (
            "You are an enterprise document question answering assistant. "
            "Based strictly on the provided context below, answer the user's question clearly, precisely, and factually.\n"
            "If the context does not contain the answer, explicitly state that the information is not available in the documents.\n"
            "For each factual statement or claim you make, cite the corresponding source using the exact format [Source ID: <id>]."
        )

        user_prompt = f"Context:\n{formatted_context}\n\nQuestion: {query}\n\nAnswer:"

        if self.provider.lower() == "openai" and self.api_key and not self.api_key.startswith("your_"):
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.api_key)
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0
                )
                raw_answer = response.choices[0].message.content.strip()
                return self._process_citations(raw_answer, contexts)
            except Exception as e:
                logger.error(f"OpenAI call failed ({e}), falling back to intelligent extractive answering.")

        # Intelligent local answering fallback
        return self._generate_extractive_answer(query, contexts)

    def _generate_extractive_answer(self, query: str, contexts: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """Synthesize a structured cited answer using the highest-ranked contexts."""
        top_chunks = contexts[:3]
        summary_lines = []
        for ctx in top_chunks:
            source_id = ctx.get("chunk_id")
            filename = ctx.get("filename", "document")
            content = ctx.get("content", "").strip()
            # Pick first 2 sentences or 250 characters
            sentences = re.split(r"(?<=[.!?])\s+", content)
            excerpt = " ".join(sentences[:2]) if sentences else content[:200]
            summary_lines.append(f"{excerpt} [Source ID: {source_id}]")

        answer_text = (
            f"Based on the retrieved enterprise documentation from {', '.join({c.get('filename', 'docs') for c in top_chunks})}:\n\n"
            + "\n\n".join(f"- {line}" for line in summary_lines)
        )
        return self._process_citations(answer_text, contexts)

    def _process_citations(self, answer_text: str, contexts: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract cited source IDs from the answer or map to the top contexts."""
        cited_ids = set(re.findall(r"\[Source ID:\s*([^\]]+)\]", answer_text))

        sources = []
        for ctx in contexts:
            cid = ctx.get("chunk_id")
            # If cited explicitly or if top context and citations weren't strictly formatted
            is_cited = (cid in cited_ids) or (not cited_ids and len(sources) < 3)
            if is_cited:
                sources.append({
                    "source_id": cid,
                    "document_id": ctx.get("document_id"),
                    "filename": ctx.get("filename"),
                    "chunk_index": ctx.get("chunk_index", 0),
                    "content": ctx.get("content", ""),
                    "relevance_score": float(ctx.get("relevance_score", 0.0)),
                    "metadata": ctx.get("metadata", {})
                })

        # Ensure all contexts are included in sources if none matched directly
        if not sources:
            for ctx in contexts[:3]:
                sources.append({
                    "source_id": ctx.get("chunk_id"),
                    "document_id": ctx.get("document_id"),
                    "filename": ctx.get("filename"),
                    "chunk_index": ctx.get("chunk_index", 0),
                    "content": ctx.get("content", ""),
                    "relevance_score": float(ctx.get("relevance_score", 0.0)),
                    "metadata": ctx.get("metadata", {})
                })

        return answer_text, sources

llm_service = LLMService()
