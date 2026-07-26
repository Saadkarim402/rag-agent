import logging
import re
from typing import List

from app.retrieval.retriever import RetrievalResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Smart Re-ranking module utilizing a neural Cross-Encoder model.

    Falls back to a semantic-syntactic token overlap heuristic if model loading fails.
    """

    def __init__(self, semantic_weight: float = 0.7) -> None:
        self.semantic_weight = semantic_weight
        self.model = None
        try:
            from sentence_transformers import CrossEncoder
            # Load the lightweight TinyBERT reranker model to prevent OOM on 512MB RAM instances
            logger.info("Initializing neural Cross-Encoder ('cross-encoder/ms-marco-TinyBERT-L-2-v2')...")
            self.model = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")
            logger.info("Neural Cross-Encoder successfully initialized.")
        except Exception as e:
            logger.warning(
                f"Failed to load neural Cross-Encoder model ({e}). "
                "Reranker will use the Jaccard-overlap fallback heuristic."
            )

    def _tokenize(self, text: str) -> set:
        return set(re.findall(r"\b\w+\b", text.lower()))

    def _fallback_rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Syntactic Jaccard-overlap fallback re-scorer."""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return results

        reranked = []
        for res in results:
            semantic_score = res.score if res.score is not None else 0.0
            doc_tokens = self._tokenize(res.document_text)
            intersection = query_tokens.intersection(doc_tokens)
            union = query_tokens.union(doc_tokens)
            
            overlap_score = len(intersection) / len(union) if union else 0.0
            blended_score = (self.semantic_weight * semantic_score) + ((1.0 - self.semantic_weight) * overlap_score)
            res.score = round(blended_score, 4)
            reranked.append(res)

        reranked.sort(key=lambda x: x.score if x.score is not None else 0.0, reverse=True)
        return reranked

    def rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Re-scores and sorts the retrieved chunks.

        Args:
            query: The user query string.
            results: List of retrieved RetrievalResult objects.

        Returns:
            Re-ranked list of RetrievalResult objects sorted by score descending.
        """
        if not results:
            return []

        # Use Jaccard fallback if Cross-Encoder model is not loaded
        if self.model is None:
            return self._fallback_rerank(query, results)

        try:
            # Prepare query-document pairs
            pairs = [(query, res.document_text) for res in results]
            
            # Run prediction (scores are logits)
            raw_scores = self.model.predict(pairs)
            
            import numpy as np
            # Apply sigmoid to normalize logit scores to 0-1 range
            scores = 1.0 / (1.0 + np.exp(-np.array(raw_scores)))

            reranked = []
            for idx, res in enumerate(results):
                res.score = float(round(scores[idx], 4))
                reranked.append(res)

            # Sort descending by re-ranked score
            reranked.sort(key=lambda x: x.score if x.score is not None else 0.0, reverse=True)
            logger.info(f"[RERANKER] Neural Cross-Encoder re-ranked {len(reranked)} chunks.")
            return reranked
        except Exception as e:
            logger.error(f"Neural Cross-Encoder reranking failed: {e}. Falling back to Jaccard-overlap.")
            return self._fallback_rerank(query, results)
