import logging
import re
from typing import List

from app.retrieval.retriever import RetrievalResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Smart Re-ranking module combining semantic similarity and exact term token overlap.

    Does not require downloading extra model files, ensuring 100% offline compatibility.
    """

    def __init__(self, semantic_weight: float = 0.7) -> None:
        self.semantic_weight = semantic_weight

    def _tokenize(self, text: str) -> set:
        return set(re.findall(r"\b\w+\b", text.lower()))

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

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return results

        reranked = []
        for res in results:
            # 1. Semantic score from bi-encoder retrieval (higher is better)
            semantic_score = res.score if res.score is not None else 0.0

            # 2. Token overlap score (Jaccard similarity on keyword tokens)
            doc_tokens = self._tokenize(res.document_text)
            intersection = query_tokens.intersection(doc_tokens)
            union = query_tokens.union(doc_tokens)
            
            overlap_score = len(intersection) / len(union) if union else 0.0

            # 3. Blended Re-rank score
            blended_score = (self.semantic_weight * semantic_score) + ((1.0 - self.semantic_weight) * overlap_score)

            res.score = round(blended_score, 4)
            reranked.append(res)

        # Sort descending by updated score
        reranked.sort(key=lambda x: x.score if x.score is not None else 0.0, reverse=True)
        logger.info(f"[RERANKER] Re-ranked {len(reranked)} chunks.")
        return reranked
