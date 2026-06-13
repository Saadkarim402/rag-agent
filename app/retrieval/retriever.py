from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from app.config import RetrievalConfig
from app.embeddings.embedder import EmbeddingManager
from app.vectordb.chroma_client import ChromaDBManager

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Structured retrieval result returned by RetrievalManager.

    Attributes:
        chunk_id: Unique identifier for the chunk in the vector store.
        document_text: The stored chunk text.
        metadata: Arbitrary metadata associated with the chunk.
        distance: Raw distance returned by the vector store (lower is closer).
        score: Optional normalized score derived from distance (higher is better).
        collection: The collection name where the chunk was found.
    """
    chunk_id: str
    document_text: str
    metadata: Dict[str, Any]
    distance: Optional[float]
    score: Optional[float]
    collection: str

    def __getitem__(self, key: str) -> Any:
        if not isinstance(key, str):
            raise KeyError(key)
        if hasattr(self, key):
            return getattr(self, key)
        if isinstance(self.metadata, dict) and key in self.metadata:
            return self.metadata[key]
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        if not isinstance(key, str):
            return False
        if hasattr(self, key):
            return True
        if isinstance(self.metadata, dict) and key in self.metadata:
            return True
        return False

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


class RetrievalManager:
    """High-level retrieval orchestration for semantic search.

    Responsibilities:
    - Normalize and validate queries.
    - Use `EmbeddingManager` to create query embeddings.
    - Ask `ChromaDBManager` for nearest neighbors and return
      structured `RetrievalResult` objects.

    The class intentionally does not know storage or transformer internals;
    it relies on the two manager abstractions.

    Example usage:
        config = RetrievalConfig(top_k=5, min_score_threshold=0.5)
        rm = RetrievalManager(config=config)
        results = rm.retrieve(query="What is the capital of France?", collection_name="my_collection")
    """

    def __init__(
        self,
        chroma: Optional[ChromaDBManager] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        config: Optional[RetrievalConfig] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        self.chroma = chroma or ChromaDBManager()
        self.embedding_manager = embedding_manager or EmbeddingManager
        self.config = config or RetrievalConfig()
        self.default_collection_name = collection_name if collection_name is not None else self.config.collection_name
        self.default_top_k = self.config.top_k
        self.default_min_score_threshold = self.config.min_score_threshold

        if self.config.embedding_model is not None and hasattr(self.embedding_manager, "_model_name"):
            self.embedding_manager._model_name = self.config.embedding_model

        logger.info(
            "[CONFIG] top_k=%s min_score_threshold=%s collection_name=%s embedding_model=%s",
            self.default_top_k,
            self.default_min_score_threshold,
            self.default_collection_name,
            self.config.embedding_model,
        )

    def _normalize_query(self, query: str) -> str:
        """Lightweight normalization for queries.

        Keeps the operation minimal and deterministic. This is a placeholder
        for more advanced normalization/tokenization/hashing later.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        q = query.strip()
        if not q:
            raise ValueError("query must be a non-empty string")
        return q

    def _to_score(self, distance: Optional[float]) -> Optional[float]:
        if distance is None:
            return None
        try:
            return 1.0 / (1.0 + float(distance))
        except Exception:
            return None

    def retrieve(
        self,
        query: str,
        collection_name: Optional[str] = None,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        min_score_threshold: Optional[float] = None,
    ) -> List[RetrievalResult]:
        """Retrieve nearest semantic neighbors for `query` from `collection_name`.

        Args:
            query: Text query to search.
            collection_name: Name of the collection to query.
            top_k: Number of neighbors to return (must be > 0).
            metadata_filter: Optional dictionary to filter stored documents by metadata.
            min_score_threshold: Optional minimum score to accept.

        Returns:
            List[RetrievalResult] ordered by increasing distance (closest first).
        """
        # === RETRIEVAL ===
        logger.info("=" * 80)
        logger.info("=== RETRIEVAL ===")
        logger.info("=" * 80)
        
        q = self._normalize_query(query)
        logger.info(f"[QUERY] {q}")

        collection_name = collection_name.strip() if isinstance(collection_name, str) and collection_name.strip() else self.default_collection_name
        if not collection_name:
            raise ValueError("collection_name must be a non-empty string")

        top_k = top_k if top_k is not None else self.default_top_k
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        threshold = self.default_min_score_threshold if min_score_threshold is None else min_score_threshold
        if not isinstance(threshold, (int, float)) or threshold < 0.0 or threshold > 1.0:
            raise ValueError("min_score_threshold must be between 0.0 and 1.0")

        if metadata_filter:
            logger.info(f"[FILTER] Applied metadata filter: {metadata_filter}")

        logger.info(f"[RETRIEVAL] Collection={collection_name} top_k={top_k} threshold={threshold}")

        try:
            embedding = self.embedding_manager.embed_text(q)
        except Exception as exc:
            raise RuntimeError(f"failed to create query embedding: {exc}") from exc

        logger.info(f"[EMBEDDING] Query vector generated (dimension={len(embedding)})")

        try:
            raw = self.chroma.query_embeddings(
                collection_name=collection_name,
                query_embeddings=[embedding],
                n_results=top_k,
                include_metadata=True,
                metadata_filter=metadata_filter,
            )
        except Exception as exc:
            raise RuntimeError(f"vector store query failed: {exc}") from exc

        try:
            ids = raw.get("ids", [[]])[0]
            documents = raw.get("documents", [[]])[0]
            metadatas = raw.get("metadatas", [[]])[0] if "metadatas" in raw else [None] * len(ids)
            distances = raw.get("distances", [[]])[0] if "distances" in raw else [None] * len(ids)
        except Exception as exc:
            raise RuntimeError(f"unexpected vector store response format: {exc}") from exc

        logger.info(f"[RETRIEVAL] Retrieved {len(ids)} candidates from vector store")

        results: List[RetrievalResult] = []
        for idx in range(len(ids)):
            cid = ids[idx]
            doc = documents[idx] if idx < len(documents) else ""
            meta = metadatas[idx] if idx < len(metadatas) and metadatas[idx] is not None else {}
            dist = distances[idx] if idx < len(distances) else None
            score = self._to_score(dist)

            results.append(
                RetrievalResult(
                    chunk_id=cid,
                    document_text=doc,
                    metadata=meta or {},
                    distance=dist,
                    score=score,
                    collection=collection_name,
                )
            )

            score_str = f"{score:.4f}" if score else "N/A"
            logger.info(
                f"[RETRIEVAL RESULT] Rank={idx+1} chunk_id={cid[:16]}... "
                f"distance={dist:.4f} score={score_str} "
                f"text_preview={doc[:50].replace(chr(10), ' ')}..."
            )

        logger.info("=" * 80)

        # Ensure deterministic ordering by distance (None values go last)
        results.sort(key=lambda r: (float("inf") if r.distance is None else r.distance))

        if threshold > 0.0:
            accepted = [r for r in results if r.score is not None and r.score >= threshold]
            rejected = len(results) - len(accepted)
            logger.info(
                "[SCORE FILTER] Threshold=%.2f Accepted=%s Rejected=%s",
                threshold,
                len(accepted),
                rejected,
            )
            results = accepted

        logger.info("[FINAL] Returned results=%s", len(results))
        return results

