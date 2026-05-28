from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from app.embeddings.embedder import EmbeddingManager
from app.vectordb.chroma_client import ChromaDBManager


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
        rm = RetrievalManager()
        results = rm.retrieve(
            query="What is the capital of France?",
            collection_name="my_collection",
            top_k=3,
        )
        for r in results:
            print(r.chunk_id, r.score, r.document_text[:100])
    """

    def __init__(
        self,
        chroma: Optional[ChromaDBManager] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
    ) -> None:
        self.chroma = chroma or ChromaDBManager()
        self.embedding_manager = embedding_manager or EmbeddingManager

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
        collection_name: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Retrieve nearest semantic neighbors for `query` from `collection_name`.

        Args:
            query: Text query to search.
            collection_name: Name of the collection to query.
            top_k: Number of neighbors to return (must be > 0).
            metadata_filter: Optional dictionary to filter stored documents by metadata.

        Returns:
            List[RetrievalResult] ordered by increasing distance (closest first).
        """
        q = self._normalize_query(query)

        if not isinstance(collection_name, str) or not collection_name.strip():
            raise ValueError("collection_name must be a non-empty string")

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        try:
            embedding = self.embedding_manager.embed_text(q)
        except Exception as exc:
            raise RuntimeError(f"failed to create query embedding: {exc}") from exc

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

        # Parse single-query response shape: lists per query.
        try:
            ids = raw.get("ids", [[]])[0]
            documents = raw.get("documents", [[]])[0]
            metadatas = raw.get("metadatas", [[]])[0] if "metadatas" in raw else [None] * len(ids)
            distances = raw.get("distances", [[]])[0] if "distances" in raw else [None] * len(ids)
        except Exception as exc:
            raise RuntimeError(f"unexpected vector store response format: {exc}") from exc

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

        # Ensure deterministic ordering by distance (None values go last)
        results.sort(key=lambda r: (float("inf") if r.distance is None else r.distance))

        return results

