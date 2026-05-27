from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from app.embeddings.embedder import EmbeddingManager
from app.vectordb.chroma_client import ChromaDBManager


class Retriever:
    """Semantic retriever that converts queries to embeddings and queries ChromaDB.

    Responsibilities:
    - Validate and encode a text query using `EmbeddingManager`.
    - Query a ChromaDB `Collection` for nearest neighbors.
    - Optionally apply metadata filters (`where`-style dictionary passed to Chroma).
    - Return structured, LLM-friendly retrieval results.

    Design notes:
    - Minimal, backend-agnostic retrieval layer: relies on ChromaDBManager only
      to obtain the collection object, then calls `collection.query(...)` so
      future backends can be substituted by changing the manager implementation.
    - Deterministic behavior is achieved by relying on the deterministic
      embedding generation of `EmbeddingManager` and deterministic ordering
      returned by the vector store for ties.
    """

    def __init__(
        self,
        chroma: Optional[ChromaDBManager] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        collection_name: str = "documents",
        top_k_default: int = 5,
    ) -> None:
        if not collection_name:
            raise ValueError("collection_name must not be empty")
        if top_k_default <= 0:
            raise ValueError("top_k_default must be > 0")

        self.chroma = chroma or ChromaDBManager()
        self.embedding_manager = embedding_manager or EmbeddingManager
        self.collection_name = collection_name
        self.top_k_default = int(top_k_default)

    def retrieve_with_scores(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-K chunks with raw distance scores.

        Returns a list of dicts with the keys: `id`, `document`, `metadata`, `distance`.
        The `distance` field is the raw distance value returned by the vector store.

        Args:
            query: The user query string to embed and search.
            top_k: Number of neighbors to return. If omitted, uses `top_k_default`.
            filters: Optional metadata filters compatible with ChromaDB `where`.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        k = int(top_k or self.top_k_default)
        if k <= 0:
            raise ValueError("top_k must be > 0")

        # generate embedding for the query
        try:
            query_embedding = self.embedding_manager.embed_text(query)
        except Exception as exc:
            raise RuntimeError(f"failed to create query embedding: {exc}") from exc

        # obtain collection and perform query; use manager to get collection
        collection = self.chroma.get_collection(self.collection_name)

        try:
            # Chroma's `query` accepts `where` for metadata filtering
            raw = collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["ids", "documents", "metadatas", "distances"],
                where=filters or {},
            )
        except Exception as exc:
            raise RuntimeError(f"vector store query failed: {exc}") from exc

        # parse results for single-query case; collection.query returns lists per query
        try:
            ids: List[str] = raw.get("ids", [[]])[0]
            documents: List[str] = raw.get("documents", [[]])[0]
            metadatas: List[Dict[str, Any]] = raw.get("metadatas", [[]])[0] if "metadatas" in raw else [None] * len(ids)
            distances: List[float] = raw.get("distances", [[]])[0] if "distances" in raw else [None] * len(ids)
        except Exception as exc:
            raise RuntimeError(f"unexpected vector store response format: {exc}") from exc

        results: List[Dict[str, Any]] = []
        for i in range(len(ids)):
            results.append(
                {
                    "id": ids[i],
                    "document": documents[i] if i < len(documents) else "",
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "distance": distances[i] if i < len(distances) else None,
                }
            )

        return results

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-K chunks and include a normalized score.

        The returned dicts contain: `id`, `document`, `metadata`, `distance`, and
        a convenience `score` computed as `1/(1+distance)` when `distance` is numeric.
        This `score` is a heuristic to convert distances to a bounded similarity-like value.
        """
        raw_results = self.retrieve_with_scores(query=query, top_k=top_k, filters=filters)

        formatted: List[Dict[str, Any]] = []
        for r in raw_results:
            dist = r.get("distance")
            score: Optional[float]
            if isinstance(dist, (int, float)):
                try:
                    score = 1.0 / (1.0 + float(dist))
                except Exception:
                    score = None
            else:
                score = None

            item = {
                "id": r.get("id"),
                "document": r.get("document"),
                "metadata": r.get("metadata"),
                "distance": dist,
                "score": score,
            }
            formatted.append(item)

        return formatted

    def format_context(self, results: Iterable[Dict[str, Any]], max_chars: Optional[int] = 2000) -> str:
        """Format retrieved chunks into a single LLM-ready context string.

        Preserves the order of `results`. Truncation (when `max_chars` is set)
        keeps whole chunks and stops before exceeding the limit to avoid
        returning partial sentences when possible.

        Args:
            results: Iterable of retrieval result dicts (as returned by `retrieve`).
            max_chars: Optional maximum number of characters in the context.
        """
        pieces: List[str] = []
        total = 0
        sep = "\n\n---\n\n"

        for idx, item in enumerate(results):
            text = item.get("document") or ""
            if not isinstance(text, str):
                continue
            piece = f"[{idx}] {text}"
            piece_len = len(piece)

            if max_chars is not None and total + piece_len > max_chars:
                # stop before adding this chunk to keep context composed of whole chunks
                break

            pieces.append(piece)
            total += piece_len

        return sep.join(pieces)
