from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Protocol

from app.retrieval.retriever import RetrievalManager, RetrievalResult


class Retriever(Protocol):
    """Protocol for retrieval implementations used by RetrievalPipeline."""

    def retrieve(
        self,
        query: str,
        collection_name: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        ...


class BaseReranker(ABC):
    """Base interface for reranker components in the retrieval pipeline."""

    @abstractmethod
    def rerank(self, query: str, results: Iterable[RetrievalResult]) -> List[RetrievalResult]:
        """Return a new ordered list of retrieval results based on reranking logic."""


class MockReranker(BaseReranker):
    """A simple deterministic reranker used for testing and pipeline proof-of-concept."""

    def __init__(
        self,
        mode: str = "preserve",
        metadata_key: Optional[str] = None,
        reverse: bool = False,
    ) -> None:
        if mode not in {"preserve", "reverse", "metadata_sort"}:
            raise ValueError("mode must be one of: preserve, reverse, metadata_sort")
        self.mode = mode
        self.metadata_key = metadata_key
        self.reverse = reverse

    def rerank(self, query: str, results: Iterable[RetrievalResult]) -> List[RetrievalResult]:
        ordered = list(results)
        if self.mode == "preserve":
            return ordered

        if self.mode == "reverse":
            return list(reversed(ordered))

        if self.mode == "metadata_sort":
            if self.metadata_key is None:
                raise ValueError("metadata_key is required for metadata_sort mode")

            return sorted(
                ordered,
                key=lambda item: (
                    item.metadata.get(self.metadata_key) is None,
                    item.metadata.get(self.metadata_key),
                    item.chunk_id,
                ),
                reverse=self.reverse,
            )

        return ordered


class RetrievalPipeline:
    """Composable retrieval orchestration for future RAG retrieval workflows."""

    def __init__(
        self,
        retriever: Retriever,
        reranker: Optional[BaseReranker] = None,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker

    def run(
        self,
        query: str,
        collection_name: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Run the retrieval pipeline from query to final ordered results."""
        results = self._retriever.retrieve(
            query=query,
            collection_name=collection_name,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

        if not results or self._reranker is None:
            return results

        return self._reranker.rerank(query=query, results=results)


def build_default_pipeline(
    collection_name: str,
    top_k: int = 5,
    chroma: Optional[Any] = None,
    embedding_manager: Optional[Any] = None,
) -> RetrievalPipeline:
    """Create a default pipeline backed by RetrievalManager.

    This helper is convenient for codebases that want a ready-to-use pipeline
    without manually instantiating the retriever.
    """
    retriever = RetrievalManager(chroma=chroma, embedding_manager=embedding_manager)
    return RetrievalPipeline(retriever=retriever)
