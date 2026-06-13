from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IngestionConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    collection_name: str = "documents"
    embedding_model: str | None = None

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if not self.collection_name or not isinstance(self.collection_name, str):
            raise ValueError("collection_name must be a non-empty string")


@dataclass
class RetrievalConfig:
    top_k: int = 5
    min_score_threshold: float = 0.0
    collection_name: str = "documents"
    embedding_model: str | None = None

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if not 0.0 <= self.min_score_threshold <= 1.0:
            raise ValueError("min_score_threshold must be between 0.0 and 1.0")
        if not self.collection_name or not isinstance(self.collection_name, str):
            raise ValueError("collection_name must be a non-empty string")
