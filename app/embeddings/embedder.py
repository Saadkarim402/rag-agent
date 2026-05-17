from __future__ import annotations

from typing import List

from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    """Manage sentence-transformer embeddings for the RAG pipeline."""

    _model: SentenceTransformer | None = None
    _model_name: str = "BAAI/bge-small-en-v1.5"

    @classmethod
    def _load_model(cls) -> SentenceTransformer:
        """Load the embedding model once and cache it for reuse."""
        if cls._model is None:
            cls._model = SentenceTransformer(cls._model_name)
        return cls._model

    @classmethod
    def embed_text(cls, text: str) -> List[float]:
        """Embed a single text string and return a vector as a Python list."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text:
            raise ValueError("text must not be empty")

        model = cls._load_model()
        embedding = model.encode(text, show_progress_bar=False)
        return embedding.tolist()

    @classmethod
    def embed_texts(cls, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts and return a list of embedding vectors."""
        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings")

        if not texts:
            raise ValueError("texts must contain at least one item")

        if any(not isinstance(item, str) for item in texts):
            raise TypeError("all items in texts must be strings")

        model = cls._load_model()
        embeddings = model.encode(texts, show_progress_bar=False)
        return [vector.tolist() for vector in embeddings]
