import hashlib
import json
from typing import Callable, List

import pytest

from app.vectordb.chroma_client import ChromaDBManager
from app.embeddings import embedder
from app.ingestion.ingest import DocumentIngestionManager


def deterministic_vector(text: str, dim: int = 8) -> List[float]:
    """Produce a deterministic but lightweight embedding for tests.

    Uses SHA1 of the text to generate stable pseudo-floats in [0,1).
    """
    h = hashlib.sha1(text.encode("utf-8")).digest()
    vec: List[float] = []
    for i in range(dim):
        b = h[i % len(h)]
        vec.append((b % 251) / 250.0)  # values in [0, 1.004) clipped effectively
    return vec


@pytest.fixture(autouse=True)
def patch_embeddings(monkeypatch: pytest.MonkeyPatch):
    """Patch the EmbeddingManager to avoid heavy external model loads."""

    def fake_embed_text(text: str) -> List[float]:
        return deterministic_vector(text)

    def fake_embed_texts(texts: List[str]) -> List[List[float]]:
        return [deterministic_vector(t) for t in texts]

    monkeypatch.setattr(embedder.EmbeddingManager, "embed_text", staticmethod(fake_embed_text))
    monkeypatch.setattr(embedder.EmbeddingManager, "embed_texts", staticmethod(fake_embed_texts))
    yield


@pytest.fixture()
def tmp_chroma_manager(tmp_path) -> ChromaDBManager:
    """Create an isolated ChromaDBManager using a temporary directory."""
    persist_dir = tmp_path / "chroma_test_db"
    cm = ChromaDBManager(persist_directory=str(persist_dir))
    return cm


@pytest.fixture()
def ingestion_manager(tmp_chroma_manager) -> DocumentIngestionManager:
    """Create an ingestion manager that writes to the temporary chroma manager."""
    return DocumentIngestionManager(chroma=tmp_chroma_manager, embedding_manager=embedder.EmbeddingManager, collection_name="test_collection", chunk_size=50, chunk_overlap=10)


@pytest.fixture()
def sample_docs() -> List[dict]:
    return [
        {
            "id": "doc-1",
            "text": "Paris is the capital of France. It is known for the Eiffel Tower and cafes.",
            "metadata": {"category": "geography", "source": "unit_test"},
        },
        {
            "id": "doc-2",
            "text": "Python is a programming language that is popular for scripting and data science.",
            "metadata": {"category": "programming", "source": "unit_test"},
        },
    ]
