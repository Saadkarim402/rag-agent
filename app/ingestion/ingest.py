from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, Iterable, List, Optional

from app.config import IngestionConfig
from app.embeddings.embedder import EmbeddingManager
from app.vectordb.chroma_client import ChromaDBManager

logger = logging.getLogger(__name__)


class DocumentIngestionManager:
    """Orchestrates deterministic document ingestion and indexing.

    This class handles raw text ingestion, text chunking, embedding generation,
    and vector storage. It is intentionally minimal and focused on ingestion
    workflows without retrieval or LLM logic.
    """

    def __init__(
        self,
        chroma: Optional[ChromaDBManager] = None,
        embedding_manager: Optional[type[EmbeddingManager]] = None,
        config: Optional[IngestionConfig] = None,
        collection_name: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> None:
        config = config or IngestionConfig()

        self.chroma = chroma or ChromaDBManager()
        self.embedding_manager = embedding_manager or EmbeddingManager
        self.collection_name = collection_name if collection_name is not None else config.collection_name
        self.chunk_size = int(chunk_size) if chunk_size is not None else config.chunk_size
        self.chunk_overlap = int(chunk_overlap) if chunk_overlap is not None else config.chunk_overlap

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if not self.collection_name or not isinstance(self.collection_name, str):
            raise ValueError("collection_name must be a non-empty string")

        if config.embedding_model is not None and hasattr(self.embedding_manager, "_model_name"):
            self.embedding_manager._model_name = config.embedding_model

        logger.info(
            "[CONFIG] chunk_size=%s chunk_overlap=%s collection_name=%s embedding_model=%s",
            self.chunk_size,
            self.chunk_overlap,
            self.collection_name,
            config.embedding_model,
        )

    def _normalize_text(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return " ".join(text.strip().split())

    def _get_overlap_words(self, words: List[str]) -> List[str]:
        if self.chunk_overlap == 0 or not words:
            return []

        overlap_words: List[str] = []
        total = 0
        for word in reversed(words):
            increment = len(word) + (1 if overlap_words else 0)
            if total + increment > self.chunk_overlap:
                break
            overlap_words.append(word)
            total += increment

        return list(reversed(overlap_words))

    def _chunk_text(self, text: str) -> List[str]:
        """Deterministically split text into readable chunks with overlap."""
        normalized = self._normalize_text(text)
        if not normalized:
            return []

        words = normalized.split(" ")
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_length = 0

        for word in words:
            next_length = current_length + (1 if current_chunk else 0) + len(word)
            if next_length > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                overlap = self._get_overlap_words(current_chunk)
                current_chunk = overlap.copy()
                current_length = len(" ".join(current_chunk))
                next_length = current_length + (1 if current_chunk else 0) + len(word)
                if next_length > self.chunk_size and not current_chunk:
                    current_chunk = [word]
                    current_length = len(word)
                    continue

            current_chunk.append(word)
            current_length = next_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _generate_chunk_id(self, doc_id: str, chunk_index: int, chunk_text: str) -> str:
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError("doc_id must be a non-empty string")

        hash_input = f"{doc_id}:{chunk_index}:{hashlib.sha1(chunk_text.encode('utf-8')).hexdigest()}"
        return hashlib.sha1(hash_input.encode("utf-8")).hexdigest()

    def ingest_text(self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Ingest a single raw text document into ChromaDB."""
        if not isinstance(doc_id, str) or not doc_id:
            raise TypeError("doc_id must be a non-empty string")
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        chunks = self._chunk_text(text)
        if not chunks:
            return []

        ids = [self._generate_chunk_id(doc_id, i, chunk) for i, chunk in enumerate(chunks)]

        try:
            embeddings = self.embedding_manager.embed_texts(chunks)
        except Exception as exc:
            raise RuntimeError(f"failed to generate embeddings: {exc}") from exc

        metadatas: List[Dict[str, Any]] = []
        for index, _ in enumerate(chunks):
            chunk_metadata: Dict[str, Any] = {"source_id": doc_id, "chunk_index": index}
            if metadata is not None:
                chunk_metadata.update(metadata)
            metadatas.append(chunk_metadata)

        try:
            self.chroma.add_documents(
                collection_name=self.collection_name,
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        except Exception as exc:
            raise RuntimeError(f"failed to store chunk vectors in ChromaDB: {exc}") from exc

        return ids

    def ingest_texts(self, documents: Iterable[Dict[str, Any]]) -> List[str]:
        """Ingest multiple raw text documents in a batch."""
        if not isinstance(documents, Iterable):
            raise TypeError("documents must be an iterable of document dicts")

        all_ids: List[str] = []
        for item in documents:
            if not isinstance(item, dict):
                raise TypeError("each document must be a dict containing 'id' and 'text'")

            doc_id = item.get("id")
            text = item.get("text")
            metadata = item.get("metadata")
            all_ids.extend(self.ingest_text(doc_id=doc_id, text=text, metadata=metadata))

        return all_ids

    ingest_document = ingest_text
    ingest_documents = ingest_texts
