from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from app.embeddings.embedder import EmbeddingManager
from app.vectordb.chroma_client import ChromaDBManager


class DocumentIngestionManager:
    """Orchestrates simple text chunking, embedding, and storage to ChromaDB.

    This class focuses solely on ingestion and indexing orchestration. It performs
    deterministic chunking, generates embeddings via `EmbeddingManager`, and stores
    vectors and documents using `ChromaDBManager`.
    """

    def __init__(
        self,
        chroma: Optional[ChromaDBManager] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        collection_name: str = "documents",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chroma = chroma or ChromaDBManager()
        self.embedding_manager = embedding_manager or EmbeddingManager
        self.collection_name = collection_name
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)

    def _chunk_text(self, text: str) -> List[str]:
        """Simple deterministic text chunking with overlap.

        - Splits by character windows while avoiding mid-word cuts when possible.
        - Preserves chunk ordering and supports a fixed overlap in characters.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text:
            return []

        text = text.strip()
        length = len(text)
        chunks: List[str] = []
        start = 0

        while start < length:
            end = start + self.chunk_size
            if end >= length:
                chunk = text[start:length]
                chunks.append(chunk)
                break

            # avoid splitting mid-word: try to move end back to last whitespace
            window = text[start:end]
            last_space = window.rfind(" ")
            if last_space > int(self.chunk_size * 0.5):
                cut = start + last_space
            else:
                # if no suitable whitespace, cut at end
                cut = end

            chunk = text[start:cut]
            chunks.append(chunk)

            # advance start by chunk_size - overlap (but ensure progress)
            start = max(cut - self.chunk_overlap, cut - (self.chunk_size - 1))

        return chunks

    def _generate_chunk_id(self, doc_id: str, chunk_index: int, chunk_text: str) -> str:
        """Generate a deterministic chunk id from document id, index, and chunk content."""
        if not doc_id:
            raise ValueError("doc_id must be provided")
        h = hashlib.sha1()
        h.update(doc_id.encode("utf-8"))
        h.update(b":")
        h.update(str(chunk_index).encode("utf-8"))
        h.update(b":")
        # include a short fingerprint of the chunk text to keep ids stable
        h.update(hashlib.sha1(chunk_text.encode("utf-8")).hexdigest().encode("utf-8"))
        return h.hexdigest()

    def ingest_document(self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Ingest a single document: chunk, embed, and store in ChromaDB.

        Returns the list of generated chunk ids.
        """
        if not isinstance(doc_id, str) or not doc_id:
            raise TypeError("doc_id must be a non-empty string")
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        chunks = self._chunk_text(text)
        if not chunks:
            return []

        # generate deterministic ids per chunk
        ids: List[str] = [self._generate_chunk_id(doc_id, i, chunk) for i, chunk in enumerate(chunks)]

        # generate embeddings
        try:
            embeddings = self.embedding_manager.embed_texts(chunks)
        except Exception as exc:  # lightweight error handling
            raise RuntimeError(f"failed to generate embeddings: {exc}") from exc

        # prepare metadatas per chunk
        metadatas: List[Dict[str, Any]] = []
        for i, _ in enumerate(chunks):
            md: Dict[str, Any] = {"source_id": doc_id, "chunk_index": i}
            if metadata:
                md.update(metadata)
            metadatas.append(md)

        # store in chroma
        try:
            self.chroma.add_documents(
                collection_name=self.collection_name,
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        except Exception as exc:
            raise RuntimeError(f"failed to store embeddings in ChromaDB: {exc}") from exc

        return ids

    def ingest_documents(self, docs: Iterable[Dict[str, Any]]) -> List[str]:
        """Ingest multiple documents.

        Each item in `docs` must be a dict with keys: `id` (str), `text` (str), optional `metadata` (dict).
        Returns a flat list of all generated chunk ids.
        """
        if not isinstance(docs, Iterable):
            raise TypeError("docs must be an iterable of dicts")

        all_ids: List[str] = []
        for item in docs:
            if not isinstance(item, dict):
                raise TypeError("each document must be a dict with keys 'id' and 'text'")
            doc_id = item.get("id")
            text = item.get("text")
            metadata = item.get("metadata")
            chunk_ids = self.ingest_document(doc_id=doc_id, text=text, metadata=metadata)
            all_ids.extend(chunk_ids)

        return all_ids
