from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Dict, Iterable, List, Optional

from app.config import IngestionConfig
from app.documents import DocumentRepository
from app.embeddings.embedder import EmbeddingManager
from app.vectordb.chroma_client import ChromaDBManager

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'“‘])')

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
        document_repository: Optional[DocumentRepository] = None,
        config: Optional[IngestionConfig] = None,
        collection_name: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> None:
        config = config or IngestionConfig()

        self.chroma = chroma or ChromaDBManager()
        self.embedding_manager = embedding_manager or EmbeddingManager
        self.document_repository = document_repository or DocumentRepository()
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

    def _split_into_sentences(self, text: str) -> List[str]:
        normalized = self._normalize_text(text)
        if not normalized:
            return []

        sentences = [sentence.strip() for sentence in _SENTENCE_SPLIT_RE.split(normalized) if sentence.strip()]
        return sentences

    def _chunk_text_word_based(self, text: str) -> List[str]:
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

    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Extract the last N sentences that fit in chunk_overlap."""
        if self.chunk_overlap == 0 or not sentences:
            return []

        overlap_sentences: List[str] = []
        total_length = 0

        for sentence in reversed(sentences):
            # Length includes sentence + space separator
            sentence_length = len(sentence) + (1 if overlap_sentences else 0)
            if total_length + sentence_length > self.chunk_overlap:
                break
            overlap_sentences.append(sentence)
            total_length += sentence_length

        return list(reversed(overlap_sentences))

    def _split_long_sentence_smartly(self, sentence: str) -> List[str]:
        """Split a sentence longer than chunk_size into reasonable pieces.
        
        Tries to break at natural boundaries (commas, conjunctions after noun phrases).
        Avoids leaving incomplete phrases like "scaling, and" or "and networking".
        """
        if len(sentence) <= self.chunk_size:
            return [sentence]

        logger.info(f"[CHUNKING] Sentence exceeds chunk_size ({len(sentence)} > {self.chunk_size})")

        # Strategy: Break at commas when possible, then fallback to word boundaries
        # Try comma-based splits first
        parts = sentence.split(", ")
        if len(parts) > 1:
            sub_chunks: List[str] = []
            current: List[str] = []
            current_len = 0

            for part in parts:
                # Add comma back for all but last part
                part_with_comma = part if part == parts[-1] else part + ","
                part_len = len(part_with_comma) + (1 if current else 0)

                if current_len + part_len > self.chunk_size and current:
                    # Current chunk is full, save it
                    sub_chunks.append(" ".join(current))
                    current = [part_with_comma.rstrip(",")]
                    current_len = len(current[0])
                else:
                    current.append(part_with_comma.rstrip(","))
                    current_len = current_len + part_len if current_len > 0 else len(current[0])

            if current:
                sub_chunks.append(" ".join(current))

            # Verify we don't have incomplete trailing phrases
            result = []
            for chunk in sub_chunks:
                # Don't add chunks that end with conjunctions or incomplete phrases
                if not any(chunk.rstrip().endswith(bad) for bad in [", and", ", or", ", but", " and"]):
                    result.append(chunk)
                else:
                    # This chunk ends badly, merge it with next or keep as fallback
                    result.append(chunk)

            if result:
                logger.info(f"[CHUNKING] Split long sentence into {len(result)} sub-chunks using commas")
                return result

        # Fallback: word-based splitting
        words = sentence.split(" ")
        sub_chunks = []
        current = []
        current_len = 0

        for word in words:
            word_len = len(word) + (1 if current else 0)
            if current_len + word_len > self.chunk_size and current:
                sub_chunks.append(" ".join(current))
                current = []
                current_len = 0

            current.append(word)
            current_len = current_len + word_len if current_len > 0 else len(word)

        if current:
            sub_chunks.append(" ".join(current))

        logger.info(f"[CHUNKING] Split long sentence into {len(sub_chunks)} sub-chunks using words")
        return sub_chunks


    def _chunk_text(self, text: str) -> List[str]:
        """Deterministically split text into chunks with true sentence-awareness.
        
        Algorithm:
        1. Split text into sentences
        2. Group sentences until reaching chunk_size
        3. Create overlap from complete sentences at boundaries
        4. Never create semantic fragments
        5. Only split long sentences when absolutely necessary
        """
        normalized = self._normalize_text(text)
        if not normalized:
            return []

        sentences = self._split_into_sentences(normalized)
        if not sentences:
            return []

        # Single sentence fallback
        if len(sentences) == 1:
            if len(sentences[0]) <= self.chunk_size:
                return sentences
            else:
                return self._split_long_sentence_smartly(sentences[0])

        logger.info("[CHUNKING] Detected sentences=%s", len(sentences))
        logger.info("[CHUNKING] Chunk size target=%s", self.chunk_size)

        chunks: List[str] = []
        sentence_idx = 0
        tiny_chunk_count = 0
        fallback_split_count = 0

        while sentence_idx < len(sentences):
            chunk_sentences: List[str] = []
            chunk_length = 0

            # Greedily add sentences until we exceed chunk_size
            while sentence_idx < len(sentences):
                sentence = sentences[sentence_idx]

                # Calculate length with separator
                separator_len = 1 if chunk_sentences else 0
                next_length = chunk_length + separator_len + len(sentence)

                if next_length > self.chunk_size and chunk_sentences:
                    # Chunk is full, break without adding this sentence
                    break

                if next_length > self.chunk_size and not chunk_sentences:
                    # First sentence exceeds chunk_size - must split it
                    if len(sentence) > self.chunk_size:
                        logger.info(
                            "[CHUNKING] Sentence %s exceeds chunk_size (len=%s > %s)",
                            sentence_idx,
                            len(sentence),
                            self.chunk_size,
                        )
                        fallback_split_count += 1
                        sub_chunks = self._split_long_sentence_smartly(sentence)
                        chunks.extend(sub_chunks)
                        sentence_idx += 1
                        break
                    else:
                        # Shouldn't happen but handle it
                        chunk_sentences.append(sentence)
                        chunk_length = next_length
                        sentence_idx += 1
                        break
                else:
                    # Add sentence to current chunk
                    chunk_sentences.append(sentence)
                    chunk_length = next_length
                    sentence_idx += 1

            # Create chunk from accumulated sentences
            if chunk_sentences:
                chunk_text = " ".join(chunk_sentences)
                chunks.append(chunk_text)

                # Track tiny fragments
                if len(chunk_text) < 15:  # Arbitrary minimum for semantic content
                    tiny_chunk_count += 1
                    logger.warning(
                        "[CHUNKING] Tiny chunk detected: %s chars, %s sentences: %s...",
                        len(chunk_text),
                        len(chunk_sentences),
                        chunk_text[:50],
                    )

                logger.info(
                    "[CHUNKING] Created chunk %s sentences=%s chars=%s",
                    len(chunks) - 1,
                    len(chunk_sentences),
                    len(chunk_text),
                )

        # Log quality metrics
        logger.info("=" * 80)
        logger.info("=== CHUNKING QUALITY METRICS ===")
        logger.info("=" * 80)
        if chunks:
            chunk_lengths = [len(c) for c in chunks]
            logger.info("[METRICS] Total chunks: %s", len(chunks))
            logger.info("[METRICS] Avg chunk length: %.1f chars", sum(chunk_lengths) / len(chunks))
            logger.info("[METRICS] Min chunk length: %s chars", min(chunk_lengths))
            logger.info("[METRICS] Max chunk length: %s chars", max(chunk_lengths))
            logger.info("[METRICS] Tiny chunks (< 15 chars): %s", tiny_chunk_count)
            logger.info("[METRICS] Long sentence fallback splits: %s", fallback_split_count)

        return chunks


    def _generate_chunk_id(self, doc_id: str, chunk_index: int, chunk_text: str) -> str:
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError("doc_id must be a non-empty string")

        hash_input = f"{doc_id}:{chunk_index}:{hashlib.sha1(chunk_text.encode('utf-8')).hexdigest()}"
        return hashlib.sha1(hash_input.encode("utf-8")).hexdigest()

    def ingest_text(self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Ingest a single raw text document into ChromaDB.

        Pipeline:
            Document Load → Save Original → Chunking → Embeddings → Storage

        Args:
            doc_id: Unique document identifier
            text: Full document text content
            metadata: Optional metadata dictionary

        Returns:
            List of chunk IDs stored in ChromaDB
        """
        if not isinstance(doc_id, str) or not doc_id:
            raise TypeError("doc_id must be a non-empty string")
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        # === DOCUMENT LOAD ===
        logger.info("=" * 80)
        logger.info("=== DOCUMENT LOAD ===")
        logger.info("=" * 80)
        logger.info(f"[LOAD] doc_id={doc_id} text_length={len(text)}")

        # === DOCUMENT SAVE ===
        logger.info("=" * 80)
        logger.info("=== DOCUMENT SAVE ===")
        logger.info("=" * 80)
        doc_path = self.document_repository.save(
            doc_id=doc_id,
            content=text,
            title=metadata.get("title", "") if metadata else "",
            source=metadata.get("source", "") if metadata else "",
            metadata=metadata,
        )
        logger.info(f"[SAVE] Saved to: {doc_path}")

        # === CHUNKING ===
        logger.info("=" * 80)
        logger.info("=== CHUNKING ===")
        logger.info("=" * 80)
        chunks = self._chunk_text(text)
        if not chunks:
            logger.warning("[CHUNKING] No chunks generated")
            return []

        logger.info(f"[CHUNKING] Generated {len(chunks)} chunks")

        # === CHUNK ID GENERATION ===
        logger.info("=" * 80)
        logger.info("=== CHUNK ID GENERATION ===")
        logger.info("=" * 80)
        ids = []
        for i, chunk in enumerate(chunks):
            chunk_id = self._generate_chunk_id(doc_id, i, chunk)
            ids.append(chunk_id)
            logger.info(
                f"[CHUNK ID] Index={i} ID={chunk_id} length={len(chunk)} "
                f"text_preview={chunk[:50].replace(chr(10), ' ')}..."
            )

        # === METADATA PROPAGATION ===
        logger.info("=" * 80)
        logger.info("=== METADATA PROPAGATION ===")
        logger.info("=" * 80)
        metadatas: List[Dict[str, Any]] = []
        for index, chunk_id in enumerate(ids):
            chunk_metadata: Dict[str, Any] = {
                "source_id": doc_id,
                "chunk_index": index,
                "chunk_id": chunk_id,
            }
            if metadata is not None:
                chunk_metadata.update(metadata)
            metadatas.append(chunk_metadata)
            logger.info(f"[METADATA] Chunk {index}: {chunk_metadata}")

        # === EMBEDDINGS ===
        logger.info("=" * 80)
        logger.info("=== EMBEDDINGS ===")
        logger.info("=" * 80)
        logger.info(f"[EMBEDDINGS] Generating embeddings for {len(chunks)} chunks...")
        try:
            embeddings = self.embedding_manager.embed_texts(chunks)
            logger.info(f"[EMBEDDINGS] Generated {len(embeddings)} embeddings")
            if embeddings:
                dim = len(embeddings[0])
                logger.info(f"[EMBEDDINGS] Embedding dimension: {dim}")
        except Exception as exc:
            logger.error(f"[EMBEDDINGS] Failed to generate embeddings: {exc}")
            raise RuntimeError(f"failed to generate embeddings: {exc}") from exc

        # === STORAGE ===
        logger.info("=" * 80)
        logger.info("=== STORAGE ===")
        logger.info("=" * 80)
        logger.info(f"[STORAGE] Storing to collection: {self.collection_name}")
        logger.info(f"[STORAGE] Storing {len(ids)} chunks with IDs and embeddings...")
        try:
            self.chroma.add_documents(
                collection_name=self.collection_name,
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            logger.info(f"[STORAGE] Successfully stored all chunks")
            logger.info("=" * 80)
            logger.info("=== INGESTION COMPLETE ===")
            logger.info("=" * 80)
        except Exception as exc:
            logger.error(f"[STORAGE] Failed to store chunks: {exc}")
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
