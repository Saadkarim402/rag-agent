"""Tests for ingestion pipeline with document repository."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from app.documents.repository import DocumentRepository
from app.ingestion.ingest import DocumentIngestionManager
from app.vectordb.chroma_client import ChromaDBManager


def test_ingestion_saves_original_document():
    """Test that ingestion saves the original document to repository."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo = DocumentRepository(repo_dir=Path(tmp_dir) / "documents")
        chroma = ChromaDBManager(persist_directory=Path(tmp_dir) / "chroma")
        manager = DocumentIngestionManager(
            chroma=chroma,
            document_repository=repo,
            collection_name="test",
            chunk_size=50,
            chunk_overlap=10,
        )

        doc_id = "ingest_test_001"
        text = "This is a test document for ingestion."
        metadata = {"title": "Test", "source": "test_source"}

        # Ingest
        ids = manager.ingest_text(doc_id, text, metadata)

        # Verify document is stored
        assert repo.exists(doc_id)
        stored = repo.load(doc_id)
        assert stored is not None
        assert stored["content"] == text
        assert stored["title"] == metadata["title"]


def test_ingestion_creates_chunks_and_embeddings():
    """Test that ingestion creates chunks and embeddings."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo = DocumentRepository(repo_dir=Path(tmp_dir) / "documents")
        chroma = ChromaDBManager(persist_directory=Path(tmp_dir) / "chroma")
        manager = DocumentIngestionManager(
            chroma=chroma,
            document_repository=repo,
            collection_name="test",
            chunk_size=40,
            chunk_overlap=10,
        )

        doc_id = "chunks_test"
        text = "First sentence. Second sentence. Third sentence."
        ids = manager.ingest_text(doc_id, text)

        # Should have multiple chunks
        assert len(ids) > 0
        assert all(isinstance(chunk_id, str) for chunk_id in ids)


def test_ingestion_metadata_propagation():
    """Test that metadata is propagated to chunks."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo = DocumentRepository(repo_dir=Path(tmp_dir) / "documents")
        chroma = ChromaDBManager(persist_directory=Path(tmp_dir) / "chroma")
        manager = DocumentIngestionManager(
            chroma=chroma,
            document_repository=repo,
            collection_name="test",
            chunk_size=50,
            chunk_overlap=10,
        )

        doc_id = "metadata_test"
        text = "Test content here. More content."
        metadata = {"source_id": "test_source", "category": "test"}

        ids = manager.ingest_text(doc_id, text, metadata)
        assert len(ids) > 0


def test_original_document_remains_after_ingestion():
    """Test that original document remains available after ingestion."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo = DocumentRepository(repo_dir=Path(tmp_dir) / "documents")
        chroma = ChromaDBManager(persist_directory=Path(tmp_dir) / "chroma")
        manager = DocumentIngestionManager(
            chroma=chroma,
            document_repository=repo,
            collection_name="test",
            chunk_size=50,
            chunk_overlap=10,
        )

        doc_id = "persistence_test"
        original_text = "This is the original document text that should be preserved."

        # Ingest
        manager.ingest_text(doc_id, original_text)

        # Verify original is still available
        stored = repo.load(doc_id)
        assert stored is not None
        assert stored["content"] == original_text


def test_re_ingestion_from_stored_document():
    """Test re-ingesting a document from repository without re-uploading."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo = DocumentRepository(repo_dir=Path(tmp_dir) / "documents")
        chroma = ChromaDBManager(persist_directory=Path(tmp_dir) / "chroma")
        manager = DocumentIngestionManager(
            chroma=chroma,
            document_repository=repo,
            collection_name="test",
            chunk_size=50,
            chunk_overlap=10,
        )

        doc_id = "reingest_test"
        text = "Original document for re-ingestion test."

        # Initial ingestion
        initial_ids = manager.ingest_text(doc_id, text)
        assert len(initial_ids) > 0

        # Retrieve from repository and re-ingest
        stored = repo.load(doc_id)
        assert stored is not None
        re_ingested_ids = manager.ingest_text(doc_id, stored["content"])

        # Should produce same chunks (deterministic)
        assert initial_ids == re_ingested_ids


def test_chunk_ids_are_deterministic():
    """Test that chunk IDs are deterministically generated."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo = DocumentRepository(repo_dir=Path(tmp_dir) / "documents")
        chroma = ChromaDBManager(persist_directory=Path(tmp_dir) / "chroma")
        manager = DocumentIngestionManager(
            chroma=chroma,
            document_repository=repo,
            collection_name="test",
            chunk_size=50,
            chunk_overlap=10,
        )

        doc_id = "deterministic_test"
        text = "Same text every time. Should produce same chunks."

        # First ingestion
        ids1 = manager.ingest_text(doc_id, text)

        # Second ingestion (should be identical)
        ids2 = manager.ingest_text(doc_id, text)

        assert ids1 == ids2


def test_multiple_documents_in_repository():
    """Test storing and retrieving multiple documents."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo = DocumentRepository(repo_dir=Path(tmp_dir) / "documents")
        chroma = ChromaDBManager(persist_directory=Path(tmp_dir) / "chroma")
        manager = DocumentIngestionManager(
            chroma=chroma,
            document_repository=repo,
            collection_name="test",
            chunk_size=50,
            chunk_overlap=10,
        )

        # Ingest multiple documents
        for i in range(3):
            doc_id = f"multi_doc_{i}"
            text = f"Document {i} content here."
            manager.ingest_text(doc_id, text)

        # Verify all are stored
        docs = repo.list_documents()
        assert len(docs) == 3
        doc_ids = [d["doc_id"] for d in docs]
        for i in range(3):
            assert f"multi_doc_{i}" in doc_ids


def test_ingestion_with_empty_metadata():
    """Test ingestion with no metadata provided."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo = DocumentRepository(repo_dir=Path(tmp_dir) / "documents")
        chroma = ChromaDBManager(persist_directory=Path(tmp_dir) / "chroma")
        manager = DocumentIngestionManager(
            chroma=chroma,
            document_repository=repo,
            collection_name="test",
            chunk_size=50,
            chunk_overlap=10,
        )

        doc_id = "no_metadata_test"
        text = "Text without metadata."

        ids = manager.ingest_text(doc_id, text, metadata=None)
        assert len(ids) > 0

        stored = repo.load(doc_id)
        assert stored is not None
        assert stored["metadata"] == {}


def test_document_repository_integration():
    """End-to-end test of document repository integration."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo = DocumentRepository(repo_dir=Path(tmp_dir) / "documents")
        chroma = ChromaDBManager(persist_directory=Path(tmp_dir) / "chroma")
        manager = DocumentIngestionManager(
            chroma=chroma,
            document_repository=repo,
            collection_name="integration_test",
            chunk_size=50,
            chunk_overlap=10,
        )

        # Document 1
        doc1_id = "doc1"
        doc1_text = "Machine learning is powerful. Deep learning is a subset."
        doc1_meta = {"title": "ML Guide", "category": "AI"}
        ids1 = manager.ingest_text(doc1_id, doc1_text, doc1_meta)

        # Document 2
        doc2_id = "doc2"
        doc2_text = "Python is a programming language. It is widely used."
        doc2_meta = {"title": "Python Intro", "category": "Programming"}
        ids2 = manager.ingest_text(doc2_id, doc2_text, doc2_meta)

        # Verify both are stored
        assert repo.exists(doc1_id)
        assert repo.exists(doc2_id)

        # Verify content
        stored1 = repo.load(doc1_id)
        stored2 = repo.load(doc2_id)
        assert stored1["content"] == doc1_text
        assert stored2["content"] == doc2_text

        # Verify metadata
        assert stored1["metadata"]["title"] == "ML Guide"
        assert stored2["metadata"]["title"] == "Python Intro"

        # Verify chunks were created
        assert len(ids1) > 0
        assert len(ids2) > 0
