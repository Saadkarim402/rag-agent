"""Tests for DocumentRepository."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from app.documents.repository import DocumentRepository


def test_repository_save_and_load():
    """Test saving and loading a document."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo = DocumentRepository(repo_dir=tmp_dir)

        doc_id = "test_doc_001"
        content = "This is test content."
        title = "Test Document"
        source = "test_source"
        metadata = {"category": "test"}

        # Save
        path = repo.save(doc_id, content, title, source, metadata)
        assert Path(path).exists()

        # Load
        loaded = repo.load(doc_id)
        assert loaded is not None
        assert loaded["doc_id"] == doc_id
        assert loaded["content"] == content
        assert loaded["title"] == title
        assert loaded["source"] == source
        assert loaded["metadata"] == metadata
        assert "created_at" in loaded


def test_repository_save_validates_input():
    """Test that save validates inputs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo = DocumentRepository(repo_dir=tmp_dir)

        with pytest.raises(ValueError, match="doc_id must be"):
            repo.save("", "content")

        with pytest.raises(ValueError, match="content must be"):
            repo.save("id", 123)  # type: ignore


def test_repository_get_content():
    """Test getting just the content of a document."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo = DocumentRepository(repo_dir=tmp_dir)

        doc_id = "content_test"
        content = "Full document content goes here."
        repo.save(doc_id, content)

        retrieved_content = repo.get_content(doc_id)
        assert retrieved_content == content

        missing = repo.get_content("missing_id")
        assert missing is None


def test_repository_list_documents():
    """Test listing all stored documents."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo = DocumentRepository(repo_dir=tmp_dir)

        # Initially empty
        docs = repo.list_documents()
        assert len(docs) == 0

        # Add documents
        repo.save("doc1", "content1", title="Doc 1", source="source1")
        repo.save("doc2", "content2", title="Doc 2", source="source2")

        # List
        docs = repo.list_documents()
        assert len(docs) == 2
        assert any(d["doc_id"] == "doc1" for d in docs)
        assert any(d["doc_id"] == "doc2" for d in docs)


def test_repository_exists():
    """Test checking if a document exists."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo = DocumentRepository(repo_dir=tmp_dir)

        doc_id = "exist_test"
        assert not repo.exists(doc_id)

        repo.save(doc_id, "content")
        assert repo.exists(doc_id)


def test_repository_delete():
    """Test deleting a document."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo = DocumentRepository(repo_dir=tmp_dir)

        doc_id = "delete_test"
        repo.save(doc_id, "content")
        assert repo.exists(doc_id)

        # Delete
        success = repo.delete(doc_id)
        assert success
        assert not repo.exists(doc_id)

        # Delete non-existent returns False
        success = repo.delete("missing")
        assert not success


def test_repository_persistence():
    """Test that saved documents persist across repository instances."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create and save
        repo1 = DocumentRepository(repo_dir=tmp_dir)
        doc_id = "persist_test"
        content = "Persistent content"
        repo1.save(doc_id, content)

        # Load with new instance
        repo2 = DocumentRepository(repo_dir=tmp_dir)
        loaded = repo2.load(doc_id)
        assert loaded is not None
        assert loaded["content"] == content


def test_repository_get_path():
    """Test getting the file path for a document."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo = DocumentRepository(repo_dir=tmp_dir)
        doc_id = "path_test"

        path = repo.get_path(doc_id)
        assert str(doc_id) in path
        assert path.endswith(".json")


def test_repository_json_format():
    """Test that documents are stored as valid JSON."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo = DocumentRepository(repo_dir=tmp_dir)

        doc_id = "json_test"
        content = "Test content"
        metadata = {"key": "value", "number": 42}
        repo.save(doc_id, content, metadata=metadata)

        # Read the JSON file directly
        file_path = Path(tmp_dir) / f"{doc_id}.json"
        with open(file_path, "r") as f:
            data = json.load(f)

        assert data["doc_id"] == doc_id
        assert data["content"] == content
        assert data["metadata"] == metadata


def test_repository_empty_content():
    """Test saving and retrieving empty content."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo = DocumentRepository(repo_dir=tmp_dir)

        doc_id = "empty_test"
        repo.save(doc_id, "")

        loaded = repo.load(doc_id)
        assert loaded is not None
        assert loaded["content"] == ""
