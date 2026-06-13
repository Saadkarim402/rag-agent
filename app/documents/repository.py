"""Document repository for storing and retrieving original documents."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DocumentRepository:
    """Manages storage and retrieval of original documents independently of chunks.

    Stores documents as JSON files in a local filesystem with deterministic naming.
    This allows re-indexing, chunking strategy changes, and evaluation datasets
    without re-uploading source content.
    """

    def __init__(self, repo_dir: str | Path = "data/documents") -> None:
        """Initialize repository with a storage directory.

        Args:
            repo_dir: Path to the documents directory. Will be created if missing.
        """
        self.repo_dir = Path(repo_dir)
        self.repo_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[DOCUMENT REPO] Initialized at {self.repo_dir}")

    def save(
        self,
        doc_id: str,
        content: str,
        title: str = "",
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save an original document to the repository.

        Args:
            doc_id: Unique document identifier.
            content: Full document text content.
            title: Human-readable document title.
            source: Source of the document (file path, URL, etc.).
            metadata: Optional metadata dictionary.

        Returns:
            Path to the saved document file.

        Raises:
            ValueError: If doc_id is empty or content is not a string.
        """
        if not doc_id or not isinstance(doc_id, str):
            raise ValueError("doc_id must be a non-empty string")
        if not isinstance(content, str):
            raise ValueError("content must be a string")

        document = {
            "doc_id": doc_id,
            "title": title,
            "source": source,
            "content": content,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
        }

        file_path = self.repo_dir / f"{doc_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(document, f, indent=2, ensure_ascii=False)

        logger.info(f"[DOCUMENT REPO] Saved document: {doc_id} ({len(content)} chars)")
        return str(file_path)

    def load(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Load a document from the repository by ID.

        Args:
            doc_id: The document identifier.

        Returns:
            Document dictionary with keys: doc_id, title, source, content, metadata, created_at.
            None if document not found.
        """
        file_path = self.repo_dir / f"{doc_id}.json"

        if not file_path.exists():
            logger.warning(f"[DOCUMENT REPO] Document not found: {doc_id}")
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                document = json.load(f)
            logger.info(f"[DOCUMENT REPO] Loaded document: {doc_id}")
            return document
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"[DOCUMENT REPO] Failed to load {doc_id}: {e}")
            return None

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all stored documents with basic metadata.

        Returns:
            List of documents with keys: doc_id, title, source, created_at.
        """
        documents = []
        for file_path in sorted(self.repo_dir.glob("*.json")):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                documents.append({
                    "doc_id": doc["doc_id"],
                    "title": doc.get("title", ""),
                    "source": doc.get("source", ""),
                    "created_at": doc.get("created_at", ""),
                })
            except (json.JSONDecodeError, IOError):
                continue

        logger.info(f"[DOCUMENT REPO] Listed {len(documents)} documents")
        return documents

    def delete(self, doc_id: str) -> bool:
        """Delete a document from the repository.

        Args:
            doc_id: The document identifier.

        Returns:
            True if deleted successfully, False if not found.
        """
        file_path = self.repo_dir / f"{doc_id}.json"

        if not file_path.exists():
            logger.warning(f"[DOCUMENT REPO] Cannot delete: document not found: {doc_id}")
            return False

        try:
            file_path.unlink()
            logger.info(f"[DOCUMENT REPO] Deleted document: {doc_id}")
            return True
        except OSError as e:
            logger.error(f"[DOCUMENT REPO] Failed to delete {doc_id}: {e}")
            return False

    def exists(self, doc_id: str) -> bool:
        """Check if a document exists in the repository.

        Args:
            doc_id: The document identifier.

        Returns:
            True if document exists, False otherwise.
        """
        return (self.repo_dir / f"{doc_id}.json").exists()

    def get_content(self, doc_id: str) -> Optional[str]:
        """Get just the content of a document.

        Args:
            doc_id: The document identifier.

        Returns:
            Document content string, or None if not found.
        """
        document = self.load(doc_id)
        return document["content"] if document else None

    def get_path(self, doc_id: str) -> str:
        """Get the file path for a document.

        Args:
            doc_id: The document identifier.

        Returns:
            Full path to the document file.
        """
        return str(self.repo_dir / f"{doc_id}.json")
