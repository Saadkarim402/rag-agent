from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import chromadb
from chromadb.api.client import Client
from chromadb.config import Settings
from chromadb.api.models import Collection


class ChromaDBManager:
    """Manage persistent ChromaDB storage and collection operations."""

    _client: Client | None = None
    _persist_directory: Path | None = None

    def __init__(self, persist_directory: str = "./chroma_db") -> None:
        self._persist_directory = Path(persist_directory).expanduser().resolve()
        self._client = self._get_client(self._persist_directory)

    @classmethod
    def _get_client(cls, persist_directory: Optional[Path] = None) -> Client:
        """Initialize and cache a persistent ChromaDB client."""
        if cls._client is None:
            if persist_directory is None:
                raise ValueError("persist_directory must be provided for first client initialization")
            cls._client = chromadb.Client(Settings(chromadb_impl="chromadb.db.duckdb.DuckDB", persist_directory=str(persist_directory)))
        return cls._client

    def _ensure_client(self) -> Client:
        if self._client is None or self._persist_directory is None:
            raise RuntimeError("ChromaDB client is not initialized")
        return self._client

    def get_collection(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> Collection: # type: ignore
        """Load an existing collection or create a new one in the persistent store."""
        if not name:
            raise ValueError("collection name must not be empty")

        client = self._ensure_client()
        if metadata is None:
            return client.get_or_create_collection(name=name)
        return client.get_or_create_collection(name=name, metadata=metadata)

    def add_documents(
        self,
        collection_name: str,
        ids: Iterable[str],
        documents: Iterable[str],
        embeddings: Iterable[List[float]],
        metadatas: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> None:
        """Add embeddings, documents, and optional metadata to a collection."""
        if not collection_name:
            raise ValueError("collection_name must not be empty")

        ids_list = list(ids)
        documents_list = list(documents)
        embeddings_list = list(embeddings)

        if len(ids_list) != len(documents_list) or len(ids_list) != len(embeddings_list):
            raise ValueError("ids, documents, and embeddings must have the same length")

        metadatas_list: Optional[List[Dict[str, Any]]] = None
        if metadatas is not None:
            metadatas_list = list(metadatas)
            if len(metadatas_list) != len(ids_list):
                raise ValueError("metadatas must have the same length as ids")

        collection = self.get_collection(collection_name)
        collection.add(ids=ids_list, documents=documents_list, embeddings=embeddings_list, metadatas=metadatas_list)

    def query_embeddings(
        self,
        collection_name: str,
        query_embeddings: Iterable[List[float]],
        n_results: int = 10,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        """Query a collection for similar embeddings."""
        if not collection_name:
            raise ValueError("collection_name must not be empty")

        query_list = list(query_embeddings)
        if not query_list:
            raise ValueError("query_embeddings must contain at least one vector")

        collection = self.get_collection(collection_name)
        include_fields = ["documents", "distances"]
        if include_metadata:
            include_fields.append("metadatas")

        return collection.query(
            query_embeddings=query_list,
            n_results=n_results,
            include=include_fields,
        )

    def delete_documents(self, collection_name: str, ids: Iterable[str]) -> None:
        """Delete documents from a collection by their IDs."""
        if not collection_name:
            raise ValueError("collection_name must not be empty")

        ids_list = list(ids)
        if not ids_list:
            raise ValueError("ids must contain at least one identifier")

        collection = self.get_collection(collection_name)
        collection.delete(ids=ids_list)

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Return collection metadata and document count."""
        if not collection_name:
            raise ValueError("collection_name must not be empty")

        collection = self.get_collection(collection_name)
        info = collection.count()
        return {
            "name": collection_name,
            "metadata": collection.metadata or {},
            "count": info,
        }
