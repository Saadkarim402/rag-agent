import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import pytest
from fastapi.testclient import TestClient

from app.api.server import app
from app.documents.repository import DocumentRepository
from app.ingestion.ingest import DocumentIngestionManager
from app.llm.base import BaseLLMClient
from app.llm.chains import RAGChain
from app.retrieval.retriever import RetrievalManager
from app.vectordb.chroma_client import ChromaDBManager


class MockLLMClient(BaseLLMClient):
    """Simple mock LLM client to return deterministic responses during API tests."""

    def __init__(self, response_text: str = "Mock REST response") -> None:
        self.response_text = response_text
        self.last_prompt: Optional[str] = None
        self.last_system_instruction: Optional[str] = None

    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        self.last_prompt = prompt
        self.last_system_instruction = system_instruction
        return self.response_text


@pytest.fixture()
def api_client(tmp_path):
    """Set up isolated resources, override FastAPI app.state, and return TestClient."""
    # Create temp paths
    db_dir = tmp_path / "test_api_db"
    repo_dir = tmp_path / "test_api_repo"
    
    # Initialize managers
    chroma = ChromaDBManager(persist_directory=str(db_dir))
    repository = DocumentRepository(repo_dir=repo_dir)
    ingest_manager = DocumentIngestionManager(
        chroma=chroma,
        document_repository=repository,
        collection_name="api_test_collection"
    )
    retriever = RetrievalManager(
        chroma=chroma,
        collection_name="api_test_collection"
    )
    llm = MockLLMClient("ChromaDB and FastAPI are integrated.")
    rag_chain = RAGChain(
        retriever=retriever,
        llm_client=llm,
        default_collection_name="api_test_collection"
    )

    # Backup original states
    old_state = app.state

    # Inject mock states
    app.state.chroma = chroma
    app.state.document_repository = repository
    app.state.ingest_manager = ingest_manager
    app.state.retriever = retriever
    app.state.llm_client = llm
    app.state.rag_chain = rag_chain

    client = TestClient(app)
    yield client

    # Restore app state
    app.state = old_state



def test_ingest_text_api(api_client):
    payload = {
        "doc_id": "test-doc",
        "text": "FastAPI is a modern web framework for building APIs with Python.",
        "metadata": {"author": "Saad", "topic": "web"}
    }
    response = api_client.post("/ingest/text", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["doc_id"] == "test-doc"
    assert len(data["chunk_ids"]) > 0


def test_ingest_file_api(api_client):
    file_content = b"Docker packages applications into portable container images."
    response = api_client.post(
        "/ingest/file",
        files={"file": ("docker_guide.txt", file_content)},
        data={"doc_id": "custom-docker-id"}
    )
    assert response.status_code == 201
    data = response.json()
    assert data["doc_id"] == "custom-docker-id"
    assert len(data["chunk_ids"]) > 0


def test_query_chat_api(api_client):
    # Ingest context document first
    ingest_payload = {
        "doc_id": "k8s-doc",
        "text": "Kubernetes coordinates container deployment, scaling, and networking across clusters.",
        "metadata": {"source_id": "k8s-doc"}
    }
    api_client.post("/ingest/text", json=ingest_payload)

    # Query RAG Chain
    query_payload = {
        "query": "What does Kubernetes coordinate?",
        "top_k": 3,
        "min_score_threshold": 0.0
    }
    response = api_client.post("/chat", json=query_payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["answer"] == "ChromaDB and FastAPI are integrated."
    assert len(data["source_nodes"]) >= 1
    assert any("Kubernetes" in node["document_text"] for node in data["source_nodes"])
    assert "k8s-doc" in data["prompt"]


def test_list_documents_api(api_client):
    # Ingest document
    payload = {
        "doc_id": "list-doc-1",
        "text": "Document text for listing.",
        "metadata": {"title": "Doc Title 1", "source": "unit-test"}
    }
    api_client.post("/ingest/text", json=payload)

    response = api_client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    doc_ids = [d["doc_id"] for d in data]
    assert "list-doc-1" in doc_ids
    assert any(d["title"] == "Doc Title 1" for d in data)


def test_delete_document_api(api_client):
    # Ingest document
    payload = {
        "doc_id": "delete-me",
        "text": "This document will be deleted.",
        "metadata": {"title": "Delete Target"}
    }
    api_client.post("/ingest/text", json=payload)

    # Verify exists
    assert app.state.document_repository.exists("delete-me")

    # Delete
    response = api_client.delete("/documents/delete-me")
    assert response.status_code == 200
    assert "deleted successfully" in response.json()["detail"]

    # Verify deleted
    assert not app.state.document_repository.exists("delete-me")

    # Double delete -> 404
    response = api_client.delete("/documents/delete-me")
    assert response.status_code == 404
