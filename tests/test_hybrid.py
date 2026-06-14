from unittest.mock import MagicMock, patch

import pytest

from app.retrieval.hybrid import BM25Retriever, HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.retriever import RetrievalResult
from app.tools.web_search import DuckDuckGoSearchTool


def test_bm25_retriever():
    retriever = BM25Retriever()
    chunks = [
        {"id": "doc1", "text": "Kubernetes is an open-source container orchestration system."},
        {
            "id": "doc2",
            "text": "Docker is a tool designed to make it easier to create and run applications in containers.",
        },
    ]

    # Search for container orchestration
    results = retriever.retrieve_scores("container orchestration", chunks)
    assert len(results) > 0
    assert results[0][0]["id"] == "doc1"

    # Verify tokenization
    tokens = retriever._tokenize("Hello World! 123")
    assert tokens == ["hello", "world", "123"]


def test_cross_encoder_reranker():
    reranker = CrossEncoderReranker(semantic_weight=0.5)
    results = [
        RetrievalResult(
            chunk_id="c1",
            document_text="Python programming language tutorial",
            metadata={},
            distance=0.2,
            score=0.8,
            collection="test",
        ),
        RetrievalResult(
            chunk_id="c2",
            document_text="Java programming language tutorial",
            metadata={},
            distance=0.3,
            score=0.7,
            collection="test",
        ),
    ]

    reranked = reranker.rerank("Python programming", results)
    assert len(reranked) == 2
    assert reranked[0].chunk_id == "c1"


def test_ddg_search_tool():
    tool = DuckDuckGoSearchTool(max_results=2)
    mock_html = """
    <div class="web-result">
        <a class="result__a" href="https://example.com/k8s">Kubernetes Documentation</a>
        <a class="result__snippet" href="https://example.com/k8s">Kubernetes is a container management system.</a>
    </div>
    <div class="web-result">
        <a class="result__a" href="https://example.com/docker">Docker Homepage</a>
        <a class="result__snippet" href="https://example.com/docker">Docker containers simplify app deployment.</a>
    </div>
    """

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = mock_html
    mock_response.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_response) as mock_get:
        hits = tool.search("k8s")
        assert len(hits) == 2
        assert hits[0]["title"] == "Kubernetes Documentation"
        assert hits[0]["url"] == "https://example.com/k8s"
        assert "container management" in hits[0]["snippet"]
        mock_get.assert_called_once()
