import pytest
from unittest.mock import patch, MagicMock
from app.ui.streamlit_app import (
    is_backend_online,
    get_documents_list,
    delete_document_by_id,
    upload_document_file,
    query_rag_engine
)

def test_is_backend_online_success():
    mock_response = MagicMock()
    mock_response.status_code = 200
    with patch("requests.get", return_value=mock_response) as mock_get:
        assert is_backend_online() is True
        mock_get.assert_called_once()

def test_is_backend_online_failure():
    with patch("requests.get", side_effect=Exception("Connection error")):
        assert is_backend_online() is False

def test_get_documents_list_success():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [{"doc_id": "doc1", "title": "Doc 1"}]
    with patch("requests.get", return_value=mock_response) as mock_get:
        docs = get_documents_list()
        assert docs == [{"doc_id": "doc1", "title": "Doc 1"}]

def test_delete_document_by_id_success():
    mock_response = MagicMock()
    mock_response.status_code = 200
    with patch("requests.delete", return_value=mock_response) as mock_delete:
        assert delete_document_by_id("doc1") is True
        mock_delete.assert_called_once_with("http://localhost:8000/documents/doc1", timeout=5)

def test_upload_document_file_success():
    mock_response = MagicMock()
    mock_response.status_code = 201
    with patch("requests.post", return_value=mock_response) as mock_post:
        assert upload_document_file(b"test data", "test.txt", "doc_override") is True
        mock_post.assert_called_once()
        # Ensure it passed override as data
        args, kwargs = mock_post.call_args
        assert kwargs["data"] == {"doc_id": "doc_override"}

def test_query_rag_engine_success():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"answer": "Paris is in France", "source_nodes": []}
    with patch("requests.post", return_value=mock_response) as mock_post:
        res = query_rag_engine("Where is Paris?", "my_collection", 3, 0.5)
        assert res == {"answer": "Paris is in France", "source_nodes": []}
        mock_post.assert_called_once_with(
            "http://localhost:8000/chat",
            json={
                "query": "Where is Paris?",
                "collection_name": "my_collection",
                "top_k": 3,
                "min_score_threshold": 0.5,
                "chat_history": None
            },
            timeout=30
        )

def test_query_rag_engine_with_history():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"answer": "It is in France.", "source_nodes": []}
    with patch("requests.post", return_value=mock_response) as mock_post:
        history = [{"role": "user", "content": "Where is Paris?"}]
        res = query_rag_engine("Is it beautiful?", "my_collection", 3, 0.5, chat_history=history)
        assert res == {"answer": "It is in France.", "source_nodes": []}
        mock_post.assert_called_once_with(
            "http://localhost:8000/chat",
            json={
                "query": "Is it beautiful?",
                "collection_name": "my_collection",
                "top_k": 3,
                "min_score_threshold": 0.5,
                "chat_history": history
            },
            timeout=30
        )
