import pytest
from unittest.mock import MagicMock, patch
import requests

from app.llm import get_llm_client, OllamaClient, GeminiClient

def test_factory_resolves_providers():
    ollama = get_llm_client("ollama", model="phi3")
    assert isinstance(ollama, OllamaClient)
    assert ollama.model == "phi3"

    gemini = get_llm_client("gemini", api_key="test-key")
    assert isinstance(gemini, GeminiClient)
    assert gemini.api_key == "test-key"

    with pytest.raises(ValueError):
        get_llm_client("unknown")

def test_ollama_client_success():
    client = OllamaClient(host="http://localhost:11434", model="llama3")
    
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "Mocked LLM response"}
    mock_response.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_response) as mock_post:
        res = client.generate(prompt="hello", system_instruction="be friendly")
        assert res == "Mocked LLM response"
        mock_post.assert_called_once_with(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": "hello",
                "stream": False,
                "system": "be friendly"
            },
            timeout=30
        )

def test_ollama_client_failure():
    client = OllamaClient(fallback_to_simulation=False)
    with patch("requests.post", side_effect=requests.RequestException("Connection error")):
        with pytest.raises(RuntimeError):
            client.generate("hello")

def test_gemini_client_missing_key():
    client = GeminiClient(api_key=None)
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="GEMINI_API_KEY is not set"):
            client.generate("hello")

def test_gemini_client_success():
    client = GeminiClient(api_key="my-key", model="gemini-1.5-flash")

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "Hello from Gemini"}
                    ]
                }
            }
        ]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_response) as mock_post:
        res = client.generate(prompt="hi", system_instruction="talk short")
        assert res == "Hello from Gemini"
        mock_post.assert_called_once_with(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=my-key",
            json={
                "contents": [
                    {
                        "parts": [
                            {"text": "hi"}
                        ]
                    }
                ],
                "systemInstruction": {
                    "parts": [
                        {"text": "talk short"}
                    ]
                }
            },
            timeout=30
        )

def test_gemini_client_failure():
    client = GeminiClient(api_key="my-key")
    with patch("requests.post", side_effect=requests.HTTPError("API error")):
        with pytest.raises(RuntimeError):
            client.generate("hi")
