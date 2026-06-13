import requests
from typing import Optional
from app.llm.base import BaseLLMClient

class OllamaClient(BaseLLMClient):
    """Client for local Ollama instance generation."""

    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3", timeout: int = 30) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        if not prompt:
            raise ValueError("prompt must not be empty")

        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system_instruction:
            payload["system"] = system_instruction

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}") from e
