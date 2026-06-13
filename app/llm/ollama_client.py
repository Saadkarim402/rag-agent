import requests
from typing import Optional
from app.llm.base import BaseLLMClient

class OllamaClient(BaseLLMClient):
    """Client for local Ollama instance generation."""

    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3", timeout: int = 30, fallback_to_simulation: bool = True) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.fallback_to_simulation = fallback_to_simulation

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
            is_network_err = isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout))
            if self.fallback_to_simulation and is_network_err:
                import logging
                import re
                logger = logging.getLogger(__name__)
                logger.warning(f"Ollama connection failed ({e}). Generating a simulated response based on context.")

                # Parse context and query from the prompt
                context_str = ""
                context_match = re.search(r"Context:\n(.*?)\n\nQuestion:", prompt, re.DOTALL)
                if context_match:
                    context_str = context_match.group(1).strip()

                if not context_str or context_str.strip() == "":
                    return "I cannot answer this based on the provided context."

                # Split the formatted context into original chunks
                chunks = re.split(r"\[\d+\]\s+\(Source:\s*[^)]+\)\s*", context_str)
                chunks = [c.strip() for c in chunks if c.strip()]

                if not chunks:
                    return "I cannot answer this based on the provided context."

                # Build a summary answer using the retrieved chunks
                answers = []
                for i, chunk in enumerate(chunks, 1):
                    # Clean/extract the first sentence or two of each chunk to make a summary
                    sentences = re.split(r"(?<=[.!?])\s+", chunk)
                    if sentences:
                        answers.append(f"{sentences[0]} [{i}].")

                summary = " ".join(answers)
                return (
                    f"🤖 **[Simulated Response - Ollama Offline]**\n\n"
                    f"Based on the retrieved context, here is the answer:\n"
                    f"{summary}"
                )

            raise RuntimeError(f"Ollama generation failed: {e}") from e

