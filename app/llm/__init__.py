from typing import Any
from app.llm.base import BaseLLMClient
from app.llm.ollama_client import OllamaClient
from app.llm.gemini_client import GeminiClient
from app.llm.chains import RAGChain, RAGResponse

def get_llm_client(provider: str, **kwargs: Any) -> BaseLLMClient:
    """Factory function to retrieve an LLM client instance.

    Args:
        provider: The name of the provider ('ollama' or 'gemini').
        **kwargs: Configuration keyword arguments for the chosen client.

    Returns:
        BaseLLMClient instance.

    Raises:
        ValueError: If provider is unsupported.
    """
    prov = provider.lower().strip()
    if prov == "ollama":
        return OllamaClient(**kwargs)
    elif prov == "gemini":
        return GeminiClient(**kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Supported providers: 'ollama', 'gemini'.")

