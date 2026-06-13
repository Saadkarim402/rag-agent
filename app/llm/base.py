from abc import ABC, abstractmethod
from typing import Optional

class BaseLLMClient(ABC):
    """Abstract base class for LLM client integrations."""

    @abstractmethod
    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Generate a response for a prompt and optional system instructions.

        Args:
            prompt: The user query or compiled prompt.
            system_instruction: Optional system instruction/persona.

        Returns:
            The generated response as a string.

        Raises:
            RuntimeError: If the API call fails or model returns error.
        """
        pass
