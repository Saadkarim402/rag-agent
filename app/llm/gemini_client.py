import os
import requests
from typing import Optional
from app.llm.base import BaseLLMClient

class GeminiClient(BaseLLMClient):
    """Client for Google Gemini REST API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash", timeout: int = 30) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set. Please provide an API key or set the GEMINI_API_KEY environment variable.")
        
        if not prompt:
            raise ValueError("prompt must not be empty")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }
        
        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [
                    {"text": system_instruction}
                ]
            }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            candidates = data.get("candidates", [])
            if not candidates:
                return ""
            
            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                return ""
                
            return parts[0].get("text", "")
        except Exception as e:
            raise RuntimeError(f"Gemini generation failed: {e}") from e
