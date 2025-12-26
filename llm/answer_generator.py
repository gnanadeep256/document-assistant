import os
from typing import Literal, Optional

from llm.offline_ollama import OllamaClient
from llm.online_gemini import GeminiClient
from llm.prompts import grounded_qa_prompt

LLMProvider = Literal["ollama", "gemini"]


class AnswerGenerator:
    """High-level wrapper that chooses between Ollama and Gemini.

    Provider can be selected via constructor or `LLM_PROVIDER` env var
    ("ollama" or "gemini"). Defaults to Ollama so it works out of the box
    if you already have an Ollama server running locally.
    """

    def __init__(self, provider: Optional[str] = None):
        raw = (provider or os.getenv("LLM_PROVIDER", "ollama")).lower()

        if raw not in ("ollama", "gemini"):
            raise ValueError(f"Unsupported LLM_PROVIDER: {raw}")

        self.provider: LLMProvider = raw  # type: ignore[assignment]

        if self.provider == "gemini":
            self.llm = GeminiClient()
        else:
            # Default path: local Ollama
            self.llm = OllamaClient()

    def generate(self, context: str, question: str) -> str:
        prompt = grounded_qa_prompt(context, question)
        return self.llm.generate(prompt)
