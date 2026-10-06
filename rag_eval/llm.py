"""Pluggable LLM layer.

Everything else in the app calls `get_llm().complete(system, user)` and never
touches a vendor SDK directly. Swapping providers becomes a config change, and
Stage 4 (Phoenix tracing) gets one place to instrument.
"""
import os
from typing import Optional

from rag_eval import config

try:  # optional: load ANTHROPIC_API_KEY / OPENAI_API_KEY from a .env file
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

PROVIDER = os.getenv("LLM_PROVIDER", getattr(config, "LLM_PROVIDER", "anthropic"))
ANTHROPIC_MODEL = os.getenv(
    "ANTHROPIC_MODEL", getattr(config, "ANTHROPIC_MODEL", "claude-sonnet-5-5")
)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", getattr(config, "OPENAI_MODEL", "gpt-4o-mini"))
MAX_TOKENS = getattr(config, "MAX_TOKENS", 800)


class AnthropicLLM:
    def __init__(self, model: str = ANTHROPIC_MODEL):
        import anthropic

        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model

    def complete(self, system: str, user: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")


class OpenAILLM:
    def __init__(self, model: str = OPENAI_MODEL):
        from openai import OpenAI

        self.client = OpenAI()  # reads OPENAI_API_KEY
        self.model = model

    def complete(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


def get_llm(provider: Optional[str] = None):
    provider = (provider or PROVIDER).lower()
    if provider == "anthropic":
        return AnthropicLLM()
    if provider == "openai":
        return OpenAILLM()
    raise ValueError(f"Unknown LLM provider: {provider!r}")
