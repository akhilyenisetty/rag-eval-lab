"""Pluggable LLM layer.

Everything else in the app calls `get_llm().complete(system, user)` and never
touches a vendor SDK directly. Swapping providers is a config change, and
tracing lives in one place: every call becomes an LLM span with the prompt,
response, model, and token counts.
"""
import os
from typing import Optional

from rag_eval import config
from rag_eval.tracing import set_attr, span

try:  # load ANTHROPIC_API_KEY / OPENAI_API_KEY from .env
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

PROVIDER = os.getenv("LLM_PROVIDER", getattr(config, "LLM_PROVIDER", "anthropic"))
ANTHROPIC_MODEL = os.getenv(
    "ANTHROPIC_MODEL", getattr(config, "LLM_MODEL", "claude-haiku-4-5-20251001")
)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", getattr(config, "OPENAI_MODEL", "gpt-4o-mini"))
MAX_TOKENS = getattr(config, "MAX_TOKENS", 800)


def _trace_llm(s, provider: str, model: str, system: str, user: str,
               text: str, prompt_tokens, completion_tokens) -> None:
    set_attr(s, "llm.provider", provider)
    set_attr(s, "llm.system", provider)
    set_attr(s, "llm.model_name", model)
    set_attr(s, "llm.input_messages.0.message.role", "system")
    set_attr(s, "llm.input_messages.0.message.content", system)
    set_attr(s, "llm.input_messages.1.message.role", "user")
    set_attr(s, "llm.input_messages.1.message.content", user)
    set_attr(s, "llm.output_messages.0.message.role", "assistant")
    set_attr(s, "llm.output_messages.0.message.content", text)
    set_attr(s, "output.value", text)
    set_attr(s, "llm.token_count.prompt", prompt_tokens)
    set_attr(s, "llm.token_count.completion", completion_tokens)
    if prompt_tokens is not None and completion_tokens is not None:
        set_attr(s, "llm.token_count.total", prompt_tokens + completion_tokens)


class AnthropicLLM:
    def __init__(self, model: str = ANTHROPIC_MODEL):
        import anthropic

        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model

    def complete(self, system: str, user: str, name: str = "generate") -> str:
        with span(name, "LLM", input_value=user) as s:
            # Note: anthropic SDK >= 1.0 rejects temperature/top_p/top_k.
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(b.text for b in resp.content if b.type == "text")
            usage = getattr(resp, "usage", None)
            _trace_llm(s, "anthropic", self.model, system, user, text,
                       getattr(usage, "input_tokens", None),
                       getattr(usage, "output_tokens", None))
            return text


class OpenAILLM:
    def __init__(self, model: str = OPENAI_MODEL):
        from openai import OpenAI

        self.client = OpenAI()  # reads OPENAI_API_KEY
        self.model = model

    def complete(self, system: str, user: str, name: str = "generate") -> str:
        with span(name, "LLM", input_value=user) as s:
            resp = self.client.chat.completions.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            text = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            _trace_llm(s, "openai", self.model, system, user, text,
                       getattr(usage, "prompt_tokens", None),
                       getattr(usage, "completion_tokens", None))
            return text


def get_llm(provider: Optional[str] = None):
    provider = (provider or PROVIDER).lower()
    if provider == "anthropic":
        return AnthropicLLM()
    if provider == "openai":
        return OpenAILLM()
    raise ValueError(f"Unknown LLM provider: {provider!r}")
