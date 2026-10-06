"""Stage 4: tracing with Arize Phoenix (optional).

Every question becomes a trace you can open in the Phoenix UI:

  rag.answer (CHAIN)            question in, answer out
    ├── retrieve (RETRIEVER)    the query, every chunk returned, its distance and source
    └── generate (LLM)          exact prompt sent, response, model, token counts

Tracing is OFF unless RAG_TRACING=1. When it is off, or when the
arize-phoenix-otel package is not installed, every span below is a no-op, so
the rest of the app never has to care whether tracing exists.

Spans use OpenInference attribute names (the convention Phoenix understands)
written by hand, rather than an auto-instrumentor that patches the Anthropic
SDK. The SDK just had a breaking 1.0 release; manual spans can't be broken by
that kind of change.
"""
import json
import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional

try:  # pick up RAG_TRACING / PHOENIX_* from .env
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

_tracer = None
_initialized = False

PROJECT = "rag-eval-lab"
DEFAULT_ENDPOINT = "http://localhost:6006/v1/traces"  # local Phoenix, HTTP


def enabled() -> bool:
    return os.getenv("RAG_TRACING", "").strip().lower() in {"1", "true", "yes", "on"}


def _get_tracer():
    """Create the tracer once, on first use. Returns None when tracing is off."""
    global _tracer, _initialized
    if _initialized:
        return _tracer
    _initialized = True
    if not enabled():
        return None
    try:
        from phoenix.otel import register
    except ImportError:
        print("[tracing] RAG_TRACING is on but arize-phoenix-otel is not installed. "
              "Run: pip install 'arize-phoenix-otel>=0.16.0'")
        return None

    project = os.getenv("PHOENIX_PROJECT", PROJECT)
    if os.getenv("PHOENIX_COLLECTOR_ENDPOINT"):
        provider = register(project_name=project, batch=False)  # reads the env var
    else:
        provider = register(project_name=project, endpoint=DEFAULT_ENDPOINT, batch=False)
    # batch=False exports each span immediately, so short CLI runs never lose spans.
    _tracer = provider.get_tracer("rag_eval")
    return _tracer


class _NoopSpan:
    def set_attribute(self, *args, **kwargs) -> None:
        pass


def set_attr(span, key: str, value: Any) -> None:
    """Set one attribute, skipping None and JSON-encoding dicts/lists."""
    if value is None:
        return
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, default=str)
    span.set_attribute(key, value)


@contextmanager
def span(name: str, kind: str, input_value: Optional[str] = None,
         metadata: Optional[dict] = None) -> Iterator[Any]:
    """Open a span. kind is an OpenInference span kind: CHAIN, RETRIEVER, LLM, ..."""
    tracer = _get_tracer()
    if tracer is None:
        yield _NoopSpan()
        return
    # start_as_current_span nests child spans automatically and records
    # exceptions on the span if the block raises.
    with tracer.start_as_current_span(name) as s:
        s.set_attribute("openinference.span.kind", kind)
        set_attr(s, "input.value", input_value)
        set_attr(s, "metadata", metadata)
        yield s
