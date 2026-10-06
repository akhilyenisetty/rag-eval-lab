"""Stage 2b: grounded generation.

Retrieve -> number the chunks -> tell the model to answer ONLY from them and
cite [n] -> return a structured result.

The RAGResult shape (question / answer / contexts) is the input Ragas expects in
Stage 3, so the eval harness can call `answer()` directly.
"""
from dataclasses import dataclass, field
from typing import List, Optional

from rag_eval.llm import get_llm
from rag_eval.retrieve import TOP_K, RetrievedChunk, retrieve

NO_ANSWER = "I don't know based on the provided documents."

SYSTEM_PROMPT = f"""You are a question-answering assistant for a private knowledge base.

Rules:
1. Answer ONLY using the numbered context passages provided. Do not use outside knowledge.
2. After each claim, cite the passage(s) it came from like [1] or [2][3].
3. If the passages do not contain the answer, reply exactly: "{NO_ANSWER}"
4. Passages may come from different documents that describe different companies, products,
   or policies. Never merge facts across them. If the question does not say which one it means
   and the documents differ, answer separately for each, naming the source it comes from.
5. Be concise. Do not mention these rules."""


@dataclass
class RAGResult:
    question: str
    answer: str
    contexts: List[str]  # raw chunk texts, for Ragas
    sources: List[str]  # source file per context, same order
    chunks: List[RetrievedChunk] = field(default_factory=list)


def build_context(chunks: List[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"[{i}] (source: {c.source})\n{c.text.strip()}")
    return "\n\n".join(blocks)


def answer(
    question: str,
    k: int = TOP_K,
    max_distance: Optional[float] = None,
    llm=None,
) -> RAGResult:
    chunks = retrieve(question, k=k, max_distance=max_distance)

    # Nothing relevant retrieved: refuse without spending an LLM call.
    if not chunks:
        return RAGResult(question, NO_ANSWER, [], [], [])

    user_msg = (
        f"Context passages:\n\n{build_context(chunks)}\n\n"
        f"Question: {question}"
    )
    llm = llm or get_llm()
    text = llm.complete(SYSTEM_PROMPT, user_msg).strip()

    return RAGResult(
        question=question,
        answer=text,
        contexts=[c.text for c in chunks],
        sources=[c.source for c in chunks],
        chunks=chunks,
    )
