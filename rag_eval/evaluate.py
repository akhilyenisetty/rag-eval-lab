"""Stage 3: evaluation harness.

Runs every question in the golden set through the full pipeline (`answer()`),
then scores each result on four things:

  retrieval   Did the chunk holding the answer come back in the top-k?  (deterministic)
  refusal     Did it refuse when it should, and only when it should?    (deterministic)
  correctness Does the answer match the reference answer?               (LLM judge)
  faithfulness Is every claim supported by the retrieved chunks?        (LLM judge)

Each row is saved with Ragas-style field names (user_input, response,
retrieved_contexts, reference) so Ragas can score the same runs later.

  python -m rag_eval.evaluate
  python -m rag_eval.evaluate --max-distance 1.2
  python -m rag_eval.evaluate --no-judge        # free: skips the LLM-judge calls
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rag_eval import config
from rag_eval.generate import answer
from rag_eval.llm import get_llm
from rag_eval.retrieve import TOP_K

ROOT = Path(getattr(config, "ROOT", Path.cwd()))
GOLDEN_PATH = ROOT / "data" / "eval" / "golden.jsonl"
RESULTS_DIR = ROOT / "results"

JUDGE_SYSTEM = """You grade answers from a question-answering system that must answer only from retrieved context.

Return ONLY a JSON object, no other text, in this exact shape:
{"correct": true | false | null, "faithful": true | false, "reason": "<one short sentence>"}

"correct": true if the RESPONSE conveys the key facts in the REFERENCE answer (extra detail is fine
if it is accurate). false if it is wrong, missing key facts, or contradicts the reference.
Facts about a different company or product than the one the reference is about do NOT contradict
it, as long as the RESPONSE clearly attributes them to their own source. Mixing up which source a
fact belongs to IS a contradiction. null if no reference answer is given.

"faithful": true if EVERY factual claim in the RESPONSE is supported by the CONTEXT passages.
false if any claim is not found in, or goes beyond, the context."""


# ---------- loading ----------

def load_golden(path: Path) -> List[Dict[str, Any]]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ---------- deterministic metrics ----------

def is_refusal(text: str) -> bool:
    # A full refusal starts with the refusal phrase. A partial answer that
    # admits one gap ("...300 kg [1]. I don't know how many offices...") is not.
    return text.strip().lower().startswith("i don't know")


def is_partial(text: str) -> bool:
    t = text.strip().lower()
    return "i don't know" in t and not t.startswith("i don't know")


def retrieval_metrics(
    contexts: List[str],
    sources: List[str],
    evidence: List[str],
    expected_source: Optional[str] = None,
) -> Tuple[Optional[bool], Optional[int]]:
    """hit: every evidence phrase appears in at least one retrieved chunk.
    rank: 1-based position of the first chunk containing the first evidence phrase.

    If expected_source is set, only chunks from that file count. Otherwise a
    distractor doc that happens to contain "multi-factor" would fake a hit."""
    if not evidence:
        return None, None
    lowered = [
        c.lower() if (expected_source is None or s == expected_source) else ""
        for c, s in zip(contexts, sources)
    ]
    hit = all(any(e.lower() in c for c in lowered) for e in evidence)
    rank = None
    for i, c in enumerate(lowered, start=1):
        if evidence[0].lower() in c:
            rank = i
            break
    return hit, rank


# ---------- LLM judge ----------

def parse_json(text: str) -> Dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return {"error": f"no JSON in judge output: {text[:200]}"}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        return {"error": f"bad JSON from judge: {e}"}


def judge(llm, question: str, reference: str, response: str,
          contexts: List[str], sources: List[str]) -> Dict[str, Any]:
    # Show the judge exactly what the generator saw, including source labels.
    ctx = "\n\n".join(f"[{i}] (source: {s})\n{c.strip()}"
                      for i, (c, s) in enumerate(zip(contexts, sources), start=1))
    user = (
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE ANSWER:\n{reference or '(none)'}\n\n"
        f"CONTEXT:\n{ctx or '(no context retrieved)'}\n\n"
        f"RESPONSE:\n{response}"
    )
    return parse_json(llm.complete(JUDGE_SYSTEM, user))


# ---------- running ----------

def evaluate_one(item, k, max_distance, judge_llm) -> Dict[str, Any]:
    t0 = time.perf_counter()
    result = answer(item["question"], k=k, max_distance=max_distance)
    latency = time.perf_counter() - t0

    refused = is_refusal(result.answer)
    hit, rank = retrieval_metrics(result.contexts, result.sources,
                                  item.get("evidence", []), item.get("source"))

    verdict: Dict[str, Any] = {}
    # A refusal makes no claims, so there is nothing to judge (and no cost).
    if judge_llm is not None and not refused:
        verdict = judge(judge_llm, item["question"], item.get("reference", ""),
                        result.answer, result.contexts, result.sources)

    return {
        "id": item["id"],
        "user_input": item["question"],
        "response": result.answer,
        "retrieved_contexts": result.contexts,
        "reference": item.get("reference", ""),
        "answerable": item.get("answerable", True),
        "sources": result.sources,
        "distances": [round(c.distance, 3) for c in result.chunks],
        "retrieval_hit": hit,
        "first_relevant_rank": rank,
        "refused": refused,
        "partial": is_partial(result.answer),
        "correct": verdict.get("correct"),
        "faithful": (True if refused else verdict.get("faithful")) if judge_llm is not None else None,
        "judge_reason": verdict.get("reason") or verdict.get("error"),
        "latency_s": round(latency, 2),
    }


def _rate(values: List[Optional[bool]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return round(sum(1 for v in vals if v) / len(vals), 3) if vals else None


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    ranks = [r["first_relevant_rank"] for r in ans]
    mrr = round(sum((1 / x) if x else 0 for x in ranks) / len(ranks), 3) if ranks else None
    return {
        "questions": len(rows),
        "retrieval_hit_rate": _rate([r["retrieval_hit"] for r in ans]),
        "mrr": mrr,
        "correct_refusal_rate": _rate([r["refused"] for r in unans]),
        "false_refusal_rate": _rate([r["refused"] for r in ans]),
        "correctness": _rate([r["correct"] for r in ans]),
        "faithfulness": _rate([r["faithful"] for r in rows]),
        "avg_latency_s": round(sum(r["latency_s"] for r in rows) / len(rows), 2) if rows else None,
    }


def yn(v) -> str:
    return "-" if v is None else ("Y" if v else "N")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate the RAG pipeline on the golden set.")
    p.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    p.add_argument("-k", type=int, default=TOP_K)
    p.add_argument("--max-distance", type=float, default=None)
    p.add_argument("--no-judge", action="store_true", help="skip LLM-judge scoring")
    args = p.parse_args()

    items = load_golden(args.golden)
    judge_llm = None if args.no_judge else get_llm()

    print(f"\nEvaluating {len(items)} questions  (k={args.k}, max_distance={args.max_distance})\n")
    print(f"{'id':<5} {'retr':>4} {'rank':>4} {'refused':>7} {'correct':>7} {'faithful':>8} {'secs':>5}")
    rows = []
    for item in items:
        row = evaluate_one(item, args.k, args.max_distance, judge_llm)
        rows.append(row)
        print(f"{row['id']:<5} {yn(row['retrieval_hit']):>4} {str(row['first_relevant_rank'] or '-'):>4} "
              f"{yn(row['refused']):>7} {yn(row['correct']):>7} {yn(row['faithful']):>8} {row['latency_s']:>5}")

    summary = summarize(rows)
    print("\nSummary")
    for key, val in summary.items():
        print(f"  {key:<22} {val}")

    # Show the failures, since those are what you act on.
    failures = [r for r in rows if r["correct"] is False or r["faithful"] is False
                or r["retrieval_hit"] is False or (r["answerable"] == r["refused"])]
    if failures:
        print("\nFailures to look at")
        for r in failures:
            print(f"  {r['id']}: {r['user_input']}")
            print(f"      answer: {r['response'][:200]}")
            if r["judge_reason"]:
                print(f"      judge:  {r['judge_reason']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({
        "settings": {"k": args.k, "max_distance": args.max_distance,
                     "judge": not args.no_judge, "golden": str(args.golden)},
        "summary": summary,
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}\n")


if __name__ == "__main__":
    main()
