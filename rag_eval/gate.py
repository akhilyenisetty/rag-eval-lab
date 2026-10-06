"""Stage 5: quality gate.

Runs the golden sets several times, pools every answer, and compares the
pooled metrics against thresholds in data/eval/thresholds.json. Exits 1 if any
threshold fails, so CI can block a pull request.

Why pool and repeat: a single run of 12 questions moves a score by ~8 points per
flipped answer, and answers vary run to run. Pooling 27 questions x 2 runs makes
one flip worth ~2 points, so thresholds can sit just under the baseline without
failing on noise.

  python -m rag_eval.gate              # uses thresholds.json (2 runs)
  python -m rag_eval.gate --runs 1     # quicker local check
"""
import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from rag_eval import llm as llm_mod
from rag_eval.evaluate import (JUDGE_MODEL, RESULTS_DIR, ROOT, evaluate_one,
                               load_golden, summarize)
from rag_eval.llm import AnthropicLLM

THRESHOLDS_PATH = ROOT / "data" / "eval" / "thresholds.json"


def preflight() -> None:
    """Fail fast with a clear message instead of a stack trace halfway through."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("gate: ANTHROPIC_API_KEY is not set (in CI, add it as a repository secret).")
    try:
        from rag_eval.retrieve import _collection

        count = _collection().count()
    except Exception as e:  # collection missing, path wrong, ...
        sys.exit(f"gate: vector index not available ({e}). Run `python -m rag_eval.ingest` first.")
    if count == 0:
        sys.exit("gate: vector index is empty. Run `python -m rag_eval.ingest` first.")


def check(summary: Dict[str, Any], cfg: Dict[str, Any]) -> List[Tuple[str, Any, str, bool]]:
    results = []
    for metric, floor in cfg.get("min", {}).items():
        val = summary.get(metric)
        results.append((metric, val, f">= {floor}", val is not None and val >= floor))
    for metric, ceiling in cfg.get("max", {}).items():
        val = summary.get(metric)
        results.append((metric, val, f"<= {ceiling}", val is not None and val <= ceiling))
    return results


def report_markdown(results, summary, settings, worst) -> str:
    passed = all(ok for *_, ok in results)
    lines = [
        f"## RAG eval gate: {'PASSED' if passed else 'FAILED'}",
        "",
        f"Generator `{settings['generator_model']}`, judge `{settings['judge_model']}`, "
        f"k={settings['k']}, {settings['runs']} runs x {settings['questions']} questions "
        f"= {summary['questions']} answers pooled.",
        "",
        "| Metric | Value | Threshold | Result |",
        "|---|---|---|---|",
    ]
    for metric, val, rule, ok in results:
        lines.append(f"| {metric} | {val} | {rule} | {'pass' if ok else '**FAIL**'} |")
    if worst:
        lines += ["", "**Questions that failed in any run**", "",
                  "| Question | Failed runs | Last judge note |", "|---|---|---|"]
        for (qid, n, note) in worst:
            lines.append(f"| {qid} | {n}/{settings['runs']} | {(note or '').replace('|', '/')[:140]} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Fail if eval metrics fall below thresholds.")
    p.add_argument("--thresholds", type=Path, default=THRESHOLDS_PATH)
    p.add_argument("--runs", type=int, default=None, help="override the number of runs")
    args = p.parse_args()

    cfg = json.loads(args.thresholds.read_text(encoding="utf-8"))
    runs = args.runs or cfg.get("runs", 2)
    k = cfg.get("k", 4)
    preflight()

    item_sets = [(path, load_golden(ROOT / path)) for path in cfg["golden_sets"]]
    n_questions = sum(len(items) for _, items in item_sets)
    judge_llm = AnthropicLLM(model=JUDGE_MODEL)
    gen_model = (llm_mod.ANTHROPIC_MODEL if llm_mod.PROVIDER == "anthropic"
                 else llm_mod.OPENAI_MODEL)
    print(f"gate: {runs} runs x {n_questions} questions | generator {gen_model} | judge {JUDGE_MODEL}")

    rows: List[Dict[str, Any]] = []
    for r in range(1, runs + 1):
        run_id = f"gate_{datetime.now():%Y%m%d_%H%M%S}_run{r}"
        for path, items in item_sets:
            for item in items:
                row = evaluate_one(item, k, None, judge_llm, run_id)
                row["golden"], row["run"] = path, r
                rows.append(row)
        print(f"  run {r}/{runs} done")

    summary = summarize(rows)
    results = check(summary, cfg)

    # Which questions failed, and how often: the first thing to read on a red build.
    def failed(row) -> bool:
        return (row["correct"] is False or row["faithful"] is False
                or row["retrieval_hit"] is False or row["answerable"] == row["refused"])

    counts, notes = Counter(), {}
    for row in rows:
        if failed(row):
            counts[row["id"]] += 1
            notes[row["id"]] = row["judge_reason"]
    worst = [(qid, n, notes[qid]) for qid, n in counts.most_common()]

    settings = {"generator_model": gen_model, "judge_model": JUDGE_MODEL, "k": k,
                "runs": runs, "questions": n_questions}
    md = report_markdown(results, summary, settings, worst)
    print("\n" + md)

    # In GitHub Actions, this shows the table on the workflow run's summary page.
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(md)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"gate_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({
        "settings": settings, "thresholds": cfg, "summary": summary,
        "checks": [{"metric": m, "value": v, "rule": rule, "passed": ok}
                   for m, v, rule, ok in results],
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"Saved: {out}")

    sys.exit(0 if all(ok for *_, ok in results) else 1)


if __name__ == "__main__":
    main()
