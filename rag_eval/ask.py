"""Ask questions against the knowledge base.

  python -m rag_eval.ask "How many PTO days do new hires get?"
  python -m rag_eval.ask "..." -k 6 --show-context
  python -m rag_eval.ask            # interactive mode, type 'quit' to exit
"""
import argparse

from rag_eval.generate import answer
from rag_eval.retrieve import TOP_K


def run(question: str, k: int, show_context: bool, max_distance) -> None:
    result = answer(question, k=k, max_distance=max_distance)

    print(f"\nQ: {result.question}\n")
    print(result.answer)

    if result.chunks:
        print("\nSources:")
        for i, c in enumerate(result.chunks, start=1):
            print(f"  [{i}] {c.source}  (distance={c.distance:.3f})")
            if show_context:
                preview = c.text.strip().replace("\n", " ")
                print(f"      {preview[:240]}{'...' if len(preview) > 240 else ''}")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="Ask the RAG knowledge base a question.")
    p.add_argument("question", nargs="?", help="omit for interactive mode")
    p.add_argument("-k", type=int, default=TOP_K, help="chunks to retrieve")
    p.add_argument("--show-context", action="store_true", help="print chunk previews")
    p.add_argument("--max-distance", type=float, default=None,
                   help="drop chunks farther than this")
    args = p.parse_args()

    if args.question:
        run(args.question, args.k, args.show_context, args.max_distance)
        return

    while True:
        q = input("Ask (or 'quit'): ").strip()
        if q.lower() in {"quit", "exit", "q"}:
            break
        if q:
            run(q, args.k, args.show_context, args.max_distance)


if __name__ == "__main__":
    main()
