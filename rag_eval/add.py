"""
Add your own files to the knowledge base, then re-index.

Usage:
    python -m rag_eval.add path/to/file.pdf path/to/notes.docx ...

Supported: .md, .txt, .pdf, .docx. Files are copied into data/docs/ and the vector index
is rebuilt so the new content is immediately searchable.
"""
import shutil
import sys
from pathlib import Path

from . import config
from .ingest import build_index
from .loaders import EXT_READERS


def main(argv):
    if not argv:
        print("Usage: python -m rag_eval.add <file> [<file> ...]")
        return
    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    added = 0
    for arg in argv:
        src = Path(arg).expanduser()
        if not src.exists():
            print(f"skip (not found): {src}")
            continue
        if src.suffix.lower() not in EXT_READERS:
            print(f"skip (unsupported type {src.suffix}; supported: {', '.join(EXT_READERS)}): {src.name}")
            continue
        dst = config.DOCS_DIR / src.name
        if src.resolve() != dst.resolve():
            shutil.copy(src, dst)
        print(f"added {src.name}")
        added += 1
    if added:
        print("re-indexing...")
        build_index()
    else:
        print("nothing added.")


if __name__ == "__main__":
    main(sys.argv[1:])
