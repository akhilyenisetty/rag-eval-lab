"""
Type-aware chunking.

Different document types split best in different ways, so instead of one global chunker we
pick a strategy per document:
  - markdown  -> split on headings (each section becomes a chunk, with context intact)
  - code      -> split on top-level definitions (functions/classes stay together)
  - prose     -> recursive: group paragraphs up to the size limit (txt / pdf / docx / unknown)
  - fixed     -> character window with overlap (the low-level fallback the others reuse)

The right chunk size/overlap is ultimately an empirical choice, tune it against the eval set
(Stage 3) rather than guessing. These are sensible defaults to start from.
"""
import re
from pathlib import Path

from . import config

CODE_EXTS = {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rb", ".cpp", ".c", ".cs"}


def fixed_chunk(text, size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    """Character window with overlap. Overlap keeps a fact that lands on a boundary whole."""
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start:start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += max(1, size - overlap)
    return chunks


def recursive_chunk(text, size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    """Prose: split on blank lines (paragraphs), then greedily pack paragraphs up to `size`,
    keeping paragraphs whole. A single oversized paragraph falls back to fixed_chunk."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(p) > size:
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.extend(fixed_chunk(p, size, overlap))
            continue
        if cur and len(cur) + len(p) + 2 > size:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def markdown_chunk(text, size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    """Markdown: start a new section at each heading (#..######). Each section (heading + body)
    is a chunk; sections larger than `size` are split further with the prose chunker."""
    sections, buf = [], []
    for line in text.splitlines():
        if re.match(r"^#{1,6}\s", line) and buf:
            sections.append("\n".join(buf).strip())
            buf = [line]
        else:
            buf.append(line)
    if buf:
        sections.append("\n".join(buf).strip())

    chunks = []
    for s in (s for s in sections if s):
        chunks.extend([s] if len(s) <= size else recursive_chunk(s, size, overlap))
    return chunks


def code_chunk(text, size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    """Code: start a new block at each top-level definition so functions/classes stay together."""
    defn = re.compile(r"^(def |class |function |public |private |protected |export |func )")
    blocks, buf = [], []
    for line in text.splitlines():
        if defn.match(line) and buf:
            blocks.append("\n".join(buf))
            buf = [line]
        else:
            buf.append(line)
    if buf:
        blocks.append("\n".join(buf))

    chunks = []
    for b in (b for b in blocks if b.strip()):
        chunks.extend([b] if len(b) <= size else fixed_chunk(b, size, overlap))
    return chunks


def _strategy(source):
    """Pick (strategy_name, fn) from the source's file extension."""
    ext = Path(source).suffix.lower()
    if ext == ".md":
        return "markdown", markdown_chunk
    if ext in CODE_EXTS:
        return "code", code_chunk
    return "recursive", recursive_chunk   # txt / pdf / docx / unknown sources


def strategy_for(source):
    return _strategy(source)[0]


def chunk_for(source, text):
    """Chunk `text` using the strategy appropriate for `source`'s type."""
    return _strategy(source)[1](text)
