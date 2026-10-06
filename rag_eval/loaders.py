"""
Pluggable data sources for the RAG knowledge base.

A "document" is just a (source, text) pair. Every loader implements `load() -> list[(source, text)]`.
To add a NEW source (Gmail, Slack, Notion, Drive, ...), write a class with that same `load()`
method and add it to `load_all()`. The rest of the pipeline (chunk -> embed -> index) is unchanged,
that's the whole point of this interface.
"""
from pathlib import Path

from . import config


# ---- file readers (one per format) ----
def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader  # imported lazily so the dep is only needed if you use PDFs
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _read_docx(path: Path) -> str:
    import docx  # python-docx
    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs)


EXT_READERS = {".md": _read_text, ".txt": _read_text, ".pdf": _read_pdf, ".docx": _read_docx}


class FileLoader:
    """Loads every supported file (md, txt, pdf, docx) from a directory, recursively."""

    def __init__(self, directory=None):
        self.directory = Path(directory or config.DOCS_DIR)

    def load(self):
        docs = []
        for path in sorted(self.directory.glob("**/*")):
            reader = EXT_READERS.get(path.suffix.lower())
            if not reader:
                continue
            try:
                text = reader(path)
            except Exception as e:
                print(f"skip (could not read {path.name}): {e}")
                continue
            if text and text.strip():
                docs.append((path.name, text))
        return docs


class GmailLoader:
    """STUB (later stage): pull emails as documents via the Gmail API.

    Implementing this is the 'connect your email' feature. It needs:
      1. A Google Cloud project with the Gmail API enabled and an OAuth consent screen.
      2. An OAuth flow to get the user's consent + a stored refresh token (scope: gmail.readonly).
      3. Use the Gmail API to list + fetch messages, extract the body text, and return
         [(f"email:{msg_id}", body_text), ...].
    Once written, just add GmailLoader() to load_all() and emails flow through the same pipeline.
    """

    def load(self):
        raise NotImplementedError("GmailLoader is a planned connector. See the docstring for the steps.")


def load_all():
    """Aggregate documents from every ENABLED source. Add new loaders to this list."""
    loaders = [
        FileLoader(),
        # GmailLoader(),      # enable once implemented
        # SlackLoader(),      # future
    ]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    return docs
