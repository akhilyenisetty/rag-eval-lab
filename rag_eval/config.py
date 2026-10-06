"""Central config. Reads from environment (.env) with sensible defaults."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "docs"      # source documents to index
CHROMA_DIR = ROOT / ".chroma"          # persistent vector store (gitignored)
COLLECTION = "knowledge"

# Local, free embedding model (no API key). Small + fast; good enough for a demo.
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

# Chunking (character-based, with overlap so ideas split at a boundary still appear whole).
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

# How many chunks to retrieve per query.
TOP_K = int(os.getenv("TOP_K", "4"))

# LLM for generation (used from Stage 2 on).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")   # anthropic | openai
LLM_MODEL = os.getenv("LLM_MODEL", "claude-3-5-haiku-20241022")
