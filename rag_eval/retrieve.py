"""Stage 2a: retrieval.

Embed the user's question with the SAME model used at ingest time, then ask
ChromaDB for the k nearest chunks. If the query and the documents are embedded
with different models, the vectors live in different "spaces" and similarity
is meaningless, so EMBED_MODEL must match what Stage 1 used.
"""
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional

import chromadb
from sentence_transformers import SentenceTransformer

from rag_eval import config

# Read from config.py, with fallbacks so this drops in without edits.
# Rename these if your config.py uses different names.
CHROMA_DIR = getattr(config, "CHROMA_DIR", "chroma_db")
COLLECTION_NAME = getattr(config, "COLLECTION", "knowledge")
EMBED_MODEL = getattr(config, "EMBED_MODEL", "all-MiniLM-L6-v2")
TOP_K = getattr(config, "TOP_K", 4)


@dataclass
class RetrievedChunk:
    text: str
    source: str
    metadata: Dict[str, Any]
    distance: float  # lower = closer


@lru_cache(maxsize=1)
def _embedder() -> SentenceTransformer:
    # Loading the model is slow (~1-2s), so do it once per process.
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def _collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION_NAME)


def retrieve(
    query: str,
    k: int = TOP_K,
    max_distance: Optional[float] = None,
    where: Optional[Dict[str, Any]] = None,
) -> List[RetrievedChunk]:
    """Return the k chunks closest to `query`.

    max_distance: optional cutoff; drop chunks that are too far away so a
        question with no good match yields nothing instead of junk context.
    where: optional Chroma metadata filter, e.g. {"source": "acme_handbook.md"}.
    """
    query_vec = _embedder().encode([query]).tolist()

    res = _collection().query(
        query_embeddings=query_vec,
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    chunks: List[RetrievedChunk] = []
    for text, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        if max_distance is not None and dist > max_distance:
            continue
        meta = meta or {}
        chunks.append(
            RetrievedChunk(
                text=text,
                source=str(meta.get("source", "unknown")),
                metadata=meta,
                distance=float(dist),
            )
        )
    return chunks
