"""
Stage 1 of the RAG pipeline: ingest documents, chunk them, embed each chunk, and
store the vectors in ChromaDB so we can retrieve them later.

Run:  python -m rag_eval.ingest
"""
import uuid

import chromadb
from sentence_transformers import SentenceTransformer

from . import config
from .chunking import chunk_for, strategy_for   # type-aware chunking
from .loaders import load_all


def build_index():
    model = SentenceTransformer(config.EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))

    # Reset the collection so re-running ingest is idempotent (no duplicate chunks).
    try:
        client.delete_collection(config.COLLECTION)
    except Exception:
        pass
    col = client.create_collection(config.COLLECTION)

    ids, documents, metadatas = [], [], []
    for name, text in load_all():   # pulls from every enabled source (files now, connectors later)
        strategy = strategy_for(name)   # type-aware: markdown / code / recursive
        for i, chunk in enumerate(chunk_for(name, text)):
            ids.append(str(uuid.uuid4()))
            documents.append(chunk)
            metadatas.append({"source": name, "chunk": i, "strategy": strategy})

    if not documents:
        print(f"No .md/.txt documents found in {config.DOCS_DIR}")
        return

    # Embed all chunks locally (normalized vectors -> cosine similarity on retrieval).
    embeddings = model.encode(documents, normalize_embeddings=True).tolist()
    col.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    n_sources = len({m["source"] for m in metadatas})
    strategies = sorted({m["strategy"] for m in metadatas})
    print(f"Indexed {len(documents)} chunks from {n_sources} document(s) into '{config.COLLECTION}'.")
    print(f"Chunking strategies used: {', '.join(strategies)}")


if __name__ == "__main__":
    build_index()
