# RAG App with Real Evaluations

A retrieval-augmented generation (RAG) app built the *right* way: not just "it answers questions,"
but with a real **evaluation harness** that measures whether the answers are faithful and relevant,
plus tracing so you can see exactly what the system retrieved and did. The point is to treat an AI
feature like a real engineering system you can measure, not a demo.

## Why this exists
Anyone can wire up a RAG bot that looks good in a demo. The hard part, and the part that matters in
production, is *knowing* it's correct and catching regressions when you change a model, prompt, or
chunking strategy. This project makes that measurable.

## Stack
- **Embeddings:** `sentence-transformers` (local, no API key)
- **Vector store:** ChromaDB (local, persistent)
- **Chunking:** type-aware (markdown by heading, code by definition, prose recursively by paragraph, fixed fallback)
- **Generation:** pluggable LLM (Anthropic or OpenAI; your key)
- **Evaluation:** Ragas (faithfulness, answer relevancy, context precision/recall)
- **Observability:** Arize Phoenix (tracing of retrieval + generation)
- **CI:** GitHub Actions runs the eval set and gates on regression

## Build plan (stages)
1. **Ingest → chunk → embed → index** (retrieval foundation)  ← done
2. Retrieval + generation (the RAG query path)
3. Evaluation harness with Ragas + a labeled eval set
4. Phoenix tracing / observability
5. CI gating + ship

## Quick start
```bash
python3.11 -m venv .venv && source .venv/bin/activate   # needs Python 3.10+
pip install -r requirements.txt
cp .env.example .env     # add your LLM key (needed from Stage 2)

# Stage 1: build the vector index from the sample docs in data/docs/
python -m rag_eval.ingest
```
You should see something like: `Indexed N chunks from 1 document(s) into 'knowledge'.`

## Bring your own data
Point the RAG at *your* content. Two ways:

**Add files (md, txt, pdf, docx):**
```bash
python -m rag_eval.add ~/Documents/my_notes.pdf ~/Downloads/handbook.docx
```
This copies the files into `data/docs/` and rebuilds the index. (Or just drop files into
`data/docs/` yourself and run `python -m rag_eval.ingest`.)

**Pluggable connector architecture.** Every data source is a small loader that implements
`load() -> list[(source, text)]` (see `rag_eval/loaders.py`). Files work today via `FileLoader`.
Other sources, like **Gmail** or Slack, plug into the exact same chunk→embed→index pipeline:
write a loader with that one method and add it to `load_all()`. A documented `GmailLoader` stub is
included showing the OAuth + Gmail API steps. *(Connectors like Gmail are a planned later stage,
they mainly add an auth flow; the ingestion side is already built for them.)*

## Layout
```
rag_eval/            # the package
  config.py          # settings (models, chunk size, top-k) from env
  loaders.py         # pluggable data sources (FileLoader now; Gmail/Slack later)
  chunking.py        # type-aware chunking (markdown / code / prose / fixed)
  ingest.py          # Stage 1: load -> chunk -> embed -> store in Chroma
  add.py             # add your own files, then re-index
data/docs/           # sample knowledge base (Acme Robotics handbook)
requirements.txt · .env.example · .gitignore
```

## Status
- ✅ Stage 1: ingestion / chunking / embeddings / indexing
- ⬜ Stage 2–5 (retrieval+generation, evals, tracing, CI)
