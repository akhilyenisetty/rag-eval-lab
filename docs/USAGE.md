# Usage guide

Everything you need to install, run, evaluate, and trace rag-eval-lab. All commands run from the repository root (the folder that contains `rag_eval/`).

## 1. Setup

### Requirements

- Python 3.10 or newer (3.11 recommended). The Anthropic SDK 1.0+ does not support 3.9.
- An API key for Anthropic (default) or OpenAI.
- About 2 GB of disk for dependencies. The first install downloads PyTorch for the embedding model.

### Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip        # newer pip resolves dependencies much faster
pip install -r requirements.txt
```

### Configure your API key

```bash
cp .env.example .env
```

Edit `.env` and set your key on one line, with no quotes and no spaces around `=`:

```
ANTHROPIC_API_KEY=sk-ant-api03-...
```

`.env` is listed in `.gitignore` and must never be committed.

### Settings

All tunable settings live in `rag_eval/config.py`:

| Setting | Default | Meaning |
|---|---|---|
| `DOCS_DIR` | `data/docs` | Folder of documents to index |
| `CHROMA_DIR` | `.chroma` | Where the vector index is stored |
| `COLLECTION` | `knowledge` | ChromaDB collection name; all documents share it |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Local embedding model; must match between ingest and query |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 500 / 80 | Size of prose chunks and overlap between neighbors |
| `TOP_K` | 4 | Chunks retrieved per question |
| `LLM_PROVIDER` | `anthropic` | `anthropic` or `openai` |
| `LLM_MODEL` | `claude-sonnet-5-5` | Anthropic model that answers questions |
| `JUDGE_MODEL` | `claude-sonnet-5-5` | Model that grades answers during evaluation; stays fixed when `LLM_MODEL` changes |

Any of `EMBED_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `LLM_PROVIDER`, `LLM_MODEL`, and `JUDGE_MODEL` can be overridden in `.env` (or for one command, as `LLM_MODEL=... python -m ...`) without editing code. `ANTHROPIC_MODEL` and `OPENAI_MODEL` override the model for a single provider. A value in `.env` beats the default in `config.py`, and a value exported in your shell beats both.

### Choosing a model

| Model | Use when | Measured on the hard eval set |
|---|---|---|
| `claude-sonnet-5-5` (default) | Answers matter; documents conflict or only partly answer questions | Correctness 1.0, false refusals 0.0, ~4.0 s per question |
| `claude-haiku-4-5-20251001` | Lower-stakes questions where speed and cost matter more | Correctness 0.90, false refusals 0.10, ~1.9 s per question |

Haiku's misses are over-caution: it sometimes declines a question the documents answer only partly. Both models receive the same retrieved passages through the same API, so the choice affects answer quality and cost, not privacy. Details: [EVALUATION.md](EVALUATION.md#6-model-comparison-sonnet-55-vs-haiku-45).

To use Haiku, set `LLM_MODEL=claude-haiku-4-5-20251001` in `.env`.

## 2. Build the knowledge base

Index everything in `data/docs/`:

```bash
python -m rag_eval.ingest
```

Add a file from anywhere on your computer. It's copied into `data/docs/`, chunked, embedded, and stored:

```bash
python -m rag_eval.add ~/Documents/benefits_policy.pdf
```

Supported formats: `.md`, `.txt`, `.pdf`, `.docx`. Every file goes into the same collection, so questions search across all of them, and each chunk keeps its source filename for citations.

Check what's indexed:

```bash
python -c "import chromadb; from rag_eval import config; c = chromadb.PersistentClient(path=str(config.CHROMA_DIR)).get_collection(config.COLLECTION); print('chunks:', c.count()); print(c.peek(2)['metadatas'])"
```

To start over, delete the index and rebuild it:

```bash
rm -rf .chroma
python -m rag_eval.ingest
```

## 3. Ask questions

```bash
python -m rag_eval.ask "How many PTO days do Acme employees get?"
```

| Option | Effect |
|---|---|
| `-k 6` | Retrieve 6 chunks instead of the default |
| `--show-context` | Print a preview of each retrieved chunk |
| `--max-distance 1.2` | Drop chunks farther than this; if none remain, refuse without calling the LLM |

Run with no question for interactive mode (type `quit` to exit):

```bash
python -m rag_eval.ask
```

### Reading the output

Each source line shows a **distance**: lower means closer. The index uses squared L2 distance on normalized embeddings, so cosine similarity is `1 - distance / 2`. On the sample knowledge base, relevant chunks land around 0.7 and unrelated ones above 1.3, which is why 1.2 works as a cutoff there. Re-check that threshold on your own documents.

## 4. Run evaluations

```bash
python -m rag_eval.evaluate                                        # easy set (15 questions)
python -m rag_eval.evaluate --golden data/eval/golden_hard.jsonl   # hard set (12 questions)
```

| Option | Effect |
|---|---|
| `--golden PATH` | Which question set to run |
| `-k N` / `--max-distance D` | Same as for `ask` |
| `--no-judge` | Skip LLM-judge calls; free, deterministic metrics only |

Each run prints a per-question table, a summary, and the failures worth reading, then saves a full JSON report to `results/eval_<timestamp>.json`. A judged run of the hard set makes about 24 LLM calls: one answer and one judge verdict per question (fewer if the distance cutoff skips some answers).

**Run each configuration 3 times before comparing.** LLM output varies between runs, and with 12 questions a single flipped answer moves a score by about 8 points.

### Writing your own golden questions

Golden sets are JSON Lines, one question per line:

```json
{"id": "x01", "type": "paraphrase", "question": "How much vacation do I get?", "reference": "20 days of PTO plus 10 holidays.", "evidence": ["20 days"], "answerable": true, "source": "acme_handbook.md"}
```

| Field | Meaning |
|---|---|
| `id` | Unique ID shown in reports |
| `type` | Free-form label for grouping failures (paraphrase, multi_hop, ...) |
| `question` | What gets asked |
| `reference` | The correct answer, for the judge; empty for unanswerable questions |
| `evidence` | Phrases that must appear in a retrieved chunk for a retrieval hit |
| `answerable` | `false` if the right response is "I don't know" |
| `source` | File the evidence must come from; prevents distractor docs faking a hit |

When you add documents, re-check existing references. A question that was unanswerable may now have an answer.

## 5. Tracing with Arize Phoenix (optional)

Tracing records every question as a trace you can inspect in a browser:

```
rag.answer (CHAIN)         question → answer
├── retrieve (RETRIEVER)   query, each chunk returned, its distance, id, and source
└── generate (LLM)         exact system + user prompt, response, model, token counts
```

During evaluation, each question is wrapped in an `eval.<id>` span that also contains the judge call, and all questions from one run share a session ID.

### Install the tracing client

The client is in `requirements.txt`, so it's already installed if you followed setup. It is small: it only sends traces.

### Start a Phoenix server

The server is optional and has its own requirements file, because it is much larger and needs tightly pinned dependencies:

```bash
pip install -r requirements-tracing.txt
```

Then open a **second terminal**, activate the same environment, and start it:

```bash
source .venv/bin/activate
phoenix serve
```

Leave that window open while you work (`Ctrl+C` stops it), and open http://localhost:6006.

Alternatives that keep the server out of the project environment entirely: `uvx arize-phoenix serve` (needs [uv](https://docs.astral.sh/uv/)), or Docker:

```bash
docker run -p 6006:6006 -p 4317:4317 arizephoenix/phoenix:latest
```

### Turn tracing on

Tracing is off by default. Enable it per command:

```bash
RAG_TRACING=1 python -m rag_eval.ask "Can I carry over 10 PTO days?"
RAG_TRACING=1 python -m rag_eval.evaluate --golden data/eval/golden_hard.jsonl
```

or for every command, by adding `RAG_TRACING=1` to `.env`. Traces appear in the `rag-eval-lab` project in Phoenix.

To send traces somewhere other than local Phoenix, set `PHOENIX_COLLECTOR_ENDPOINT` (and `PHOENIX_API_KEY` if required) in `.env`.

### What to look for

- **A wrong answer:** open its `retrieve` span first. If the right chunk isn't there, it's a retrieval problem; if it is, look at the `generate` span's prompt and response.
- **Slow questions:** compare span durations. Usually the LLM span dominates; a slow `retrieve` on the first question is the embedding model loading.
- **Cost:** token counts on each `generate` and `judge` span.

If `RAG_TRACING=1` is set but the client isn't installed, the app prints a warning and runs normally without tracing.

## 6. Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `Collection [...] does not exist` | Nothing ingested yet, or `COLLECTION` changed | Run `python -m rag_eval.ingest` |
| `401 invalid x-api-key` | Wrong or placeholder key | Check `.env`; run `unset ANTHROPIC_API_KEY` if an old value was exported in the shell, which overrides `.env` |
| `unexpected keyword argument 'temperature'` | Old code against Anthropic SDK 1.0+ | The SDK removed sampling parameters; pull the latest code |
| `pip install` runs forever | Python 3.9 or old pip forcing dependency backtracking | Use Python 3.11 and `pip install --upgrade pip` |
| `SameFileError` from `add` | File is already in `data/docs/` | Use `python -m rag_eval.ingest` for files already there |
| `You are sending unauthenticated requests to the HF Hub` | Harmless warning from the embedding model download check | Ignore, or set `HF_TOKEN` |
| No traces in Phoenix, or `Connection refused` on port 6006 | Server not running, or tracing off | Start `phoenix serve` in a second terminal and keep it open; check `RAG_TRACING=1`; the UI is on port 6006 |
