# RAG Chatbot — Groq + FAISS + BM25 Hybrid Retrieval

A complete, end-to-end Retrieval-Augmented Generation chatbot covering the
features discussed in design: hybrid retrieval, reranking, query rewriting,
conversation memory, web-search fallback, source citations, guardrails,
and logging.

## Architecture

```
User question
     │
     ▼
Sanitize input (guardrail)
     │
     ▼
Rewrite as standalone question (using chat history) ── Groq LLM call
     │
     ▼
Hybrid retrieval
   ├── FAISS vector search (semantic)
   └── BM25 keyword search (exact terms)
     │
     ▼
Merge + dedupe candidates
     │
     ▼
Cross-encoder reranking → confidence score
     │
     ├── score ≥ threshold ──► Answer from DOCUMENTS (Groq LLM, cited sources)
     │
     └── score < threshold ──► Web search fallback ──► Answer from WEB
                                  (clearly labeled as not from your documents)
     │
     ▼
Log query/scores/answer + rough groundedness check
     │
     ▼
Streamlit chat UI (with source citations shown)
```

## Why this design

- **FAISS alone misses exact matches** (codes, names, numbers) because
  embeddings capture meaning, not exact text — BM25 catches those.
- **Cross-encoder reranking** re-scores the merged candidates with a model
  that looks at the query and chunk *together*, which is far more accurate
  than cosine similarity for final ranking.
- **Confidence-based fallback** is the answer to "why does it just say
  'not enough info'?" — instead of refusing outright, the bot searches the
  web when document confidence is low, but **always tells the user which
  source the answer came from** so it's never misleading.
- **Query rewriting** makes follow-up questions ("what about 2023?") work,
  since they get expanded into standalone questions before retrieval.
- **Prompt-injection guardrail**: the system prompt tells the LLM to treat
  retrieved text as data, not instructions — important if a malicious
  instruction is hidden inside an uploaded document.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env and add your real GROQ_API_KEY
streamlit run app.py
```

Get a Groq API key at https://console.groq.com. Check
https://console.groq.com/docs/models for the current list of available
model IDs — Groq updates this periodically, so confirm `GROQ_MODEL` in
`.env` is still valid before relying on it.

## Using it

1. Upload PDF/DOCX/TXT/CSV files in the sidebar and click **Build / Update
   Knowledge Base**.
2. Ask questions in the chat box.
3. Each answer shows its sources — document chunks (with relevance score)
   or web links, depending on which path was used.
4. Toggle **Enable web search fallback** off if you want strict
   "documents-only" behavior again.

## Tuning

- `RERANK_CONFIDENCE_THRESHOLD` (config.py) — controls how confident the
  retriever must be before trusting the documents. Lower it if the bot
  falls back to the web too often; raise it if it answers from weak
  document matches. Watch the `rag_app.log` file (logs the top score for
  every query) to calibrate this for your own documents.
- `CHUNK_SIZE` / `CHUNK_OVERLAP` — larger chunks keep more context together
  but reduce retrieval precision. Start at 500/80 words and adjust.
- `FINAL_TOP_N` — how many chunks get passed to the LLM as context.

## What's intentionally simplified (and how to harden it for production)

| This project uses | Production upgrade |
|---|---|
| FAISS + pickle files on disk | Chroma, Qdrant, Weaviate, or Pinecone — easier incremental updates, filtering, multi-user scaling |
| DuckDuckGo search (no API key) | Tavily, Serper, or Bing Search API — more reliable, higher rate limits |
| Word-count chunking | Token-aware or semantic chunking |
| Word-overlap groundedness heuristic | RAGAS or a dedicated LLM-as-judge evaluation pipeline |
| Local file logging | Structured logging to a monitoring stack (e.g. ELK, Datadog) |
| Single-process Streamlit app | FastAPI backend + separate frontend, with auth and rate limiting |

## File overview

- `config.py` — all tunable settings in one place
- `ingest.py` — document loaders, chunking, FAISS + BM25 index building
- `retriever.py` — hybrid search + cross-encoder reranking
- `web_fallback.py` — DuckDuckGo web search fallback
- `rag_chain.py` — orchestration: query rewriting, fallback decision, Groq generation, guardrails, logging
- `app.py` — Streamlit chat UI
