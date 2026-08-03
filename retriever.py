"""
Hybrid retrieval pipeline:

  1. Semantic search via FAISS (catches paraphrases / meaning-based matches)
  2. Keyword search via BM25 (catches exact terms, codes, names that
     embeddings often miss)
  3. Merge + dedupe candidates from both
  4. Cross-encoder reranking for a final, more accurate relevance ordering

The reranked top score is also used by rag_chain.py to decide whether the
documents actually contain a good answer, or whether to fall back to the web.
"""

import math
import logging
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import config

logger = logging.getLogger("rag_retriever")

_embedder = None
_reranker = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _embedder


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(config.RERANKER_MODEL_NAME)
    return _reranker


def vector_search(query, faiss_index, metadata, k=config.VECTOR_TOP_K):
    embedder = get_embedder()
    q_vec = embedder.encode([query], normalize_embeddings=True)
    q_vec = np.array(q_vec, dtype="float32")
    scores, indices = faiss_index.search(q_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({**metadata[idx], "vector_score": float(score)})
    return results


def bm25_search(query, bm25, metadata, k=config.BM25_TOP_K):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_idx = np.argsort(scores)[::-1][:k]

    results = []
    for idx in top_idx:
        if scores[idx] <= 0:
            continue
        results.append({**metadata[idx], "bm25_score": float(scores[idx])})
    return results


def merge_candidates(vector_results, bm25_results):
    """Dedupe by chunk_id, keeping both score types where available."""
    merged = {}
    for r in vector_results:
        merged[r["chunk_id"]] = {**r}
    for r in bm25_results:
        if r["chunk_id"] in merged:
            merged[r["chunk_id"]]["bm25_score"] = r["bm25_score"]
        else:
            merged[r["chunk_id"]] = {**r}
    return list(merged.values())


def rerank(query, candidates, top_n=config.FINAL_TOP_N):
    """Cross-encoder reranking. Scores are sigmoid-normalized to [0, 1]
    so they can be compared against a fixed confidence threshold."""
    if not candidates:
        return []

    reranker = get_reranker()
    pairs = [[query, c["text"]] for c in candidates]
    raw_scores = reranker.predict(pairs)

    for c, s in zip(candidates, raw_scores):
        c["rerank_score"] = float(1 / (1 + math.exp(-s)))  # sigmoid normalize

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:top_n]


def retrieve(query, faiss_index, metadata, bm25, source_filter=None):
    """Full hybrid retrieval pipeline. Returns the final reranked chunks."""
    vector_results = vector_search(query, faiss_index, metadata)
    bm25_results = bm25_search(query, bm25, metadata)
    candidates = merge_candidates(vector_results, bm25_results)

    if source_filter:
        candidates = [c for c in candidates if c["source"] == source_filter]

    ranked = rerank(query, candidates)
    logger.info(f"Retrieved {len(ranked)} chunks for query: {query!r}")
    return ranked


def best_score(ranked_chunks):
    if not ranked_chunks:
        return 0.0
    return ranked_chunks[0]["rerank_score"]