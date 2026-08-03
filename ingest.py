"""
Document ingestion pipeline: load files -> clean -> chunk -> embed -> persist.

Supports PDF, DOCX, TXT, and CSV. Builds both a FAISS vector index (semantic
search) and a BM25 index (keyword search) so the retriever can do hybrid
search instead of relying on embeddings alone.
"""

import os
import csv
import pickle
import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from pypdf import PdfReader
import docx
import config

logger = logging.getLogger("rag_ingest")


# ---------------- Loaders ----------------
# Each loader returns a list of (text, metadata) tuples. PDFs return one
# tuple per page (so page numbers can be cited); other formats return a
# single tuple for the whole file.

def load_pdf(path):
    sections = []
    reader = PdfReader(path)
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            sections.append((text, {"page": i + 1}))
    return sections


def load_docx(path):
    doc = docx.Document(path)
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [(full_text, {"page": None})]


def load_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [(f.read(), {"page": None})]


def load_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(", ".join(row))
    return [("\n".join(rows), {"page": None})]


LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".txt": load_txt,
    ".csv": load_csv,
}


def load_document(path): 
    ext = os.path.splitext(path)[1].lower()
    if ext not in LOADERS:
        raise ValueError(f"Unsupported file type: {ext}")
    return LOADERS[ext](path)


# ---------------- Chunking ----------------

def chunk_text(text, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    """Word-based sliding-window chunking with overlap.

    This is intentionally simple (word count, not tokens). For production
    use, consider a tokenizer-aware or semantic chunker instead.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap
    return chunks


# ---------------- Index building ----------------

def build_index(file_paths, persist=True):
    """
    Build a FAISS vector index + BM25 index from a list of document paths.

    Returns (faiss_index, chunk_metadata_list, bm25_index).
    chunk_metadata_list[i] corresponds to embedding row i in the FAISS index
    and contains: source, page, chunk_id, text.
    """
    embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    all_chunks = []
    all_metadata = []

    for path in file_paths:
        filename = os.path.basename(path)
        try:
            sections = load_document(path)
        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")
            continue

        for text, meta in sections:
            for idx, chunk in enumerate(chunk_text(text)):
                all_chunks.append(chunk)
                all_metadata.append({
                    "source": filename,
                    "page": meta.get("page"),
                    "chunk_id": len(all_metadata),
                    "text": chunk,
                })

    if not all_chunks:
        raise ValueError("No text could be extracted from the provided documents.")

    logger.info(f"Total chunks created: {len(all_chunks)}")

    # ---- Embeddings + FAISS (cosine similarity via normalized inner product) ----
    embeddings = embedder.encode(all_chunks, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # ---- BM25 keyword index ----
    tokenized_corpus = [c.lower().split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    if persist:
        os.makedirs(config.VECTORSTORE_DIR, exist_ok=True)
        faiss.write_index(index, config.FAISS_INDEX_PATH)
        with open(config.METADATA_PATH, "wb") as f:
            pickle.dump(all_metadata, f)
        with open(config.BM25_PATH, "wb") as f:
            pickle.dump(bm25, f)
        logger.info("Vector store, metadata, and BM25 index persisted to disk.")

    return index, all_metadata, bm25


def load_index():
    """Load a previously persisted index from disk. Returns None if not found."""
    paths_exist = (
        os.path.exists(config.FAISS_INDEX_PATH)
        and os.path.exists(config.METADATA_PATH)
        and os.path.exists(config.BM25_PATH)
    )
    if not paths_exist:
        return None

    index = faiss.read_index(config.FAISS_INDEX_PATH)
    with open(config.METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    with open(config.BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)
    return index, metadata, bm25