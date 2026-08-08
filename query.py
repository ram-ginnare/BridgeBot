import os
import streamlit as st
from dotenv import load_dotenv
# from langchain_community.tools import DuckDuckGoSearchRun
from ddgs import DDGS
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from rank_bm25 import BM25Okapi
import joblib
from reranker import rerank
import time
from utils.logger import (rag_logger, performance_logger, error_logger, log_performance)

load_dotenv(override=True)


# ------------------------------------
# Configuration
# ------------------------------------

CHROMA_DB_PATH = "chroma_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "llama-3.3-70b-versatile"

TOP_K = 3

BM25_INDEX = "bm25/bm25_index.pkl"

# ------------------------------------
# Embeddings
# ------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

# ------------------------------------
# Load Vector Database
# ------------------------------------

vector_db = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embeddings
)

# ------------------------------------
# Groq LLM
# ------------------------------------

llm = ChatGroq(
    model=LLM_MODEL,
    temperature=0
)

# ------------------------------------
# DuckDuckGo Search
# ------------------------------------

# search = DuckDuckGoSearchRun()
search = DDGS()

# ------------------------------------
# Ask Question
# ------------------------------------

def ask(
        question,
        use_web_search=False,
        selected_documents=None,
        selected_category=None,
        owner=None,
        department=None,
        team=None,
        visibility="Private"
):

    request_start = time.perf_counter()

    rag_logger.info("=" * 80)
    rag_logger.info("New Query")
    rag_logger.info("Question : %s", question)
    rag_logger.info("Web Search : %s", use_web_search)

    try:

        # ---------------------------------------------------
        # Hybrid Search
        # ---------------------------------------------------

        start = time.perf_counter()

        docs = hybrid_search(
            question=question,
            selected_documents=selected_documents,
            selected_category=selected_category,
            owner=owner,
            department=department,
            team=team,
            visibility=visibility
        )

        performance_logger.info("Hybrid Search returned %d chunks in %.3f sec", len(docs), time.perf_counter() - start)

        # ---------------------------------------------------
        # Cross Encoder
        # ---------------------------------------------------

        start = time.perf_counter()

        docs, rerank_scores = rerank(
            question,
            docs,
            top_k=3
        )

        performance_logger.info("Cross Encoder selected %d chunks in %.3f sec", len(docs), time.perf_counter() - start)

        context = ""

        pages = set()

        documents = set()

        rag_logger.info("\n\n=====start=========docs===============\n\n")

        rag_logger.info(docs)

        rag_logger.info("\n\n=====end=========docs===============\n\n")

        for doc in docs:

            context += doc.page_content + "\n\n"

            pages.add(
                doc.metadata.get("page", "Unknown")
            )

            source = doc.metadata.get("document_name", "")

            if source:
                documents.add(
                    os.path.basename(source)
                )

        rag_logger.info("Documents : %s", ", ".join(sorted(documents)))

        rag_logger.info("Pages : %s", sorted(pages))

        # ---------------------------------------------------
        # Optional Web Search
        # ---------------------------------------------------

        web_context = ""

        web_sources = []

        if use_web_search:

            try:

                start = time.perf_counter()

                rag_logger.info("DuckDuckGo Search Started")

                # web_context = search.run(question)
                with DDGS() as ddgs:
                    web_results = list(ddgs.text(question, max_results=1))
                    web_context = web_results[0]["body"]

                performance_logger.info("Web Search completed in %.3f sec", time.perf_counter() - start)

                if web_context:

                    context += "\n\n========== WEB SEARCH ==========\n\n"

                    context += web_context

                    web_sources.append("DuckDuckGo")

                    rag_logger.info("Web Context Size : %d chars", len(web_context))

            except Exception as e:

                error_logger.exception(f"Web Search Failed: {e}")

        if context.strip() == "":

            rag_logger.warning("No Context Found")

            return {

                "answer": "No relevant information found.",

                "llm": LLM_MODEL,

                "vector_db": "ChromaDB",

                "documents": [],

                "pages": [],

                "chunks": 0,

                "metadata_filter": st.session_state.metadata_filter,

                "web_used": use_web_search,

                "web_sources": web_sources,

                "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",

                "rerank_scores": []

            }

        prompt = f"""
You are an expert AI assistant.

Use the context below to answer the user's question.

Priority:
1. Use uploaded PDF content first.
2. If web search content exists, use it only when required.
3. If the answer isn't available anywhere, say so.

========================
Context
========================

{context}

========================
Question
========================

{question}

========================
Answer
========================
"""

        # ---------------------------------------------------
        # LLM
        # ---------------------------------------------------

        start = time.perf_counter()

        rag_logger.info("Sending Prompt To Groq")

        response = llm.invoke(prompt)

        llm_time = time.perf_counter() - start

        performance_logger.info("Groq Response Time : %.3f sec", llm_time)

        rag_logger.info("Answer Length : %d characters", len(response.content))

        total_time = time.perf_counter() - request_start

        performance_logger.info("Total Query Time : %.3f sec", total_time)

        rag_logger.info("=" * 80)

        return {

            "answer": response.content,

            "llm": LLM_MODEL,

            "vector_db": "ChromaDB",

            "documents": sorted(documents),

            "pages": sorted(pages),

            "chunks": len(docs),

            "web_used": use_web_search,

            "web_sources": web_sources,

            "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",

            "rerank_scores": rerank_scores

        }

    except Exception as ex:

        error_logger.exception("Query Failed")

        return {

            "answer": str(ex),

            "llm": LLM_MODEL,

            "vector_db": "ChromaDB",

            "documents": [],

            "pages": [],

            "chunks": 0,

            "web_used": use_web_search,

            "web_sources": [],

            "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",

            "rerank_scores": []

        }

def bm25_search(question, k=5,
                selected_documents=None,
                selected_category=None,
                owner=None,
                department=None,
                team=None,
                visibility=None):

    rag_logger.info("BM25 Search Started")

    if not os.path.exists(BM25_INDEX):
        return []

    index = joblib.load(BM25_INDEX)

    bm25 = index["bm25"]

    docs = index["documents"]

    filtered_documents = []
    filtered_tokens = []

    for doc in docs:
        metadata = doc.metadata
        if selected_documents:

            if metadata.get("document_name") not in selected_documents:
                continue

        if selected_category:

            if metadata.get("category") != selected_category:
                continue

        if owner:

            if metadata.get("owner") != owner:
                continue

        if department:

            if metadata.get("department") != department:
                continue

        if team:

            if metadata.get("team") != team:
                continue

        if visibility:

            if metadata.get("visibility") != visibility:
                continue

        filtered_documents.append(doc)
        filtered_tokens.append(doc.page_content.lower().split())

    bm25 = BM25Okapi(filtered_tokens)

    # ------------------------------------------
    # BM25 Search : code for return docs only with score > 0.10
    # ------------------------------------------

    query_tokens = question.lower().split()

    scores = bm25.get_scores(query_tokens)

    ranked = sorted(zip(scores, filtered_documents), key=lambda x: x[0], reverse=True)

    rag_logger.info("BM25 Retrieved %d relevant documents (score > 0)", len(ranked))

    # Log individual scores (optional)
    for score, doc in ranked:

        rag_logger.debug(
            "BM25 Score: %.4f | Document: %s | Page: %s",
            score,
            doc.metadata.get("document_name"),
            doc.metadata.get("page")
        )

    return [doc for score, doc in ranked[:k]]


def hybrid_search(
        question,
        selected_documents=None,
        selected_category=None,
        owner=None,
        department=None,
        team=None,
        visibility=None
):

    rag_logger.info("Hybrid Search Started")

    # ----------------------------------------
    # Metadata Filter
    # ----------------------------------------

    metadata_filter = None

    conditions = []

    if selected_documents:
        conditions.append({
            "document_name": {
                "$in": selected_documents
            }
        })

    if selected_category:
        conditions.append({
            "category": selected_category
        })

    if st.session_state.department:
        conditions.append({
            "department": st.session_state.department
        })

    if st.session_state.user:
        conditions.append({
            "uploaded_by": st.session_state.user
        })

    #TODO: update visibility condition
    # if visibility:
    #     conditions.append({
    #         "visibility": visibility
    #     })

    if len(conditions) == 1:
        metadata_filter = conditions[0]

    elif len(conditions) > 1:
        metadata_filter = {
            "$and": conditions
        }

    rag_logger.info("Metadata Filter : %s", metadata_filter)
    st.session_state.metadata_filter = metadata_filter

    if metadata_filter:
        dense_docs = vector_db.similarity_search(question, k=5, filter=metadata_filter)
    else:
        dense_docs = vector_db.similarity_search(question, k=5)

    rag_logger.info("Dense Search : %d chunks", len(dense_docs))

    sparse_docs = bm25_search(
        question,
        k=20,
        selected_documents=selected_documents,
        selected_category=selected_category,
        owner=owner,
        department=department,
        team=team,
        visibility=visibility
    )

    if selected_documents:
        sparse_docs = [doc for doc in sparse_docs if doc.metadata.get("document_name") in selected_documents]

    if selected_category:
        sparse_docs = [doc for doc in sparse_docs if doc.metadata.get("category") == selected_category]


    rag_logger.info("BM25 Search : %d chunks", len(sparse_docs))

    rag_logger.info("Metadata Filtered Dense Docs : %d", len(dense_docs))

    rag_logger.info("Metadata Filtered BM25 Docs : %d", len(sparse_docs))

    results = []

    seen = set()

    for doc in dense_docs + sparse_docs:

        text = doc.page_content

        if text not in seen:

            results.append(doc)

            seen.add(text)

    rag_logger.info("Hybrid Search Final : %d chunks", len(results))

    return results


# ------------------------------------
# Test
# ------------------------------------

if __name__ == "__main__":

    result = ask(
        "What is Spring Boot?",
        use_web_search=True
    )

    print(result["answer"])
    print(result["documents"])
    print(result["pages"])
    print(result["web_used"])
    print(result["web_sources"])