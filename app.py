"""
Streamlit chat UI for the Groq-powered RAG chatbot.

Run with: streamlit run app.py
"""

import os
import streamlit as st
import config
import ingest

st.set_page_config(page_title="RAG Chatbot (Groq)", page_icon="💬", layout="wide")

# Fail fast with a clear message if the Groq key isn't set up correctly,
# instead of letting the first chat message crash with a raw 401 error.
try:
    config.validate_groq_key()
except RuntimeError as e:
    st.error(f"⚠️ Groq API key problem:\n\n{e}")
    st.stop()

import rag_chain

if "history" not in st.session_state:
    st.session_state.history = []  # list of {"user", "assistant", "sources", "used_web_fallback"}
if "index_data" not in st.session_state:
    st.session_state.index_data = ingest.load_index()  # (faiss_index, metadata, bm25) or None





# ==================== Sidebar: knowledge base management ====================

st.sidebar.title("📁 Knowledge Base")

uploaded_files = st.sidebar.file_uploader(
    "Upload documents (PDF, DOCX, TXT, CSV)",
    type=["pdf", "docx", "txt", "csv"],
    accept_multiple_files=True,
)

if st.sidebar.button("Build / Update Knowledge Base", use_container_width=True):
    if not uploaded_files:
        st.sidebar.warning("Please upload at least one file first.")
    else:
        os.makedirs(config.DATA_DIR, exist_ok=True)
        saved_paths = []
        for f in uploaded_files:
            path = os.path.join(config.DATA_DIR, f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())
            saved_paths.append(path)

        with st.spinner("Indexing documents... this can take a minute for large files."):
            try:
                index, metadata, bm25 = ingest.build_index(saved_paths)
                st.session_state.index_data = (index, metadata, bm25)
                st.sidebar.success(f"Indexed {len(saved_paths)} file(s) -> {len(metadata)} chunks.")
            except Exception as e:
                st.sidebar.error(f"Failed to build index: {e}")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Settings")
enable_web_fallback = st.sidebar.checkbox(
    "Enable web search fallback",
    value=True,
    help="If the documents don't contain a good answer, search the web instead of just refusing.",
)

source_filter = None
if st.session_state.index_data:
    _, metadata, _ = st.session_state.index_data
    sources = sorted(set(m["source"] for m in metadata))
    source_choice = st.sidebar.selectbox("Filter by document (optional)", ["All"] + sources)
    source_filter = None if source_choice == "All" else source_choice

if st.sidebar.button("🗑️ Clear conversation", use_container_width=True):
    st.session_state.history = []
    st.rerun()





# ==================== Main chat area ====================

st.title("💬 RAG Chatbot")
st.caption("Groq LLM • FAISS + BM25 hybrid retrieval • cross-encoder reranking • web fallback")

if not st.session_state.index_data:
    st.info("👈 Upload documents and click **Build / Update Knowledge Base** to get started.")


def render_sources(used_web_fallback, sources):
    if used_web_fallback:
        st.caption("🌐 This answer used a web search — it is not based on your uploaded documents.")
        if sources:
            with st.expander("🔗 Web sources"):
                for s in sources:
                    st.markdown(f"- [{s.get('title', 'source')}]({s.get('href', '')})")
    elif sources:
        with st.expander("📑 Document sources"):
            for s in sources:
                page_info = f", page {s['page']}" if s.get("page") else ""
                st.markdown(f"**{s['source']}{page_info}** — relevance {s.get('rerank_score', 0):.2f}")
                st.text(s["text"][:300] + ("..." if len(s["text"]) > 300 else ""))


for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["user"])
    with st.chat_message("assistant"):
        st.write(turn["assistant"])
        render_sources(turn.get("used_web_fallback", False), turn.get("sources", []))

user_query = st.chat_input("Ask a question about your documents...")

if user_query:
    with st.chat_message("user"):
        st.write(user_query)

    if not st.session_state.index_data:
        result = {
            "answer": "Please upload and index documents first using the sidebar.",
            "sources": [],
            "used_web_fallback": False,
            "groundedness": None,
        }
    else:
        index, metadata, bm25 = st.session_state.index_data
        with st.spinner("Thinking..."):
            result = rag_chain.answer_query(
                user_query,
                index,
                metadata,
                bm25,
                history=st.session_state.history,
                source_filter=source_filter,
                enable_web_fallback=enable_web_fallback,
            )

    with st.chat_message("assistant"):
        st.write(result["answer"])
        render_sources(result["used_web_fallback"], result["sources"])

    st.session_state.history.append({
        "user": user_query,
        "assistant": result["answer"],
        "sources": result["sources"],
        "used_web_fallback": result["used_web_fallback"],
    })