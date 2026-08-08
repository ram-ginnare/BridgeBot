import json
import os
import pandas as pd
import streamlit as st
from ingest import ingest_github_pdf
from ingest import ingest_google_drive_pdf
from ingest import ingest_pdf
from query import ask
import time
from utils.logger import (rag_logger, performance_logger, error_logger, log_performance)
#login
from auth.login import login_page
from auth.session import (is_logged_in, logout)

st.set_page_config(
    page_title="BridgeBot",
    layout="wide"
)

# -------------------------------
# Authentication
# -------------------------------

if not is_logged_in():

    login_page()

    st.stop()

# -------------------------------
# Logged-in User
# -------------------------------

st.sidebar.success(
    f"👤 {st.session_state.user}"
)

st.sidebar.write(
    f"Role : {st.session_state.role}"
)

st.sidebar.divider()

if st.sidebar.button("🚪 Logout"):

    logout()

    st.rerun()

st.sidebar.divider()

# ---------------------------------------------------
# Configuration
# ---------------------------------------------------

PDF_FOLDER = "pdfs"
REGISTRY_FILE = "document_registry.json"

os.makedirs(PDF_FOLDER, exist_ok=True)

st.set_page_config(
    page_title="PDF Chat using Groq",
    page_icon="📄",
    layout="wide"
)

# ---------------------------------------------------
# Session State
# ---------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------

def load_uploaded_documents():

    if not os.path.exists(REGISTRY_FILE):
        return pd.DataFrame()

    with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    rag_logger.info("Loaded %d documents from registry", len(data) )

    return pd.DataFrame(data)


def knowledge_base_ready():

    ud = load_uploaded_documents()

    return not ud.empty


# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

with st.sidebar:

    st.title("📚 Knowledge Base")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        pdf_path = os.path.join(
            PDF_FOLDER,
            uploaded_file.name
        )

        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success("PDF uploaded successfully.")

        rag_logger.info("PDF Uploaded : %s", uploaded_file.name)

        if st.button("🚀 Prepare Knowledge Base"):

            with st.spinner("Creating Knowledge Base..."):

                rag_logger.info("Preparing Knowledge Base : %s", uploaded_file.name)

                start = time.perf_counter()

                owner = st.session_state.user
                department = st.session_state.department
                team = st.session_state.team
                # TODO:  take visibility from UI in dropdown
                visibility = "Private"

                status = ingest_pdf(pdf_path,owner=owner, department=department, team=team, visibility=visibility)

                elapsed = time.perf_counter() - start

                rag_logger.info("Knowledge Base Status : %s", status)

                performance_logger.info("Knowledge Base Creation Time : %.3f sec", elapsed)

            st.success("Knowledge Base Ready!")

            st.rerun()

    st.divider()

    st.sidebar.subheader("GitHub PDF")

    github_url = st.sidebar.text_input(
        "GitHub Raw PDF URL"
    )

    if st.sidebar.button("Prepare KB From GitHub"):

        if github_url.strip():

            with st.spinner("Preparing Knowledge Base..."):

                status = ingest_github_pdf(github_url)

            if status:

                st.success("Knowledge Base Created Successfully")

            else:

                st.error("Knowledge Base Creation Failed")


    st.divider()

    st.sidebar.subheader("Google Drive PDF")

    drive_url = st.sidebar.text_input(
        "Public Google Drive PDF URL"
    )

    if st.sidebar.button("Prepare KB From Google Drive"):

        if drive_url.strip():

            with st.spinner("Preparing Knowledge Base..."):

                status = ingest_google_drive_pdf(drive_url)

            if status:

                st.success("Knowledge Base Created Successfully")

            else:

                st.error("Knowledge Base Creation Failed")


    st.divider()

    st.subheader("📚 Uploaded Documents")

    uploaded_documents = load_uploaded_documents()

    if not uploaded_documents.empty:

        st.dataframe(
            uploaded_documents,
            use_container_width=True,
            hide_index=True
        )

        st.success(f"Knowledge Base Ready ({len(uploaded_documents)} document(s))")

        selected_documents = st.sidebar.multiselect(

            "Select Documents",

            uploaded_documents["document_name"].tolist()

        )

        selected_category = st.sidebar.selectbox(
            "Category",
            ["All"] +
            sorted(uploaded_documents["category"].unique())
        )

        if selected_category == "All":
            selected_category = None

    else:
        st.warning("No documents uploaded.")

    st.divider()

    if st.button("🗑 Clear Chat"):

        st.session_state.messages = []

        rag_logger.info("Chat History Cleared")

        st.rerun()


# ---------------------------------------------------
# Main Screen
# ---------------------------------------------------

st.title("📄 Chat with PDF")

st.caption(
    "Groq + LangChain + ChromaDB + HuggingFace Embeddings"
)

# ---------------------------------------------------
# Display Chat History
# ---------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ---------------------------------------------------
# Chat Input
# ---------------------------------------------------
use_web_search = st.checkbox(
    "🌐 Enable Web Search",
    value=False
)
rag_logger.info("Web Search Enabled : %s", use_web_search)

question = st.chat_input(
    "Ask anything about your documents...",
    disabled=not knowledge_base_ready()
)

if question:

    request_start = time.perf_counter()

    rag_logger.info("=" * 80)

    rag_logger.info( "New Question Received" )

    rag_logger.info( "Question : %s", question )

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching Knowledge Base..."):

            query_start = time.perf_counter()
            try:
                result = ask(
                    question,
                    use_web_search=use_web_search,
                    selected_documents=selected_documents,
                    selected_category=selected_category,
                    owner=st.session_state.user,
                    department=st.session_state.department,
                    team=st.session_state.team,
                    visibility="Private"
                )

            except Exception as e:

                error_logger.exception(f"Application exception : {e}")

                st.error("Unexpected Error")

            rag_logger.info( "Retrieved Documents : %s", ", ".join(result["documents"]) )

            rag_logger.info( "Retrieved Pages : %s", result["pages"] )

            rag_logger.info("Reranker : %s", result["reranker"])

            rag_logger.info("Reranker Scores : %s", result["rerank_scores"])

            rag_logger.info("Web Search Used : %s", result["web_used"])

            if result["web_used"]:

                rag_logger.info("Web Sources : %s", result["web_sources"])

            rag_logger.info("Answer Length : %d characters", len(result["answer"]))

            query_time = time.perf_counter() - query_start

            performance_logger.info( "Query Processing Time : %.3f sec", query_time )

        st.markdown(result["answer"])

        with st.expander("📌 Sources Used", expanded=False):

            st.write(f"**LLM :** {result['llm']}")

            st.write(f"**Vector Database :** {result['vector_db']}")

            st.write(f"**Documents :** {result['documents']}")

            st.write(f"**Chunks size :** {result['chunks']}")

            if result["pages"]:

                pages = ", ".join(
                    str(page + 1)
                    for page in result["pages"]
                )

                st.write(f"**Pages :** {pages}")

            else:

                st.write("**Pages :** N/A")

            if result["web_used"]:

                st.write("**Web Search :** Used")

                for url in result["web_sources"]:

                    st.write(url)

            else:

                st.write("**Web Search :** Not Used")

            # Cross encoder - Reranker score
            st.write(f"**Reranker:** {result['reranker']}")

            st.write("**Reranker Scores:**")

            for score in result["rerank_scores"]:
                st.write(f"{score:.4f}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"]
        }
    )

    total = time.perf_counter() - request_start

    performance_logger.info("Total Request Time : %.3f sec", total)

    rag_logger.info("=" * 80)


# ---------------------------------------------------
# Empty Knowledge Base Message
# ---------------------------------------------------

if not knowledge_base_ready():

    st.info(
        """
### 📄 No Knowledge Base Found

Please follow these steps:

1. Upload one or more PDF documents.
2. Click **🚀 Prepare Knowledge Base**.
3. Wait until indexing is complete.
4. Start asking questions.

"""
    )