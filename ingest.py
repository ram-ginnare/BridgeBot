import json
import uuid
import os
import streamlit as st
from urllib.parse import urlparse, unquote
from datetime import datetime
from rank_bm25 import BM25Okapi
import joblib
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from models.document_metadata import create_document_metadata
import time
import tempfile
import requests
import re

from utils.logger import (rag_logger, performance_logger, error_logger, log_performance)

load_dotenv()

# -----------------------------------
# Configuration
# -----------------------------------

CHROMA_DB_PATH = "chroma_db"
REGISTRY_FILE = "document_registry.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

BM25_INDEX = "bm25/bm25_index.pkl"

os.makedirs("bm25", exist_ok=True)

# -----------------------------------
# Registry Functions
# -----------------------------------

def load_registry():

    if not os.path.exists(REGISTRY_FILE):
        return []

    with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(data):

    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


# -----------------------------------
# Ingest PDF
# -----------------------------------

def ingest_pdf(pdf_path,
               document_name=None,
               source_type=None,
               source_url=None,
               owner=None,
               department=None,
               team=None,
               visibility="Private"):

    request_start = time.perf_counter()

    if document_name is None:
        document_name = os.path.basename(pdf_path)

    rag_logger.info("=" * 80)
    rag_logger.info("Knowledge Base Creation Started")
    rag_logger.info("Document : %s", document_name)

    try:

        registry = load_registry()

        # --------------------------------------------------------
        # Duplicate Check
        # --------------------------------------------------------

        if any(doc["name"] == document_name for doc in registry):

            logger.warning("Duplicate document detected : %s", document_name)

            return False

        # --------------------------------------------------------
        # Load PDF
        # --------------------------------------------------------

        start = time.perf_counter()

        loader = PyPDFLoader(pdf_path)

        documents = loader.load()

        pdf_load_time = time.perf_counter() - start

        total_pages = len(documents)

        rag_logger.info("Pages : %d", total_pages)

        performance_logger.info("PDF Loaded in %.3f sec", pdf_load_time)

        # --------------------------------------------------------
        # Chunking
        # --------------------------------------------------------

        start = time.perf_counter()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)


        # --------------------------------------------------------
        # Add Metadata to Every Chunk
        # --------------------------------------------------------

        #document_id = os.path.splitext(document_name)[0]
        document_id = str(uuid.uuid4())

        document_without_ext = os.path.splitext(document_name)[0]
        category = document_without_ext.split("_")[0]

        upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        file_size_mb = round(os.path.getsize(pdf_path) / (1024 * 1024), 2)

        # doc_metadata = create_document_metadata(
        #     document_name=document_name,
        #     owner=st.session_state.user,
        #     department="AI",
        #     team="BridgeBot",
        #     visibility="Private",
        #     source_type="Local"
        # )

        for index, chunk in enumerate(chunks):

            chunk.metadata["document_id"] = document_id

            chunk.metadata["document"] = document_name

            chunk.metadata["owner"] = owner

            chunk.metadata["department"] = department

            chunk.metadata["team"] = team

            chunk.metadata["visibility"] = visibility

            chunk.metadata["uploaded_by"] = owner

            chunk.metadata["category"] = category

            chunk.metadata["source_url"] = source_url

            chunk.metadata["source_type"] = source_type

            chunk.metadata["uploaded_date"] = upload_time

            chunk.metadata["chunk_id"] = index + 1

            chunk.metadata["total_pages"] = total_pages

            chunk.metadata["file_size_mb"] = file_size_mb

            chunk.metadata["embedding_model"] = EMBEDDING_MODEL

        rag_logger.info("Metadata Added To %d Chunks", len(chunks))

        chunk_time = time.perf_counter() - start

        rag_logger.info("Chunks Created : %d", len(chunks))

        performance_logger.info("Chunking and Metadata Time : %.3f sec", chunk_time)

        # --------------------------------------------------------
        # BM25 Index
        # --------------------------------------------------------

        start = time.perf_counter()

        tokenized_corpus = [
            chunk.page_content.lower().split()
            for chunk in chunks
        ]

        bm25 = BM25Okapi(tokenized_corpus)

        joblib.dump(
            {
                "bm25": bm25,
                "documents": chunks
            },
            BM25_INDEX
        )

        bm25_time = time.perf_counter() - start

        rag_logger.info("BM25 Index Created")

        performance_logger.info("BM25 Time : %.3f sec", bm25_time)

        # --------------------------------------------------------
        # ChromaDB
        # --------------------------------------------------------

        start = time.perf_counter()

        if os.path.exists(CHROMA_DB_PATH):

            vector_db = Chroma(
                persist_directory=CHROMA_DB_PATH,
                embedding_function=embeddings
            )

            vector_db.add_documents(chunks)

            vector_db.persist()

            rag_logger.info("Documents Added To Existing ChromaDB")

        else:

            Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=CHROMA_DB_PATH
            )

            rag_logger.info("New ChromaDB Created")

        chroma_time = time.perf_counter() - start

        performance_logger.info("ChromaDB Time : %.3f sec", chroma_time)

        # --------------------------------------------------------
        # Registry
        # --------------------------------------------------------

        registry.append(
            {
                "document_id": document_id,
                "document_name": document_name,
                "owner": owner,
                "department": department,
                "team": team,
                "visibility": visibility,
                "uploaded_by": owner,
                "uploaded_date": upload_time,
                "source_type": source_type,
                "source_url": source_url,
                "category": category,
                "pages": total_pages,
                "chunks": len(chunks),
                "file_size_mb": file_size_mb,
                "embedding_model": EMBEDDING_MODEL,
                "vector_db": "ChromaDB",
                "status": "Indexed"
            }
        )

        save_registry(registry)

        rag_logger.info("Document Registry Updated")
        rag_logger.info("Document ID : %s", document_id)
        rag_logger.info("Category : %s", category)
        rag_logger.info("File Size : %.2f MB", file_size_mb)

        total_time = time.perf_counter() - request_start

        rag_logger.info("=" * 80)
        rag_logger.info("Knowledge Base Created Successfully")
        performance_logger.info("Total Time : %.3f sec", total_time)
        rag_logger.info("=" * 80)

        return True

    except Exception as e:

        error_logger.exception(f"Knowledge Base Creation Failed {e}")

        return False


# -----------------------------------
# Ingest GitHub PDF
# -----------------------------------
def ingest_github_pdf(url):
    """
    Downloads a PDF from GitHub,
    creates a temporary file,
    prepares the knowledge base,
    and deletes the temporary file.
    """

    request_start = time.perf_counter()

    rag_logger.info("=" * 80)
    rag_logger.info("GitHub PDF Knowledge Base Creation Started")
    rag_logger.info("GitHub URL : %s", url)

    temp_file = None

    try:

        # --------------------------------------------------------
        # Download PDF
        # --------------------------------------------------------

        start = time.perf_counter()

        rag_logger.info("Downloading PDF from GitHub...")

        response = requests.get(url, timeout=60)

        response.raise_for_status()

        download_time = time.perf_counter() - start

        performance_logger.info("GitHub PDF Download Time : %.3f sec", download_time)

        rag_logger.info("HTTP Status : %s", response.status_code)

        rag_logger.info("Downloaded Size : %.2f KB", len(response.content) / 1024)

        # --------------------------------------------------------
        # Create Temporary File
        # --------------------------------------------------------

        parsed_url = urlparse(url)

        document_name = os.path.basename(unquote(parsed_url.path))

        rag_logger.info("-------->document_name----->", document_name)

        temp_file = os.path.join(tempfile.gettempdir(), document_name)

        with open(temp_file, "wb") as file:

            file.write(response.content)

        rag_logger.info("Temporary PDF Created : %s", temp_file)

        # --------------------------------------------------------
        # Prepare Knowledge Base
        # --------------------------------------------------------

        rag_logger.info("Preparing Knowledge Base...")

        start = time.perf_counter()

        owner = st.session_state.user
        department = st.session_state.department
        team = st.session_state.team
        # TODO get from UI dd
        visibility = "Private"

        status = ingest_pdf(temp_file, document_name, "GitHub", url,
                            owner=owner,
                            department=department,
                            team=team,
                            visibility=visibility)

        kb_time = time.perf_counter() - start

        performance_logger.info("Knowledge Base Creation Time : %.3f sec", kb_time)

        rag_logger.info("Knowledge Base Status : %s", status)

        total_time = time.perf_counter() - request_start

        performance_logger.info("Total GitHub Ingestion Time : %.3f sec", total_time)

        rag_logger.info("GitHub PDF Knowledge Base Creation Completed")

        rag_logger.info("=" * 80)

        return status

    except requests.exceptions.RequestException as ex:

        error_logger.exception("Failed to download PDF from GitHub : %s", ex)

        return False

    except Exception as ex:

        error_logger.exception("GitHub PDF Ingestion Failed : %s", ex)

        return False

    finally:

        if temp_file and os.path.exists(temp_file):

            try:

                os.remove(temp_file)

                rag_logger.info("Temporary PDF Deleted : %s", temp_file)

            except Exception as ex:

                error_logger.exception("Failed to delete temporary PDF : %s", ex)

# -----------------------------------
# Ingest Google drive PDF
# -----------------------------------

def ingest_google_drive_pdf(drive_url):

    request_start = time.perf_counter()

    rag_logger.info("=" * 80)
    rag_logger.info("Google Drive PDF Knowledge Base Creation Started")
    rag_logger.info("Drive URL : %s", drive_url)

    temp_file = None

    try:

        # ------------------------------------------
        # Extract File ID
        # ------------------------------------------

        file_id = extract_drive_file_id(drive_url)

        if not file_id:

            rag_logger.error("Invalid Google Drive URL")

            return False

        rag_logger.info("File ID : %s", file_id)

        download_url = get_drive_download_url(file_id)

        rag_logger.info("Download URL Generated")

        # ------------------------------------------
        # Download PDF
        # ------------------------------------------

        start = time.perf_counter()

        response = requests.get(download_url, timeout=120)

        response.raise_for_status()

        performance_logger.info("Google Drive Download Time : %.3f sec", time.perf_counter() - start)

        content_type = response.headers.get("Content-Type", "")

        rag_logger.info("Content-Type : %s", content_type)

        if not response.content.startswith(b"%PDF-"):

            error_logger.error("Downloaded file is not a valid PDF.")

            return False

        # ------------------------------------------
        # Original Filename
        # ------------------------------------------

        document_name = f"{file_id}.pdf"

        disposition = response.headers.get("Content-Disposition", "")

        if "filename=" in disposition:

            document_name = (disposition.split("filename=")[-1].replace('"', ''))

        rag_logger.info("Document Name : %s", document_name)

        # ------------------------------------------
        # Temporary File
        # ------------------------------------------

        temp_file = os.path.join(tempfile.gettempdir(), document_name)

        with open(temp_file, "wb") as f:

            f.write(response.content)

        rag_logger.info("Temporary PDF Created")

        # ------------------------------------------
        # Prepare Knowledge Base
        # ------------------------------------------

        owner = st.session_state.user
        department = st.session_state.department
        team = st.session_state.team
        # TODO get from UI dd
        visibility = "Private"

        status = ingest_pdf(temp_file, document_name, "Google Drive", drive_url, owner=owner,
                            department=department,
                            team=team,
                            visibility=visibility)

        performance_logger.info("Total Google Drive Ingestion : %.3f sec", time.perf_counter() - request_start)

        rag_logger.info("Google Drive Knowledge Base Completed")

        return status

    except Exception:

        error_logger.exception("Google Drive Ingestion Failed")

        return False

    finally:

        if temp_file and os.path.exists(temp_file):

            try:

                os.remove(temp_file)

                rag_logger.info("Temporary PDF Deleted")

            except Exception:

                error_logger.exception("Unable to delete temp file")

def extract_drive_file_id(drive_url):
    """
    Extract Google Drive File ID from a public sharing URL.
    """

    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)"
    ]

    for pattern in patterns:

        match = re.search(pattern, drive_url)

        if match:
            return match.group(1)

    return None

def get_drive_download_url(file_id):
    """
    Build Google Drive direct download URL.
    """

    return (
        f"https://drive.google.com/uc?export=download&id={file_id}"
    )



# -----------------------------------
# Run Individually
# -----------------------------------

if __name__ == "__main__":

    ingest_pdf("pdfs/sample.pdf")