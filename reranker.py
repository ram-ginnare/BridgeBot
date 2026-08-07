import time

from sentence_transformers import CrossEncoder

from utils.logger import (rag_logger, performance_logger, error_logger, log_performance)

# --------------------------------------------------------
# Configuration
# --------------------------------------------------------

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --------------------------------------------------------
# Load Cross Encoder
# --------------------------------------------------------

rag_logger.info("Loading CrossEncoder Model : %s", MODEL_NAME)

reranker = CrossEncoder(MODEL_NAME)

rag_logger.info("CrossEncoder Loaded Successfully")

# --------------------------------------------------------
# Rerank
# --------------------------------------------------------

def rerank(question, documents, top_k=3):
    """
    Rerank retrieved documents using a CrossEncoder.

    Args:
        question: User question
        documents: Retrieved documents
        top_k: Number of chunks to keep

    Returns:
        reranked_docs,
        reranked_scores
    """

    request_start = time.perf_counter()

    rag_logger.info("-" * 60)
    rag_logger.info("CrossEncoder Reranking Started")
    rag_logger.info("Question : %s", question)
    rag_logger.info("Input Chunks : %d", len(documents))
    rag_logger.info("Top K : %d", top_k)

    try:

        if len(documents) == 0:

            rag_logger.warning("No documents received for reranking.")

            return [], []

        # --------------------------------------------------------
        # Build Question/Chunk Pairs
        # --------------------------------------------------------

        start = time.perf_counter()

        pairs = [
            (question, doc.page_content)
            for doc in documents
        ]

        performance_logger.info("Prepared %d question-document pairs in %.3f sec", len(pairs), time.perf_counter() - start)

        # --------------------------------------------------------
        # Predict Scores
        # --------------------------------------------------------

        start = time.perf_counter()

        scores = reranker.predict(pairs)

        prediction_time = time.perf_counter() - start

        performance_logger.info("CrossEncoder Prediction Time : %.3f sec", prediction_time)

        # --------------------------------------------------------
        # Ranking
        # --------------------------------------------------------

        ranked = sorted(
            zip(scores, documents),
            key=lambda x: x[0],
            reverse=True
        )

        reranked_docs = [
            doc
            for score, doc in ranked[:top_k]
        ]

        reranked_scores = [
            float(score)
            for score, doc in ranked[:top_k]
        ]

        # --------------------------------------------------------
        # Statistics
        # --------------------------------------------------------

        rag_logger.info("Selected %d chunks", len(reranked_docs))

        rag_logger.info("Highest Score : %.4f", reranked_scores[0])

        rag_logger.info("Average Score : %.4f", sum(reranked_scores) / len(reranked_scores))

        rag_logger.info("Lowest Score : %.4f", reranked_scores[-1])

        total_time = time.perf_counter() - request_start

        performance_logger.info("Total Reranking Time : %.3f sec", total_time)

        rag_logger.info("CrossEncoder Reranking Completed")
        rag_logger.info("-" * 60)

        return reranked_docs, reranked_scores

    except Exception as e:

        error_logger.exception(f"CrossEncoder Reranking Failed : {e}")

        return [], []