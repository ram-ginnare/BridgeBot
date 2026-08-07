"""
Production Grade Logging Utility
Author : Bharat Soni RAG Project
"""

import logging
import os
from logging.handlers import RotatingFileHandler

# --------------------------------------------------------
# Configuration
# --------------------------------------------------------

LOG_DIR = "logs"

os.makedirs(LOG_DIR, exist_ok=True)

LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)-8s | "
    "%(name)-20s | "
    "%(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# --------------------------------------------------------
# Internal Logger Builder
# --------------------------------------------------------

def _create_logger(
        logger_name,
        file_name,
        level=logging.INFO,
        console=True
):
    logger = logging.getLogger(logger_name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        LOG_FORMAT,
        DATE_FORMAT
    )

    # ----------------------------------------------------
    # File Handler
    # ----------------------------------------------------

    file_handler = RotatingFileHandler(
        filename=os.path.join(LOG_DIR, file_name),
        maxBytes=10 * 1024 * 1024,      # 10 MB
        backupCount=10,
        encoding="utf-8"
    )

    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    # ----------------------------------------------------
    # Console Handler
    # ----------------------------------------------------

    if console:

        console_handler = logging.StreamHandler()

        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    logger.propagate = False

    return logger


# --------------------------------------------------------
# Application Logger
# --------------------------------------------------------

rag_logger = _create_logger(
    logger_name="RAG",
    file_name="rag.log",
    level=logging.INFO
)

# --------------------------------------------------------
# Performance Logger
# --------------------------------------------------------

performance_logger = _create_logger(
    logger_name="PERFORMANCE",
    file_name="performance.log",
    level=logging.INFO
)

# --------------------------------------------------------
# Error Logger
# --------------------------------------------------------

error_logger = _create_logger(
    logger_name="ERROR",
    file_name="error.log",
    level=logging.ERROR
)


# --------------------------------------------------------
# Compatibility Function
# (For existing code)
# --------------------------------------------------------

def get_logger(name):
    """
    Existing code compatibility.

    This returns the application logger.

    Old code:
        logger = get_logger(__name__)

    will continue to work.
    """

    return rag_logger


# --------------------------------------------------------
# Performance Logging Helper
# --------------------------------------------------------

def log_performance(operation, elapsed_time):

    performance_logger.info(
        "%-35s : %.3f sec",
        operation,
        elapsed_time
    )


# --------------------------------------------------------
# Error Logging Helper
# --------------------------------------------------------

def log_exception(message):

    error_logger.exception(message)


# --------------------------------------------------------
# Banner Helpers
# --------------------------------------------------------

def log_banner(title):

    rag_logger.info("=" * 90)
    rag_logger.info(title)
    rag_logger.info("=" * 90)


def log_separator():

    rag_logger.info("-" * 90)