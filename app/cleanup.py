"""
Cleanup Utilities

Responsible for:
- Upload Cleanup
- Temporary Folder Cleanup
- Vector Store Cleanup
- Session Cleanup
"""

import shutil
from pathlib import Path

from app.config import (
    UPLOAD_DIR,
    TEMP_DIR,
    FAISS_DIR,
    CHROMA_DIR,
    LOG_DIR
)


# ============================================================
# Generic Directory Cleanup
# ============================================================

def delete_directory(directory_path):
    """
    Delete and recreate a directory.
    """

    directory = Path(directory_path)

    if directory.exists():
        shutil.rmtree(directory)

    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# Upload Cleanup
# ============================================================

def cleanup_uploads():
    """
    Remove uploaded files.
    """

    delete_directory(UPLOAD_DIR)


# ============================================================
# Vector Store Cleanup
# ============================================================

def cleanup_faiss():
    """
    Remove FAISS vector store.
    """

    delete_directory(FAISS_DIR)


def cleanup_chroma():
    """
    Remove Chroma vector store.
    """

    delete_directory(CHROMA_DIR)


# ============================================================
# Temporary Cleanup
# ============================================================

def cleanup_temp():
    """
    Remove temporary content.
    """

    delete_directory(TEMP_DIR)


# ============================================================
# Log Cleanup
# ============================================================

def cleanup_logs():
    """
    Remove log files.
    """

    delete_directory(LOG_DIR)


# ============================================================
# Full Session Cleanup
# ============================================================

def cleanup_session():
    """
    Cleanup all runtime artifacts.
    """

    cleanup_uploads()
    cleanup_faiss()
    cleanup_chroma()
    cleanup_temp()

    print(
        "Session cleanup completed."
    )


# ============================================================
# Startup Cleanup
# ============================================================

def startup_cleanup():
    """
    Ensure clean startup state.
    """

    cleanup_uploads()
    cleanup_faiss()
    cleanup_chroma()

    print(
        "Startup cleanup completed."
    )