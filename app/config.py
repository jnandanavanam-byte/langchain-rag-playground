"""
Configuration Settings
LangChain RAG Playground
"""

# ============================================================
# Application
# ============================================================

APP_NAME = "LangChain RAG Playground"
APP_VERSION = "2.0"

# ============================================================
# Upload Configuration
# ============================================================

MAX_FILES = 3

RECOMMENDED_FILE_SIZE_MB = 5

MAX_FILE_SIZE_MB = 10

SUPPORTED_FILE_TYPES = [
    "pdf",
    "docx",
    "xlsx",
    "csv"
]

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".xlsx",
    ".csv"
}

UPLOAD_DIR = "uploads"

# ============================================================
# Temporary Directories
# ============================================================

TEMP_DIR = "temp"

FAISS_DIR = "temp/faiss"

CHROMA_DIR = "temp/chroma"

LOG_DIR = "logs"

# ============================================================
# Chunking Options
# ============================================================

CHUNKING_OPTIONS = {
    "Recursive Character": {
        "chunk_size": 1000,
        "chunk_overlap": 200
    },

    "Character Splitter": {
        "chunk_size": 1000,
        "chunk_overlap": 100
    },

    "Token Splitter": {
        "chunk_size": 500,
        "chunk_overlap": 50
    },

    "Markdown Splitter": {
        "chunk_size": 1000,
        "chunk_overlap": 200
    }
}

DEFAULT_CHUNKER = "Recursive Character"

# ============================================================
# Embedding Models
# ============================================================

EMBEDDING_MODELS = {
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "BGE Small": "BAAI/bge-small-en-v1.5",
    "E5 Small": "intfloat/e5-small-v2"
}

DEFAULT_EMBEDDING = "MiniLM"

# ============================================================
# Vector Databases
# ============================================================

VECTOR_DATABASES = [
    "FAISS",
    "Chroma"
]

DEFAULT_VECTOR_DB = "FAISS"

# ============================================================
# Ollama Models
# ============================================================

LLM_MODELS = {
    "Llama 3.2": "llama3.2",
    "Mistral": "mistral",
    "Phi-3": "phi3"
}

DEFAULT_LLM = "Llama 3.2"

# ============================================================
# Retrieval Settings
# ============================================================

DEFAULT_TOP_K = 5

MIN_TOP_K = 1

MAX_TOP_K = 20

SEARCH_TYPE = "similarity"

# ============================================================
# UI Layout
# ============================================================

LEFT_PANEL_RATIO = 1

CENTER_PANEL_RATIO = 1

RIGHT_PANEL_RATIO = 2

# ============================================================
# UI Messages
# ============================================================

UPLOAD_HELP = (
    f"Upload up to {MAX_FILES} files. "
    f"Supported formats: PDF, DOCX, XLSX and CSV. "
    f"Recommended size: <= {RECOMMENDED_FILE_SIZE_MB} MB per file."
)

READY_MESSAGE = (
    "Knowledge Base is ready. "
    "You can now ask questions."
)

# ============================================================
# Session Management
# ============================================================

AUTO_CLEANUP_ON_START = True

AUTO_CLEANUP_ON_EXIT = True

# ============================================================
# Hardware Acceleration
# ============================================================

USE_GPU_IF_AVAILABLE = True