"""
Vector Database Factory

Supported Vector Stores:
- FAISS
- Chroma
"""

from app.config import (
    DEFAULT_VECTOR_DB,
    DEFAULT_TOP_K
)

from app.vectorstores.faiss_store import (
    create_faiss_store,
    get_faiss_retriever
)

from app.vectorstores.chroma_store import (
    create_chroma_store,
    get_chroma_retriever
)


# ============================================================
# Vector Store Factory
# ============================================================

def create_vector_store(
    documents,
    embeddings,
    vector_db_name=DEFAULT_VECTOR_DB,
    persist_directory=None
):
    """
    Create selected vector store.
    """

    if vector_db_name == "FAISS":

        return create_faiss_store(
            documents=documents,
            embeddings=embeddings
        )

    if vector_db_name == "Chroma":

        return create_chroma_store(
            documents=documents,
            embeddings=embeddings,
            persist_directory=persist_directory
        )

    raise ValueError(
        f"Unsupported Vector Store: {vector_db_name}"
    )


# ============================================================
# Retriever Factory
# ============================================================

def get_retriever(
    vector_store,
    top_k=DEFAULT_TOP_K
):
    """
    Create retriever from vector store.
    """

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": top_k
        }
    )

    return retriever


# ============================================================
# Vector Store Information
# ============================================================

def get_vectorstore_info(
    vector_db_name
):
    """
    Return vector store metadata.
    """

    return {
        "vector_store": vector_db_name
    }


# ============================================================
# Statistics
# ============================================================

def get_vectorstore_statistics(
    vector_db_name,
    chunk_count
):
    """
    Statistics used in Streamlit UI.
    """

    return {
        "vector_store": vector_db_name,
        "indexed_chunks": chunk_count
    }