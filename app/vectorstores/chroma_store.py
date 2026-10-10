"""
Chroma Vector Store

Provides:
- Chroma database creation
- Persistence support
- Loading existing databases
- Retriever generation
- Statistics for UI reporting
"""

from langchain_chroma import Chroma


# ============================================================
# Create Chroma Vector Store
# ============================================================

def create_chroma_store(
    documents,
    embeddings,
    persist_directory=None
):
    """
    Create and populate a Chroma vector store.

    Parameters
    ----------
    documents : list
        LangChain Document objects

    embeddings : Embeddings
        Embedding model instance

    persist_directory : str, optional
        Directory used for persistence

    Returns
    -------
    Chroma
        Initialized vector store
    """

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    return vector_store


# ============================================================
# Load Existing Chroma Store
# ============================================================

def load_chroma_store(
    persist_directory,
    embeddings
):
    """
    Load an existing Chroma vector store.
    """

    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )


# ============================================================
# Retriever Factory
# ============================================================

def get_chroma_retriever(
    vector_store,
    top_k=5,
    search_type="similarity"
):
    """
    Create a retriever from the Chroma store.
    """

    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={
            "k": top_k
        }
    )


# ============================================================
# Search Utility
# ============================================================

def similarity_search(
    vector_store,
    query,
    top_k=5
):
    """
    Execute similarity search directly.

    Parameters
    ----------
    vector_store : Chroma

    query : str

    top_k : int

    Returns
    -------
    list
        Retrieved documents
    """

    return vector_store.similarity_search(
        query=query,
        k=top_k
    )


# ============================================================
# Database Information
# ============================================================

def get_chroma_info(
    persist_directory=None
):
    """
    Return metadata describing the Chroma store.
    """

    return {
        "vector_store": "Chroma",
        "persisted": persist_directory is not None,
        "persist_directory": persist_directory
    }


# ============================================================
# Statistics
# ============================================================

def get_chroma_statistics(
    document_count=0,
    chunk_count=0
):
    """
    Statistics used by Streamlit UI.
    """

    return {
        "vector_store": "Chroma",
        "documents": document_count,
        "chunks": chunk_count
    }