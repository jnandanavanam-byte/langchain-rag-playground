"""
Custom FAISS Vector Store
"""

import uuid

import faiss
import numpy as np


# ============================================================
# FAISS Store
# ============================================================

class FaissStore:

    def __init__(
        self,
        embedding_model
    ):

        self.embedding_model = embedding_model

        self.index = None

        self.documents = []

        self.ids = []

    # ========================================================
    # Add Documents
    # ========================================================

    def add_documents(
        self,
        documents
    ):

        self.documents.extend(
            documents
        )

        vectors = (
            self.embedding_model
            .embed_documents(
                [
                    doc.page_content
                    for doc in documents
                ]
            )
        )

        vectors = np.array(
            vectors,
            dtype="float32"
        )

        dimension = vectors.shape[1]

        if self.index is None:

            self.index = (
                faiss.IndexFlatL2(
                    dimension
                )
            )

        self.index.add(
            vectors
        )

        self.ids.extend(
            [
                str(uuid.uuid4())
                for _ in documents
            ]
        )

    # ========================================================
    # Similarity Search
    # ========================================================

    def similarity_search(
        self,
        query,
        k=5
    ):

        query_vector = (
            self.embedding_model
            .embed_query(
                query
            )
        )

        query_vector = np.array(
            [query_vector],
            dtype="float32"
        )

        distances, indices = (
            self.index.search(
                query_vector,
                k
            )
        )

        results = []

        for idx in indices[0]:

            if idx < len(
                self.documents
            ):

                results.append(
                    self.documents[idx]
                )

        return results

    # ========================================================
    # Retriever
    # =======================================================

    def as_retriever(
        self,
        search_type="similarity",
        search_kwargs=None
    ):

        k = 5

        if (
            search_kwargs
            and "k" in search_kwargs
        ):
            k = search_kwargs["k"]

        return FaissRetriever(
            self,
            k
        )


# ============================================================
# Retriever
# ============================================================

class FaissRetriever:

    def __init__(
        self,
        faiss_store,
        k=5
    ):

        self.faiss_store = (
            faiss_store
        )

        self.k = k

    def invoke(
        self,
        query
    ):

        return (
            self.faiss_store
            .similarity_search(
                query=query,
                k=self.k
            )
        )


# ============================================================
# Factory
# ============================================================

def create_faiss_store(
    documents,
    embeddings
):

    store = FaissStore(
        embedding_model=embeddings
    )

    store.add_documents(
        documents
    )

    return store


# ============================================================
# Retriever Factory
# ============================================================

def get_faiss_retriever(
    vector_store,
    top_k=5
):

    return FaissRetriever(
        vector_store,
        top_k
    )


# ============================================================
# Statistics
# ============================================================

def get_faiss_statistics(
    document_count=0,
    chunk_count=0
):

    return {
        "vector_store": "FAISS",
        "documents": document_count,
        "chunks": chunk_count
    }