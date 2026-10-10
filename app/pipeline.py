"""
RAG Pipeline

Orchestrates:

Documents
    ↓
Chunking
    ↓
Embeddings
    ↓
Vector Store
    ↓
Retriever
    ↓
Prompt
    ↓
LLM
"""

from dataclasses import dataclass
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate

import time

from app.config import (
    DEFAULT_CHUNKER,
    DEFAULT_EMBEDDING,
    DEFAULT_VECTOR_DB,
    DEFAULT_LLM,
    DEFAULT_TOP_K
)

from app.chunkers import (
    chunk_documents,
    get_chunk_statistics
)

from app.embeddings import (
    get_embedding_model,
    get_embedding_statistics
)

from app.vectordb import (
    create_vector_store,
    get_retriever,
    get_vectorstore_statistics
)

from app.llms import (
    get_llm,
    get_llm_statistics
)


# ============================================================
# Pipeline Configuration
# ============================================================

@dataclass
class PipelineConfig:

    chunker: str = DEFAULT_CHUNKER

    embedding: str = DEFAULT_EMBEDDING

    vector_db: str = DEFAULT_VECTOR_DB

    llm: str = DEFAULT_LLM

    top_k: int = DEFAULT_TOP_K

    persist_directory: Optional[str] = None


# ============================================================
# RAG Pipeline
# ============================================================

class RagPipeline:

    def __init__(
        self,
        config: PipelineConfig
    ):

        self.config = config

        self.chunks = []

        self.embeddings = None

        self.vector_store = None

        self.retriever = None

        self.llm = None

        self.pipeline_stats = {}

        self.ready = False

    # ========================================================
    # Build Pipeline
    # ========================================================

    def build(
        self,
        documents
    ):
        """
        Build complete pipeline.
        """

        # ----------------------------------------------------
        # Chunking
        # ----------------------------------------------------

        self.chunks = chunk_documents(
            documents=documents,
            chunker_name=self.config.chunker
        )

        self.pipeline_stats["chunking"] = (
            get_chunk_statistics(self.chunks)
        )

        # ----------------------------------------------------
        # Embeddings
        # ----------------------------------------------------

        self.embeddings = get_embedding_model(
            self.config.embedding
        )

        self.pipeline_stats["embedding"] = (
            get_embedding_statistics(
                self.chunks,
                self.config.embedding
            )
        )

        # ----------------------------------------------------
        # Vector Store
        # ----------------------------------------------------

        self.vector_store = create_vector_store(
            documents=self.chunks,
            embeddings=self.embeddings,
            vector_db_name=self.config.vector_db,
            persist_directory=self.config.persist_directory
        )

        self.pipeline_stats["vectorstore"] = (
            get_vectorstore_statistics(
                self.config.vector_db,
                len(self.chunks)
            )
        )

        # ----------------------------------------------------
        # Retriever
        # ----------------------------------------------------

        self.retriever = get_retriever(
            self.vector_store,
            self.config.top_k
        )

        # ----------------------------------------------------
        # LLM
        # ----------------------------------------------------

        self.llm = get_llm(
            self.config.llm
        )

        self.pipeline_stats["llm"] = (
            get_llm_statistics(
                self.config.llm
            )
        )

        self.ready = True

        return self.pipeline_stats

    # ========================================================
    # Status
    # ========================================================

    def is_ready(self):
        """
        Indicates whether the pipeline is ready.
        """

        return self.ready

    # ========================================================
    # Context Builder
    # ========================================================

    def _build_context(
        self,
        documents
    ):
        """
        Convert retrieved documents into context.
        """

        return "\n\n".join(
            doc.page_content
            for doc in documents
        )

    # ========================================================
    # Source Extraction
    # ========================================================

    def _extract_sources(
        self,
        documents
    ):
        """
        Extract document source metadata.
        """

        sources = []

        for doc in documents:

            metadata = getattr(
                doc,
                "metadata",
                {}
            )

            source = metadata.get(
                "source",
                "Unknown"
            )

            page = metadata.get("page")

            sheet = metadata.get("sheet")

            if page is not None:

                source = (
                    f"{source} "
                    f"(Page {page})"
                )

            elif sheet is not None:

                source = (
                    f"{source} "
                    f"(Sheet {sheet})"
                )

            sources.append(source)

        return sorted(
            set(sources)
        )

    # ========================================================
    # Ask Question
    # ========================================================

    def ask(
        self,
        question
    ):
        """
        Execute RAG query.
        """

        if not self.ready:

            raise RuntimeError(
                "Pipeline has not been built yet."
            )

        # ----------------------------------------------------
        # Retrieve Content
        # ----------------------------------------------------

        documents = self.retriever.invoke(
            question
        )

        retrieved_chunks = len(
            documents
        )

        context = self._build_context(
            documents
        )

        # ----------------------------------------------------
        # Prompt
        # ----------------------------------------------------

        prompt = ChatPromptTemplate.from_template(
            """
You are a helpful AI assistant.

Answer ONLY from the provided context.

If the answer cannot be found in the context, respond exactly with:

"I could not find the answer in the uploaded documents."

Context:
{context}

Question:
{question}

Answer:
"""
        )

        chain = prompt | self.llm

        start_time = time.perf_counter()

        response = chain.invoke(
            {
                "context": context,
                "question": question
            }
        )

        response_time = round(
            time.perf_counter() - start_time,
            2
        )

        answer = (
            response.content
            if hasattr(response, "content")
            else str(response)
        )

        sources = self._extract_sources(
            documents
        )

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        return {
            "question": question,
            "answer": answer,
            "response_time": response_time,
            "answer_length": len(answer),
            "source_count": len(sources),
            "context": context,
            "sources": sources,
            "retrieved_chunks": retrieved_chunks,
            "pipeline": {
                "chunker": self.config.chunker,
                "embedding": self.config.embedding,
                "vector_db": self.config.vector_db,
                "llm": self.config.llm,
                "top_k": self.config.top_k
            }
        }

    # ========================================================
    # Statistics
    # ========================================================

    def get_statistics(self):

        return self.pipeline_stats

    # ========================================================
    # Configuration
    # ========================================================

    def get_configuration(self):

        return {
            "chunker": self.config.chunker,
            "embedding": self.config.embedding,
            "vector_db": self.config.vector_db,
            "llm": self.config.llm,
            "top_k": self.config.top_k,
            "persist_directory": self.config.persist_directory
        }

    # ========================================================
    # String Representation
    # ========================================================

    def __repr__(self):

        return (
            f"RagPipeline("
            f"chunker='{self.config.chunker}', "
            f"embedding='{self.config.embedding}', "
            f"vector_db='{self.config.vector_db}', "
            f"llm='{self.config.llm}', "
            f"top_k={self.config.top_k})"
        )