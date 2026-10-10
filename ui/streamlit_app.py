"""
LangChain RAG Playground
RAG Benchmarking & Comparison Platform
"""

import os
import tempfile

import pandas as pd
import streamlit as st
import torch

from app.config import APP_NAME

from app.loaders import load_documents

from app.pipeline import PipelineConfig

from app.rag import RagComparison

from app.cleanup import (
    startup_cleanup,
    cleanup_session
)


# ============================================================
# Session State
# ============================================================

def initialize_session():

    if "comparison" not in st.session_state:
        st.session_state.comparison = None

    if "ready" not in st.session_state:
        st.session_state.ready = False

    if "document_stats" not in st.session_state:
        st.session_state.document_stats = None

    if "build_stats" not in st.session_state:
        st.session_state.build_stats = None

    if "responses" not in st.session_state:
        st.session_state.responses = None

    if "last_question" not in st.session_state:
        st.session_state.last_question = ""


# ============================================================
# GPU Status
# ============================================================

def render_gpu_status():

    if torch.cuda.is_available():

        st.success(
            f"🚀 CUDA Enabled | "
            f"{torch.cuda.get_device_name(0)}"
        )

    else:

        st.warning(
            "⚠ Running on CPU"
        )


# ============================================================
# Upload Handling
# ============================================================

def save_uploaded_files(uploaded_files):

    saved_files = []

    for uploaded_file in uploaded_files:

        extension = os.path.splitext(
            uploaded_file.name
        )[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:

            temp_file.write(
                uploaded_file.getbuffer()
            )

            saved_files.append(
                temp_file.name
            )

    return saved_files


# ============================================================
# Build Environment
# ============================================================

def build_environment(
    uploaded_files,
    pipeline1_config,
    pipeline2_config
):

    file_paths = save_uploaded_files(
        uploaded_files
    )

    documents, document_stats = (
        load_documents(file_paths)
    )

    comparison = RagComparison(
        pipeline1_config=pipeline1_config,
        pipeline2_config=pipeline2_config
    )

    build_stats = comparison.build(
        documents
    )

    st.session_state.comparison = comparison

    st.session_state.document_stats = (
        document_stats
    )

    st.session_state.build_stats = (
        build_stats
    )

    st.session_state.ready = True


# ============================================================
# Question Handling
# ============================================================

def ask_question(question):

    st.session_state.last_question = (
        question
    )

    responses = (
        st.session_state
        .comparison
        .ask(question)
    )

    st.session_state.responses = (
        responses
    )



# ============================================================
# Response Rendering
# ============================================================

def render_response_section():

    responses = st.session_state.responses

    if not responses:
        return

    st.header("❓ Question")

    st.info(
        st.session_state.last_question
    )

    render_response_benchmark()

    left, right = st.columns(2)

    # ========================================================
    # Pipeline 1 Response
    # ========================================================

    with left:

        st.subheader(
            "Pipeline 1 Response"
        )

        st.write(
            responses["pipeline1"]["answer"]
        )

        st.markdown(
            "### Sources"
        )

        for source in responses["pipeline1"]["sources"]:

            st.write(
                f"• {source}"
            )

    # ========================================================
    # Pipeline 2 Response
    # ========================================================

    with right:

        st.subheader(
            "Pipeline 2 Response"
        )

        st.write(
            responses["pipeline2"]["answer"]
        )

        st.markdown(
            "### Sources"
        )

        for source in responses["pipeline2"]["sources"]:

            st.write(
                f"• {source}"
            )


# ============================================================
# Main Application
# ============================================================

def render_build_dashboard():

    if not st.session_state.build_stats:
        return

    stats = st.session_state.build_stats

    st.header("📊 Comparison Dashboard")

    comparison_df = pd.DataFrame(
        {
            "Metric": [
                "Device",
                "Chunks",
                "Vectors",
                "Average Chunk Size"
            ],
            "Pipeline 1": [
                stats["pipeline1"]["embedding"]["device"],
                stats["pipeline1"]["chunking"]["chunks"],
                stats["pipeline1"]["embedding"]["vectors_created"],
                stats["pipeline1"]["chunking"]["avg_chunk_size"]
            ],
            "Pipeline 2": [
                stats["pipeline2"]["embedding"]["device"],
                stats["pipeline2"]["chunking"]["chunks"],
                stats["pipeline2"]["embedding"]["vectors_created"],
                stats["pipeline2"]["chunking"]["avg_chunk_size"]
            ]
        }
    )

    st.dataframe(
        comparison_df,
        use_container_width=True
    )



def main():

    st.set_page_config(
        page_title=APP_NAME,
        page_icon="🤖",
        layout="wide"
    )

    initialize_session()

    st.title(
        "🤖 LangChain RAG Playground"
    )

    render_gpu_status()

    st.markdown(
        """
Compare different Chunking Strategies,
Embedding Models, Vector Databases
and LLMs using the same document set.
"""
    )

    st.divider()

    # ========================================================
    # Upload Section
    # ========================================================

    st.header("📄 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, XLSX or CSV files",
        accept_multiple_files=True,
        type=[
            "pdf",
            "docx",
            "xlsx",
            "csv"
        ]
    )

    st.divider()

    # ========================================================
    # Pipeline Configuration
    # ========================================================

    left, right = st.columns(2)

    with left:

        st.subheader("Pipeline 1")

        chunker1 = st.selectbox(
            "Chunking Strategy",
            [
                "Recursive Character",
                "Character Splitter",
                "Token Splitter",
                "Markdown Splitter"
            ],
            key="chunker1"
        )

        embedding1 = st.selectbox(
            "Embedding Model",
            [
                "MiniLM",
                "BGE Small",
                "E5 Small"
            ],
            key="embedding1"
        )

        vector_db1 = st.selectbox(
            "Vector Store",
            [
                "FAISS",
                "Chroma"
            ],
            key="vector_db1"
        )

        llm1 = st.selectbox(
            "LLM",
            [
                "Llama 3.2",
                "Mistral",
                "Phi-3"
            ],
            key="llm1"
        )

    with right:

        st.subheader("Pipeline 2")

        chunker2 = st.selectbox(
            "Chunking Strategy",
            [
                "Recursive Character",
                "Character Splitter",
                "Token Splitter",
                "Markdown Splitter"
            ],
            key="chunker2"
        )

        embedding2 = st.selectbox(
            "Embedding Model",
            [
                "MiniLM",
                "BGE Small",
                "E5 Small"
            ],
            key="embedding2"
        )

        vector_db2 = st.selectbox(
            "Vector Store",
            [
                "FAISS",
                "Chroma"
            ],
            key="vector_db2"
        )

        llm2 = st.selectbox(
            "LLM",
            [
                "Llama 3.2",
                "Mistral",
                "Phi-3"
            ],
            key="llm2"
        )

    st.divider()

    # ========================================================
    # Build Pipelines
    # ========================================================

    if st.button(
        "🚀 Build Comparison Environment",
        use_container_width=True
    ):

        if not uploaded_files:

            st.warning(
                "Please upload documents first."
            )

        else:

            pipeline1_config = (
                PipelineConfig(
                    chunker=chunker1,
                    embedding=embedding1,
                    vector_db=vector_db1,
                    llm=llm1
                )
            )

            pipeline2_config = (
                PipelineConfig(
                    chunker=chunker2,
                    embedding=embedding2,
                    vector_db=vector_db2,
                    llm=llm2
                )
            )

            with st.spinner(
                "Building Pipelines..."
            ):

                build_environment(
                    uploaded_files,
                    pipeline1_config,
                    pipeline2_config
                )

            st.success(
                "✅ Both pipelines built successfully."
            )

    # ========================================================
    # Dashboard
    # ========================================================

    if st.session_state.ready:

        render_build_dashboard()

        st.divider()

        question = st.chat_input(
            "Ask a question..."
        )

        if question:

            ask_question(
                question
            )

        render_response_section()
# ============================================================
# Response Benchmark Dashboard
# ============================================================

def render_response_benchmark():

    responses = st.session_state.responses

    if not responses:
        return

    benchmark_df = pd.DataFrame(
        {
            "Metric": [
                "Response Time (sec)",
                "Retrieved Chunks",
                "Source Count",
                "Answer Length"
            ],
            "Pipeline 1": [
                responses["pipeline1"]["response_time"],
                responses["pipeline1"]["retrieved_chunks"],
                responses["pipeline1"]["source_count"],
                responses["pipeline1"]["answer_length"]
            ],
            "Pipeline 2": [
                responses["pipeline2"]["response_time"],
                responses["pipeline2"]["retrieved_chunks"],
                responses["pipeline2"]["source_count"],
                responses["pipeline2"]["answer_length"]
            ]
        }
    )

    st.subheader(
        "⚡ Response Benchmark"
    )

    st.dataframe(
        benchmark_df,
        use_container_width=True
    )

    st.divider()

    if st.button(
        "🗑 Clear Session",
        use_container_width=True
    ):

        cleanup_session()

        st.session_state.clear()

        st.rerun()

   
# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    startup_cleanup()

    main()