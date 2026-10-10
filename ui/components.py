"""
UI Components
"""

import streamlit as st

from app.config import (
    UPLOAD_HELP,
    MAX_FILES,
    SUPPORTED_FILE_TYPES,
    CHUNKING_OPTIONS,
    EMBEDDING_MODELS,
    VECTOR_DATABASES,
    LLM_MODELS
)


# ============================================================
# Upload Section
# ============================================================

def render_upload_section():

    st.subheader("📄 Document Upload")

    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=SUPPORTED_FILE_TYPES,
        accept_multiple_files=True,
        help=UPLOAD_HELP
    )

    if uploaded_files and len(uploaded_files) > MAX_FILES:

        st.error(
            f"Maximum {MAX_FILES} files allowed."
        )

        return []

    return uploaded_files


# ============================================================
# Chunker Selection
# ============================================================

def render_chunking_selection():

    return st.selectbox(
        "Chunking Strategy",
        list(CHUNKING_OPTIONS.keys())
    )


# ============================================================
# Embedding Selection
# ============================================================

def render_embedding_selection():

    return st.selectbox(
        "Embedding Model",
        list(EMBEDDING_MODELS.keys())
    )


# ============================================================
# Vector Store Selection
# ============================================================

def render_vectordb_selection():

    return st.selectbox(
        "Vector Store",
        VECTOR_DATABASES
    )


# ============================================================
# LLM Selection
# ============================================================

def render_llm_selection():

    return st.selectbox(
        "LLM",
        list(LLM_MODELS.keys())
    )


# ============================================================
# Progress
# ============================================================

def render_progress(
    percentage,
    status_message
):

    st.progress(percentage)

    st.caption(status_message)


# ============================================================
# Metric Card
# ============================================================

def render_metric_card(
    label,
    value
):

    st.metric(
        label=label,
        value=value
    )


# ============================================================
# Statistics Panel
# ============================================================

def render_statistics_panel(
    statistics
):

    if not statistics:
        return

    st.subheader("📊 Statistics")

    for key, value in statistics.items():

        st.metric(
            label=key.replace(
                "_",
                " "
            ).title(),
            value=value
        )


# ============================================================
# Pipeline Status
# ============================================================

def render_status(
    message,
    status="success"
):

    if status == "success":
        st.success(message)

    elif status == "warning":
        st.warning(message)

    elif status == "error":
        st.error(message)

    else:
        st.info(message)


# ============================================================
# Sources
# ============================================================

def render_sources(
    sources
):

    st.markdown("### Sources")

    if not sources:

        st.info(
            "No sources available."
        )

        return

    for source in sources:

        st.write(
            f"• {source}"
        )


# ============================================================
# Response Panel
# ============================================================

def render_response_panel(
    response
):

    if not response:

        st.info(
            "No response available."
        )

        return

    st.markdown(
        response["answer"]
    )

    render_sources(
        response["sources"]
    )


# ============================================================
# Ready Banner
# ============================================================

def render_ready_banner():

    st.success(
        "✅ Knowledge Base Ready"
    )


# ============================================================
# Cleanup Button
# ============================================================

def render_cleanup_button():

    return st.button(
        "🗑 Clear Session",
        use_container_width=True
    )