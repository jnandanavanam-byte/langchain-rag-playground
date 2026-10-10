"""
Embedding Model Factory

Supported Models:
- MiniLM
- BGE Small
- E5 Small
"""

import torch

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import (
    EMBEDDING_MODELS,
    DEFAULT_EMBEDDING,
    USE_GPU_IF_AVAILABLE
)


# ============================================================
# Embedding Factory
# ============================================================

def get_embedding_model(
    embedding_name=DEFAULT_EMBEDDING
):
    """
    Return the selected embedding model.
    """

    if embedding_name not in EMBEDDING_MODELS:

        raise ValueError(
            f"Unsupported embedding model: {embedding_name}"
        )

    model_name = EMBEDDING_MODELS[
        embedding_name
    ]

    device = "cpu"

    if (
        USE_GPU_IF_AVAILABLE
        and torch.cuda.is_available()
    ):
        device = "cuda"
        print(
            f"Embedding Device: {device} | "
            f"GPU: {torch.cuda.get_device_name(0)}"
        )
    else:
        print("Embedding Device: cpu")

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            "device": device
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )


# ============================================================
# Embedding Information
# ============================================================

def get_embedding_info(
    embedding_name
):
    """
    Return embedding model metadata.
    """

    if embedding_name not in EMBEDDING_MODELS:

        raise ValueError(
            f"Unsupported embedding model: {embedding_name}"
        )

    return {
        "display_name": embedding_name,
        "model_name": EMBEDDING_MODELS[
            embedding_name
        ]
    }


# ============================================================
# Embedding Statistics
# ============================================================

def get_embedding_statistics(
    chunks,
    embedding_name
):
    """
    Generate embedding metrics for UI.
    """

    vector_count = len(chunks)

    device = (
        "CUDA"
        if (
            USE_GPU_IF_AVAILABLE
            and torch.cuda.is_available()
        )
        else "CPU"
    )

    return {
        "embedding_model": embedding_name,
        "vectors_created": vector_count,
        "device": device
    }
