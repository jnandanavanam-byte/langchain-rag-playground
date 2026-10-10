"""
LLM Factory

Supported LLMs:
- Llama 3.2
- Mistral
- Phi-3
"""

from langchain_ollama import OllamaLLM

from app.config import (
    LLM_MODELS,
    DEFAULT_LLM
)


# ============================================================
# LLM Factory
# ============================================================

def get_llm(
    llm_name=DEFAULT_LLM
):
    """
    Return the selected Ollama LLM.
    """

    if llm_name not in LLM_MODELS:

        raise ValueError(
            f"Unsupported LLM: {llm_name}"
        )

    model_name = LLM_MODELS[
        llm_name
    ]

    return OllamaLLM(
        model=model_name,
        temperature=0
    )


# ============================================================
# LLM Information
# ============================================================

def get_llm_info(
    llm_name
):
    """
    Return LLM metadata.
    """

    if llm_name not in LLM_MODELS:

        raise ValueError(
            f"Unsupported LLM: {llm_name}"
        )

    return {
        "display_name": llm_name,
        "model_name": LLM_MODELS[
            llm_name
        ]
    }


# ============================================================
# LLM Statistics
# ============================================================

def get_llm_statistics(
    llm_name
):
    """
    Generate LLM metrics for UI.
    """

    if llm_name not in LLM_MODELS:

        raise ValueError(
            f"Unsupported LLM: {llm_name}"
        )

    return {
        "llm_name": llm_name,
        "provider": "Ollama",
        "model": LLM_MODELS[
            llm_name
        ]
    }


# ============================================================
# Direct Response
# ============================================================

def generate_response(
    llm,
    prompt
):
    """
    Generate response from LLM.
    """

    response = llm.invoke(
        prompt
    )

    return (
        response.content
        if hasattr(response, "content")
        else str(response)
    )