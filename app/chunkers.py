"""
Document Chunking Strategies

Supported Chunkers:
- Recursive Character Splitter
- Character Splitter
- Token Splitter
- Markdown Header Splitter
"""

from app.config import CHUNKING_OPTIONS

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter
)


# ============================================================
# Recursive Character Splitter
# ============================================================

def recursive_chunk(documents):
    """
    Recursive Character Chunking.
    Recommended default chunker.
    """

    config = CHUNKING_OPTIONS["Recursive Character"]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"]
    )

    return splitter.split_documents(documents)


# ============================================================
# Character Splitter
# ============================================================

def character_chunk(documents):
    """
    Character-based Chunking.
    """

    config = CHUNKING_OPTIONS["Character Splitter"]

    splitter = CharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"]
    )

    return splitter.split_documents(documents)


# ============================================================
# Token Splitter
# ============================================================

def token_chunk(documents):
    """
    Token-based Chunking.
    """

    config = CHUNKING_OPTIONS["Token Splitter"]

    splitter = TokenTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"]
    )

    return splitter.split_documents(documents)


# ============================================================
# Markdown Header Splitter
# ============================================================

def markdown_chunk(documents):
    """
    Markdown Header Chunking.

    Useful for structured Markdown documents.
    Falls back to original document if splitting fails.
    """

    headers = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3")
    ]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers
    )

    chunks = []

    for doc in documents:

        try:

            sections = splitter.split_text(
                doc.page_content
            )

            chunks.extend(sections)

        except Exception:
            chunks.append(doc)

    return chunks


# ============================================================
# Generic Chunking Interface
# ============================================================

def chunk_documents(
    documents,
    chunker_name
):
    """
    Main chunking entry point.
    """

    if chunker_name == "Recursive Character":
        return recursive_chunk(documents)

    if chunker_name == "Character Splitter":
        return character_chunk(documents)

    if chunker_name == "Token Splitter":
        return token_chunk(documents)

    if chunker_name == "Markdown Splitter":
        return markdown_chunk(documents)

    raise ValueError(
        f"Unsupported chunker: {chunker_name}"
    )


# ============================================================
# Chunk Statistics
# ============================================================

def get_chunk_statistics(chunks):
    """
    Generate chunking metrics for Streamlit UI.
    """

    chunk_count = len(chunks)

    total_characters = sum(
        len(chunk.page_content)
        for chunk in chunks
    )

    avg_chunk_size = (
        total_characters / chunk_count
        if chunk_count > 0
        else 0
    )

    return {
        "chunks": chunk_count,
        "characters": total_characters,
        "avg_chunk_size": round(
            avg_chunk_size,
            2
        )
    }