"""Prompt helpers for generation."""

from __future__ import annotations

from src.models import RetrievedChunk


def build_generation_prompt(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
) -> str:
    """Build the current answer-generation prompt.

    Args:
        question: User question to answer.
        retrieved_chunks: Ranked retrieved chunks to include as context.

    Returns:
        One formatted prompt string for answer generation.
    """
    context_sections = [
        f"Context {index}:\n{chunk.text}"
        for index, chunk in enumerate(retrieved_chunks, start=1)
    ]
    context_text = "\n\n".join(context_sections)
    return (
        "Based on the following context(s), answer the question. "
        'If the answer cannot be found in the contexts, say "I don\'t have enough '
        'information to answer this question."\n\n'
        f"{context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer:\n"
    )
