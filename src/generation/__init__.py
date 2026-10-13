"""Capítulo 4 (e seção 5.3) — inferência e geração autoregressiva de texto."""
from src.generation.generate import (
    text_to_token_ids,
    token_ids_to_text,
    generate_text_simple,
    generate,
    next_token_distribution,
)

__all__ = [
    "text_to_token_ids",
    "token_ids_to_text",
    "generate_text_simple",
    "generate",
    "next_token_distribution",
]
