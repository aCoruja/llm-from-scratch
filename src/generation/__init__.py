"""Capítulo 4 (e seção 5.3) — inferência e geração autoregressiva de texto."""

# Sprint 4 — Geração básica
from src.generation.generate import (
    text_to_token_ids,
    token_ids_to_text,
    generate_text_simple,
    generate,
    next_token_distribution,
)

# Sprint 5 — Estratégias de decodificação (Entregável 5)
from src.generation.strategies import (
    generate_greedy,
    generate_with_temperature,
    generate_top_k,
    run_comparison,
)

__all__ = [
    # Sprint 4
    "text_to_token_ids",
    "token_ids_to_text",
    "generate_text_simple",
    "generate",
    "next_token_distribution",
    # Sprint 5
    "generate_greedy",
    "generate_with_temperature",
    "generate_top_k",
    "run_comparison",
]
