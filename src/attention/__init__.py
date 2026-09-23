"""Capítulo 3 — Mecanismos de Atenção."""
from src.attention.scaled_dot_product import scaled_dot_product_attention
from src.attention.self_attention import SelfAttention
from src.attention.causal_attention import CausalAttention
from src.attention.multi_head_attention import MultiHeadAttention

__all__ = [
    "scaled_dot_product_attention",
    "SelfAttention",
    "CausalAttention",
    "MultiHeadAttention",
]
