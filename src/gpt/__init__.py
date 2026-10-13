"""Capítulo 4 — modelo GPT integrado."""
from src.gpt.config import GPT_CONFIG_124M, GPT_CONFIG_MINI, GPT2_VARIANTS
from src.gpt.gpt_model import GPTModel, count_parameters, model_size_mb

__all__ = [
    "GPT_CONFIG_124M",
    "GPT_CONFIG_MINI",
    "GPT2_VARIANTS",
    "GPTModel",
    "count_parameters",
    "model_size_mb",
]
