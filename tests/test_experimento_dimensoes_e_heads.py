"""
Testes dos Experimentos: Dimensões de Embedding e Heads
1. Diferentes dimensões de embedding
2. Diferentes números de heads
3. Diferentes dimensões de head (d_k)
"""
import torch
import pytest
from src.attention.multi_head_attention import MultiHeadAttention


def test_diferentes_dimensoes_de_embedding():
    """Experimento 1: Valida o funcionamento com dimensões de embedding variadas."""
    dimensoes = [16, 32, 64, 128, 256]
    lote, tokens = 2, 4
    
    for d in dimensoes:
        mha = MultiHeadAttention(d_in=d, d_out=d, context_length=tokens, num_heads=4)
        x = torch.randn(lote, tokens, d)
        out, _ = mha(x)
        
        # A saída deve manter exatamente a mesma dimensão de embedding configurada
        assert out.shape == (lote, tokens, d), f"Falha na dimensão de embedding {d}"


def test_diferentes_numeros_de_heads():
    """Experimento 2: Testa com 1, 2, 4 e 8 cabeças mantendo d_out fixo."""
    d_fixo = 64
    tokens = 4
    heads_list = [1, 2, 4, 8]
    x = torch.randn(1, tokens, d_fixo)
    
    params_esperados = 16448  # Total teórico de parâmetros para d=64
    
    for h in heads_list:
        mha = MultiHeadAttention(d_in=d_fixo, d_out=d_fixo, context_length=tokens, num_heads=h)
        out, weights = mha(x)
        
        # 1. Verifica se a saída preserva o formato
        assert out.shape == (1, tokens, d_fixo)
        # 2. Verifica se a matriz de pesos tem a dimensão de heads correta
        assert weights.shape == (1, h, tokens, tokens)
        # 3. Verifica a invariância de parâmetros
        total_p = sum(p.numel() for p in mha.parameters())
        assert total_p == params_esperados, f"Número de parâmetros divergiu para h={h}"


def test_diferentes_dimensoes_de_head():
    """Experimento 3: Valida se a dimensão interna da cabeça d_k = d_out // num_heads é exata."""
    configuracoes = [
        {"d_out": 32,  "heads": 4, "d_k_esperado": 8},
        {"d_out": 64,  "heads": 2, "d_k_esperado": 32},
        {"d_out": 64,  "heads": 4, "d_k_esperado": 16},
        {"d_out": 128, "heads": 8, "d_k_esperado": 16},
    ]
    
    for cfg in configuracoes:
        mha = MultiHeadAttention(
            d_in=cfg["d_out"],
            d_out=cfg["d_out"],
            context_length=4,
            num_heads=cfg["heads"]
        )
        assert mha.head_dim == cfg["d_k_esperado"], (
            f"Esperado d_k={cfg['d_k_esperado']}, obtido {mha.head_dim}"
        )
