"""
Testes dos Experimentos: Máscara Causal e Sequências
7. Comportamento da máscara causal
8. Diferentes sequências de entrada
"""
import torch
import pytest
from src.attention.causal_attention import CausalAttention
from src.attention.multi_head_attention import MultiHeadAttention


def test_comportamento_mascara_causal():
    """Experimento 7: Garante a anulação estrita do futuro e a soma unitária das linhas."""
    tam_seq = 4
    causal = CausalAttention(d_in=16, d_out=16, context_length=tam_seq)
    x = torch.randn(1, tam_seq, 16)
    _, weights = causal(x)
    
    matriz = weights[0]
    
    # 1. Posições futuras (j > i) devem ser estritamente zero
    for i in range(tam_seq):
        for j in range(i + 1, tam_seq):
            assert matriz[i, j].item() == 0.0
            
    # 2. A primeira posição só olha para si mesma (probabilidade 1.0)
    assert torch.isclose(matriz[0, 0], torch.tensor(1.0), atol=1e-5)
    
    # 3. Cada linha deve somar 1.0
    somas = matriz.sum(dim=-1)
    assert torch.allclose(somas, torch.ones(tam_seq), atol=1e-5)


def test_diferentes_sequencias_de_entrada():
    """Experimento 8: Valida se o modelo aceita sequências de comprimentos variados (T=2, 5, 9, 14)."""
    contexto_maximo = 32
    mha = MultiHeadAttention(d_in=16, d_out=32, context_length=contexto_maximo, num_heads=4)
    
    tamanhos_seq = [2, 5, 9, 14, 25]
    
    for T in tamanhos_seq:
        x_var = torch.randn(1, T, 16)
        out, weights = mha(x_var)
        
        # O modelo deve se adaptar dinamicamente ao tamanho T de cada frase
        assert out.shape == (1, T, 32), f"Erro no formato de saída para T={T}"
        assert weights.shape == (1, 4, T, T), f"Erro no formato dos pesos para T={T}"
