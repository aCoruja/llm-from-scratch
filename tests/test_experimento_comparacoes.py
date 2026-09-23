"""
Testes dos Experimentos: Comparações e Visualização
4. Comparação entre Self-Attention e Multi-Head Attention
5. Visualização da matriz de atenção
6. Comparação entre Attention com e sem escala
"""
import torch
import pytest
from src.attention.self_attention import SelfAttention
from src.attention.multi_head_attention import MultiHeadAttention
from src.attention.scaled_dot_product import scaled_dot_product_attention


def test_comparacao_self_attention_e_multi_head():
    """Experimento 4: Compara formatos entre Single-Head e Multi-Head."""
    d_in, d_out, tokens = 16, 32, 5
    x = torch.randn(1, tokens, d_in)
    
    # 1. Single-Head
    sa = SelfAttention(d_in=d_in, d_out=d_out)
    out_sa, weights_sa = sa(x)
    
    # 2. Multi-Head (4 cabeças)
    mha = MultiHeadAttention(d_in=d_in, d_out=d_out, context_length=tokens, num_heads=4)
    out_mha, weights_mha = mha(x)
    
    # Saídas têm a mesma forma
    assert out_sa.shape == out_mha.shape == (1, tokens, d_out)
    # Self-Attention tem 1 matriz (1, T, T); MHA tem 4 matrizes (1, 4, T, T)
    assert weights_sa.shape == (1, tokens, tokens)
    assert weights_mha.shape == (1, 4, tokens, tokens)


def test_visualizacao_matriz_de_atencao():
    """Experimento 5: Valida se a matriz gerada é válida para visualização (sem NaN ou Inf)."""
    mha = MultiHeadAttention(d_in=16, d_out=16, context_length=4, num_heads=2)
    x = torch.randn(1, 4, 16)
    _, weights = mha(x)
    
    # Converte para numpy como na visualização de heatmaps
    matriz_np = weights[0, 0].detach().cpu().numpy()
    
    # Não pode conter valores nulos (NaN) ou infinitos (Inf)
    assert not torch.isnan(torch.tensor(matriz_np)).any()
    assert not torch.isinf(torch.tensor(matriz_np)).any()
    # Todos os valores devem estar na faixa de probabilidade [0.0, 1.0]
    assert (matriz_np >= 0.0).all() and (matriz_np <= 1.0).all()


def test_comparacao_com_e_sem_escala():
    """Experimento 6: Valida a saturação do Softmax na ausência do fator sqrt(d_k)."""
    d_k = 64
    seq = 4
    Q = torch.randn(1, seq, d_k) * 2.0
    K = torch.randn(1, seq, d_k) * 2.0
    V = torch.randn(1, seq, d_k)
    
    _, w_sem_escala = scaled_dot_product_attention(Q, K, V, scale=False)
    _, w_com_escala = scaled_dot_product_attention(Q, K, V, scale=True)
    
    max_sem = w_sem_escala.max().item()
    max_com = w_com_escala.max().item()
    
    # Sem escala, a distribuição deve saturar em valores mais extremos
    assert max_sem > max_com, f"Sem escala ({max_sem:.4f}) deveria ser maior que com escala ({max_com:.4f})"
    assert max_com < 0.85, f"Com escala deveria ser suave, mas deu {max_com:.4f}"
