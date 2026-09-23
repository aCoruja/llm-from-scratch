"""
Scaled Dot-Product Attention
=============================
Calcula a atenção básica através da fórmula:
Attention(Q, K, V) = Softmax( (Q @ K^T) / sqrt(d_k) + M ) @ V
"""
import math
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, mask=None, scale=True, dropout_layer=None):
    """
    Parâmetros:
    -----------
    Q, K, V : torch.Tensor
        Tensores de Consulta, Chave e Valor. Formato: (..., seq_len, d_k).
    mask : torch.Tensor, opcional
        Máscara booleana ou binária (1 para manter, 0 para mascarar).
    scale : bool
        Se True, divide os scores por sqrt(d_k).
    dropout_layer : nn.Dropout, opcional
        Camada de dropout aplicada aos pesos de atenção.
    """
    d_k = Q.size(-1)

    # 1. Similaridade entre tokens: Q @ K^T
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # 2. Escalonamento por sqrt(d_k)
    if scale:
        scores = scores / math.sqrt(d_k)

    # 3. Aplicação da máscara (substitui zeros por -infinito)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 4. Normalização Softmax (probabilidades somam 1.0 por linha)
    attention_weights = F.softmax(scores, dim=-1)

    # 5. Dropout aplicado sobre os pesos antes da ponderação
    if dropout_layer is not None:
        attention_weights = dropout_layer(attention_weights)

    # 6. Média ponderada dos vetores V
    output = torch.matmul(attention_weights, V)
    return output, attention_weights


if __name__ == "__main__":
    torch.manual_seed(123)
    Q = torch.randn(1, 4, 32)
    K = torch.randn(1, 4, 32)
    V = torch.randn(1, 4, 32)
    
    out, weights = scaled_dot_product_attention(Q, K, V)
    print("=== Demonstração: Scaled Dot-Product Attention ===")
    print(f"Shape Saída: {tuple(out.shape)}")
    print(f"Shape Pesos: {tuple(weights.shape)}")
    print(f"Soma da linha 0 dos pesos: {weights[0, 0].sum().item():.4f}")
