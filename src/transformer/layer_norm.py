"""
Layer Normalization
===================
Normaliza as ativações de cada token ao longo da dimensão de embedding
(última dimensão), deixando-as com média 0 e variância 1:

    LayerNorm(x) = scale * (x - média) / sqrt(variância + eps) + shift

`scale` e `shift` são parâmetros treináveis (um valor por dimensão), que
permitem ao modelo desfazer parcialmente a normalização se isso for útil.

Diferente da BatchNorm, a estatística é calculada por token, e não ao longo
do lote, então o resultado independe do tamanho do batch.
"""
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # x: (batch, tokens, emb_dim) -> estatísticas: (batch, tokens, 1)
        mean = x.mean(dim=-1, keepdim=True)
        # unbiased=False (divide por n) para compatibilidade com o GPT-2 original
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


if __name__ == "__main__":
    torch.manual_seed(123)
    x = torch.randn(2, 5)
    ln = LayerNorm(emb_dim=5)
    out = ln(x)

    print("=== Demonstração: Layer Normalization ===")
    print(f"Média antes:     {x.mean(dim=-1).tolist()}")
    print(f"Variância antes: {x.var(dim=-1, unbiased=False).tolist()}")
    print(f"Média depois:     {out.mean(dim=-1).tolist()}")
    print(f"Variância depois: {out.var(dim=-1, unbiased=False).tolist()}")
