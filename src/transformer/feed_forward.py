"""
GELU e FeedForward Network
==========================
GELU (Gaussian Error Linear Unit), na aproximação por tanh usada no GPT-2:

    GELU(x) ≈ 0.5 * x * (1 + tanh( sqrt(2/π) * (x + 0.044715 * x³) ))

Diferente da ReLU, é suave e devolve valores pequenos (não nulos) para
entradas negativas, o que mantém gradiente nessa região.

A FeedForward é uma MLP aplicada a cada token de forma independente:

    (batch, tokens, emb_dim) -> Linear -> (batch, tokens, ff_mult * emb_dim)
                             -> GELU
                             -> Linear -> (batch, tokens, emb_dim)

Ela expande a representação para um espaço maior (4x no GPT-2), aplica a
não linearidade e contrai de volta, preservando o formato da entrada.
"""
import math

import torch
import torch.nn as nn


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # ff_mult é opcional no cfg (padrão do GPT-2 = 4) para permitir experimentos
        hidden_dim = cfg.get("ff_mult", 4) * cfg["emb_dim"]
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], hidden_dim),
            GELU(),
            nn.Linear(hidden_dim, cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    torch.manual_seed(123)
    cfg = {"emb_dim": 768}
    ffn = FeedForward(cfg)
    x = torch.rand(2, 3, 768)
    hidden = ffn.layers[1](ffn.layers[0](x))
    out = ffn(x)

    print("=== Demonstração: FeedForward ===")
    print(f"Entrada:       {tuple(x.shape)}")
    print(f"Camada oculta: {tuple(hidden.shape)}")
    print(f"Saída:         {tuple(out.shape)}")
    print(f"Parâmetros:    {sum(p.numel() for p in ffn.parameters()):,}")

    gelu, relu = GELU(), nn.ReLU()
    xs = torch.tensor([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0])
    print("\n=== GELU vs ReLU ===")
    for v, g, r in zip(xs.tolist(), gelu(xs).tolist(), relu(xs).tolist()):
        print(f"x={v:+.1f}  GELU={g:+.4f}  ReLU={r:+.4f}")
