"""
Transformer Block
=================
Bloco básico do GPT, repetido n_layers vezes. Usa Pre-LayerNorm (a norma
vem antes de cada subcamada) e uma conexão residual em volta de cada uma:

    x = x + Dropout( MultiHeadAttention( LayerNorm1(x) ) )
    x = x + Dropout( FeedForward( LayerNorm2(x) ) )

Fluxo de dimensões, com B = batch, T = tokens, D = emb_dim:

    entrada                     (B, T, D)
    LayerNorm1                  (B, T, D)
    MultiHeadAttention          (B, T, D)  internamente (B, n_heads, T, D / n_heads)
    + residual                  (B, T, D)
    LayerNorm2                  (B, T, D)
    FeedForward                 (B, T, D)  internamente (B, T, ff_mult * D)
    + residual                  (B, T, D)

A atenção mistura informação entre os tokens; a FeedForward transforma
cada token isoladamente. A saída tem o mesmo formato da entrada, o que
permite empilhar blocos sem ajustar dimensões.
"""
import torch
import torch.nn as nn

from src.attention.multi_head_attention import MultiHeadAttention
from src.transformer.layer_norm import LayerNorm
from src.transformer.feed_forward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Dropouts separados são opcionais (exercício 4.3); caem em drop_rate se ausentes
        drop_attn = cfg.get("drop_rate_attn", cfg["drop_rate"])
        drop_shortcut = cfg.get("drop_rate_shortcut", cfg["drop_rate"])

        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=drop_attn,
            bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(drop_shortcut)

    def forward(self, x):
        # Subcamada 1: atenção com conexão residual
        shortcut = x
        x = self.norm1(x)
        x, _ = self.att(x)  # a MHA da Sprint 3 devolve (saída, pesos)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # Subcamada 2: feed forward com conexão residual
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


if __name__ == "__main__":
    from src.gpt.config import GPT_CONFIG_124M

    torch.manual_seed(123)
    x = torch.rand(2, 4, 768)
    block = TransformerBlock(GPT_CONFIG_124M)
    out = block(x)

    print("=== Demonstração: Transformer Block ===")
    print(f"Entrada: {tuple(x.shape)}")
    print(f"Saída:   {tuple(out.shape)}")
    n_att = sum(p.numel() for p in block.att.parameters())
    n_ff = sum(p.numel() for p in block.ff.parameters())
    n_ln = sum(p.numel() for p in block.norm1.parameters()) * 2
    print(f"Parâmetros MHA:         {n_att:,}")
    print(f"Parâmetros FeedForward: {n_ff:,}")
    print(f"Parâmetros LayerNorms:  {n_ln:,}")
    print(f"Total do bloco:         {n_att + n_ff + n_ln:,}")
