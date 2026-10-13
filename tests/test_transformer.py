"""
Testes do Transformer Block e seus componentes (Sprint 4)
1. LayerNorm normaliza para média 0 e variância 1
2. GELU se aproxima da ReLU nos extremos e é não nula para negativos
3. FeedForward expande ff_mult vezes e preserva o formato
4. TransformerBlock preserva o formato e respeita a causalidade
5. Conexões residuais mantêm o gradiente nas primeiras camadas
"""
import torch
import torch.nn as nn

from src.gpt.config import GPT_CONFIG_MINI, GPT_CONFIG_124M
from src.transformer import LayerNorm, GELU, FeedForward, TransformerBlock


def test_layer_norm_media_zero_variancia_um():
    torch.manual_seed(0)
    x = torch.randn(3, 5, 16) * 7 + 3
    out = LayerNorm(16)(x)
    assert torch.allclose(out.mean(dim=-1), torch.zeros(3, 5), atol=1e-5)
    assert torch.allclose(out.var(dim=-1, unbiased=False), torch.ones(3, 5), atol=1e-3)


def test_layer_norm_parametros_treinaveis():
    ln = LayerNorm(16)
    assert ln.scale.shape == (16,) and ln.shift.shape == (16,)
    assert ln.scale.requires_grad and ln.shift.requires_grad


def test_layer_norm_equivale_ao_pytorch():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 32)
    ref = nn.LayerNorm(32, eps=1e-5)
    assert torch.allclose(LayerNorm(32)(x), ref(x), atol=1e-5)


def test_gelu():
    gelu = GELU()
    x = torch.tensor([-10.0, -1.0, 0.0, 10.0])
    y = gelu(x)
    assert abs(y[0].item()) < 1e-4          # ~0 para negativos grandes
    assert y[1].item() < 0                  # não nula para negativos moderados
    assert y[2].item() == 0.0
    assert abs(y[3].item() - 10.0) < 1e-4   # ~identidade para positivos grandes
    ref = nn.GELU(approximate="tanh")
    xs = torch.linspace(-3, 3, 50)
    assert torch.allclose(gelu(xs), ref(xs), atol=1e-6)


def test_feed_forward_dimensoes():
    cfg = {"emb_dim": 32}
    ffn = FeedForward(cfg)
    x = torch.randn(2, 3, 32)
    assert ffn.layers[0].out_features == 4 * 32
    assert ffn(x).shape == x.shape

    ffn2 = FeedForward({"emb_dim": 32, "ff_mult": 2})
    assert ffn2.layers[0].out_features == 64


def test_transformer_block_preserva_formato():
    torch.manual_seed(123)
    block = TransformerBlock(GPT_CONFIG_MINI)
    x = torch.randn(2, 7, GPT_CONFIG_MINI["emb_dim"])
    assert block(x).shape == x.shape


def test_transformer_block_parametros_124m():
    block = TransformerBlock(GPT_CONFIG_124M)
    total = sum(p.numel() for p in block.parameters())
    # MHA (3*768*768 + 768*768 + 768) + FF (768*3072 + 3072 + 3072*768 + 768) + 2 LN (2*2*768)
    assert total == 2_360_064 + 4_722_432 + 3_072


def test_transformer_block_causal():
    """Alterar um token futuro não pode mudar a saída das posições anteriores."""
    torch.manual_seed(0)
    block = TransformerBlock(GPT_CONFIG_MINI).eval()
    x = torch.randn(1, 6, GPT_CONFIG_MINI["emb_dim"])
    x2 = x.clone()
    x2[:, -1] += 5.0
    out1, out2 = block(x), block(x2)
    assert torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-5)
    assert not torch.allclose(out1[:, -1], out2[:, -1])


def test_residual_mantem_gradiente():
    """Sem residuais o gradiente some nas primeiras camadas (seção 4.4)."""
    def gradiente_primeira_camada(use_shortcut):
        torch.manual_seed(123)
        layers = nn.ModuleList([nn.Sequential(nn.Linear(3, 3), GELU()) for _ in range(5)])
        x = torch.tensor([[1.0, 0.0, -1.0]])
        for layer in layers:
            out = layer(x)
            x = x + out if use_shortcut else out
        x.sum().backward()
        return layers[0][0].weight.grad.abs().mean().item()

    assert gradiente_primeira_camada(True) > 10 * gradiente_primeira_camada(False)
