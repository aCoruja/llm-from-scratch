"""
Testes da inferência e geração autoregressiva (Sprint 4)
1. Cada iteração acrescenta exatamente um token
2. O contexto inicial é preservado
3. Greedy é determinístico e igual a generate com temperature=0
4. Top-k restringe a escolha aos k tokens mais prováveis
5. O contexto é recortado quando excede context_size

Testes da Sprint 5
6. Salvar e carregar o checkpoint reproduz exatamente o modelo
7. Temperatura 0 e top-k=1 equivalem a greedy
"""
import pytest
import tiktoken
import torch

from src.gpt import GPTModel, GPT_CONFIG_MINI
from src.generation import (
    generate,
    generate_text_simple,
    text_to_token_ids,
    token_ids_to_text,
)

CTX = GPT_CONFIG_MINI["context_length"]


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(123)
    return GPTModel(GPT_CONFIG_MINI).eval()


@pytest.fixture(scope="module")
def tokenizer():
    return tiktoken.get_encoding("gpt2")


def test_texto_ida_e_volta(tokenizer):
    ids = text_to_token_ids("Hello, I am", tokenizer)
    assert ids.shape == (1, 4)
    assert token_ids_to_text(ids, tokenizer) == "Hello, I am"


def test_gera_um_token_por_iteracao(model, tokenizer):
    idx = text_to_token_ids("Hello, I am", tokenizer)
    out = generate_text_simple(model, idx, max_new_tokens=5, context_size=CTX)
    assert out.shape == (1, 4 + 5)
    assert torch.equal(out[:, :4], idx)


def test_greedy_deterministico(model, tokenizer):
    idx = text_to_token_ids("Every effort moves you", tokenizer)
    a = generate_text_simple(model, idx, 5, CTX)
    b = generate_text_simple(model, idx, 5, CTX)
    c = generate(model, idx, 5, CTX, temperature=0.0)
    assert torch.equal(a, b) and torch.equal(a, c)


def test_greedy_escolhe_argmax(model, tokenizer):
    idx = text_to_token_ids("Every effort moves you", tokenizer)
    out = generate_text_simple(model, idx, 1, CTX)
    with torch.no_grad():
        esperado = model(idx)[:, -1, :].argmax(dim=-1)
    assert out[0, -1].item() == esperado.item()


def test_top_k_restringe_escolha(model, tokenizer):
    idx = text_to_token_ids("Every effort moves you", tokenizer)
    with torch.no_grad():
        top3 = set(torch.topk(model(idx)[:, -1, :], 3).indices[0].tolist())
    for seed in range(20):
        torch.manual_seed(seed)
        out = generate(model, idx, 1, CTX, temperature=5.0, top_k=3)
        assert out[0, -1].item() in top3


def test_contexto_recortado(model):
    idx = torch.randint(0, GPT_CONFIG_MINI["vocab_size"], (1, CTX + 10))
    out = generate_text_simple(model, idx, max_new_tokens=2, context_size=CTX)
    assert out.shape == (1, CTX + 12)


def test_lote_com_varias_sequencias(model):
    idx = torch.randint(0, GPT_CONFIG_MINI["vocab_size"], (3, 4))
    out = generate_text_simple(model, idx, max_new_tokens=3, context_size=CTX)
    assert out.shape == (3, 7)


# ── Sprint 5 ──────────────────────────────────────────────────────────────────
def test_checkpoint_roundtrip(tmp_path):
    from src.training.checkpoint import save_checkpoint, load_checkpoint
    cfg = dict(GPT_CONFIG_MINI)
    torch.manual_seed(123)
    m = GPTModel(cfg).eval()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    p = str(tmp_path / "ck.pt")
    save_checkpoint(m, opt, cfg, 1, 1.0, 2.0, path=p, history={"train_losses": [1.0]})
    m2, ck = load_checkpoint(p, GPTModel, "cpu")
    x = torch.randint(0, 1000, (1, 8))
    with torch.no_grad():
        assert torch.allclose(m(x), m2(x), atol=1e-6)
    assert ck["history"]["train_losses"] == [1.0]


def test_temperature_zero_and_topk1_equal_greedy():
    from src.generation.strategies import (generate_greedy, generate_with_temperature,
                                           generate_top_k)
    torch.manual_seed(123)
    m = GPTModel(GPT_CONFIG_MINI).eval()
    x = torch.randint(0, 1000, (1, 4))
    g = generate_greedy(m, x.clone(), 5, 256)
    assert torch.equal(g, generate_with_temperature(m, x.clone(), 5, 256, temperature=0.0))
    assert torch.equal(g, generate_top_k(m, x.clone(), 5, 256, top_k=1))
