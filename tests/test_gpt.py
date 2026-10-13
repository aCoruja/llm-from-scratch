"""
Testes do modelo GPT (Sprint 4)
1. Token IDs (B, T) produzem logits (B, T, vocab_size)
2. Número de parâmetros do GPT-2 124M confere com o livro
3. Configuração controla o número de blocos
4. Sequências maiores que context_length são rejeitadas
5. Modo eval é determinístico (dropout desligado)
"""
import pytest
import torch

from src.gpt import GPTModel, GPT_CONFIG_MINI, GPT_CONFIG_124M, count_parameters, model_size_mb


@pytest.fixture(scope="module")
def mini_model():
    torch.manual_seed(123)
    return GPTModel(GPT_CONFIG_MINI)


def test_formato_dos_logits(mini_model):
    idx = torch.randint(0, GPT_CONFIG_MINI["vocab_size"], (2, 5))
    logits = mini_model(idx)
    assert logits.shape == (2, 5, GPT_CONFIG_MINI["vocab_size"])


def test_numero_de_blocos(mini_model):
    assert len(mini_model.trf_blocks) == GPT_CONFIG_MINI["n_layers"]


def test_contexto_excedido(mini_model):
    idx = torch.zeros(1, GPT_CONFIG_MINI["context_length"] + 1, dtype=torch.long)
    with pytest.raises(AssertionError):
        mini_model(idx)


def test_eval_deterministico(mini_model):
    mini_model.eval()
    idx = torch.randint(0, GPT_CONFIG_MINI["vocab_size"], (1, 8))
    assert torch.equal(mini_model(idx), mini_model(idx))
    mini_model.train()


def test_parametros_gpt2_124m():
    model = GPTModel(GPT_CONFIG_124M)
    total, tied = count_parameters(model)
    assert total == 163_009_536
    assert tied == 124_412_160
    assert round(model_size_mb(model), 2) == 621.83
