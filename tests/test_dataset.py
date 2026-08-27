"""Testes da janela deslizante e do DataLoader (Capitulo 2, Sprint 2)."""

import torch

from src.tokenizer import BPETokenizer, GPTDatasetV1, create_dataloader_v1

CORPUS = "The quick brown fox jumps over the lazy dog. " * 20


def test_dataset_gera_pares_entrada_alvo_deslocados_em_uma_posicao():
    bpe = BPETokenizer()
    ds = GPTDatasetV1(CORPUS, bpe, max_length=4, stride=4)
    input_ids, target_ids = ds[0]
    # o alvo e a entrada deslocada de uma posicao para a direita
    assert target_ids[:-1].tolist() == input_ids[1:].tolist()
    assert input_ids.shape == target_ids.shape == (4,)


def test_dataset_tamanho_bate_com_formula_da_janela_deslizante():
    bpe = BPETokenizer()
    token_ids = bpe.encode(CORPUS)
    max_length, stride = 8, 8
    ds = GPTDatasetV1(CORPUS, bpe, max_length=max_length, stride=stride)
    esperado = len(range(0, len(token_ids) - max_length, stride))
    assert len(ds) == esperado


def test_stride_menor_gera_mais_amostras_sobrepostas():
    bpe = BPETokenizer()
    ds_sem_sobreposicao = GPTDatasetV1(CORPUS, bpe, max_length=16, stride=16)
    ds_com_sobreposicao = GPTDatasetV1(CORPUS, bpe, max_length=16, stride=4)
    assert len(ds_com_sobreposicao) > len(ds_sem_sobreposicao)


def test_dataloader_produz_lotes_com_forma_batch_x_contexto():
    dl = create_dataloader_v1(CORPUS, batch_size=4, max_length=8, stride=8, shuffle=False)
    inputs, targets = next(iter(dl))
    assert inputs.shape == (4, 8)
    assert targets.shape == (4, 8)
    assert isinstance(inputs, torch.Tensor)


def test_dataloader_drop_last_descarta_lote_incompleto():
    dl = create_dataloader_v1(CORPUS, batch_size=7, max_length=8, stride=8, shuffle=False, drop_last=True)
    for inputs, _ in dl:
        assert inputs.shape[0] == 7
