"""Testes das camadas de embedding e positional embedding (Capitulo 2, Sprint 2)."""

import torch

from src.embeddings import make_token_embedding_layer, make_positional_embedding_layer, build_input_embeddings

VOCAB_SIZE, OUTPUT_DIM, CONTEXT_LENGTH, BATCH_SIZE = 100, 16, 4, 2


def test_token_embedding_layer_produz_forma_correta():
    layer = make_token_embedding_layer(VOCAB_SIZE, OUTPUT_DIM)
    ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, CONTEXT_LENGTH))
    out = layer(ids)
    assert out.shape == (BATCH_SIZE, CONTEXT_LENGTH, OUTPUT_DIM)


def test_positional_embedding_layer_tem_uma_linha_por_posicao():
    layer = make_positional_embedding_layer(CONTEXT_LENGTH, OUTPUT_DIM)
    assert layer.weight.shape == (CONTEXT_LENGTH, OUTPUT_DIM)


def test_input_embeddings_soma_token_e_posicional():
    tok_layer = make_token_embedding_layer(VOCAB_SIZE, OUTPUT_DIM)
    pos_layer = make_positional_embedding_layer(CONTEXT_LENGTH, OUTPUT_DIM)
    ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, CONTEXT_LENGTH))

    result = build_input_embeddings(ids, tok_layer, pos_layer)
    esperado = tok_layer(ids) + pos_layer(torch.arange(CONTEXT_LENGTH))
    assert torch.allclose(result, esperado)
    assert result.shape == (BATCH_SIZE, CONTEXT_LENGTH, OUTPUT_DIM)


def test_mesmo_token_id_em_posicoes_diferentes_gera_vetores_diferentes():
    tok_layer = make_token_embedding_layer(VOCAB_SIZE, OUTPUT_DIM)
    pos_layer = make_positional_embedding_layer(CONTEXT_LENGTH, OUTPUT_DIM)

    token_id = torch.tensor([7])
    vetor_puro = tok_layer(token_id)[0]
    vetor_pos0 = vetor_puro + pos_layer(torch.tensor([0]))[0]
    vetor_pos2 = vetor_puro + pos_layer(torch.tensor([2]))[0]

    assert not torch.allclose(vetor_pos0, vetor_pos2)
