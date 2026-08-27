"""Testes do vocabulario e da conversao Token <-> Token ID (Capitulo 2, Sprint 2)."""

import pytest

from src.tokenizer import build_vocabulary, Vocabulary


CORPUS = "the cat sat on the mat. the dog sat too."


def test_build_vocabulary_tem_apenas_tokens_unicos():
    vocab = build_vocabulary(CORPUS, special_tokens=None)
    assert len(vocab) == len(set(vocab.keys()))


def test_build_vocabulary_ordena_alfabeticamente_antes_dos_especiais():
    vocab = build_vocabulary(CORPUS, special_tokens=["<|unk|>", "<|endoftext|>"])
    sem_especiais = {k: v for k, v in vocab.items() if k not in ("<|unk|>", "<|endoftext|>")}
    ordenado = sorted(sem_especiais, key=lambda k: sem_especiais[k])
    assert ordenado == sorted(sem_especiais)


def test_vocabulary_ids_sao_unicos_e_sequenciais():
    vocab = Vocabulary.from_text(CORPUS)
    ids = sorted(vocab.str_to_int.values())
    assert ids == list(range(len(vocab)))


def test_token_to_id_e_id_to_token_sao_inversos():
    vocab = Vocabulary.from_text(CORPUS)
    for token in ["the", "cat", "."]:
        token_id = vocab.token_to_id(token)
        assert vocab.id_to_token(token_id) == token


def test_token_desconhecido_cai_em_unk():
    vocab = Vocabulary.from_text(CORPUS)
    assert vocab.token_to_id("palavra-inexistente-xyz") == vocab.str_to_int["<|unk|>"]


def test_vocabulary_sem_unk_levanta_keyerror_para_token_desconhecido():
    vocab = Vocabulary(build_vocabulary(CORPUS, special_tokens=None))
    with pytest.raises(KeyError):
        vocab.token_to_id("palavra-inexistente-xyz")


def test_encode_decode_ida_e_volta_preserva_tokens_conhecidos():
    vocab = Vocabulary.from_text(CORPUS)
    frase = "the cat sat"
    ids = vocab.encode(frase)
    assert vocab.decode(ids) == frase
