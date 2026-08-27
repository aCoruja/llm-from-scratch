"""Testes do componente de tokenizacao (Capitulo 2, Sprint 2)."""

from src.tokenizer import split_into_tokens, SimpleTokenizerV1, SimpleTokenizerV2, BPETokenizer


def test_split_into_tokens_separa_palavras_e_pontuacao():
    tokens = split_into_tokens("Hello, do you like tea?")
    assert tokens == ["Hello", ",", "do", "you", "like", "tea", "?"]


def test_split_into_tokens_remove_espacos_em_branco():
    tokens = split_into_tokens("  a   b  ")
    assert tokens == ["a", "b"]


def test_simple_tokenizer_v1_roundtrip():
    vocab = {tok: i for i, tok in enumerate(sorted(set(split_into_tokens("Hello, do you like tea?"))))}
    tok = SimpleTokenizerV1(vocab)
    ids = tok.encode("Hello, do you like tea?")
    assert tok.decode(ids) == "Hello, do you like tea?"


def test_simple_tokenizer_v2_usa_unk_para_palavra_desconhecida():
    vocab = {tok: i for i, tok in enumerate(sorted(set(split_into_tokens("do you like tea?"))))}
    vocab["<|unk|>"] = len(vocab)
    tok = SimpleTokenizerV2(vocab)
    ids = tok.encode("Hello, do you like tea?")
    decoded = tok.decode(ids)
    assert "<|unk|>" in decoded
    assert "tea" in decoded


def test_bpe_tokenizer_roundtrip_para_palavra_fora_do_vocabulario():
    bpe = BPETokenizer()
    texto = "Akwirw ier"
    ids = bpe.encode(texto)
    assert bpe.decode(ids) == texto
    # nao existe <|unk|>: o BPE sempre consegue representar a string
    assert all(isinstance(i, int) for i in ids)


def test_bpe_tokenizer_endoftext_e_reconhecido():
    bpe = BPETokenizer()
    ids = bpe.encode("Hello<|endoftext|>World", allowed_special={"<|endoftext|>"})
    assert bpe.decode(ids) == "Hello<|endoftext|>World"
