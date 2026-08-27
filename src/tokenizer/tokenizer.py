"""
Sprint 2 - Tokenizacao
======================

Implementa duas familias de tokenizador discutidas no Capitulo 2
("Working with Text Data") do livro de referencia do projeto:

1. Um tokenizador simples baseado em expressoes regulares (SimpleTokenizerV1
   e SimpleTokenizerV2), que divide o texto em palavras e sinais de
   pontuacao. Serve para entender o conceito de "token" sem a complexidade
   de um algoritmo de subword.

2. Um tokenizador Byte Pair Encoding (BPE), usando a biblioteca `tiktoken`
   com o vocabulario do GPT-2. E o tokenizador realmente usado pelos
   modelos GPT, capaz de lidar com palavras fora do vocabulario ao
   quebra-las em subpalavras.

Ambos expoem a mesma interface minima: `encode(text) -> list[str]` para a
etapa de tokenizacao pura (texto -> tokens) e, no caso do BPE, tambem
`encode_to_ids` / `decode` diretamente sobre Token IDs, ja que o BPE
constroi o vocabulario internamente.
"""

from __future__ import annotations

import re

import tiktoken


def split_into_tokens(text: str) -> list[str]:
    """Tokenizacao baseada em regex (Secao 2.2 do capitulo).

    Divide o texto em palavras, espacos e sinais de pontuacao, depois
    remove os tokens que sao apenas espaco em branco. Essa e a forma mais
    simples de tokenizacao: cada unidade resultante e um "token".
    """
    # separa em: pontuacao comum, travessao duplo, e demais espacos
    pieces = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    return [piece.strip() for piece in pieces if piece.strip()]


class SimpleTokenizerV1:
    """Tokenizador didatico que exige que todo token exista no vocabulario.

    Reproduz o comportamento da primeira versao apresentada no capitulo:
    qualquer palavra fora do vocabulario causa uma excecao. Isso evidencia,
    por contraste, por que tokens especiais e o BPE sao necessarios.
    """

    def __init__(self, vocab: dict[str, int]):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        tokens = split_into_tokens(text)
        return [self.str_to_int[token] for token in tokens]

    def decode(self, ids: list[int]) -> str:
        text = " ".join(self.int_to_str[i] for i in ids)
        # remove o espaco inserido antes de sinais de pontuacao
        text = re.sub(r'\s+([,.:;?_!"()\'])', r"\1", text)
        return text


class SimpleTokenizerV2:
    """Extensao do tokenizador simples com tratamento de tokens especiais.

    Palavras desconhecidas sao mapeadas para ``<|unk|>`` em vez de causar
    erro, e o token ``<|endoftext|>`` pode ser usado para concatenar
    documentos distintos sem que o modelo confunda o fim de um texto com
    o inicio do proximo.
    """

    UNK = "<|unk|>"
    EOT = "<|endoftext|>"

    def __init__(self, vocab: dict[str, int]):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        tokens = split_into_tokens(text)
        tokens = [t if t in self.str_to_int else self.UNK for t in tokens]
        return [self.str_to_int[t] for t in tokens]

    def decode(self, ids: list[int]) -> str:
        text = " ".join(self.int_to_str[i] for i in ids)
        text = re.sub(r'\s+([,.:;?_!"()\'])', r"\1", text)
        return text


class BPETokenizer:
    """Wrapper fino sobre o tokenizador BPE do GPT-2 (via `tiktoken`).

    Diferente do tokenizador simples, o BPE nao depende de um vocabulario
    construido a partir de um corpus especifico: ele ja vem treinado e
    consegue representar qualquer string, inclusive palavras nunca vistas,
    dividindo-as em subpalavras ou ate bytes individuais.
    """

    def __init__(self, encoding_name: str = "gpt2"):
        self._enc = tiktoken.get_encoding(encoding_name)

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab

    def encode(self, text: str, allowed_special: set[str] | None = None) -> list[int]:
        allowed_special = allowed_special or {"<|endoftext|>"}
        return self._enc.encode(text, allowed_special=allowed_special)

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)

    def tokens(self, text: str, allowed_special: set[str] | None = None) -> list[str]:
        """Retorna a representacao textual de cada subpalavra (para inspecao)."""
        ids = self.encode(text, allowed_special=allowed_special)
        return [self._enc.decode([i]) for i in ids]


if __name__ == "__main__":
    amostras = [
        "Hello, do you like tea?",
        "In the sunlit terraces of the palace.",
        "Akwirw ier",  # palavra inventada: forca a quebra em subpalavras
    ]

    print("=== Tokenizacao baseada em regex ===")
    for frase in amostras[:2]:
        print(f"{frase!r} -> {split_into_tokens(frase)}")

    print("\n=== Tokenizacao BPE (GPT-2, via tiktoken) ===")
    bpe = BPETokenizer()
    for frase in amostras:
        ids = bpe.encode(frase)
        print(f"{frase!r}")
        print(f"  tokens : {bpe.tokens(frase)}")
        print(f"  ids    : {ids}")
        print(f"  decode : {bpe.decode(ids)!r}")
