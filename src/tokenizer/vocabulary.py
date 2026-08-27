"""
Sprint 2 - Vocabulario e Token IDs
===================================

Implementa a construcao do vocabulario a partir de um corpus de texto
(Secao 2.3 e 2.4 do capitulo) e a classe `Vocabulary`, que centraliza a
relacao:

    Token <-> Token ID <-> Vocabulario

O vocabulario e simplesmente um dicionario que associa cada token unico
(ordenado alfabeticamente, por convencao do livro) a um inteiro. Esse
inteiro - o Token ID - e a unica coisa que a rede neural manipula
diretamente; o texto em si nunca entra na rede.
"""

from __future__ import annotations

from src.tokenizer.tokenizer import split_into_tokens

SPECIAL_TOKENS = ["<|unk|>", "<|endoftext|>"]


def build_vocabulary(text: str, special_tokens: list[str] | None = None) -> dict[str, int]:
    """Constroi o vocabulario a partir de um texto bruto.

    Passos:
        1. tokeniza o texto;
        2. reduz a um conjunto de tokens unicos;
        3. ordena alfabeticamente (reprodutibilidade);
        4. acrescenta os tokens especiais ao final;
        5. atribui um Token ID sequencial a cada token.
    """
    tokens = split_into_tokens(text)
    unique_tokens = sorted(set(tokens))
    if special_tokens:
        unique_tokens.extend(special_tokens)
    return {token: idx for idx, token in enumerate(unique_tokens)}


class Vocabulary:
    """Encapsula o dicionario token->id e sua inversa id->token."""

    def __init__(self, str_to_int: dict[str, int]):
        self.str_to_int = str_to_int
        self.int_to_str = {i: s for s, i in str_to_int.items()}

    @classmethod
    def from_text(cls, text: str, special_tokens: list[str] | None = SPECIAL_TOKENS) -> "Vocabulary":
        return cls(build_vocabulary(text, special_tokens))

    def __len__(self) -> int:
        return len(self.str_to_int)

    def token_to_id(self, token: str) -> int:
        """Converte um token em seu Token ID, usando <|unk|> como fallback."""
        if token in self.str_to_int:
            return self.str_to_int[token]
        if "<|unk|>" in self.str_to_int:
            return self.str_to_int["<|unk|>"]
        raise KeyError(f"Token {token!r} nao encontrado no vocabulario e nao ha <|unk|>.")

    def id_to_token(self, token_id: int) -> str:
        """Recupera o token original a partir do Token ID."""
        return self.int_to_str[token_id]

    def encode(self, text: str) -> list[int]:
        return [self.token_to_id(t) for t in split_into_tokens(text)]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.id_to_token(i) for i in ids)


if __name__ == "__main__":
    with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    vocab = Vocabulary.from_text(raw_text)
    print(f"Tamanho do vocabulario (com tokens especiais): {len(vocab)}")

    amostra = list(vocab.str_to_int.items())[:10]
    print("Primeiras entradas do vocabulario (token -> id):")
    for token, idx in amostra:
        print(f"  {token!r:>15} -> {idx}")

    print("\nTokens especiais:")
    for tok in SPECIAL_TOKENS:
        print(f"  {tok!r:>15} -> {vocab.str_to_int[tok]}")

    print("\nToken <-> Token ID <-> Vocabulario:")
    frase = "Hello, do you like tea?"
    ids = vocab.encode(frase)
    print(f"  texto      : {frase!r}")
    print(f"  token ids  : {ids}")
    print(f"  decodificado: {vocab.decode(ids)!r}  (note o uso de <|unk|>)")
