"""
Sprint 2 - Pipeline integrado
================================

Amarra todas as etapas implementadas nesta Sprint em um unico fluxo,
reproduzindo o diagrama do enunciado:

    Texto -> Tokenizacao -> Tokens -> Token IDs -> Sequencias de
    treinamento -> Embeddings -> Positional Embeddings -> Lote de dados
    -> Entrada do modelo

Este script e o ponto de entrada usado tanto para inspecao manual quanto
pelos experimentos da Sprint (ver `experimentos/notebooks/sprint-02-experimentos.ipynb`
e `experimentos/resultados/sprint-02-experimentos.md`), de forma que os componentes
desenvolvidos aqui possam ser reaproveitados nas proximas Sprints (mecanismo de
atencao, blocos Transformer etc.).
"""

from __future__ import annotations

import torch

from src.tokenizer import BPETokenizer, split_into_tokens, Vocabulary, create_dataloader_v1
from src.embeddings import (
    make_token_embedding_layer,
    make_positional_embedding_layer,
    build_input_embeddings,
)

GPT2_VOCAB_SIZE = 50257


def run_pipeline(
    text: str,
    context_length: int = 4,
    batch_size: int = 8,
    output_dim: int = 256,
    stride: int | None = None,
    verbose: bool = True,
) -> dict:
    """Executa o fluxo completo e retorna as estruturas intermediarias.

    Retorna um dicionario com os artefatos de cada etapa, para permitir
    que os experimentos inspecionem formas e valores sem duplicar a
    logica do pipeline.
    """
    stride = stride or context_length

    # 1) Texto -> Tokens (visao didatica, baseada em regex)
    tokens_regex = split_into_tokens(text)

    # 2) Tokens -> Token IDs, via vocabulario proprio do corpus
    vocab = Vocabulary.from_text(text)

    # 3) Texto -> Token IDs, via BPE (o tokenizador real do GPT)
    bpe = BPETokenizer()
    token_ids_bpe = bpe.encode(text, allowed_special={"<|endoftext|>"})

    # 4) Sequencias de treinamento (entrada, alvo) via janela deslizante + DataLoader
    dataloader = create_dataloader_v1(
        text, batch_size=batch_size, max_length=context_length,
        stride=stride, shuffle=False,
    )
    inputs, targets = next(iter(dataloader))

    # 5) Token IDs -> Embeddings
    tok_emb_layer = make_token_embedding_layer(GPT2_VOCAB_SIZE, output_dim)
    token_embeddings = tok_emb_layer(inputs)

    # 6) + Positional Embeddings -> Entrada do modelo
    pos_emb_layer = make_positional_embedding_layer(context_length, output_dim)
    input_embeddings = build_input_embeddings(inputs, tok_emb_layer, pos_emb_layer)

    resultado = {
        "num_chars": len(text),
        "num_tokens_regex": len(tokens_regex),
        "vocab_size_proprio": len(vocab),
        "num_tokens_bpe": len(token_ids_bpe),
        "num_amostras_dataset": len(dataloader.dataset),
        "batch_inputs_shape": tuple(inputs.shape),
        "batch_targets_shape": tuple(targets.shape),
        "token_embeddings_shape": tuple(token_embeddings.shape),
        "input_embeddings_shape": tuple(input_embeddings.shape),
        "context_length": context_length,
        "batch_size": batch_size,
        "output_dim": output_dim,
        "stride": stride,
    }

    if verbose:
        print("=== Pipeline Sprint 2: Texto -> Entrada do modelo ===")
        for chave, valor in resultado.items():
            print(f"  {chave:>24}: {valor}")

        print("\nExemplo de par (entrada, alvo) do primeiro lote:")
        print(f"  entrada (ids) : {inputs[0].tolist()}")
        print(f"  alvo    (ids) : {targets[0].tolist()}")
        print(f"  entrada (texto): {bpe.decode(inputs[0].tolist())!r}")
        print(f"  alvo    (texto): {bpe.decode(targets[0].tolist())!r}")

    return resultado


if __name__ == "__main__":
    with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    run_pipeline(raw_text)
