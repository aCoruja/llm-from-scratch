"""
Sprint 2 - Embeddings e Positional Embeddings
================================================

Implementa as duas ultimas etapas do pipeline desta Sprint (Secoes 2.7 e
2.8 do capitulo): a transformacao de Token IDs (inteiros) em vetores
continuos (embeddings), e a incorporacao da posicao de cada token dentro
da sequencia.

Um Token ID sozinho e apenas um indice - ele nao carrega nenhuma nocao de
significado ou semelhanca entre tokens. A camada de embedding (`nn.Embedding`)
e essencialmente uma tabela de consulta (lookup table) de tamanho
`(vocab_size, output_dim)`, treinavel, que associa a cada Token ID um
vetor denso. E esse vetor - e nao mais o inteiro - que passa a representar
o token dentro da rede.

Como a camada de embedding sozinha nao diferencia a posicao de um token
dentro da sequencia (o mesmo Token ID gera sempre o mesmo vetor,
independente de onde aparece), soma-se a ele um segundo vetor - o
positional embedding -, obtido de uma segunda tabela de consulta indexada
pela posicao (0, 1, 2, ..., context_length-1). O resultado, chamado aqui
de "input embedding", e o que efetivamente alimenta os blocos do
Transformer.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def make_token_embedding_layer(vocab_size: int, output_dim: int, seed: int = 123) -> nn.Embedding:
    """Cria a camada de token embeddings (tabela vocab_size x output_dim)."""
    torch.manual_seed(seed)
    return nn.Embedding(vocab_size, output_dim)


def make_positional_embedding_layer(context_length: int, output_dim: int, seed: int = 123) -> nn.Embedding:
    """Cria a camada de positional embeddings (tabela context_length x output_dim).

    Segue a abordagem de posicionamento absoluto e aprendido usada pelo
    GPT: cada posicao possui seu proprio vetor treinavel, em vez de uma
    formula fixa (como o encoding senoidal do Transformer original).
    """
    torch.manual_seed(seed)
    return nn.Embedding(context_length, output_dim)


def build_input_embeddings(
    token_ids: torch.Tensor,
    token_embedding: nn.Embedding,
    positional_embedding: nn.Embedding,
) -> torch.Tensor:
    """Combina token embeddings e positional embeddings.

    `token_ids` tem forma (batch_size, context_length). O resultado tem
    forma (batch_size, context_length, output_dim): cada token da
    sequencia passa a ser representado por um vetor que soma "o que o
    token e" (token embedding) com "onde o token esta" (positional
    embedding).
    """
    batch_size, context_length = token_ids.shape

    tok_embeds = token_embedding(token_ids)                                   # (B, T, D)
    positions = torch.arange(context_length)
    pos_embeds = positional_embedding(positions)                              # (T, D)

    return tok_embeds + pos_embeds  # broadcasting sobre a dimensao de batch


if __name__ == "__main__":
    from src.tokenizer.dataset import create_dataloader_v1

    with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    VOCAB_SIZE = 50257  # tamanho do vocabulario BPE do GPT-2
    OUTPUT_DIM = 256
    CONTEXT_LENGTH = 4
    BATCH_SIZE = 8

    dataloader = create_dataloader_v1(
        raw_text, batch_size=BATCH_SIZE, max_length=CONTEXT_LENGTH,
        stride=CONTEXT_LENGTH, shuffle=False,
    )
    inputs, targets = next(iter(dataloader))
    print(f"Token IDs de entrada, forma: {tuple(inputs.shape)}")

    tok_emb_layer = make_token_embedding_layer(VOCAB_SIZE, OUTPUT_DIM)
    pos_emb_layer = make_positional_embedding_layer(CONTEXT_LENGTH, OUTPUT_DIM)

    token_embeddings = tok_emb_layer(inputs)
    print(f"Token embeddings, forma : {tuple(token_embeddings.shape)}  (batch x contexto x dim)")

    positions = torch.arange(CONTEXT_LENGTH)
    positional_embeddings = pos_emb_layer(positions)
    print(f"Positional embeddings, forma: {tuple(positional_embeddings.shape)}  (contexto x dim)")

    input_embeddings = build_input_embeddings(inputs, tok_emb_layer, pos_emb_layer)
    print(f"Input embeddings (token + posicional), forma: {tuple(input_embeddings.shape)}")

    print("\nEvidencia de que a posicao importa:")
    print("O mesmo Token ID em posicoes diferentes recebe vetores diferentes,")
    print("pois o positional embedding somado depende apenas da posicao, nao do token.")
    id_exemplo = inputs[0, 0].item()
    print(f"Token ID escolhido para o teste: {id_exemplo}")
    vetor_puro = tok_emb_layer(torch.tensor([id_exemplo]))[0]
    vetor_pos0 = vetor_puro + pos_emb_layer(torch.tensor([0]))[0]
    vetor_pos2 = vetor_puro + pos_emb_layer(torch.tensor([2]))[0]
    diferem = not torch.allclose(vetor_pos0, vetor_pos2)
    print(f"  vetor na posicao 0 == vetor na posicao 2 ? {'NAO' if diferem else 'SIM'}")
