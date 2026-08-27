"""
Sprint 2 - Preparacao das sequencias de treinamento e DataLoader
==================================================================

Implementa a etapa de "sliding window" (Secao 2.6 do capitulo): a partir
da sequencia completa de Token IDs de um corpus, gera pares
(entrada, alvo) onde o alvo e a propria entrada deslocada em uma posicao.
E esse deslocamento que transforma um corpus de texto bruto em um dataset
de treinamento supervisionado para a tarefa de "prever o proximo token",
sem qualquer rotulacao manual.

    entrada : [x1, x2, x3, x4]
    alvo    : [x2, x3, x4, x5]

`GPTDatasetV1` encapsula essa logica como um `torch.utils.data.Dataset`,
e `create_dataloader_v1` monta o `DataLoader` correspondente, responsavel
por agrupar exemplos em lotes (batches) e por embaralhar o dataset a cada
epoca.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader

from src.tokenizer.tokenizer import BPETokenizer


class GPTDatasetV1(Dataset):
    """Gera pares (entrada, alvo) por janela deslizante sobre um texto.

    Parametros
    ----------
    text : str
        Corpus de texto bruto.
    tokenizer : BPETokenizer
        Tokenizador usado para converter o texto em Token IDs.
    max_length : int
        Tamanho do contexto (numero de tokens) de cada amostra de entrada.
    stride : int
        Passo do deslizamento da janela. `stride == max_length` gera
        amostras sem sobreposicao; `stride < max_length` gera amostras
        sobrepostas (mais amostras, porem mais redundantes).
    """

    def __init__(self, text: str, tokenizer: BPETokenizer, max_length: int, stride: int):
        self.input_ids: list[torch.Tensor] = []
        self.target_ids: list[torch.Tensor] = []

        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    text: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Monta o DataLoader que fornece lotes de (entrada, alvo) ao modelo."""
    tokenizer = BPETokenizer()
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    dataloader = create_dataloader_v1(
        raw_text, batch_size=4, max_length=8, stride=8, shuffle=False
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)

    print(f"Numero de amostras no dataset: {len(dataloader.dataset)}")
    print(f"Forma do lote de entrada : {tuple(inputs.shape)}  (batch_size x max_length)")
    print(f"Forma do lote de alvo    : {tuple(targets.shape)}")
    print("\nPrimeiro par (entrada, alvo) do lote:")
    print(f"  entrada: {inputs[0].tolist()}")
    print(f"  alvo   : {targets[0].tolist()}")
