"""
Modelo GPT
==========
Integra os componentes das Sprints 2, 3 e 4 em uma arquitetura GPT:

    Token IDs                    (B, T)
    Token Embedding              (B, T, D)    nn.Embedding(vocab_size, D)
    + Positional Embedding       (B, T, D)    nn.Embedding(context_length, D)
    Dropout                      (B, T, D)
    n_layers x TransformerBlock  (B, T, D)
    LayerNorm final              (B, T, D)
    Language Model Head          (B, T, V)    nn.Linear(D, vocab_size, bias=False)

A saída são os logits: para cada posição t, um vetor de V valores não
normalizados, um por token do vocabulário, que pontuam qual deve ser o
próximo token depois de x[0..t].
"""
import torch
import torch.nn as nn

from src.transformer.transformer_block import TransformerBlock
from src.transformer.layer_norm import LayerNorm


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg.get("drop_rate_emb", cfg["drop_rate"]))

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        assert seq_len <= self.cfg["context_length"], (
            f"Sequência com {seq_len} tokens excede context_length={self.cfg['context_length']}"
        )

        tok_embeds = self.tok_emb(in_idx)                                         # (B, T, D)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))   # (T, D)
        x = tok_embeds + pos_embeds                                               # (B, T, D)
        x = self.drop_emb(x)
        x = self.trf_blocks(x)                                                    # (B, T, D)
        x = self.final_norm(x)
        logits = self.out_head(x)                                                 # (B, T, V)
        return logits


def count_parameters(model):
    """Total de parâmetros e total considerando weight tying (out_head = tok_emb)."""
    total = sum(p.numel() for p in model.parameters())
    tied = total - sum(p.numel() for p in model.out_head.parameters())
    return total, tied


def model_size_mb(model, bytes_per_param=4):
    """Memória ocupada pelos parâmetros (float32 = 4 bytes)."""
    total, _ = count_parameters(model)
    return total * bytes_per_param / (1024 * 1024)


if __name__ == "__main__":
    import tiktoken
    from src.gpt.config import GPT_CONFIG_124M

    tokenizer = tiktoken.get_encoding("gpt2")
    batch = torch.stack([
        torch.tensor(tokenizer.encode("Every effort moves you")),
        torch.tensor(tokenizer.encode("Every day holds a")),
    ])

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    logits = model(batch)
    total, tied = count_parameters(model)

    print("=== Demonstração: GPTModel (GPT-2 124M) ===")
    print(f"Token IDs de entrada:\n{batch}")
    print(f"Shape da entrada: {tuple(batch.shape)}")
    print(f"Shape dos logits: {tuple(logits.shape)}")
    print(f"Parâmetros totais:             {total:,}")
    print(f"Parâmetros com weight tying:   {tied:,}")
    print(f"Tamanho (float32):             {model_size_mb(model):.2f} MB")
    print(f"Embedding de tokens:           {tuple(model.tok_emb.weight.shape)}")
    print(f"Language Model Head:           {tuple(model.out_head.weight.shape)}")
