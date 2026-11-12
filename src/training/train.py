"""
Script Principal de Treinamento do GPT

Corpus utilizados:
  1. data/data_description.txt — fornecido pela disciplina 
  2. data/the-verdict.txt      — corpus canônico do livro (Cap. 5, Raschka)
"""

import os
import copy
import urllib.request
import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
from src.gpt.gpt_model import GPTModel
from src.training.trainer import train_model
from src.training.checkpoint import (save_checkpoint, load_checkpoint,
                                     verify_checkpoint, demo_inference)
from src.generation.strategies import compare_weights


# ══════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════
class GPTDataset(Dataset):
    """
    Dataset com janela deslizante para modelagem autorregressiva.

    Para cada posição i no texto tokenizado, cria um par:
      X = tokens[i   : i + max_length]     (entrada)
      Y = tokens[i+1 : i + max_length + 1] (alvo — próximo token)

    O stride controla a sobreposição entre janelas consecutivas.
    stride = max_length → sem sobreposição
    stride = max_length // 2 → 50% de sobreposição (mais amostras)
    """
    def __init__(self, text, tokenizer, max_length=64, stride=32):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(
                torch.tensor(token_ids[i : i + max_length], dtype=torch.long)
            )
            self.target_ids.append(
                torch.tensor(token_ids[i + 1 : i + max_length + 1], dtype=torch.long)
            )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# ══════════════════════════════════════════════════════════════════════
# CARREGAMENTO DO CORPUS
# ══════════════════════════════════════════════════════════════════════
def load_corpus():
    """Carrega os corpus de texto disponíveis e devolve a lista de textos."""
    os.makedirs("data", exist_ok=True)
    texts = []

    # Corpus 1: data_description.txt (dados da disciplina)
    desc_path = os.path.join("data", "data_description.txt")
    if os.path.exists(desc_path):
        with open(desc_path, "r", encoding="utf-8") as f:
            texts.append(f.read())
        print(f"  [OK] '{desc_path}' carregado ({os.path.getsize(desc_path):,} bytes)")
    else:
        print(f"  [--] '{desc_path}' não encontrado (opcional)")

    # Corpus 2: the-verdict.txt (canônico do livro)
    verdict_path = os.path.join("data", "the-verdict.txt")
    if not os.path.exists(verdict_path):
        print("  Baixando 'the-verdict.txt'...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/"
            "main/ch02/01_main-chapter-code/the-verdict.txt",
            verdict_path,
        )
    with open(verdict_path, "r", encoding="utf-8") as f:
        texts.append(f.read())
    print(f"  [OK] '{verdict_path}' carregado ({os.path.getsize(verdict_path):,} bytes)")

    return texts


# ══════════════════════════════════════════════════════════════════════
# FUNÇÃO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}\n")

    # ── 1. Corpus (90/10 DENTRO de cada corpus: a validação representa os dois) ──
    print("Carregando corpus:")
    texts = load_corpus()
    tokenizer = tiktoken.get_encoding("gpt2")
    train_text = "\n\n".join(t[:int(0.9 * len(t))] for t in texts)
    val_text = "\n\n".join(t[int(0.9 * len(t)):] for t in texts)
    total_tokens = len(tokenizer.encode(train_text + val_text,
                                        allowed_special={"<|endoftext|>"}))
    print(f"  Total: {len(train_text) + len(val_text):,} caracteres | "
          f"{total_tokens:,} tokens BPE\n")

    # ── 2. Hiperparâmetros de dados ──────────────────────────────────
    MAX_LENGTH = 64      # Janela de contexto em tokens
    STRIDE = 32          # Deslocamento da janela deslizante
    BATCH_SIZE = 4       # Amostras por lote

    train_ds = GPTDataset(train_text, tokenizer, MAX_LENGTH, STRIDE)
    val_ds = GPTDataset(val_text, tokenizer, MAX_LENGTH, STRIDE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ── 3. Hiperparâmetros de treinamento ────────────────────────────
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 0.1
    NUM_EPOCHS = 15
    EARLY_EPOCHS = 3     # snapshot "pouco treinado" para comparar pesos
    SEED = 42

    # ── 4. Arquitetura (qkv_bias é obrigatório: transformer_block.py lê cfg["qkv_bias"]) ──
    cfg = {
        "vocab_size":      50257,
        "context_length":  MAX_LENGTH,
        "emb_dim":         256,
        "n_heads":         4,
        "n_layers":        4,
        "drop_rate":       0.1,
        "qkv_bias":        False,
    }

    # ── 5. Registro dos parâmetros ───────────────────────────────────
    print("Registro de Parâmetros de Treinamento:")
    print("  Corpus                  : data_description.txt + the-verdict.txt")
    print("  Particionamento         : 90% treino / 10% validação (dentro de cada corpus)")
    print(f"  Janela de contexto      : {MAX_LENGTH} tokens")
    print(f"  Stride                  : {STRIDE} tokens")
    print(f"  Batch size              : {BATCH_SIZE}")
    print(f"  Amostras de treino      : {len(train_ds)}")
    print(f"  Amostras de validação   : {len(val_ds)}")
    print(f"  Lotes por época (treino): {len(train_loader)}")
    print(f"  Número de épocas        : {NUM_EPOCHS}")
    print(f"  Taxa de aprendizado (lr): {LEARNING_RATE}")
    print(f"  Weight decay            : {WEIGHT_DECAY}")
    print("  Otimizador              : AdamW")
    print(f"  Seed                    : {SEED}")
    print(f"  Arquitetura             : {cfg['n_layers']} camadas, "
          f"{cfg['emb_dim']} emb_dim, {cfg['n_heads']} cabeças")

    # ── 6. Modelo e otimizador ───────────────────────────────────────
    torch.manual_seed(SEED)
    model = GPTModel(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parâmetros totais       : {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )

    # ── 7. Treinamento em duas etapas (guarda 3 estados de pesos) ────
    untrained = copy.deepcopy(model).eval()           # pesos aleatórios
    kw = dict(
        train_loader=train_loader, val_loader=val_loader, optimizer=optimizer,
        device=device, context_length=MAX_LENGTH, eval_freq=1,
        start_context="Every effort moved", tokenizer=tokenizer,
    )
    h1 = train_model(model=model, num_epochs=EARLY_EPOCHS, **kw)
    early = copy.deepcopy(model).eval()               # pouco treinado
    h2 = train_model(model=model, num_epochs=NUM_EPOCHS - EARLY_EPOCHS, **kw)
    history = {k: h1[k] + h2[k] for k in h1}          # histórico completo

    # ── 8. Salvar, verificar e usar o checkpoint (Entregável 6) ──────
    CKPT = "gpt_trained.pt"
    save_checkpoint(
        model, optimizer, cfg, NUM_EPOCHS,
        history["train_losses"][-1], history["val_losses"][-1],
        path=CKPT, history=history,
        extra=dict(
            seed=SEED, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
            batch_size=BATCH_SIZE, max_length=MAX_LENGTH, stride=STRIDE,
            early_epochs=EARLY_EPOCHS,
            corpus=["data/data_description.txt", "data/the-verdict.txt"],
            split="90/10 dentro de cada corpus", tokenizer="gpt2 (tiktoken)",
        ),
    )
    verify_checkpoint(model, CKPT, GPTModel, device)
    loaded, _ = load_checkpoint(CKPT, GPTModel, device)
    demo_inference(loaded, device, context_length=MAX_LENGTH)

    # ── 9. Estratégias × pesos (Entregável 5) ────────────────────────
    compare_weights(
        {"sem treino": untrained,
         f"{EARLY_EPOCHS} épocas": early,
         f"{NUM_EPOCHS} épocas": loaded},
        tokenizer, device, context_length=MAX_LENGTH,
        prompt="Every effort moved", max_new_tokens=60, num_samples=5,
        seed=123, corpus_text=train_text,
    )


if __name__ == "__main__":
    main()
