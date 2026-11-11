"""
Script Principal de Treinamento do GPT

Corpus utilizados:
  1. data/data_description.txt — fornecido pela disciplina via Moodle
  2. data/the-verdict.txt      — corpus canônico do livro (Cap. 5, Raschka)
"""

import os
import urllib.request
import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
from src.gpt.gpt_model import GPTModel
from src.training.trainer import train_model


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
    """Carrega e concatena os corpus de texto disponíveis."""
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

    return "\n\n".join(texts)


# ══════════════════════════════════════════════════════════════════════
# FUNÇÃO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}\n")

    # ── 1. Corpus ────────────────────────────────────────────────────
    print("Carregando corpus:")
    text = load_corpus()

    tokenizer = tiktoken.get_encoding("gpt2")
    total_tokens = len(tokenizer.encode(text, allowed_special={"<|endoftext|>"}))
    print(f"  Total: {len(text):,} caracteres | {total_tokens:,} tokens BPE\n")

    # ── 2. Hiperparâmetros de Dados ─────────────────────────────────
    MAX_LENGTH = 64      # Janela de contexto em tokens
    STRIDE = 32          # Deslocamento da janela deslizante
    BATCH_SIZE = 4       # Amostras por lote

    split_idx = int(0.9 * len(text))
    train_ds = GPTDataset(text[:split_idx], tokenizer, MAX_LENGTH, STRIDE)
    val_ds = GPTDataset(text[split_idx:], tokenizer, MAX_LENGTH, STRIDE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ── 3. Hiperparâmetros de Treinamento ───────────────────────────
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 0.1
    NUM_EPOCHS = 15

    # ── 4. Configuração Arquitetural do GPT ─────────────────────────
    cfg = {
        "vocab_size":      50257,
        "context_length":  MAX_LENGTH,
        "emb_dim":         256,
        "n_heads":         4,
        "n_layers":        4,
        "drop_rate":       0.1,
    }

    # ── 5. Registro dos Parâmetros ──────────────────────────────────
    print("Registro de Parâmetros de Treinamento:")
    print(f"  Corpus                  : data_description.txt + the-verdict.txt")
    print(f"  Particionamento         : 90% treino / 10% validação")
    print(f"  Janela de contexto      : {MAX_LENGTH} tokens")
    print(f"  Stride                  : {STRIDE} tokens")
    print(f"  Batch size              : {BATCH_SIZE}")
    print(f"  Amostras de treino      : {len(train_ds)}")
    print(f"  Amostras de validação   : {len(val_ds)}")
    print(f"  Lotes por época (treino): {len(train_loader)}")
    print(f"  Número de épocas        : {NUM_EPOCHS}")
    print(f"  Taxa de aprendizado (lr): {LEARNING_RATE}")
    print(f"  Weight decay            : {WEIGHT_DECAY}")
    print(f"  Otimizador              : AdamW")
    print(f"  Arquitetura             : {cfg['n_layers']} camadas, "
          f"{cfg['emb_dim']} emb_dim, {cfg['n_heads']} cabeças")

    # ── 6. Instanciação do Modelo ───────────────────────────────────
    torch.manual_seed(42)
    model = GPTModel(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parâmetros totais       : {total_params:,}")

    # ── 7. Otimizador AdamW ─────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ── 8. Execução do Treinamento ──────────────────────────────────
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=NUM_EPOCHS,
        context_length=MAX_LENGTH,
        eval_freq=1,
        start_context="Every effort moved",
        tokenizer=tokenizer,
    )

    # ── 9. Salvamento dos Pesos ─────────────────────────────────────
    torch.save(model.state_dict(), "gpt_trained.pt")
    print(f"\n[SUCESSO] Pesos do modelo salvos em 'gpt_trained.pt'")


if __name__ == "__main__":
    main()
