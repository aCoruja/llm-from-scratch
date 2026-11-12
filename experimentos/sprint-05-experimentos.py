"""
Sprint 5 — Experimentos de treinamento (Entregável 4) + Avaliação (Entregável 2)
================================================================================
Treina o GPT (versão reduzida, em CPU) sobre o corpus `the-verdict.txt`
variando a taxa de aprendizado, o número de épocas e a capacidade do modelo,
e registra, para cada experimento:

  - perda (cross-entropy) de treino e validação época a época;
  - perplexidade (PPL = exp(loss)) correspondente;
  - estado inicial (época 0, pesos aleatórios) — base de comparação do
    Entregável 2;
  - custo computacional (tempo de parede por época e total);
  - uma amostra de texto gerada (greedy) ao final, para avaliação qualitativa.

Os resultados brutos são gravados em
`experimentos/resultados/sprint-05-dados.json` e as curvas de perda em
`experimentos/resultados/figuras/`. A interpretação está no relatório
`experimentos/resultados/sprint-05-experimentos.md` e no README da sprint.

Uso:
    PYTHONPATH=. python experimentos/sprint-05-experimentos.py
"""
import os
import sys
import json
import time
import math

import torch
import tiktoken
from torch.utils.data import DataLoader

from src.gpt.gpt_model import GPTModel, count_parameters
from src.training.loss import calc_loss_loader, calc_loss_batch, calc_perplexity
from src.training.train import GPTDataset

RESULT_DIR = os.path.join("experimentos", "resultados")
FIG_DIR = os.path.join(RESULT_DIR, "figuras")
DATA_JSON = os.path.join(RESULT_DIR, "sprint-05-dados.json")

DEVICE = torch.device("cpu")
SEED = 42
MAX_LENGTH = 64          # janela de contexto (tokens)
STRIDE = 32              # deslizamento da janela
BATCH_SIZE = 4
WEIGHT_DECAY = 0.1
START_CONTEXT = "Every effort moved"

# Configurações de arquitetura usadas nos experimentos
CFG_SMALL = {"vocab_size": 50257, "context_length": MAX_LENGTH,
             "emb_dim": 128, "n_heads": 4, "n_layers": 2,
             "drop_rate": 0.1, "qkv_bias": False}
CFG_MED = {"vocab_size": 50257, "context_length": MAX_LENGTH,
           "emb_dim": 256, "n_heads": 4, "n_layers": 4,
           "drop_rate": 0.1, "qkv_bias": False}


def build_loaders(tokenizer):
    with open(os.path.join("data", "the-verdict.txt"), "r", encoding="utf-8") as f:
        raw = f.read()
    split = int(0.9 * len(raw))
    train_text, val_text = raw[:split], raw[split:]
    train_ds = GPTDataset(train_text, tokenizer, MAX_LENGTH, STRIDE)
    val_ds = GPTDataset(val_text, tokenizer, MAX_LENGTH, STRIDE)
    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            drop_last=False)
    meta = {"chars": len(raw), "tokens": len(tokenizer.encode(raw)),
            "train_samples": len(train_ds), "val_samples": len(val_ds),
            "batches_per_epoch": len(train_loader)}
    return train_loader, val_loader, train_text, meta


def sample_greedy(model, tokenizer, prompt=START_CONTEXT, max_new_tokens=25):
    model.eval()
    ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long,
                       device=DEVICE).unsqueeze(0)
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(ids[:, -MAX_LENGTH:])[:, -1, :]
        ids = torch.cat((ids, logits.argmax(dim=-1, keepdim=True)), dim=1)
    return tokenizer.decode(ids.squeeze(0).tolist())


def run_experiment(name, cfg, lr, num_epochs, train_loader, val_loader, tokenizer):
    """Treina um modelo e devolve o histórico completo + custo + amostra."""
    print(f"\n{'='*70}\n>>> {name}  (lr={lr}, épocas={num_epochs}, "
          f"emb={cfg['emb_dim']}, L={cfg['n_layers']})\n{'='*70}")
    torch.manual_seed(SEED)
    model = GPTModel(cfg).to(DEVICE)
    n_params = count_parameters(model)[0]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=WEIGHT_DECAY)

    # Estado inicial (Entregável 2): pesos aleatórios, época 0
    init_tr = calc_loss_loader(train_loader, model, DEVICE)
    init_va = calc_loss_loader(val_loader, model, DEVICE)
    print(f"  Época 00 | treino {init_tr:.4f} (PPL {calc_perplexity(init_tr):8.1f}) "
          f"| val {init_va:.4f} (PPL {calc_perplexity(init_va):8.1f})")

    epochs, tr_losses, va_losses = [0], [init_tr], [init_va]
    t0 = time.time()
    for epoch in range(1, num_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(xb, yb, model, DEVICE)
            loss.backward()
            optimizer.step()
        tr = calc_loss_loader(train_loader, model, DEVICE)
        va = calc_loss_loader(val_loader, model, DEVICE)
        epochs.append(epoch); tr_losses.append(tr); va_losses.append(va)
        print(f"  Época {epoch:02d} | treino {tr:.4f} (PPL {calc_perplexity(tr):8.1f}) "
              f"| val {va:.4f} (PPL {calc_perplexity(va):8.1f})")
    elapsed = time.time() - t0

    sample = sample_greedy(model, tokenizer)
    print(f"  Tempo: {elapsed:.1f}s ({elapsed/num_epochs:.1f}s/época) | "
          f"params {n_params:,}")
    print(f"  Amostra: {sample!r}")

    # Época de menor val loss (indício de onde parar antes do sobreajuste)
    best_idx = min(range(len(va_losses)), key=lambda i: va_losses[i])
    return {
        "name": name, "lr": lr, "num_epochs": num_epochs, "cfg": cfg,
        "n_params": n_params,
        "epochs": epochs,
        "train_losses": tr_losses, "val_losses": va_losses,
        "train_ppls": [calc_perplexity(x) for x in tr_losses],
        "val_ppls": [calc_perplexity(x) for x in va_losses],
        "init_train_loss": init_tr, "init_val_loss": init_va,
        "final_train_loss": tr_losses[-1], "final_val_loss": va_losses[-1],
        "best_val_epoch": epochs[best_idx], "best_val_loss": va_losses[best_idx],
        "gap_final": va_losses[-1] - tr_losses[-1],
        "elapsed_s": elapsed, "s_per_epoch": elapsed / num_epochs,
        "sample": sample,
    }


def plot_curves(results, meta):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[aviso] matplotlib indisponível ({e}); pulando figuras.")
        return []
    os.makedirs(FIG_DIR, exist_ok=True)
    figs = []

    # 1. Varredura de taxa de aprendizado (val loss)
    lr_runs = [r for r in results if r["name"].startswith("lr_")]
    if lr_runs:
        plt.figure(figsize=(7, 4.5))
        for r in lr_runs:
            plt.plot(r["epochs"], r["val_losses"], marker="o", ms=3,
                     label=f"lr={r['lr']:g}")
        plt.xlabel("Época"); plt.ylabel("Perda de validação (cross-entropy)")
        plt.title("Efeito da taxa de aprendizado na convergência")
        plt.ylim(0, 11.5); plt.grid(alpha=0.3); plt.legend()
        p = os.path.join(FIG_DIR, "sprint-05-lr-sweep.png")
        plt.tight_layout(); plt.savefig(p, dpi=120); plt.close(); figs.append(p)

    # 2. Sobreajuste: treino vs validação na corrida longa
    over = next((r for r in results if r["name"] == "overfitting"), None)
    if over:
        plt.figure(figsize=(7, 4.5))
        plt.plot(over["epochs"], over["train_losses"], marker="o", ms=3,
                 label="treino")
        plt.plot(over["epochs"], over["val_losses"], marker="s", ms=3,
                 label="validação")
        plt.axvline(over["best_val_epoch"], color="gray", ls="--",
                    label=f"menor val (ép. {over['best_val_epoch']})")
        plt.xlabel("Época"); plt.ylabel("Perda (cross-entropy)")
        plt.title("Sobreajuste: treino cai, validação sobe")
        plt.grid(alpha=0.3); plt.legend()
        p = os.path.join(FIG_DIR, "sprint-05-overfitting.png")
        plt.tight_layout(); plt.savefig(p, dpi=120); plt.close(); figs.append(p)

    # 3. Capacidade do modelo (val loss)
    cap = [r for r in results if r["name"] in ("baseline", "capacidade_maior")]
    if len(cap) == 2:
        plt.figure(figsize=(7, 4.5))
        for r in cap:
            plt.plot(r["epochs"], r["val_losses"], marker="o", ms=3,
                     label=f"{r['name']} ({r['n_params']:,} par.)")
        plt.xlabel("Época"); plt.ylabel("Perda de validação (cross-entropy)")
        plt.title("Capacidade do modelo vs convergência")
        plt.grid(alpha=0.3); plt.legend()
        p = os.path.join(FIG_DIR, "sprint-05-capacidade.png")
        plt.tight_layout(); plt.savefig(p, dpi=120); plt.close(); figs.append(p)

    return figs


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    tokenizer = tiktoken.get_encoding("gpt2")
    train_loader, val_loader, train_text, meta = build_loaders(tokenizer)
    print("Corpus the-verdict.txt:", meta)
    print(f"ln(vocab_size) = ln(50257) = {math.log(50257):.4f} "
          "(perda esperada de um modelo aleatório)")

    results = []
    # --- Experimento 1: varredura de taxa de aprendizado (SMALL, 15 épocas) ---
    for lr in (1e-4, 5e-4, 1e-3, 5e-3):
        results.append(run_experiment(f"lr_{lr:g}", CFG_SMALL, lr, 15,
                                      train_loader, val_loader, tokenizer))

    # 'baseline' = a corrida lr=5e-4 (apelido, para o gráfico de capacidade)
    baseline = next(r for r in results if r["name"] == "lr_0.0005")
    baseline_alias = dict(baseline); baseline_alias["name"] = "baseline"
    results.append(baseline_alias)

    # --- Experimento 2: capacidade do modelo (MED, 15 épocas, lr=5e-4) ---
    results.append(run_experiment("capacidade_maior", CFG_MED, 5e-4, 15,
                                  train_loader, val_loader, tokenizer))

    # --- Experimento 3: sobreajuste (SMALL, 40 épocas, lr=5e-4) ---
    results.append(run_experiment("overfitting", CFG_SMALL, 5e-4, 40,
                                  train_loader, val_loader, tokenizer))

    figs = plot_curves(results, meta)

    payload = {
        "meta": meta,
        "constantes": {
            "seed": SEED, "max_length": MAX_LENGTH, "stride": STRIDE,
            "batch_size": BATCH_SIZE, "weight_decay": WEIGHT_DECAY,
            "optimizer": "AdamW", "device": str(DEVICE),
            "ln_vocab": math.log(50257), "start_context": START_CONTEXT,
            "torch_version": torch.__version__,
        },
        "figuras": figs,
        "experimentos": results,
    }
    with open(DATA_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Dados salvos em {DATA_JSON}")
    print(f"[OK] Figuras: {figs}")


if __name__ == "__main__":
    sys.exit(main())
