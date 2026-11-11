"""
Rotina de Treinamento do Modelo GPT 
"""
import torch
from src.training.loss import calc_loss_batch, calc_loss_loader, calc_perplexity


def generate_sample(model, tokenizer, device, context_length=64,
                    prompt="MSSubClass", max_new_tokens=20):
    """Gera uma amostra de texto para validação qualitativa do aprendizado."""
    model.eval()
    tokens = torch.tensor(
        tokenizer.encode(prompt), dtype=torch.long, device=device
    ).unsqueeze(0)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(tokens[:, -context_length:])
        logits = logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        tokens = torch.cat((tokens, next_token), dim=1)

    return tokenizer.decode(tokens.squeeze(0).tolist())


def train_model(model, train_loader, val_loader, optimizer, device, num_epochs,
                context_length=64, eval_freq=1,
                start_context="MSSubClass", tokenizer=None):
    """
    Executa a iteração por épocas e lotes, cálculo da perda,
    retropropagação (backward pass) e atualização de pesos (optimizer.step).
    """
    train_losses, val_losses = [], []
    train_ppls, val_ppls = [], []

    print("\n" + "=" * 68)
    print(" INICIANDO A ROTINA DE TREINAMENTO DO GPT (ENTREGÁVEL 3)")
    print("=" * 68)

    # Avaliação Inicial (Época 0 — Pesos Aleatórios)
    init_tr = calc_loss_loader(train_loader, model, device)
    init_va = calc_loss_loader(val_loader, model, device)
    print(f"Estado Inicial (Época 0):")
    print(f"  Treino Loss: {init_tr:.4f} | Perplexidade: {calc_perplexity(init_tr):.2f}")
    print(f"  Valid. Loss:  {init_va:.4f} | Perplexidade: {calc_perplexity(init_va):.2f}")
    if tokenizer:
        print(f"  Texto (sem treino):\n    '{generate_sample(model, tokenizer, device, context_length, start_context)}'")
    print("-" * 68)

    # Loop de Treinamento
    for epoch in range(1, num_epochs + 1):
        model.train()
        for input_batch, target_batch in train_loader:
            # 1. Zerar os gradientes acumulados
            optimizer.zero_grad()

            # 2. Forward pass e cálculo da função de perda
            loss = calc_loss_batch(input_batch, target_batch, model, device)

            # 3. Retropropagação dos gradientes (Backpropagation)
            loss.backward()

            # 4. Atualização dos parâmetros do modelo
            optimizer.step()

        # Avaliação ao fim de cada época
        if epoch % eval_freq == 0:
            train_loss = calc_loss_loader(train_loader, model, device)
            val_loss = calc_loss_loader(val_loader, model, device)
            train_ppl = calc_perplexity(train_loss)
            val_ppl = calc_perplexity(val_loss)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_ppls.append(train_ppl)
            val_ppls.append(val_ppl)

            print(f"Época [{epoch:02d}/{num_epochs:02d}] | "
                  f"Treino Loss: {train_loss:.4f} (PPL: {train_ppl:.2f}) | "
                  f"Val Loss: {val_loss:.4f} (PPL: {val_ppl:.2f})")

            if tokenizer:
                sample = generate_sample(model, tokenizer, device, context_length, start_context)
                print(f"  [Amostra]: '{sample}'")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_ppls": train_ppls,
        "val_ppls": val_ppls
    }
