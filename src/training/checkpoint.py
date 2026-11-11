"""
 Salvamento e Carregamento dos Pesos do Modelo

Funções:
  save_checkpoint : salva pesos + configuração + metadados em .pt
  load_checkpoint : carrega e reconstrói o modelo a partir do .pt
  demo_inference  : demonstra inferência com o modelo carregado
"""

import os
import json
import torch
import tiktoken


def save_checkpoint(model, optimizer, cfg, epoch, train_loss, val_loss,
                    path="gpt_checkpoint.pt"):
    """
    Salva o estado completo do modelo para permitir reprodução futura.

    O arquivo .pt contém:
      - model_state_dict  : pesos e biases de todas as camadas
      - optimizer_state   : momentos do AdamW (permite retomar treinamento)
      - config            : hiperparâmetros arquiteturais do GPT
      - epoch             : época em que o checkpoint foi gerado
      - train_loss        : perda de treinamento no momento do salvamento
      - val_loss          : perda de validação no momento do salvamento

    Por que salvar a configuração junto com os pesos?
    Sem ela, seria impossível recriar a arquitetura exata do modelo
    para carregar os pesos corretamente. Pesos e arquitetura são inseparáveis.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state":  optimizer.state_dict(),
        "config":           cfg,
        "epoch":            epoch,
        "train_loss":       train_loss,
        "val_loss":         val_loss,
    }
    torch.save(checkpoint, path)
    print(f"[✓] Checkpoint salvo em '{path}'")
    print(f"    Época: {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Salva também a configuração como JSON legível (útil para o README)
    cfg_path = path.replace(".pt", "_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"[✓] Configuração salva em '{cfg_path}'")
    return path


def load_checkpoint(path, model_class, device=None):
    """
    Carrega um checkpoint e reconstrói o modelo completo.

    Passos internos:
      1. Carrega o dicionário do .pt para a CPU/GPU especificada.
      2. Lê a configuração salva para recriar a arquitetura idêntica.
      3. Instancia um novo GPTModel com essa configuração.
      4. Carrega os pesos (state_dict) no modelo recém-criado.
      5. Coloca o modelo no device correto e em modo de avaliação.

    Retorna o modelo pronto para inferência e os metadados do checkpoint.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint não encontrado: '{path}'")

    print(f"Carregando checkpoint de '{path}'...")
    checkpoint = torch.load(path, map_location=device)

    cfg = checkpoint["config"]
    epoch = checkpoint["epoch"]
    train_loss = checkpoint["train_loss"]
    val_loss = checkpoint["val_loss"]

    # Reconstrói a arquitetura com a configuração salva
    model = model_class(cfg).to(device)

    # Carrega os pesos treinados
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"[✓] Modelo reconstruído e pesos carregados com sucesso.")
    print(f"    Configuração: emb_dim={cfg['emb_dim']}, "
          f"n_layers={cfg['n_layers']}, n_heads={cfg['n_heads']}")
    print(f"    Checkpoint da Época {epoch}: "
          f"Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}")

    return model, checkpoint


def demo_inference(model, device, context_length=64,
                   prompts=None, max_new_tokens=40):
    """
    Demonstra que o modelo carregado realiza inferência corretamente.

    Usa top-k sampling (k=10, T=0.8) para gerar texto de qualidade
    a partir de múltiplos prompts, confirmando que os pesos foram
    carregados e estão operacionais.
    """
    from src.generation.strategies import generate_top_k

    tokenizer = tiktoken.get_encoding("gpt2")
    model.eval()

    if prompts is None:
        prompts = [
            "Every effort moved",
            "MSSubClass identifies",
            "The house has",
        ]

    print("\n" + "=" * 65)
    print(" DEMONSTRAÇÃO DE INFERÊNCIA COM MODELO CARREGADO (Entregável 6)")
    print("=" * 65)

    for prompt in prompts:
        encoded = tokenizer.encode(prompt)
        input_ids = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)

        output = generate_top_k(
            model, input_ids, max_new_tokens, context_length,
            top_k=10, temperature=0.8
        )
        generated = tokenizer.decode(output.squeeze(0).tolist())
        print(f"\n  Prompt : '{prompt}'")
        print(f"  Gerado : '{generated}'")

    print("\n[✓] Inferência com pesos carregados: CONFIRMADA\n")
