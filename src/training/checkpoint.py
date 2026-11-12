"""
Salvamento e Carregamento dos Pesos do Modelo

Funções:
  save_checkpoint   : salva pesos + configuração + histórico + metadados em .pt
  load_checkpoint   : carrega e reconstrói o modelo a partir do .pt
  verify_checkpoint : prova que o modelo carregado é idêntico ao original
  demo_inference    : demonstra inferência com o modelo carregado
"""

import os
import json
import torch
import tiktoken


def save_checkpoint(model, optimizer, cfg, epoch, train_loss, val_loss,
                    path="gpt_checkpoint.pt", history=None, extra=None):
    """
    Salva o estado completo do modelo para permitir reprodução futura.

    O arquivo .pt contém:
      - model_state_dict : pesos e biases de todas as camadas
      - optimizer_state  : momentos do AdamW (permite retomar o treinamento)
      - config           : hiperparâmetros arquiteturais (sem ela não há como
                           recriar a arquitetura para receber os pesos)
      - epoch, train_loss, val_loss : estado no momento do salvamento
      - history          : perdas e perplexidades por época
      - extra            : seed, hiperparâmetros, corpus, tokenizador
      - torch_version    : versão do PyTorch usada
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state":  optimizer.state_dict(),
        "config":           cfg,
        "epoch":            epoch,
        "train_loss":       train_loss,
        "val_loss":         val_loss,
        "history":          history,
        "extra":            extra,
        "torch_version":    torch.__version__,
    }
    torch.save(checkpoint, path)
    print(f"[✓] Checkpoint salvo em '{path}'")
    print(f"    Época: {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Configuração em JSON legível (útil para o README)
    base, _ = os.path.splitext(path)
    cfg_path = base + "_config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "epoch": epoch, "extra": extra}, f, indent=2, ensure_ascii=False)
    print(f"[✓] Configuração salva em '{cfg_path}'")
    return path


def load_checkpoint(path, model_class, device=None):
    """
    Carrega um checkpoint e reconstrói o modelo completo.

    1. Carrega o dicionário do .pt para o dispositivo.
    2. Lê a configuração salva e recria a arquitetura idêntica.
    3. Carrega os pesos (state_dict) no modelo novo.
    4. Deixa o modelo em modo de avaliação.

    Retorna (modelo pronto para inferência, dicionário do checkpoint).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint não encontrado: '{path}'")

    print(f"Carregando checkpoint de '{path}'...")
    # weights_only=False: o checkpoint guarda o dicionário de configuração e a
    # versão do PyTorch (objetos Python), além dos pesos. A partir do PyTorch 2.6
    # o padrão de torch.load passou a ser weights_only=True, que recusaria esse
    # conteúdo. Carregamos com weights_only=False por ser um checkpoint próprio e
    # confiável, gerado por save_checkpoint() neste mesmo repositório.
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if "config" not in checkpoint or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"'{path}' não é um checkpoint completo (faltam 'config'/'model_state_dict'). "
            "Foi salvo com torch.save(model.state_dict())? Use save_checkpoint()."
        )

    cfg = checkpoint["config"]
    model = model_class(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("[✓] Modelo reconstruído e pesos carregados com sucesso.")
    print(f"    Configuração: emb_dim={cfg['emb_dim']}, "
          f"n_layers={cfg['n_layers']}, n_heads={cfg['n_heads']}")
    print(f"    Checkpoint da Época {checkpoint['epoch']}: "
          f"Train Loss={checkpoint['train_loss']:.4f} | Val Loss={checkpoint['val_loss']:.4f}")
    return model, checkpoint


def verify_checkpoint(model, path, model_class, device, prompt="Every effort moved"):
    """Compara logits e texto greedy do modelo original com o carregado do arquivo."""
    from src.generation.strategies import generate_greedy

    tokenizer = tiktoken.get_encoding("gpt2")
    loaded, _ = load_checkpoint(path, model_class, device)
    model.eval()
    loaded.eval()

    ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        same_logits = torch.allclose(model(ids), loaded(ids), atol=1e-6)

    ctx = model.cfg["context_length"]
    a = generate_greedy(model, ids.clone(), 30, ctx)
    b = generate_greedy(loaded, ids.clone(), 30, ctx)
    same_text = torch.equal(a, b)

    print(f"[verificação] logits idênticos: {same_logits} | texto greedy idêntico: {same_text}")
    print(f"[verificação] texto: {tokenizer.decode(b.squeeze(0).tolist())!r}")
    return bool(same_logits and same_text)


def demo_inference(model, device, context_length=64, prompts=None, max_new_tokens=40):
    """Demonstra que o modelo carregado gera texto (top-k=10, T=0.8)."""
    from src.generation.strategies import generate_top_k

    tokenizer = tiktoken.get_encoding("gpt2")
    model.eval()

    if prompts is None:
        prompts = ["Every effort moved", "MSSubClass identifies", "The house has"]

    print("\n" + "=" * 65)
    print(" DEMONSTRAÇÃO DE INFERÊNCIA COM MODELO CARREGADO (Entregável 6)")
    print("=" * 65)

    for prompt in prompts:
        ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        torch.manual_seed(123)
        out = generate_top_k(model, ids, max_new_tokens, context_length, top_k=10, temperature=0.8)
        print(f"\n  Prompt : '{prompt}'")
        print(f"  Gerado : '{tokenizer.decode(out.squeeze(0).tolist())}'")

    print("\n[✓] Inferência com pesos carregados: CONFIRMADA\n")
