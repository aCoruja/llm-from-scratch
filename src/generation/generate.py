"""
Inferência e geração autoregressiva
===================================
Cada passo da geração:

    1. Recorta o contexto para os últimos context_size tokens    idx: (B, T)
    2. Executa o modelo                                          logits: (B, T, V)
    3. Mantém só a última posição (previsão do próximo token)   logits: (B, V)
    4. Converte logits em probabilidades com softmax             probas: (B, V)
    5. Seleciona o próximo token                                 idx_next: (B, 1)
    6. Concatena ao contexto e repete                            idx: (B, T + 1)

Estratégias de seleção (passo 5):
    - greedy (temperature = 0): argmax, sempre o token mais provável;
      determinística. Como softmax preserva a ordem, argmax(logits) é igual
      a argmax(probas).
    - amostragem com temperatura: softmax(logits / T) e sorteio. T < 1 deixa
      a distribuição mais concentrada; T > 1 a deixa mais uniforme.
    - top-k: antes do softmax, mantém só os k maiores logits (os demais
      viram -inf), então só esses k tokens podem ser sorteados.

generate_text_simple segue o Listing 4.8 (greedy); generate acrescenta
temperatura, top-k e parada por eos_id (seção 5.3 do livro).
"""
import torch


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded).unsqueeze(0)  # adiciona a dimensão de batch: (1, T)


def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.squeeze(0).tolist())


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """Geração gulosa (Listing 4.8). idx: (B, T) com os Token IDs do contexto."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)                               # (B, T, V)
        logits = logits[:, -1, :]                                  # (B, V)
        probas = torch.softmax(logits, dim=-1)                     # (B, V)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)      # (B, 1)
        idx = torch.cat((idx, idx_next), dim=1)                    # (B, T + 1)
    return idx


def generate(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None):
    """Geração com temperatura e top-k. temperature=0 equivale à geração gulosa."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1:]
            logits = torch.where(logits < min_val, torch.full_like(logits, float("-inf")), logits)

        if temperature > 0.0:
            probas = torch.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probas, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and (idx_next == eos_id).all():
            break

        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def next_token_distribution(model, idx, context_size, temperature=1.0, top_k=None, k_show=5):
    """Inspeciona um único passo: devolve os k_show tokens mais prováveis e suas probabilidades."""
    with torch.no_grad():
        logits = model(idx[:, -context_size:])[:, -1, :]
    if top_k is not None:
        top_logits, _ = torch.topk(logits, top_k)
        logits = torch.where(logits < top_logits[:, -1:], torch.full_like(logits, float("-inf")), logits)
    probas = torch.softmax(logits / max(temperature, 1e-8), dim=-1)
    values, indices = torch.topk(probas, k_show, dim=-1)
    return values.squeeze(0), indices.squeeze(0)


if __name__ == "__main__":
    import tiktoken
    from src.gpt.config import GPT_CONFIG_124M
    from src.gpt.gpt_model import GPTModel

    tokenizer = tiktoken.get_encoding("gpt2")
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()  # desliga o dropout durante a inferência

    start_context = "Hello, I am"
    encoded = text_to_token_ids(start_context, tokenizer)
    print("=== Demonstração: geração autoregressiva (modelo sem treino) ===")
    print(f"Contexto inicial: {start_context!r} -> {encoded.tolist()}  shape {tuple(encoded.shape)}")

    values, indices = next_token_distribution(model, encoded, GPT_CONFIG_124M["context_length"])
    print("\nTop-5 candidatos para o próximo token:")
    for p, i in zip(values.tolist(), indices.tolist()):
        print(f"  {i:>6}  {tokenizer.decode([i])!r:<16} p={p:.6f}")

    out = generate_text_simple(model, encoded, max_new_tokens=6,
                               context_size=GPT_CONFIG_124M["context_length"])
    print(f"\nGreedy:  {out.tolist()}")
    print(f"Texto:   {token_ids_to_text(out, tokenizer)!r}")

    torch.manual_seed(123)
    out = generate(model, encoded, max_new_tokens=6,
                   context_size=GPT_CONFIG_124M["context_length"],
                   temperature=1.4, top_k=25)
    print(f"\nT=1.4, top-k=25: {token_ids_to_text(out, tokenizer)!r}")
    print("\nSem treino os pesos são aleatórios, então o texto não tem sentido (esperado).")
