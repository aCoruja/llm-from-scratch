"""
 Estratégias de Geração de Texto


Estratégias implementadas:
  1. Greedy Decoding     — Argmax: seleciona sempre o token mais provável.
  2. Temperature Scaling — Controla "suavidade" da distribuição antes de amostrar.
  3. Top-k Sampling      — Restringe amostragem aos k tokens mais prováveis.
  4. Top-k + Temperatura — Combinação das duas estratégias anteriores.
"""

import torch
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# ESTRATÉGIA 1: GREEDY DECODING
# ══════════════════════════════════════════════════════════════════════════════
def generate_greedy(model, idx, max_new_tokens, context_length):
    """
    Decodificação Gulosa (Greedy Decoding).

    A cada passo seleciona o token com MAIOR logit via argmax.
    - Completamente DETERMINÍSTICO: mesma entrada → mesma saída sempre.
    - Vantagem: coerente e rápido.
    - Desvantagem: tende a repetir tokens/frases em sequências longas,
      pois ignora toda distribuição de probabilidade fora do máximo.
    """
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_length:]
        with torch.no_grad():
            logits = model(idx_cond)[:, -1, :]          # logits do último token
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, next_token), dim=1)
    return idx


# ══════════════════════════════════════════════════════════════════════════════
# ESTRATÉGIA 2: TEMPERATURA
# ══════════════════════════════════════════════════════════════════════════════
def generate_with_temperature(model, idx, max_new_tokens, context_length,
                               temperature=1.0):
    """
    Amostragem com Escalonamento de Temperatura.

    Divide os logits por T antes do softmax:
      logits_scaled = logits / T  →  probs = softmax(logits_scaled)

    Efeito de T:
      T → 0   : concentra probabilidade no token mais provável (≈ greedy)
      T = 1.0 : distribuição original do modelo, sem alteração
      T > 1.0 : achata a distribuição, tokens raros ganham mais chance
      T → ∞   : distribuição uniforme (completamente aleatório)

    A seleção é ESTOCÁSTICA (multinomial), não argmax.
    """
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_length:]
        with torch.no_grad():
            logits = model(idx_cond)[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)
    return idx


# ══════════════════════════════════════════════════════════════════════════════
# ESTRATÉGIA 3 e 4: TOP-K SAMPLING (com temperatura opcional)
# ══════════════════════════════════════════════════════════════════════════════
def generate_top_k(model, idx, max_new_tokens, context_length,
                   top_k=10, temperature=1.0):
    """
    Top-k Sampling com Temperatura.

    Passo a passo:
      1. Calcula os logits para o próximo token.
      2. Aplica temperatura: logits = logits / T
      3. Mantém apenas os top_k maiores logits;
         todos os outros recebem -inf (probabilidade zero após softmax).
      4. Converte em probabilidades via softmax.
      5. Amostra multinomialmente entre os k candidatos.

    Efeito de k:
      k = 1       → idêntico ao greedy (só 1 candidato)
      k pequeno   → textos conservadores e coerentes
      k médio     → equilíbrio diversidade/coerência  ← recomendado
      k grande    → mais criativo, pode gerar incoerências
      k = vocab   → amostragem pura (sem filtro)

    A temperatura é aplicada ANTES do filtro top-k, controlando a
    concentração da distribuição ENTRE os k candidatos selecionados.
    """
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_length:]
        with torch.no_grad():
            logits = model(idx_cond)[:, -1, :] / temperature

        # Identifica os k maiores logits e seus índices
        top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)

        # Mascara todos os tokens fora do top-k com -inf
        logits_filtered = torch.full_like(logits, float("-inf"))
        logits_filtered.scatter_(1, top_k_indices, top_k_logits)

        probs = F.softmax(logits_filtered, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)
    return idx


# ══════════════════════════════════════════════════════════════════════════════
# COMPARAÇÃO SISTEMÁTICA (mesmo prompt para todas as estratégias)
# ══════════════════════════════════════════════════════════════════════════════
def run_comparison(model, tokenizer, device, context_length=64,
                   prompt="Every effort moved", max_new_tokens=30,
                   num_samples=3):
    """
    Executa a comparação sistemática entre todas as estratégias.

    Usa o MESMO prompt inicial para todas as estratégias, isolando
    o efeito da decodificação do efeito dos pesos do modelo.
    """
    model.eval()
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    decode = lambda ids: tokenizer.decode(ids.squeeze(0).tolist())

    # ── GREEDY ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(" ESTRATÉGIA 1: GREEDY DECODING (argmax — determinístico)")
    print("=" * 70)
    for i in range(num_samples):
        out = generate_greedy(model, input_ids.clone(), max_new_tokens, context_length)
        print(f"  Amostra {i+1}: '{decode(out)}'")
    print("  → Todas as amostras são IDÊNTICAS (determinístico).")

    # ── TEMPERATURA ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(" ESTRATÉGIA 2: TEMPERATURA (estocástico — sem filtro top-k)")
    print("=" * 70)
    for temp in [0.3, 0.7, 1.0, 1.5]:
        print(f"\n  ── Temperatura = {temp} ──")
        for i in range(num_samples):
            out = generate_with_temperature(
                model, input_ids.clone(), max_new_tokens, context_length, temp
            )
            print(f"    Amostra {i+1}: '{decode(out)}'")

    # ── TOP-K ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(" ESTRATÉGIA 3: TOP-K SAMPLING (temperatura = 1.0)")
    print("=" * 70)
    for k in [3, 10, 25, 50]:
        print(f"\n  ── Top-k = {k} ──")
        for i in range(num_samples):
            out = generate_top_k(
                model, input_ids.clone(), max_new_tokens, context_length,
                top_k=k, temperature=1.0
            )
            print(f"    Amostra {i+1}: '{decode(out)}'")

    # ── TOP-K + TEMPERATURA ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(" ESTRATÉGIA 4: TOP-K + TEMPERATURA (combinado — recomendado)")
    print("=" * 70)
    for k, t in [(5, 0.7), (10, 1.0), (10, 1.5), (25, 0.5)]:
        print(f"\n  ── Top-k = {k}, Temperatura = {t} ──")
        for i in range(num_samples):
            out = generate_top_k(
                model, input_ids.clone(), max_new_tokens, context_length,
                top_k=k, temperature=t
            )
            print(f"    Amostra {i+1}: '{decode(out)}'")

    # ── ANÁLISE ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(" ANÁLISE COMPARATIVA")
    print("=" * 70)
    print("""
  ┌──────────────────────┬─────────────┬─────────────┬──────────────────┐
  │ Estratégia           │ Diversidade │ Repetição   │ Coerência        │
  ├──────────────────────┼─────────────┼─────────────┼──────────────────┤
  │ Greedy               │ Nenhuma     │ Alta        │ Alta (repetitiva)│
  │ Temperatura T=0.3    │ Muito baixa │ Média-alta  │ Alta             │
  │ Temperatura T=0.7    │ Média       │ Baixa       │ Boa              │
  │ Temperatura T=1.0    │ Alta        │ Muito baixa │ Moderada         │
  │ Temperatura T=1.5    │ Muito alta  │ Muito baixa │ Baixa            │
  │ Top-k k=3            │ Baixa       │ Média       │ Alta             │
  │ Top-k k=10           │ Média       │ Baixa       │ Boa              │
  │ Top-k k=50           │ Alta        │ Muito baixa │ Moderada         │
  │ Top-k=10, T=0.7      │ Média       │ Baixa       │ Boa ← recomendado│
  └──────────────────────┴─────────────┴─────────────┴──────────────────┘

  DISTINÇÃO IMPORTANTE:
  As diferenças acima decorrem EXCLUSIVAMENTE da estratégia de decodificação.
  Os pesos do modelo são IDÊNTICOS em todos os casos — a decodificação NÃO
  os altera. O treinamento define a distribuição de probabilidades; a
  decodificação define como SELECIONAR um token a partir dessa distribuição.
    """)
