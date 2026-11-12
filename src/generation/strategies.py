"""
Estratégias de Geração de Texto

Estratégias implementadas:
  1. Greedy Decoding     — Argmax: seleciona sempre o token mais provável.
  2. Temperature Scaling — Controla a "suavidade" da distribuição antes de amostrar.
  3. Top-k Sampling      — Restringe a amostragem aos k tokens mais prováveis.
  4. Top-k + Temperatura — Combinação das duas anteriores.

Além disso: métricas objetivas (diversidade, repetição, memorização) e a
comparação entre estados de pesos, para separar o efeito da ESTRATÉGIA do
efeito do TREINAMENTO.
"""

import torch
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# ESTRATÉGIA 1: GREEDY DECODING
# ══════════════════════════════════════════════════════════════════════════════
def generate_greedy(model, idx, max_new_tokens, context_length):
    """
    Decodificação Gulosa: a cada passo seleciona o token de MAIOR logit (argmax).
    - Determinística: mesma entrada → mesma saída.
    - Tende a repetir tokens/frases em sequências longas.
    """
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_length:]
        with torch.no_grad():
            logits = model(idx_cond)[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, next_token), dim=1)
    return idx


# ══════════════════════════════════════════════════════════════════════════════
# ESTRATÉGIA 2: TEMPERATURA
# ══════════════════════════════════════════════════════════════════════════════
def generate_with_temperature(model, idx, max_new_tokens, context_length,
                              temperature=1.0):
    """
    Amostragem com escalonamento de temperatura: softmax(logits / T) + sorteio.
      T → 0   : concentra a probabilidade no token mais provável (≈ greedy)
      T = 1.0 : distribuição original do modelo
      T > 1.0 : achata a distribuição; tokens raros ganham chance
    T <= 0 é tratado como greedy (evita divisão por zero).
    """
    if temperature <= 0:
        return generate_greedy(model, idx, max_new_tokens, context_length)

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
    Top-k Sampling com temperatura.
      1. logits = logits / T
      2. mantém só os k maiores logits; o resto vira -inf
      3. softmax e sorteio entre os k candidatos
    k = 1 (ou T <= 0) equivale a greedy; k >= vocabulário equivale a
    amostragem pura. A temperatura é aplicada antes do filtro.
    """
    if temperature <= 0 or top_k == 1:
        return generate_greedy(model, idx, max_new_tokens, context_length)

    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_length:]
        with torch.no_grad():
            logits = model(idx_cond)[:, -1, :] / temperature

        k = min(top_k, logits.size(-1))
        top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)

        logits_filtered = torch.full_like(logits, float("-inf"))
        logits_filtered.scatter_(1, top_k_indices, top_k_logits)

        probs = F.softmax(logits_filtered, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)
    return idx


# ══════════════════════════════════════════════════════════════════════════════
# MÉTRICAS
# ══════════════════════════════════════════════════════════════════════════════
def _ngrams(ids, n):
    return [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def distinct_n(ids, n):
    """Fração de n-gramas únicos: 1.0 = sem repetição; perto de 0 = muita repetição."""
    g = _ngrams(ids, n)
    return len(set(g)) / len(g) if g else 0.0


def _overlap(ids, corpus_ngrams, n=4):
    """Fração dos n-gramas gerados que existem literalmente no corpus de treino
    (indício de memorização: efeito dos PESOS, não da estratégia)."""
    g = _ngrams(ids, n)
    return sum(x in corpus_ngrams for x in g) / len(g) if g else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# COMPARAÇÃO SISTEMÁTICA (mesmo prompt e mesmas seeds para todas as estratégias)
# ══════════════════════════════════════════════════════════════════════════════
def _strategies():
    s = [("greedy", generate_greedy, {})]
    s += [(f"T={t}", generate_with_temperature, {"temperature": t})
          for t in (0.3, 0.7, 1.0, 1.5)]
    s += [(f"top-k={k}", generate_top_k, {"top_k": k, "temperature": 1.0})
          for k in (3, 10, 25, 50)]
    s += [(f"top-k={k}, T={t}", generate_top_k, {"top_k": k, "temperature": t})
          for k, t in ((5, 0.7), (10, 1.0), (10, 1.5), (25, 0.5))]
    return s


def _print_table(titulo, rows):
    print(f"\n### {titulo}\n")
    print("| Estratégia | distinct-1 | distinct-2 | repet. trigramas | "
          "distinct-2 entre amostras | amostras únicas | overlap 4-gramas c/ corpus |")
    print("|:--|--:|--:|--:|--:|--:|--:|")
    for r in rows:
        ov = "—" if r["overlap4"] is None else f"{r['overlap4']:.2f}"
        print(f"| {r['estrategia']} | {r['distinct1']:.2f} | {r['distinct2']:.2f} | "
              f"{r['rep3']:.2f} | {r['inter_distinct2']:.2f} | {r['amostras_unicas']:.2f} | {ov} |")


def run_comparison(model, tokenizer, device, context_length=64,
                   prompt="Every effort moved", max_new_tokens=60,
                   num_samples=5, seed=123, corpus_text=None, verbose=True,
                   titulo="Comparação de estratégias"):
    """
    Compara todas as estratégias com o MESMO prompt e as mesmas seeds (seed + i).
    O greedy roda uma vez (é determinístico). Retorna uma lista de dicts.
    corpus_text (texto de TREINO) habilita a métrica de memorização.
    """
    model.eval()
    ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    n0 = ids.shape[1]
    corpus4 = (set(_ngrams(tokenizer.encode(corpus_text, allowed_special={"<|endoftext|>"}), 4))
               if corpus_text else None)

    rows = []
    for name, fn, kw in _strategies():
        runs = 1 if fn is generate_greedy else num_samples
        gens = []
        for i in range(runs):
            torch.manual_seed(seed + i)
            out = fn(model, ids.clone(), max_new_tokens, context_length, **kw)
            gens.append(out[0].tolist()[n0:])

        bigrams = [b for g in gens for b in _ngrams(g, 2)]
        rows.append({
            "estrategia": name,
            "distinct1": _mean([distinct_n(g, 1) for g in gens]),
            "distinct2": _mean([distinct_n(g, 2) for g in gens]),
            "rep3": _mean([1 - distinct_n(g, 3) for g in gens]),
            "inter_distinct2": len(set(bigrams)) / len(bigrams) if bigrams else 0.0,
            "amostras_unicas": len({tuple(g) for g in gens}) / runs,
            "overlap4": (_mean([_overlap(g, corpus4) for g in gens])
                         if corpus4 is not None else None),
            "texto": tokenizer.decode(gens[0]),
        })

    if verbose:
        _print_table(titulo, rows)
        print("\nAmostras (1ª de cada estratégia):\n")
        for r in rows:
            print(f"**{r['estrategia']}**: {r['texto']!r}\n")
    return rows


def compare_weights(models, tokenizer, device, **kw):
    """
    models: {"nome": modelo}. Mesmo prompt, mesmas seeds e mesmas estratégias.
      - Efeito da ESTRATÉGIA: compare as LINHAS de uma mesma tabela (pesos fixos).
      - Efeito do TREINAMENTO: compare as COLUNAS da tabela cruzada (estratégia fixa).
    """
    res = {n: run_comparison(m, tokenizer, device, verbose=False, **kw)
           for n, m in models.items()}
    for n, rows in res.items():
        _print_table(f"Pesos: {n}", rows)

    nomes = list(res)
    for chave, titulo in (("rep3", "repetição de trigramas"),
                          ("inter_distinct2", "diversidade entre amostras"),
                          ("overlap4", "overlap de 4-gramas com o corpus")):
        print(f"\n### {titulo}: mesma estratégia, pesos diferentes\n")
        print("| Estratégia | " + " | ".join(nomes) + " |")
        print("|:--|" + "--:|" * len(nomes))
        for i, r in enumerate(res[nomes[0]]):
            vals = [res[n][i][chave] for n in nomes]
            print(f"| {r['estrategia']} | "
                  + " | ".join("—" if v is None else f"{v:.2f}" for v in vals) + " |")

    print("\nAmostras do último estado de pesos (1ª de cada estratégia):\n")
    for r in res[nomes[-1]]:
        print(f"**{r['estrategia']}**: {r['texto']!r}\n")
    return res
