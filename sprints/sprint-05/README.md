# Sprint 05 — Treinamento e Avaliação do GPT

**Capítulo:** 5 — *Pretraining on Unlabeled Data* (*Build a Large Language Model (From Scratch)*, Sebastian Raschka).
**Foco:** função de perda (cross-entropy), perplexidade, rotina de treinamento (épocas, lotes, backward, otimizador), experimentos de convergência e sobreajuste, estratégias de geração e salvamento/carregamento dos pesos.

Esta sprint fecha o ciclo de *pré-treinamento*: a Sprint 4 deixou o modelo GPT
completo, porém com pesos aleatórios (texto sem sentido). Aqui o modelo
**aprende** a prever o próximo token, é **avaliado** quantitativamente e tem seu
estado **salvo** para inferência posterior.

> **Documentação técnica (Entregável 7).** Este README consolida o código, os
> experimentos, os registros de perda, as amostras de texto e as conclusões. O
> relatório completo de resultados está em
> [`experimentos/resultados/sprint-05-experimentos.md`](../../experimentos/resultados/sprint-05-experimentos.md),
> com dados brutos em
> [`sprint-05-dados.json`](../../experimentos/resultados/sprint-05-dados.json) e
> figuras em [`experimentos/resultados/figuras/`](../../experimentos/resultados/figuras/).

---

## Mapa dos entregáveis

| Entregável | Descrição | Arquivos |
|:--|:--|:--|
| **2 — Avaliação** | Cross-entropy, perda treino/validação, perplexidade | [`src/training/loss.py`](../../src/training/loss.py) |
| **3 — Treinamento** | Loop de épocas/lotes, backward, AdamW | [`src/training/trainer.py`](../../src/training/trainer.py), [`src/training/train.py`](../../src/training/train.py) |
| **4 — Experimentos** | Taxa de aprendizado, capacidade, sobreajuste | [`experimentos/sprint-05-experimentos.py`](../../experimentos/sprint-05-experimentos.py), [`resultados/sprint-05-experimentos.md`](../../experimentos/resultados/sprint-05-experimentos.md) |
| **5 — Geração** | Greedy, temperatura, top-k | [`src/generation/strategies.py`](../../src/generation/strategies.py) |
| **6 — Checkpoint** | Salvamento/carregamento e inferência | [`src/training/checkpoint.py`](../../src/training/checkpoint.py) |
| **7 — Documentação** | Este README + relatório de resultados | este arquivo |

> Esta entrega focou os **Entregáveis 2, 4 e 7**. Os módulos 3, 5 e 6 já estavam
> implementados em commits anteriores da Sprint 5 e são reutilizados e
> documentados aqui; um ajuste de compatibilidade foi aplicado ao Entregável 6
> (ver [Correções](#correções)).

---

## Entregável 2 — Avaliação do modelo

### Cross-entropy

Para cada posição, o modelo prevê o próximo token. A perda é o log-verossimilhança
negativo médio do token correto:

```
L = − (1/N) Σ log p_θ(y_i | x_<i)
```

implementada em `calc_loss_batch` / `calc_loss_loader` com
`F.cross_entropy(logits.flatten(0,1), targets.flatten())`.

### Perplexidade

`PPL = exp(L)` — o número efetivo de tokens entre os quais o modelo hesita por
passo. `calc_perplexity(loss)`.

### Estado inicial (pesos aleatórios)

| Conjunto | Perda | Perplexidade |
|:--|--:|--:|
| Treino | 10,97 | ≈ 58 000 |
| Validação | 10,97 | ≈ 58 000 |

Isso **confirma a teoria**: um modelo aleatório prevê de forma quase uniforme,
então a perda tende a `ln(V) = ln(50257) = 10,83` e a perplexidade ao tamanho do
vocabulário. A avaliação está correta — a perda inicial não é arbitrária.

---

## Entregável 3 — Rotina de treinamento

`train_model` ([`trainer.py`](../../src/training/trainer.py)) executa, por época e
por lote:

1. `optimizer.zero_grad()` — zera gradientes acumulados;
2. `loss = calc_loss_batch(...)` — forward + cross-entropy;
3. `loss.backward()` — retropropagação;
4. `optimizer.step()` — atualização dos pesos (AdamW).

Ao fim de cada época registra perda e perplexidade de treino e validação, e gera
uma amostra para acompanhamento qualitativo. O script principal
[`train.py`](../../src/training/train.py) registra todos os parâmetros (épocas,
batch, `lr`, otimizador, seed, arquitetura) e guarda três estados de pesos (sem
treino, pouco treinado, treinado) para o Entregável 5.

---

## Entregável 4 — Experimentos

Resumo (detalhes e gráficos no
[relatório](../../experimentos/resultados/sprint-05-experimentos.md)):

- **Taxa de aprendizado** (15 épocas, modelo 13,3 M): `lr` maior → convergência
  mais rápida, porém sobreajuste mais cedo. A época de menor perda de validação
  recua de **15 → 12 → 7 → 3** conforme `lr` sobe de 1·10⁻⁴ a 5·10⁻³.
  `lr = 5·10⁻⁴` foi o melhor equilíbrio; `5·10⁻³` é instável (val PPL ≈ 14 000).
- **Capacidade**: o modelo maior (28,9 M) atinge perda de validação menor e mais
  cedo (5,98 na época 5), mas sobreajusta antes e custa ~1,9× mais por época.
- **Sobreajuste** (40 épocas): treino → PPL 1,1 (memorização); validação atinge
  o mínimo na **época 12** e depois sobe (PPL 438 → 3 682). *Early stopping* ≈
  época 12. A amostra final é **texto literal do corpus**.

### Amostras de texto geradas (contexto `"Every effort moved"`, greedy)

| Configuração | Amostra |
|:--|:--|
| Sem treino | `Every effort movedcision Playstation perceiveDateUb Ket Guantanamo...` (sem sentido) |
| `lr=1e-4`, 15 ép. (subajuste) | `Every effort moved,,,,,,, the` + quebras |
| `lr=5e-4`, 15 ép. | `Every effort moved, the picture, and he was a little of the picture...` |
| `lr=5e-4`, 40 ép. (sobreajuste) | `Every effort moved room at the price of loathing my work."...` (literal) |

---

## Entregável 5 — Estratégias de geração

Em [`src/generation/strategies.py`](../../src/generation/strategies.py): greedy,
amostragem por temperatura e top-k, com função comparativa `compare_weights` que
distingue o efeito do **treino dos pesos** do efeito da **estratégia de
decodificação**.

---

## Entregável 6 — Salvamento e carregamento

[`src/training/checkpoint.py`](../../src/training/checkpoint.py): `save_checkpoint`
(pesos + configuração + histórico + metadados), `load_checkpoint` (reconstrói a
arquitetura a partir da config salva), `verify_checkpoint` (prova que o modelo
carregado é idêntico) e `demo_inference` (gera texto com os pesos carregados).

---

## Configuração de referência

| Parâmetro | Valor |
|:--|:--|
| Corpus | `data/the-verdict.txt` (5 145 tokens BPE) |
| Partição | 90 % treino / 10 % validação |
| Contexto / stride | 64 / 32 tokens |
| Batch size | 4 |
| Otimizador | AdamW, weight decay 0,1 |
| Taxa de aprendizado | 5·10⁻⁴ (baseline) |
| Arquitetura baseline | emb 128, 2 camadas, 4 cabeças (13,27 M) |
| Seed | 42 |
| Dispositivo | CPU (PyTorch 2.15.0+cpu) |

---

## Limitações

- **Corpus pequeno** (5 mil tokens): o sobreajuste é inevitável com modelos de
  13 M+ parâmetros; a validação degrada após ~12 épocas. É a limitação dominante.
- **Treino em CPU**: inviabiliza o GPT-2 124M completo; usou-se uma versão
  reduzida. Os resultados são qualitativamente fiéis ao livro, mas a qualidade
  absoluta do texto é modesta.
- O corpus final do projeto (texto técnico de eletrônica em PT-BR, ver README
  raiz) ainda não foi incorporado ao treino.

---

## Como reproduzir

```bash
# ambiente
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch tiktoken numpy matplotlib pytest

# experimentos (Entregáveis 2 e 4) — gera JSON + figuras
PYTHONPATH=. python experimentos/sprint-05-experimentos.py

# treino completo com checkpoint (Entregáveis 3, 5 e 6)
PYTHONPATH=. python src/training/train.py

# testes
PYTHONPATH=. pytest tests/ -q
```

---

## Correções

- **Entregável 6 / `checkpoint.py`**: `load_checkpoint` passou a usar
  `torch.load(..., weights_only=False)`. A partir do PyTorch 2.6 o padrão mudou
  para `weights_only=True`, que recusaria o checkpoint (ele guarda a configuração
  e a versão do PyTorch, não só os pesos). O carregamento é de um checkpoint
  próprio e confiável. Com o ajuste, `tests/test_generation.py` passa 9/9.
