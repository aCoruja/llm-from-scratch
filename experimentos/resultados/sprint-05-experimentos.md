# Relatório Técnico — Avaliação e Experimentos de Treinamento do GPT

> **Sprint 05 — Capítulo 5 (*Pretraining on Unlabeled Data*)**
> Entregável 2 (Avaliação do modelo) e Entregável 4 (Experimentos de treinamento)

---

## 1. Objetivo

Este relatório registra e interpreta a **avaliação quantitativa** do modelo GPT
(Entregável 2) e os **experimentos de treinamento** (Entregável 4) realizados na
Sprint 5. Todos os números abaixo foram produzidos pelo script
[`experimentos/sprint-05-experimentos.py`](../sprint-05-experimentos.py) e
gravados em [`sprint-05-dados.json`](sprint-05-dados.json); as figuras estão em
[`figuras/`](figuras/).

### Ambiente e protocolo fixo

| Item | Valor |
|:--|:--|
| Corpus | `data/the-verdict.txt` (*The Verdict*, Edith Wharton, 1908 — domínio público) |
| Tamanho do corpus | 20 479 caracteres · **5 145 tokens BPE** (GPT-2) |
| Partição | 90 % treino / 10 % validação |
| Amostras | 143 (treino) · 15 (validação), janela deslizante |
| Janela de contexto | 64 tokens · stride 32 |
| Batch size | 4 (35 lotes/época no treino) |
| Otimizador | **AdamW**, weight decay 0,1 |
| Tokenizador | BPE GPT-2 (`tiktoken`), vocabulário V = 50 257 |
| Dispositivo | CPU · PyTorch 2.15.0+cpu · seed 42 |
| Contexto de geração | `"Every effort moved"` (greedy) |

O corpus é propositalmente pequeno (o canônico do livro), o que torna o
**sobreajuste** o fenômeno central a observar: o modelo tem capacidade de sobra
para **memorizar** 5 mil tokens.

---

## 2. Entregável 2 — Avaliação do modelo

### 2.1 Como a cross-entropy mede a qualidade da previsão

A cada posição, o modelo produz logits `(B, T, V)` e a tarefa é prever o próximo
token. A perda é a **entropia cruzada** entre a distribuição prevista (após
softmax) e o token verdadeiro (one-hot):

$$
\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \log p_\theta\!\left(y_i \mid x_{<i}\right)
$$

É o **log-verossimilhança negativo médio** do token correto. Se o modelo
atribui probabilidade alta ao token certo, $\log p \to 0$ e a perda é pequena;
se atribui probabilidade baixa, a perda cresce. A implementação está em
[`src/training/loss.py`](../../src/training/loss.py) (`calc_loss_batch`,
`calc_loss_loader`), usando `torch.nn.functional.cross_entropy` sobre os logits
achatados para `(B·T, V)` e os alvos para `(B·T,)`.

### 2.2 Perplexidade

$$
\mathrm{PPL} = e^{\mathcal{L}}
$$

A perplexidade é a **cross-entropy exponenciada**. Interpretação: o número
efetivo de tokens entre os quais o modelo "hesita" em cada passo. PPL = 1
significa certeza total; PPL = V significa indecisão total (escolha uniforme
sobre o vocabulário). Implementada em `calc_perplexity`.

### 2.3 Estado inicial — o modelo sem treino

Antes de qualquer época, com pesos aleatórios, os logits são quase uniformes.
A previsão é, na prática, um sorteio uniforme sobre V tokens, então a teoria
prevê:

$$
\mathcal{L}_0 \approx \ln V = \ln 50257 = 10{,}825
\qquad
\mathrm{PPL}_0 \approx V = 50257
$$

**Medido** (média das 6 corridas, pesos aleatórios):

| Conjunto | Perda inicial | Perplexidade inicial |
|:--|--:|--:|
| Treino | **10,97** | ≈ 58 000 |
| Validação | **10,97** | ≈ 58 000 |

A perda medida (10,97) fica muito próxima do limite teórico $\ln V = 10,83$, e a
perplexidade (~58 000) é da ordem do tamanho do vocabulário. **Isto valida a
avaliação**: um modelo não treinado tem a incerteza máxima possível, e a perda
inicial nada tem de arbitrária — é $\ln V$.

### 2.4 Evolução durante o treinamento

A perda de treino cai de forma monótona em todas as configurações. A perda de
**validação** cai no início e depois **sobe** — a assinatura do sobreajuste
(Seção 4). A tabela completa por época está no JSON; a Seção 3 resume os pontos
de interesse.

---

## 3. Entregável 4 — Experimentos de treinamento

Três eixos foram variados, sempre sobre o mesmo corpus e protocolo:

1. **Taxa de aprendizado** (`lr`) — convergência e estabilidade;
2. **Capacidade do modelo** (nº de parâmetros) — convergência × custo;
3. **Número de épocas** — sobreajuste (treino × validação).

### Experimento 1 — Taxa de aprendizado

Modelo fixo: `emb_dim=128`, `n_layers=2`, `n_heads=4` (**13,27 M** parâmetros),
15 épocas. Ver figura [`figuras/sprint-05-lr-sweep.png`](figuras/sprint-05-lr-sweep.png).

| `lr` | Treino (final) | PPL treino | Val (final) | PPL val | Menor val (época) | Tempo |
|:--|--:|--:|--:|--:|:--:|--:|
| 1·10⁻⁴ | 5,98 | 396 | 6,59 | 730 | 6,59 (**ép. 15**) | 138 s |
| 5·10⁻⁴ | 2,57 | 13,1 | 6,15 | 469 | 6,08 (**ép. 12**) | 125 s |
| 1·10⁻³ | 0,49 | 1,6 | 7,29 | 1 463 | 6,11 (**ép. 7**) | 128 s |
| 5·10⁻³ | 0,065 | 1,1 | 9,55 | 13 985 | 6,21 (**ép. 3**) | 129 s |

**Leitura:**

- **lr = 1·10⁻⁴ (baixa)**: converge devagar; em 15 épocas ainda não atingiu o
  melhor ponto (a menor val é a última época). *Subajuste*. O texto gerado é
  degenerado (`"Every effort moved,,,,,,, the"` + quebras de linha), sinal de que
  o modelo ainda não aprendeu a estrutura da língua.
- **lr = 5·10⁻⁴ (equilibrada)**: melhor compromisso. Menor perda de validação
  (6,08) entre as corridas de 15 épocas, atingida na **época 12**. Texto já com
  sintaxe reconhecível.
- **lr = 1·10⁻³ (alta)**: o treino despenca (PPL 1,6), mas a validação piora
  (7,29); o melhor ponto de validação ocorre cedo (**época 7**) — converge rápido
  e **sobreajusta rápido**.
- **lr = 5·10⁻³ (excessiva)**: o treino colapsa para ~0 (memorização completa),
  mas a validação **explode** (PPL ≈ 14 000); o melhor ponto é já na **época 3**.
  Regime instável/divergente para a generalização.

**Padrão central:** quanto maior a `lr`, mais rápido a perda de treino cai, mas
mais cedo e mais forte vem o sobreajuste — a época de menor validação recua de
**15 → 12 → 7 → 3** à medida que a taxa sobe.

### Experimento 2 — Capacidade do modelo

Mesma `lr` (5·10⁻⁴), 15 épocas. Ver [`figuras/sprint-05-capacidade.png`](figuras/sprint-05-capacidade.png).

| Modelo | emb / L | Parâmetros | Menor val (época) | Val final | Tempo/época |
|:--|:--:|--:|:--:|--:|--:|
| baseline | 128 / 2 | 13,27 M | 6,08 (ép. 12) | 6,15 | 8,3 s |
| maior | 256 / 4 | 28,90 M | **5,98 (ép. 5)** | 7,30 | **15,6 s** |

**Leitura:** o modelo maior atinge uma perda de validação **menor** (5,98 vs
6,08) e **mais cedo** (época 5 vs 12), mas **sobreajusta antes** e custa **~1,9×**
mais tempo por época. Em um corpus de 5 mil tokens, capacidade extra se traduz
sobretudo em memorização mais veloz, não em melhor generalização sustentada.

### Experimento 3 — Sobreajuste (número de épocas)

Modelo baseline (128/2, lr 5·10⁻⁴), **40 épocas**. Ver
[`figuras/sprint-05-overfitting.png`](figuras/sprint-05-overfitting.png).

| Época | Treino (loss / PPL) | Validação (loss / PPL) | Gap (val − treino) |
|:--:|--:|--:|--:|
| 0 (inicial) | 10,97 / 58 000 | 10,97 / 58 000 | ~0 |
| **12** (menor val) | 3,48 / 32,5 | **6,08 / 438** | 2,60 |
| 20 | 1,34 / 3,8 | 6,57 / 712 | 5,23 |
| 30 | 0,30 / 1,4 | 7,48 / 1 767 | 7,17 |
| 40 (final) | **0,088 / 1,1** | **8,21 / 3 682** | **8,12** |

**Leitura:** o treino cai até PPL ≈ 1,1 — o modelo praticamente **memorizou** o
corpus. A validação, porém, atinge o mínimo na **época 12** e a partir daí **sobe
continuamente**: o *gap* entre treino e validação cresce de ~0 para **8,1**. A
amostra gerada na época 40 é **texto literal do corpus** (`"Every effort moved
room at the price of loathing my work." I made a deprecating gesture, which he
negatived`), confirmando memorização em vez de generalização. O ponto de
**parada antecipada** (*early stopping*) correto fica em torno da **época 12**.

---

## 4. Treino × Validação — diagnóstico de sobreajuste

- **Antes do ponto ótimo** (≈ época 12): treino e validação caem juntos — o
  modelo aprende padrões genéricos da língua que valem fora do treino.
- **Depois do ponto ótimo**: treino continua caindo, validação sobe — o modelo
  passa a decorar tokens específicos do conjunto de treino.
- O sinal mensurável é o **gap val − treino** e a **inflexão da curva de
  validação**. Em todas as corridas longas ou com `lr` alta, esse gap domina.

**Mitigações coerentes com os dados:** parada antecipada (~época 12), `lr`
moderada (5·10⁻⁴), menos capacidade (o modelo pequeno generaliza de forma
comparável por muito mais tempo) e, acima de tudo, **mais dados** — 5 mil tokens
são insuficientes para evitar memorização com 13 M+ parâmetros.

---

## 5. Custo computacional

| Modelo | Parâmetros | Tempo/época (CPU, 12 threads) |
|:--|--:|--:|
| 128 / 2 camadas | 13,27 M | **8,3 s** |
| 256 / 4 camadas | 28,90 M | **15,6 s** |

O custo por época escala com a profundidade e a largura (os blocos Transformer
dominam o cálculo). Dobrar aproximadamente a capacidade dobra o tempo. Como todo
o treino roda em CPU, a versão reduzida foi a adotada como baseline para iterar
com rapidez.

---

## 6. Conclusões

1. **Avaliação (Entregável 2):** a cross-entropy inicial (10,97) confirma a
   teoria ($\ln V = 10,83$) e a perplexidade inicial (~58 000 ≈ V) mostra que o
   modelo sem treino é maximamente incerto. A perda e a perplexidade são
   métricas fiéis da evolução do aprendizado.
2. **Taxa de aprendizado (Entregável 4):** governa a velocidade de convergência
   e a velocidade do sobreajuste. 5·10⁻⁴ foi o melhor equilíbrio; 5·10⁻³ é
   instável para a generalização.
3. **Capacidade:** mais parâmetros convergem mais rápido e custam mais, mas
   sobreajustam antes neste corpus pequeno.
4. **Sobreajuste:** inevitável com 40 épocas sobre 5 mil tokens; o treino chega a
   PPL 1,1 (memorização) enquanto a validação triplica. *Early stopping* ≈ época
   12.
5. **Limitação principal:** o tamanho do corpus. O comportamento qualitativo
   (convergência, divergência treino/validação, efeito da `lr`) é o esperado e
   reproduz o do livro, mas a qualidade absoluta do texto é limitada pelos dados
   e pelo treino em CPU.

---

## 7. Reprodução

```bash
# a partir da raiz do repositório, com o ambiente virtual ativo
PYTHONPATH=. python experimentos/sprint-05-experimentos.py
```

Gera `sprint-05-dados.json` e as três figuras em `figuras/`. Seed fixa (42),
resultados determinísticos em CPU. O treino completo com checkpoint
(Entregáveis 3 e 6) está em [`src/training/train.py`](../../src/training/train.py).
