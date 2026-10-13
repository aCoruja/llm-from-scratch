# Sprint 04

**Capítulo:** 4 — *Implementing a GPT Model from Scratch to Generate Text* (e seção 5.3, *Decoding strategies*).
**Foco:** Transformer Block (LayerNorm, FeedForward com GELU, conexões residuais), arquitetura GPT completa e geração autoregressiva de texto.
**Entrega:** 29/10.

## O que foi entregue

| Entregável | Arquivos |
|:--|:--|
| 2 — Transformer Block | `src/transformer/layer_norm.py`, `src/transformer/feed_forward.py`, `src/transformer/transformer_block.py`, `tests/test_transformer.py` |
| 3 — Modelo GPT | `src/gpt/config.py`, `src/gpt/gpt_model.py`, `tests/test_gpt.py` |
| 4 — Inferência e geração | `src/generation/generate.py`, `tests/test_generation.py` |

As notas de leitura e a análise desta sprint estão no fim deste arquivo, nas seções [Notas de leitura orientada](#notas-de-leitura-orientada) e [Análise dos resultados](#análise-dos-resultados).

O `MultiHeadAttention` reutilizado é o da Sprint 3 (`src/attention/multi_head_attention.py`), e os embeddings seguem a mesma ideia da Sprint 2 (token + posição), agora como camadas internas do modelo.

Notação usada abaixo: **B** = batch, **T** = número de tokens, **D** = `emb_dim`, **H** = `n_heads`, **V** = `vocab_size`. Valores numéricos referem-se ao `GPT_CONFIG_124M` (D = 768, H = 12, 12 blocos, V = 50257).

---

## Entregável 2 — Transformer Block

### Estrutura (Pre-LayerNorm)

```
x ─────────────────────────────┐
│                              │ (residual)
LayerNorm 1                    │
Masked Multi-Head Attention    │
Dropout                        │
(+) ◄──────────────────────────┘
│
x ─────────────────────────────┐
│                              │ (residual)
LayerNorm 2                    │
FeedForward                    │
Dropout                        │
(+) ◄──────────────────────────┘
│
saída
```

Em código (`TransformerBlock.forward`):

```python
x = x + drop(att(norm1(x)))
x = x + drop(ff(norm2(x)))
```

### Função de cada componente

| Componente | O que faz | Por que está no bloco |
|:--|:--|:--|
| **LayerNorm** | Normaliza cada vetor de token para média 0 e variância 1 na dimensão D; depois aplica `scale` e `shift` treináveis. | Mantém as ativações em uma escala estável ao longo de muitos blocos, evitando gradientes que explodem ou desaparecem. Aplicada *antes* de cada subcamada (Pre-LN), que treina de forma mais estável que a Post-LN do Transformer original. |
| **Masked Multi-Head Attention** | Cada token combina informação dos tokens anteriores (máscara causal), em H cabeças paralelas. | É a única parte do bloco que troca informação *entre* tokens. |
| **FeedForward** | MLP de duas camadas aplicada a cada token separadamente: expande D → 4D, aplica GELU, contrai 4D → D. | Processa o que a atenção reuniu; concentra cerca de 2/3 dos parâmetros do bloco. |
| **GELU** | Ativação suave: ≈ identidade para x positivo, ≈ 0 para x muito negativo, levemente negativa perto de zero. | Diferente da ReLU, tem gradiente não nulo para entradas negativas, o que facilita a otimização. |
| **Dropout** | Zera aleatoriamente uma fração (`drop_rate`) das ativações durante o treino. | Regularização. Desligado com `model.eval()`. |
| **Conexão residual** | Soma a entrada da subcamada à sua saída (`x + f(x)`). | Cria um caminho direto para o gradiente; sem ela as primeiras camadas recebem gradiente quase nulo (ver `test_residual_mantem_gradiente`). Também exige que entrada e saída tenham o mesmo formato. |

### Dimensões dos tensores

| Etapa | Formato | 124M |
|:--|:--|:--|
| Entrada do bloco | (B, T, D) | (B, T, 768) |
| LayerNorm 1 | (B, T, D) | (B, T, 768) |
| Q, K, V após projeção e divisão em cabeças | (B, H, T, D/H) | (B, 12, T, 64) |
| Pesos de atenção | (B, H, T, T) | (B, 12, T, T) |
| Saída da atenção (cabeças concatenadas + `out_proj`) | (B, T, D) | (B, T, 768) |
| Após residual 1 | (B, T, D) | (B, T, 768) |
| FeedForward: 1ª Linear + GELU | (B, T, 4D) | (B, T, 3072) |
| FeedForward: 2ª Linear | (B, T, D) | (B, T, 768) |
| Saída do bloco (após residual 2) | (B, T, D) | (B, T, 768) |

### Parâmetros de um bloco (124M)

| Parte | Cálculo | Parâmetros |
|:--|:--|--:|
| MHA | W_q, W_k, W_v: 3 × 768 × 768 + `out_proj`: 768 × 768 + 768 | 2.360.064 |
| FeedForward | 768 × 3072 + 3072 + 3072 × 768 + 768 | 4.722.432 |
| 2 LayerNorms | 2 × (768 + 768) | 3.072 |
| **Total** | | **7.085.568** |

A FeedForward tem o dobro de parâmetros da atenção (exercício 4.1 do livro).

---

## Entregável 3 — Modelo GPT

### Organização (`GPTModel`)

```python
self.tok_emb    = nn.Embedding(V, D)              # Token Embedding
self.pos_emb    = nn.Embedding(context_length, D) # Positional Embedding
self.drop_emb   = nn.Dropout(drop_rate)
self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(n_layers)])
self.final_norm = LayerNorm(D)
self.out_head   = nn.Linear(D, V, bias=False)     # Language Model Head
```

### Fluxo de uma sequência

| Etapa | Formato | Exemplo (2 textos de 4 tokens) |
|:--|:--|:--|
| Token IDs | (B, T) | (2, 4) |
| Token embeddings | (B, T, D) | (2, 4, 768) |
| + Positional embeddings (`arange(T)` → (T, D), broadcast) | (B, T, D) | (2, 4, 768) |
| Dropout | (B, T, D) | (2, 4, 768) |
| 12 × TransformerBlock | (B, T, D) | (2, 4, 768) |
| LayerNorm final | (B, T, D) | (2, 4, 768) |
| Language Model Head → **logits** | (B, T, V) | (2, 4, 50257) |

A posição `t` dos logits pontua o próximo token depois de `x[0..t]`. Por causa da máscara causal, ela não depende dos tokens seguintes.

### Tamanho do modelo

| Medida | Valor |
|:--|--:|
| Parâmetros totais | 163.009.536 |
| Parâmetros com *weight tying* (`out_head` compartilha pesos com `tok_emb`, como no GPT-2 original) | 124.412.160 |
| Memória em float32 | 621,83 MB |

Os dois embeddings de vocabulário (`tok_emb` e `out_head`, 50257 × 768 cada) somam 77,2 M parâmetros, quase metade do modelo. Esses valores conferem com o livro e são verificados em `tests/test_gpt.py`.

`GPT_CONFIG_MINI` (D = 128, 4 cabeças, 2 blocos, contexto 256) é uma versão reduzida da mesma arquitetura, usada nos testes e nos experimentos em CPU.

---

## Entregável 4 — Inferência e geração autoregressiva

### Procedimento (`generate_text_simple` / `generate`)

```
contexto: "Hello, I am" → [15496, 11, 314, 716]           idx (1, 4)
repetir max_new_tokens vezes:
    idx_cond = idx[:, -context_size:]                      recorta ao contexto máximo
    logits   = model(idx_cond)                             (1, T, 50257)
    logits   = logits[:, -1, :]                            (1, 50257)  só a última posição
    probas   = softmax(logits)                             (1, 50257)  soma = 1
    próximo  = argmax(probas)  ou  sorteio(probas)         (1, 1)
    idx      = concat(idx, próximo)                        (1, T + 1)
```

### Logits → probabilidades → próximo token

- **Logits** são as pontuações brutas que o Language Model Head devolve, um número real por token do vocabulário, sem limite de escala.
- **Softmax** converte esses números em uma distribuição de probabilidade: `p_i = exp(z_i) / Σ_j exp(z_j)`. Todos ficam entre 0 e 1 e somam 1.
- **Seleção do próximo token:**
  - *Greedy* (`temperature = 0`): pega o de maior probabilidade. Como a softmax preserva a ordem, `argmax(probas) = argmax(logits)`. O resultado é determinístico, mas tende a repetir trechos.
  - *Temperatura* (`temperature > 0`): sorteia de `softmax(logits / T)`. Com T < 1 a distribuição fica mais concentrada (mais conservadora); com T > 1 fica mais plana (mais variada e mais propensa a erros).
  - *Top-k*: antes da softmax, só os k maiores logits são mantidos e os demais viram `-inf` (probabilidade 0). Isso impede o sorteio de tokens muito improváveis.
- **Geração autoregressiva**: o token escolhido é concatenado à entrada e o modelo é executado de novo. Cada previsão passa a condicionar as seguintes. Quando a sequência passa de `context_length`, só os últimos tokens são mantidos.

`model.eval()` desliga o dropout, e `torch.no_grad()` dispensa o cálculo de gradientes. Os dois juntos caracterizam a **inferência**.

### Resultado com o modelo sem treino

```
Contexto inicial: 'Hello, I am' -> [[15496, 11, 314, 716]]

Top-5 candidatos para o próximo token:
   27018  ' Feature'       p=0.000205
   40179  '677'            p=0.000150
   42473  'TRUMP'          p=0.000143
   14888  ' participated'  p=0.000143
   22592  'retch'          p=0.000143

Greedy:  'Hello, I am Featureiman Byeswickattribute argue'
T=1.4, top-k=25: 'Hello, I am streaming Repair volunte Jew Thib Tank'
```

Com pesos aleatórios a distribuição é quase uniforme: a maior probabilidade (0,0002) fica perto de 1/50257 ≈ 0,00002. Por isso o texto não tem sentido. É o mesmo resultado do livro, e só passa a fazer sentido depois do treinamento (Sprint 5).

---

## Notas de leitura orientada

**Capítulo:** 4 — *Implementing a GPT Model from Scratch to Generate Text*, de *Build a Large Language Model (From Scratch)* (Sebastian Raschka, Manning, 2025), mais a seção 5.3 (*Decoding strategies*).
Estas notas priorizam como as peças se encaixam, não a reprodução do código (esse está em `src/transformer/`, `src/gpt/`, `src/generation/` e nos testes de `tests/`).

### O que faltava depois da Sprint 3

A Sprint 2 transformou texto em vetores (token + posição). A Sprint 3 deu a esses vetores um jeito de trocar informação entre si (atenção causal). Ainda faltavam três coisas para ter um modelo de linguagem:

1. **Profundidade.** Uma única camada de atenção é rasa demais. É preciso empilhar muitas, e pilhas profundas são difíceis de treinar.
2. **Processamento por token.** A atenção só *mistura* os vetores (é uma média ponderada). Falta uma etapa não linear que transforme o que cada token reuniu.
3. **Saída.** O modelo precisa terminar em uma pontuação para cada token do vocabulário, e é preciso um procedimento para transformar essas pontuações em texto.

A pergunta central do Capítulo 4 é: **como montar um modelo profundo, estável e que produza texto a partir dessas peças?**

### A construção progressiva no capítulo

1. **Esqueleto do GPT (`DummyGPTModel`).** O livro começa com a arquitetura completa, mas com blocos vazios, para fixar o fluxo `token IDs → embeddings → blocos → norma final → logits` antes de preencher cada parte.

2. **Layer Normalization.** Em redes profundas, a escala das ativações deriva de camada em camada. A LayerNorm recentra e reescala cada vetor de token (média 0, variância 1) e depois aplica `scale` e `shift` treináveis. Ela é calculada por token, não pelo lote, então funciona igual com batch 1 ou 1000.

3. **GELU e FeedForward.** A FeedForward é uma MLP aplicada a cada token separadamente: expande D → 4D, aplica GELU e contrai de volta para D. A GELU é uma versão suave da ReLU: deixa passar valores positivos e devolve valores pequenos (não nulos) para negativos próximos de zero, o que mantém algum gradiente nessa região.

4. **Conexões residuais (*shortcut*).** Em vez de `x = f(x)`, cada subcamada faz `x = x + f(x)`. A soma cria um caminho direto pelo qual o gradiente volta às primeiras camadas sem ser multiplicado por todas as derivadas do meio. É o que permite empilhar 12, 24 ou 48 blocos.

5. **Transformer Block.** Junta tudo no padrão Pre-LN: `x = x + drop(att(norm1(x)))` e depois `x = x + drop(ff(norm2(x)))`. A atenção troca informação *entre* tokens e a FeedForward transforma *cada* token. A entrada e a saída têm o mesmo formato, então os blocos podem ser empilhados livremente.

6. **GPTModel.** Embeddings de token e posição → dropout → `n_layers` blocos → LayerNorm final → `out_head` (Linear D → V). A saída são logits `(B, T, V)`. Cada posição `t` pontua o próximo token depois de `x[0..t]`.

7. **Geração de texto.** O modelo prevê um token por vez. Ele roda, pega os logits da última posição, escolhe um token, anexa à entrada e roda de novo. A seção 5.3 acrescenta temperatura e top-k à escolha.

### A cadeia de transformações de tensores

```text
Token IDs (B, T)
      ↓
Token Embedding + Positional Embedding → (B, T, D)
      ↓
Dropout
      ↓
Transformer Block (× n_layers):
    x + Dropout(MHA causal(LayerNorm 1(x)))                       → (B, T, D)
    x + Dropout(Linear 4D→D(GELU(Linear D→4D(LayerNorm 2(x)))))   → (B, T, D)
      ↓
LayerNorm final → (B, T, D)
      ↓
Language Model Head (Linear D → V) → logits (B, T, V)
      ↓
Última posição → (B, V) → softmax / temperatura / top-k → próximo token (B, 1)
      ↓
Concatena à entrada → (B, T + 1) e repete
```

### Treino e inferência

| | Treino (Sprint 5) | Inferência (esta sprint) |
|:--|:--|:--|
| Modo | `model.train()` (dropout ligado) | `model.eval()` (dropout desligado) |
| Gradientes | calculados | `torch.no_grad()` |
| Posições usadas | todas as T posições (cada uma prevê o próximo token) | só a última |
| Entrada | janela fixa do corpus | contexto que cresce um token por passo |

### O que fica para a Sprint 5

O modelo está completo, mas os pesos são aleatórios, então o texto gerado não tem sentido. A Sprint 5 vai comparar os logits `(B, T, V)` com os alvos deslocados do DataLoader da Sprint 2 usando entropia cruzada, treinar os pesos e depois carregar os pesos pré-treinados do GPT-2. As funções de `src/generation/` vão ser reutilizadas para ver o texto melhorar ao longo do treino.

---

## Análise dos resultados

Os valores abaixo vêm dos testes desta sprint (`tests/test_transformer.py`, `tests/test_gpt.py`, `tests/test_generation.py`), das demonstrações dos módulos e de contagens de parâmetros feitas com o modelo instanciado no dispositivo `meta` do PyTorch (que cria a estrutura sem alocar memória para os pesos).

### 1. Por que a LayerNorm é necessária e por que ela vem antes de cada subcamada?

Ao passar por muitas camadas, as ativações podem crescer ou encolher sem controle, e com elas os gradientes. A LayerNorm põe cada vetor de token de volta em média 0 e variância 1 antes da subcamada. O teste `test_layer_norm_media_zero_variancia_um` parte de dados com média 3 e desvio 7 e confirma que a saída tem média 0 (tolerância 1e-5) e variância 1 (tolerância 1e-3). A implementação também coincide com `nn.LayerNorm` do PyTorch (`test_layer_norm_equivale_ao_pytorch`).

No GPT-2 a norma vem *antes* da atenção e da FeedForward (Pre-LN), e não depois da soma residual como no Transformer original (Post-LN). Assim, o caminho residual `x + f(x)` não passa por nenhuma normalização, e o gradiente percorre a pilha sem obstáculos. Isso deixa o treino mais estável. Por isso o modelo tem uma LayerNorm *final* antes do `out_head`: sem ela, a saída do último bloco chegaria sem normalização.

### 2. Qual é a diferença prática entre GELU e ReLU?

| x | −3,0 | −1,0 | −0,5 | 0,0 | 0,5 | 1,0 | 3,0 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| GELU | −0,0036 | −0,1588 | −0,1543 | 0,0000 | 0,3457 | 0,8412 | 2,9964 |
| ReLU | 0 | 0 | 0 | 0 | 0 | 1 | 3 |

Nos extremos as duas praticamente coincidem. A diferença está perto de zero: a GELU é suave e fica levemente negativa para x < 0 (mínimo de cerca de −0,17 em x ≈ −0,75). Com a ReLU, um neurônio que recebe entradas negativas tem gradiente exatamente zero e pode parar de aprender. Com a GELU ainda sobra um gradiente pequeno. A implementação com a aproximação por tanh coincide com `nn.GELU(approximate="tanh")` até 1e-6 (`test_gelu`).

### 3. Por que a FeedForward expande para 4D e depois contrai?

A atenção só faz médias ponderadas dos vetores de Value, ou seja, é linear em V. A FeedForward é onde entra a não linearidade por token. Expandir para 4D = 3072 dimensões dá à GELU um espaço maior para separar padrões, e a contração devolve o formato `(B, T, D)` exigido pela soma residual.

Esse espaço custa caro: em um bloco do 124M, a FeedForward tem **4.722.432** parâmetros, contra **2.360.064** da atenção (exatamente o dobro) e 3.072 das duas LayerNorms (exercício 4.1, verificado em `test_transformer_block_parametros_124m`).

### 4. Qual é o efeito das conexões residuais no gradiente?

O teste `test_residual_mantem_gradiente` reproduz o experimento da seção 4.4: cinco camadas `Linear(3, 3) + GELU`, mesma semente, medindo a média do valor absoluto do gradiente dos pesos de cada camada.

| Camada | 1 | 2 | 3 | 4 | 5 |
|:--|--:|--:|--:|--:|--:|
| Sem residual | 0,0025 | 0,0015 | 0,0104 | 0,0193 | 0,0361 |
| Com residual | 0,4770 | 0,5024 | 0,9546 | 0,8206 | 0,5029 |

Sem residual, o gradiente que chega à primeira camada é cerca de 15 vezes menor que o da última. Ele foi multiplicado por todas as derivadas do caminho (*vanishing gradient*). Com residual, todas as camadas recebem gradientes da mesma ordem, e a primeira recebe quase **200 vezes** mais gradiente do que na versão sem atalho. Com apenas 5 camadas o efeito já é grande. Com os 24 sublayers do GPT-2 small (2 por bloco × 12 blocos), sem residuais as primeiras camadas praticamente não aprenderiam.

### 5. Como o Transformer Block preserva a causalidade?

A LayerNorm, a FeedForward e o dropout atuam em cada token separadamente. A única parte que mistura posições é a atenção, que usa a máscara causal da Sprint 3. Logo, o bloco inteiro é causal. O teste `test_transformer_block_causal` soma 5 ao último token da entrada e confirma que as saídas de todas as posições anteriores não mudam (tolerância 1e-5), enquanto a saída da última muda. Empilhar blocos causais mantém a causalidade do modelo inteiro. É isso que permite treinar todas as T posições em paralelo sem que uma posição veja a resposta que deveria prever.

### 6. Onde estão os parâmetros do GPT-2?

| Modelo | D | Blocos | Cabeças | Parâmetros totais | Com weight tying | float32 |
|:--|--:|--:|--:|--:|--:|--:|
| gpt2-small | 768 | 12 | 12 | 163.009.536 | 124.412.160 | 621,83 MB |
| gpt2-medium | 1024 | 24 | 16 | 406.212.608 | 354.749.440 | 1.549,58 MB |
| gpt2-large | 1280 | 36 | 20 | 838.220.800 | 773.891.840 | 3.197,56 MB |
| gpt2-xl | 1600 | 48 | 25 | 1.637.792.000 | 1.557.380.800 | 6.247,68 MB |

(Exercício 4.2. Os tamanhos são só dos pesos, sem contar ativações nem estados do otimizador.)

- No small, `tok_emb` e `out_head` têm 50.257 × 768 = 38.597.376 parâmetros cada. Juntos são 77,2 M, ou **47%** do total. Os 12 blocos somam 12 × 7.085.568 = 85,0 M.
- O nome "124M" do GPT-2 conta o modelo com *weight tying*: o `out_head` reaproveita a matriz do `tok_emb`. Faz sentido porque as duas camadas fazem o mesmo mapeamento (token ↔ vetor de D dimensões), em sentidos opostos. Nossa implementação mantém as duas separadas, como no livro, e `count_parameters` informa os dois números.
- Nas versões maiores, os blocos passam a dominar: o custo dos embeddings cresce com D, mas o dos blocos cresce com D² × número de blocos.
- No `GPT_CONFIG_MINI` (D = 128, 2 blocos), o desequilíbrio é extremo: de 13.294.592 parâmetros, 12,9 M (97%) estão nos dois embeddings de vocabulário e só 395.776 nos blocos. O vocabulário de 50.257 tokens é caro em qualquer tamanho de modelo.

### 7. Como os logits viram o próximo token?

O `out_head` devolve logits `(B, T, V)`. Na geração, só a última posição interessa, pois é a única que prevê um token que ainda não existe. Os logits `(B, V)` passam pela softmax e viram probabilidades que somam 1. A escolha depende da estratégia:

- **Greedy:** `argmax`. Como a softmax preserva a ordem, `argmax(probas) = argmax(logits)`, e o softmax em `generate_text_simple` é só didático. O teste `test_greedy_escolhe_argmax` confirma que o token gerado é o argmax dos logits, e `test_greedy_deterministico` confirma que duas execuções (e `generate` com `temperature=0`) dão a mesma sequência.
- **Temperatura:** divide os logits por T antes da softmax. T < 1 aumenta as diferenças entre logits (distribuição mais concentrada) e T > 1 as achata (mais variedade e mais erros).
- **Top-k:** os logits fora dos k maiores viram −∞, então a probabilidade deles é exatamente 0. Em `test_top_k_restringe_escolha`, com `top_k=3` e temperatura 5 (que achata muito a distribuição), o token sorteado ficou entre os 3 mais prováveis nas 20 sementes testadas.

### 8. Por que o modelo sem treino gera texto sem sentido?

Com pesos aleatórios, os logits são quase iguais para todos os tokens. No exemplo "Hello, I am", o candidato mais provável (`' Feature'`) tem probabilidade 0,000205, e os cinco primeiros ficam entre 0,000143 e 0,000205. A distribuição uniforme daria 1/50.257 ≈ 0,00002 a cada token. O topo está só cerca de 10 vezes acima disso, enquanto um modelo treinado costuma concentrar boa parte da probabilidade em poucos tokens. O greedy gera `'Featureiman Byeswickattribute argue'` e a amostragem gera algo igualmente sem sentido. O resultado é o esperado: a arquitetura funciona (formatos, causalidade e geração estão corretos, como mostram os testes), mas o conhecimento só vem com o treino da Sprint 5.

### 9. Por que o contexto precisa ser recortado durante a geração?

O embedding de posição é uma tabela com `context_length` linhas. Uma sequência maior não tem vetor de posição para os tokens excedentes, e o `GPTModel` rejeita a entrada com um `assert` (`test_contexto_excedido`). Por isso a geração usa `idx[:, -context_size:]`: o modelo vê só os últimos `context_length` tokens, e o texto antigo é esquecido. O teste `test_contexto_recortado` começa com `context_length + 10` tokens e confirma que a geração continua sem erro. Como a atenção é O(T²) (Sprint 3), cada passo também fica mais caro à medida que o contexto cresce, até o limite.

### 10. O que esta sprint entrega para a Sprint 5?

- `GPTModel` produz logits `(B, T, V)` que se comparam diretamente com os alvos `(B, T)` do DataLoader da Sprint 2 via `cross_entropy` (depois de achatar para `(B·T, V)` e `(B·T,)`).
- As chaves do `GPT_CONFIG_124M` seguem os nomes do livro, o que facilita carregar os pesos pré-treinados do GPT-2 na mesma estrutura.
- `generate` (com temperatura, top-k e `eos_id`) e `text_to_token_ids`/`token_ids_to_text` vão servir para acompanhar a qualidade do texto durante o treino.
- Em uma máquina com pouca memória, o 124M já ocupa ~622 MB só de pesos. Treinar exige ainda gradientes e dois estados do AdamW (≈ 4 × 622 MB ≈ 2,5 GB, sem as ativações). O `GPT_CONFIG_MINI` existe para essa situação.

---

## Como reproduzir

```bash
# a partir da raiz do repositório
source .venv/bin/activate   # ou: python3.11 -m venv .venv && pip install -r requirements.txt

# demonstrações de cada módulo
python3 -m src.transformer.layer_norm
python3 -m src.transformer.feed_forward
python3 -m src.transformer.transformer_block
python3 -m src.gpt.gpt_model
python3 -m src.generation.generate

# testes desta sprint
pytest tests/test_transformer.py tests/test_gpt.py tests/test_generation.py -v
```
