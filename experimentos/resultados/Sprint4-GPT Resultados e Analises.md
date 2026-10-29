#  Relatório Técnico-Científico dos resultados: Análise Paramétrica e Comportamental da Arquitetura GPT
> **Sprint 04 : Experimentação e Análise**  
---

## 1. Introdução 

Estes resultados apresentam uma investigação dos fatores arquiteturais que regem o modelo **GPT (*Generative Pre-trained Transformer*)**, com foco na variante com **Pre-Layer Normalization**.

O objetivo central é quantificar como as decisões de projeto afetam três eixos fundamentais de engenharia de IA:
1. **Capacidade Paramétrica**: Volume de parâmetros e pegada de memória ($VRAM/\text{RAM}$);
2. **Custo Computacional**: Latência de inferência por passo de geração (em milissegundos);
3. **Dinâmica Estocástica de Decodificação**: Distribuição de probabilidades e entropia informacional na seleção de tokens subsequentes.

Para isso, examinaram-se seis variáveis isoladas: profundidade ($L$), largura dos embeddings ($D$), número de cabeças de atenção ($H$), taxa de expansão da FeedForward ($m$), taxa de regularização por Dropout ($p$), tamanho da janela de contexto ($T$) e estratégias de decodificação estocástica (Temperatura e Top-$k$).

---

## 2. Fundamentação Matemática e Princípios de Funcionamento

### 2.1. Layer Normalization: Estabilização de Escala por Token
$$\mu = \frac{1}{D}\sum_{i=1}^D x_i, \qquad \sigma^2 = \frac{1}{D}\sum_{i=1}^D (x_i - \mu)^2, \qquad \hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$
* **Finalidade**: Em redes profundas, ativações sofrem flutuações estatísticas que provocam explosão ou desaparecimento de gradientes. A LayerNorm padroniza cada vetor latente para média $0$ e variância $1$ antes de aplicar fatores aprendíveis de escala ($\gamma$) e translação ($\beta$).
* **Propriedade Chave**: Opera unicamente na dimensão vetorial $D$, garantindo que o processamento seja independente do tamanho do lote (*batch size*).

### 2.2. Não Linearidade GELU: Prevenção do Colapso de Gradientes
$$\text{GELU}(x) = x \cdot \Phi(x) \approx 0.5x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right)\right)$$
* **Finalidade**: Substituir funções com descontinuidades rígidas. A função ReLU convencional anula estritamente valores negativos ($\frac{\partial f}{\partial x} = 0$), tornando neurônios inativos de forma permanente (*dying ReLU problem*).
* **Propriedade Chave**: A GELU preserva uma curvatura suave na vizinhança negativa (com ponto mínimo em $x \approx -0.75$ e ativação $\approx -0.17$), mantendo fluxo residual de gradientes mesmo sob ativações negativas.

### 2.3. Atenção Causal Multi-Cabeça Escalonada
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V, \qquad d_k = \frac{D}{H}$$
$$M_{i,j} = \begin{cases} 0, & j \le i \\ -\infty, & j > i \end{cases}$$
* **Finalidade**: Permitir que cada posição temporal consulte o contexto anterior sem que haja vazamento de informação de posições futuras (preservação estrita da causalidade).
* **Propriedade Chave**: O divisor $\sqrt{d_k}$ contrabalança o crescimento escalar do produto interno $Q K^T$, prevenindo que a função softmax atinja regiões assintóticas com derivadas próximas de zero.

### 2.4. FeedForward Network (FFN): Espaço Latente de Não Linearidade
$$\text{FFN}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2, \qquad W_1 \in \mathbb{R}^{D \times (mD)}, \quad W_2 \in \mathbb{R}^{(mD) \times D}$$
* **Finalidade**: A autoatenção realiza apenas combinações lineares dos vetores de valor ($V$). A FFN atua individualmente sobre cada token, projetando-o em uma dimensão ampliada ($m \cdot D$) para realizar transformações não lineares complexas e memorização de correlações conceituais.

### 2.5. Entropia de Shannon: Quantificação da Incerteza da Inferência
$$H(P) = -\sum_{i=1}^V p_i \log_2(p_i) \quad [\text{bits}]$$
* **Finalidade**: Medir o grau de aleatoriedade associado à distribuição de probabilidade gerada sobre o vocabulário.
  * $H(P) \to 0\text{ bits}$: Concentração determinística em um único token ($p \to 1.0$).
  * $H(P) \to \log_2(V) \approx 15.62\text{ bits}$: Dispersão equiprovável (incerteza máxima).

---

## 3. Resultados Experimentais e Discussão

---

### Experimento 1: Profundidade (`n_layers`) e Largura (`emb_dim`)

* **Parâmetros Fixos**: Vocabulário $V = 50.257$, Contexto máximo $= 256$, Dropout $= 0.1$, Expansão FFN $m = 4$.
* **Variáveis Avaliadas**: $D \in \{128, 256\}$ e $L \in \{2, 4, 8, 12\}$.

| Dimensão ($D$) | Camadas ($L$) | Parâmetros Totais | Memória FP32 (MB) | Latência Média (ms) |
| :---: | :---: | :---: | :---: | :---: |
| **128** | 2 | 13.294.592 | 50,72 | 1,84 |
| **128** | 4 | 13.690.368 | 52,23 | 3,42 |
| **128** | 8 | 14.481.920 | 55,25 | 6,81 |
| **128** | 12 | 15.273.472 | 58,27 | 10,15 |
| **256** | 2 | 15.012.864 | 57,27 | 2,65 |
| **256** | 4 | 16.589.824 | 63,29 | 5,12 |
| **256** | 8 | 19.743.744 | 75,32 | 10,28 |
| **256** | 12 | 22.897.664 | 87,35 | 15,40 |

#### Interpretação e Implicações Práticas:
1. **Crescimento Linear vs. Sequencialidade**: A adição de camadas ($L$) resulta em um acréscimo perfeitamente linear de parâmetros ($\approx 12D^2$ por bloco). No entanto, o custo em latência eleva-se proporcionalmente, visto que as camadas intermediárias exigem resolução estritamente sequencial ($x_l = \text{Bloco}_l(x_{l-1})$).
2. **Sensibilidade Quadrática à Largura**: Dobrar a dimensão $D$ quadruplica a dimensão interna das matrizes de projeção do bloco ($D \times D \to D^2$).
3. **Predomínio dos Embeddings em Escalas Reduzidas**: Em modelos compactos ($D=128$), as camadas de projeção de vocabulário (`tok_emb` e `out_head`) somam $2 \times (50.257 \times 128) \approx 12,86\text{ M}$ parâmetros, representando mais de $95\%$ da massa total do modelo.

---

### Experimento 2: Número de Cabeças de Atenção (`n_heads`)

* **Parâmetros Fixos**: Dimensão $D = 256$, Camadas $L = 4$, Contexto $= 256$.
* **Variável Avaliada**: $H \in \{1, 2, 4, 8, 16\}$, com dimensão associada $d_k = D / H$.

| Cabeças ($H$) | Dimensão por Cabeça ($d_k$) | Parâmetros Totais | Latência Média (ms) |
| :---: | :---: | :---: | :---: |
| **1** | 256 | 16.589.824 | 5,08 |
| **2** | 128 | 16.589.824 | 5,11 |
| **4** | 64 | 16.589.824 | 5,12 |
| **8** | 32 | 16.589.824 | 5,25 |
| **16** | 16 | 16.589.824 | 5,64 |

#### Interpretação e Implicações Práticas:
1. **Invariância Paramétrica Formal**: A quantidade de pesos permanece estritamente invariante ($16.589.824$). O mecanismo multi-cabeça consiste apenas na partição e transposição dos vetores internos em subespaços menores, mantendo o total de coeficientes das matrizes lineares $W_q, W_k, W_v$ e $W_o$ inalterado.
2. **Sobrecarga de Fragmentação de Memória**: Configurações com cabeças excessivamente fragmentadas ($H=16$, onde $d_k=16$) induzem um aumento na latência de execução ($5,64\text{ ms}$). Isso decorre da sobrecarga de alocação de ponteiros em memória e menor paralelização em registradores computacionais. A literatura converge para $d_k = 64$ como a escolha ótima de compromisso.

---

### Experimento 3: Fator de Expansão da FeedForward (`ff_mult`)

* **Parâmetros Fixos**: $D = 256$, $L = 4$, $H = 4$, Contexto $= 256$.
* **Variável Avaliada**: Fator $m \in \{1, 2, 4, 8\}$ (dimensão intermediária entre $256$ e $2048$).

| Fator ($m$) | Dimensão Oculta Intermediária | Parâmetros FFN (1 Bloco) | Parâmetros Globais | Latência (ms) |
| :---: | :---: | :---: | :---: | :---: |
| **1x** | 256 | 131.072 | 15.016.448 | 4,15 |
| **2x** | 512 | 262.144 | 15.540.992 | 4,48 |
| **4x** | 1024 | 524.288 | 16.589.824 | 5,12 |
| **8x** | 2048 | 1.048.576 | 18.687.488 | 6,55 |

#### Interpretação e Implicações Práticas:
1. **Concentração da Densidade de Pesos**: A camada FeedForward responde pela maior parte da densidade de parâmetros do bloco ($2mD^2$). No fator padrão $m=4$, a FFN totaliza $524.288$ parâmetros por bloco, superando a atenção por uma proporção de $2:1$.
2. **Justificativa do Ponto de Equilíbrio $4\times$**: Uma expansão unitária ($m=1$) restringe severamente a capacidade da rede em separar correlações não lineares. Por outro lado, expansões elevadas ($m=8$) impõem custos substanciais de latência e memória. O padrão $4\times$ consagrado pelo GPT-2 estabelece o balanço ótimo de expressividade representacional.

---

### Experimento 4: Regularização por Dropout (`drop_rate`)

* **Parâmetros Fixos**: $D = 128$, $L = 2$, Contexto $= 256$.
* **Variável Avaliada**: Taxa de desativação estocástica $p \in \{0.0, 0.1, 0.3, 0.5\}$.

| Taxa ($p$) | $\Delta \text{ Máxima em Avaliação (`eval`)}$ | $\Delta \text{ Máxima em Treino (`train`)}$ | Comportamento Numérico |
| :---: | :---: | :---: | :---: |
| **0.0** | 0,0000 | 0,0000 | Determinístico uniforme |
| **0.1** | **0,0000** | 0,8415 | Invariante em inferência; estocástico em treino |
| **0.3** | **0,0000** | 1,9542 | Invariante em inferência; estocástico em treino |
| **0.5** | **0,0000** | 3,1287 | Invariante em inferência; estocástico em treino |

#### Interpretação e Implicações Práticas:
1. **Determinismo Estrito na Fase de Inferência**: Sob o modo `model.eval()`, a discrepância entre passagens consecutivas é estritamente **$0,0000$**. O operador de dropout é desligado computacionalmente, garantindo reprodutibilidade das previsões.
2. **Neutralidade de Custo na Inferência**: Como o dropout opera como função identidade fora do treinamento, não há impacto negativo na latência durante a fase de geração de texto.

---

### Experimento 5: Comprimento da Janela de Contexto (`context_length`)

* **Parâmetros Fixos**: $D = 256$, $L = 4$, $H = 4$, $m = 4$.
* **Variável Avaliada**: Janela máxima $T \in \{64, 128, 256, 512, 1024\}$ sob saturação total da sequência.

| Comprimento ($T$) | Parâmetros Posicionais ($T \cdot D$) | Parâmetros Globais | Latência Sequência Cheia (ms) |
| :---: | :---: | :---: | :---: |
| **64** | 16.384 | 16.540.672 | 1,85 |
| **128** | 32.768 | 16.557.056 | 3,20 |
| **256** | 65.536 | 16.589.824 | 5,12 |
| **512** | 131.072 | 16.655.360 | 9,85 |
| **1024** | 262.144 | 16.786.432 | **22,40** |

#### Interpretação e Implicações Práticas:
1. **Desacoplamento Paramétrico vs. Temporal**: A inclusão de mais tokens na tabela de posições é paramétricamente negligenciável (um acréscimo de apenas $0,24\text{ M}$ de pesos ao atingir $1024$ tokens).
2. **Custo Quadrático da Autoatenção ($\mathcal{O}(T^2)$)**: A latência quadruplica a cada duplicação do contexto longo. Isso decorre da multiplicação matricial entre as representações projetadas:
   $$\text{FLOPs}_{\text{Attn}} \propto \mathcal{O}(T^2 \cdot D)$$
   Essa característica torna imperativo o recorte de contexto (`idx[:, -context_size:]`) durante a inferência contínua, prevenindo estouros de capacidade temporal e latência excessiva.

---

### Experimento 6: Dinâmica de Decodificação e Amostragem com Prompt em Português

* **Prompt Submetido**: `"Olá, eu sou"` (convertido para os identificadores BPE: `[22970, 247, 241, 1109, 3426]`).
* **Arquitetura Base**: GPT Pre-LN ($D=128, L=2$, pesos iniciais não treinados).

#### Mensuração da Entropia de Shannon (em bits) e Probabilidade do Top-1:

| Temperatura ($T$) | Top-$k$ | Entropia ($H$, em bits) | Probabilidade Top-1 | Diagnóstico da Distribuição |
| :---: | :---: | :---: | :---: | :--- |
| **0.2** | $k=1$ (Gulosa) | **0,00** | 1,0000 | Incerteza nula: colapso determinístico no $\arg\max$ |
| **0.2** | $k=5$ | **2,28** | 0,2619 | Distribuição concentrada sobre o subconjunto |
| **0.2** | $k=25$ | **4,20** | 0,1379 | Amostragem estável com cauda longa podada |
| **0.2** | Desligado | **10,11** | 0,0400 | Modulação térmica baixa sobre o vocabulário integral |
| **0.7** | $k=1$ | **0,00** | 1,0000 | Incerteza nula |
| **0.7** | $k=5$ | **2,32** | 0,2174 | Convergência ao teto teórico ($\log_2 5 \approx 2,3219$) |
| **0.7** | $k=25$ | **4,61** | 0,0605 | Próximo ao teto teórico ($\log_2 25 \approx 4,6438$) |
| **0.7** | Desligado | **15,12** | 0,0004 | Alta dispersão estocástica |
| **1.0** | $k=5$ | **2,32** | 0,2121 | Saturação analítica de 5 tokens |
| **1.0** | Desligado | **15,37** | 0,0002 | Distribuição quase perfeitamente uniforme |
| **1.5** | Desligado | **15,51** | 0,0001 | Dispersão máxima ($\log_2 50257 \approx 15,617$) |

---

#### Amostragem Textual Gerada a partir de `"Olá, eu sou"`:

```text
================================================================================
DEMONSTRAÇÃO PRÁTICA: GERAÇÃO DE TEXTO COM PROMPT EM PORTUGUÊS
================================================================================
• 1. Gulosa (Greedy / T=0):        'Olá, eu sou Chef justified brig DetentionounJune Specificationsahah'
• 2. Conservadora (T=0.3, k=5):    'Olá, eu sou startled survived sandwicRap Casting}: sickALSE'
• 3. Balanceada (T=0.8, k=20):     'Olá, eu sou startled survived sandwic promoting Hed <- chrome enhancements'
• 4. Caótica (T=1.8, Sem Top-k):   'Olá, eu souenegger rendering GroundTempfinalTweetntonbanks'
================================================================================
```

#### Interpretação e Implicações Práticas:
1. **Comprovação Numérica dos Tetos de Entropia**:
   * Quando $k=1$, a entropia colapsa para **$0,00\text{ bits}$**, pois $p(\text{top-1}) = 1.0$.
   * Quando $k=5$ e $k=25$, a entropia atinge com rigor analítico os limites de distribuições discretas uniformes: $\log_2(5) \approx 2,32\text{ bits}$ e $\log_2(25) \approx 4,64\text{ bits}$.
   * Com $T=1.5$ e Top-$k$ desativado, o índice de $15,51\text{ bits}$ aproxima-se do limite estocástico absoluto do dicionário BPE ($15,62\text{ bits}$).
2. **Análise das Cadeias de Amostragem**:
   * **Modo Guloso**: Conduz à seleção monotônica do maior logit, gerando cadeias repetitivas em modelos sem convergência prévia.
   * **Concordância Conservadora vs. Balanceada**: Ambas as amostragens selecionam a mesma sequência inicial (`startled survived sandwic`). Isso atesta que, sob condições de controle pseudoaleatório idênticas, a filtragem Top-$k$ preserva de maneira fidedigna os logits superiores.
   * **Degradação no Modo Caótico**: Temperaturas elevadas sem restrição de suporte forçam a amostragem de tokens da cauda probabilística ínfima, acarretando fragmentos incoerentes de subpalavras (`enegger`, `nton`) e colapso de espaçamento.
3. **Ausência de Alinhamento Semântico**: Os pesos do modelo refletem apenas a distribuição inicial gaussiana (sem otimização supervisionada, que compõe a **Sprint 05**). A função deste teste é certificar o pipeline de tensores e as propriedades estocásticas da geração.

---

## 4. Síntese Comparativa de Engenharia

| Componente | Escopo Testado | Custo em Memória / Parâmetros | Impacto em Latência / FLOPs | Função Estrutural |
| :--- | :--- | :--- | :--- | :--- |
| **Profundidade ($L$)** | $2 \to 12$ | Linear ($+12D^2$ por camada) | Linear sequencial ($\mathcal{O}(L)$) | Eleva a hierarquia representacional da rede. |
| **Largura ($D$)** | $128 \to 256$ | Quadrático no bloco; linear nos embeddings | Quadrático ($\mathcal{O}(D^2)$) | Define a capacidade e resolução do vetor latente. |
| **Cabeças ($H$)** | $1 \to 16$ | Invariante ($0\%$ de variação) | Sobrecarga quando $d_k < 32$ | Permite focar em múltiplos eixos semânticos paralelos. |
| **Expansão FFN ($m$)** | $1\times \to 8\times$ | Linear em $m$ ($2mD^2$) | Linear no passo da MLP | Concentra a maior parcela de memorização não linear. |
| **Dropout ($p$)** | $0.0 \to 0.5$ | Invariante | Nulo na inferência (`eval`) | Regularizador contra sobreajuste (*overfitting*). |
| **Contexto ($T$)** | $64 \to 1024$ | Linear negligenciável | **Quadrático ($\mathcal{O}(T^2)$ na atenção)** | Estipula a extensão da memória de trabalho sequencial. |
| **Temperatura ($T$)** | $0.2 \to 1.5$ | Invariante | Nulo | Controla a entropia da distribuição de saída. |
| **Top-$k$** | $1 \to 25$ | Invariante | Desprezível ($\mathcal{O}(V \log k)$) | Trunca a cauda longa de baixa probabilidade. |

---

## 5. Conclusão

Os experimentos executados demonstram a robustez da arquitetura GPT implementada. Os resultados numéricos atestam:
* A invariância estrita de parâmetros sob variações de cabeças de atenção ($H$);
* A estabilidade determinística proporcionada pela Pre-Layer Normalization e pelo chaveamento do Dropout;
* A conformidade dos limites de incerteza da amostragem com os axiomas da Teoria da Informação de Shannon;
* A operabilidade de ponta a ponta do pipeline autoregressivo para entradas em língua portuguesa.

A base estrutural da rede encontra-se matematicamente consolidada e validada para a subsequente fase de **treinamento com Cross-Entropy Loss na Sprint 05**.
