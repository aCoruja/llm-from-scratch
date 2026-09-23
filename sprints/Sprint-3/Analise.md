# Sprint 03 — Análise dos resultados

**Capítulo:** 3 — *Coding Attention Mechanisms*.  

---

## 1. Por que os embeddings estáticos da Sprint 2 são insuficientes sem o mecanismo de atenção?
A camada de embedding da Sprint 2 associa a cada Token ID um vetor denso fixo. Isso significa que uma palavra polissêmica como "banco" recebe rigorosamente o mesmo vetor numérico em "sentei no banco da praça" e em "fui ao banco pagar a conta". O mecanismo de atenção resolve essa limitação tornando as representações **dinâmicas e contextuais**: cada vetor de saída passa a ser uma média ponderada dos vetores de todas as palavras ao redor, adaptando o significado do token ao seu contexto imediato.

## 2. Qual é a intuição dos papéis de Query ($Q$), Key ($K$) e Value ($V$)?
O mecanismo se inspira em sistemas de recuperação de informação (bancos de dados):
* **Query ($Q$):** Representa a "pergunta" ou o que o token atual está buscando no restante da frase.
* **Key ($K$):** Funciona como uma "etiqueta" ou índice que descreve as propriedades de cada token para responder às consultas. O produto $Q \cdot K^T$ mede a afinidade entre a busca e a etiqueta.
* **Value ($V$):** Contém a informação semântica bruta que será efetivamente transmitida e combinada na saída com base nos pesos da afinidade.

## 3. Qual é o papel da divisão por $\sqrt{d_k}$ no Scaled Dot-Product Attention?
Quando multiplicamos dois vetores aleatórios independentes de dimensão $d_k$, a variância da soma de seus produtos acumula proporcionalmente a $d_k$. Para dimensões elevadas (ex: $d_k = 64$), os produtos escalares resultantes atingem magnitudes altas (scores acima de 28 nos testes). A divisão por $\sqrt{d_k} = \sqrt{64} = 8.0$ normaliza a variância de volta para aproximadamente 1.0, mantendo os scores em uma escala numérica estável antes da aplicação do Softmax.

## 4. O que acontece com os gradientes quando a atenção não utiliza escala?
Sem a divisão por $\sqrt{d_k}$, o valor máximo do score dispara ($28.45$ no teste E1), fazendo com que a exponencial do Softmax concentre quase $100\%$ da probabilidade em um único token ($0.9998$) e anule as demais ($0.0000$). Como a derivada da função Softmax saturada tende a zero ($s_i(1 - s_i) \approx 0$), o gradiente que retropropaga pelas camadas da rede desaparece (*vanishing gradient*), paralisando o treinamento dos pesos das projeções lineares $W_q, W_k, W_v$.

## 5. Como a máscara causal impede o vazamento de informação do futuro (*data leakage*)?
Em modelos geradores autoregressivos (como a família GPT), o treinamento supervisionado consiste em prever o próximo token $x_{t+1}$ conhecendo apenas o passado $x_1, \dots, x_t$. A máscara causal substitui todas as pontuações onde $j > i$ pelo valor $-10^9$ antes do Softmax. Como $e^{-10^9} = 0$, as probabilidades de atenção para posições futuras tornam-se estritamente $0.000$ (verificado numericamente na tabela do experimento E2), impedindo que o modelo "copie" a resposta da posição à frente.

## 6. Por que a máscara causal é registrada como *buffer* (`register_buffer`) e não como parâmetro?
A máscara causal é uma matriz constante de uns e zeros (`torch.tril`), cujos valores são puramente estruturais e nunca devem ser alterados pelo otimizador durante o treinamento. Ao usar `self.register_buffer("mask", ...)`, o PyTorch inclui o tensor no estado do modelo (salvando-o junto aos checkpoints e movendo-o automaticamente para a GPU junto com a rede), mas não o lista em `model.parameters()`, poupando memória e tempo de cálculo de gradientes.

## 7. Qual é a vantagem do Multi-Head Attention sobre a Self-Attention simples?
No Self-Attention de cabeça única, existe apenas um mapa de probabilidade para ponderar toda a frase. No Multi-Head Attention, as projeções lineares dividem o espaço de representação em $h$ subespaços menores ($d_k = d_{\text{out}} / h$). Isso permite que diferentes cabeças se especializem em fenômenos linguísticos distintos simultaneamente — por exemplo, uma cabeça monitora a concordância sujeito-verbo, outra rastreia pronomes distantes e uma terceira foca em pontuação (como observado visualmente nos mapas de calor do experimento E5).

## 8. Aumentar o número de cabeças ($h$) aumenta o tamanho ou a quantidade de parâmetros do modelo?
**Não.** Conforme comprovado no experimento E4, a camada `MultiHeadAttention` com $d_{\text{out}} = 64$ manteve rigorosamente **16.448 parâmetros treináveis** quer fosse configurada com 1, 2, 4 ou 8 cabeças. Isso ocorre porque o aumento no número de cabeças é compensado pela redução proporcional da dimensão individual de cada cabeça ($d_k = 64/h$), mantendo a soma das operações de matrizes constante ($h \cdot (d \times \frac{d}{h}) = d^2$).

## 9. Como o comprimento da sequência ($T$) afeta a complexidade do cálculo da atenção?
A atenção compara cada token de entrada contra todos os outros tokens da sequência, produzindo uma matriz de ponderação de dimensão $T \times T$. Portanto, tanto o consumo de memória quanto o número de multiplicações escalam quadraticamente: $O(T^2)$. Conforme medido no experimento E6, multiplicar a sequência por 4 (de $T=16$ para $T=64$) multiplicou o número de elementos da matriz por 16 (de 256 para 4.096), elevando a latência de execução de $0.28\text{ ms}$ para $1.28\text{ ms}$.

## 10. Quais estruturas produzidas nesta Sprint serão conectadas ao modelo completo na Sprint 4?
A camada `MultiHeadAttention` implementada nesta Sprint recebe tensores com formato `(batch_size, context_length, d_in)` e devolve tensores de mesmo formato `(batch_size, context_length, d_out)`. Essa invariância dimensional é o que permite, na Sprint 4, empilhar a atenção com conexões residuais (*Skip Connections*), camadas de normalização (*LayerNorm*) e redes neurais *Feed-Forward* (FFN) para formar os blocos do Transformer (arquitetura GPT-2).
