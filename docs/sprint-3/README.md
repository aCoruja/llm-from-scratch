# Capítulo 3: Mecanismos de Atenção em LLMs (Coding Attention Mechanisms)

Este módulo aborda a fundamentação teórica, formulação matemática e implementação prática em PyTorch dos **Mecanismos de Atenção**, elemento central na arquitetura dos *Large Language Models* (LLMs) e *Transformers*.

---

## 📌 Visão Geral do Capítulo

Historicamente, o processamento de linguagem natural utilizava Redes Neurais Recorrentes (RNNs), que sofriam com o afunilamento de sequências longas (*RNN Bottleneck*) e o desvanecimento de gradientes. O mecanismo de atenção surge para permitir acesso direto e dinâmico a todo o contexto de uma sequência.

Neste capítulo, constrói-se progressivamente o pipeline de atenção:
1. **Autoatenção Básica (*Self-Attention*)**: Cálculo de similaridade por produto escalar e normalização Softmax.
2. **Projeções Treináveis ($Q, K, V$)**: Desacoplamento entre Consulta (*Query*), Chave (*Key*) e Valor (*Value*) com matrizes de pesos lineares.
3. **Escalonamento por $\sqrt{d_k}$**: Estabilização numérica para mitigar a saturação do Softmax em altas dimensões.
4. **Atenção Causal (*Masked Attention*)**: Mascaramento autorregressivo triangular com $-\infty$ para geração de texto passo a passo.
5. **Atenção Multi-Head (*Multi-Head Attention — MHA*)**: Paralelização em múltiplos subespaços lineares usando manipulações tensoriais quadridimensionais de alta performance.

---

## 📐 Formulação Matemática Central

A equação fundamental da **Atenção Produto-Escalar Escalonada** com máscara causal é expressa por:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$

Onde:
* $Q = X W_q \in \mathbb{R}^{T \times d_k}$ (Matriz de Consultas)
* $K = X W_k \in \mathbb{R}^{T \times d_k}$ (Matriz de Chaves)
* $V = X W_v \in \mathbb{R}^{T \times d_v}$ (Matriz de Valores)
* $d_k$: Dimensionalidade do canal das chaves (fator estabilizador $\sqrt{d_k}$).
* $M$: Matriz de máscara causal triangular estrita preenchida com $0$ na diagonal inferior e $-\infty$ nas posições futuras.

---

## 📚 Conceitos Abordados

| Conceito | Descrição Técnica |
| :--- | :--- |
| **RNN Bottleneck** | Perda de contexto provocada pela compressão de sequências longas em um único vetor de estado oculto final. |
| **Atenção de Bahdanau** | Introdução do alinhamento dinâmico entre codificador e decodificador via similaridade vetorial. |
| **Autoatenção (*Self-Attention*)** | Mecanismo onde tokens de uma mesma sequência calculam afinidade mútua entre si. |
| **Vetor de Contexto ($z$)** | Combinação linear dos vetores de entrada/valores ponderada pelos pesos de atenção normalizados. |
| **Pontuações de Atenção ($\omega$)** | Produtos escalares brutos medindo a força de alinhamento antes da normalização. |
| **Similaridade por Produto Escalar** | Operação algébrica básica para quantificar alinhamento no hiperespaço latente. |
| **Pesos de Atenção ($\alpha$)** | Distribuição de probabilidade ($\sum \alpha = 1.0$) resultante da aplicação do Softmax. |
| **Escalonamento ($\sqrt{d_k}$)** | Fator divisor que evita a saturação do Softmax e o desvanecimento de gradientes. |
| **$W_q, W_k, W_v$** | Matrizes de pesos treináveis que mapeiam as entradas nos subespaços de Query, Key e Value. |
| **Camadas `nn.Linear`** | Módulos otimizados do PyTorch que executam as transformações lineares sem viés (`bias=False`). |
| **Atenção Causal** | Restrição autorregressiva que impede que tokens correntes consultem informações futuras. |
| **Truque do $-\infty$** | Preenchimento de posições com `-torch.inf` para que o Softmax resulte exatamente em probabilidade $0$. |
| **Vazamento de Informação** | Garantia analítica de que tokens posteriores não exercem influência residual sobre estados passados. |
| **Projeção de Saída (`out_proj`)** | Camada linear final que combina e integra as representações dos múltiplos cabeçotes. |

