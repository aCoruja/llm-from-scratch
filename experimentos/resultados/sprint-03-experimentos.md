# Resultados dos Experimentos — Sprint 3
## E1. Atenção com Escala vs. Sem Escala (Saturação do Softmax)
Avaliação do impacto da divisão por $\sqrt{d_k}$ sobre a magnitude das pontuações e a saturação da distribuição de probabilidade ($d_k = 64$, sequência $T=4$).

| Configuração | Escala ($1/\sqrt{d_k}$) | Score Máximo ($Q \cdot K^T$) | Menor Probabilidade | Maior Probabilidade | Estado do Softmax | Comportamento do Gradiente |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Sem Escala | Não | 28.45 | 0.0000 | 0.9998 | Saturado | $\approx 0$ (Desvanecimento) |
| Com Escala | Sim ($\div 8.0$) | 3.55 | 0.0812 | 0.4520 | Suave / Distribuído | Saudável ($\ne 0$) |

> **Observação:** Sem o escalonamento, a probabilidade colapsa quase inteiramente para um único token ($> 0.999$), tornando os gradientes das demais posições virtualmente nulos durante a retropropagação.

---

## E2. Comportamento da Máscara Causal (Atenção Autoregressiva)
Matriz numérica de pesos de atenção obtida para o primeiro lote da sequência de 4 tokens: `["O", "modelo", "gera", "texto"]`.

| Posição / Token | Token 1 ("O") | Token 2 ("modelo") | Token 3 ("gera") | Token 4 ("texto") | Soma da Linha ($\sum$) |
|---|:---:|:---:|:---:|:---:|:---:|
| **1 ("O")** | 1.000 | 0.000 | 0.000 | 0.000 | **1.000** |
| **2 ("modelo")** | 0.472 | 0.528 | 0.000 | 0.000 | **1.000** |
| **3 ("gera")** | 0.421 | 0.237 | 0.342 | 0.000 | **1.000** |
| **4 ("texto")** | 0.225 | 0.192 | 0.221 | 0.362 | **1.000** |

> **Validação:** Todos os elementos estritamente acima da diagonal principal ($j > i$) resultam em $0.000$, comprovando que o modelo é impedido de consultar o futuro no modo causal.

---

## E3. Dimensão do Embedding × Parâmetros da Camada e Latência
Medição de parâmetros treináveis da camada `MultiHeadAttention` ($W_q, W_k, W_v, W_o$) e tempo médio de inferência para lote $B=4$, contexto $T=8$ e $h=4$ cabeças.

| $d_{\text{in}} = d_{\text{out}}$ | Forma da Saída | Fórmula dos Parâmetros ($4 \cdot d^2 + d$) | Parâmetros Treináveis | Tempo de Execução (ms) |
|:---:|:---:|:---:|:---:|:---:|
| **16** | (4, 8, 16) | $4 \cdot 16^2 + 16$ | 1,040 | 0.142 |
| **32** | (4, 8, 32) | $4 \cdot 32^2 + 32$ | 4,128 | 0.185 |
| **64** | (4, 8, 64) | $4 \cdot 64^2 + 64$ | 16,448 | 0.260 |
| **128** | (4, 8, 128) | $4 \cdot 128^2 + 128$ | 65,664 | 0.485 |
| **256** | (4, 8, 256) | $4 \cdot 256^2 + 256$ | 262,400 | 1.025 |
| **512** | (4, 8, 512) | $4 \cdot 512^2 + 512$ | 1,049,088 | 2.840 |

> **Relação:** A contagem de parâmetros escala quadraticamente em relação à dimensão do embedding ($O(d^2)$) devido às matrizes de projeção linear.

---

## E4. Número de Heads × Dimensão Individual de Head ($d_k$)
Fixando $d_{\text{in}} = d_{\text{out}} = 64$ e contexto $T=4$, variando a partição das cabeças de atenção.

| Número de Heads ($h$) | Dimensão da Head ($d_k = d_{\text{out}} / h$) | Formato do Tensor de Pesos | Total de Parâmetros | Variação de Custo |
|:---:|:---:|:---:|:---:|:---:|
| **1** | 64 | (1, 1, 4, 4) | 16,448 | 0% (Base) |
| **2** | 32 | (1, 2, 4, 4) | 16,448 | 0% (Invariante) |
| **4** | 16 | (1, 4, 4, 4) | 16,448 | 0% (Invariante) |
| **8** | 8 | (1, 8, 4, 4) | 16,448 | 0% (Invariante) |

> **Constatação:** O número de parâmetros é estritamente constante independentemente da quantidade de cabeças, pois a soma das dimensões das cabeças sempre recomõe a dimensão total $d_{\text{out}}$.

---

## E5. Comparação Estrutural: Self-Attention vs. Multi-Head Attention
Comparativo direto entre o modelo com 1 cabeça versus 4 cabeças para a entrada `["A", "chave", "abriu", "a", "porta"]` ($T=5$, dimensão 32).

| Característica | Self-Attention (Single-Head) | Multi-Head Attention ($h=4$) |
|---|:---:|:---:|
| **Formato da Saída** | `(1, 5, 32)` | `(1, 5, 32)` |
| **Formato dos Pesos** | `(1, 5, 5)` | `(1, 4, 5, 5)` |
| **Padrões de Atenção** | Apenas 1 matriz global | 4 matrizes especializadas |
| **Subespaços Latentes** | 1 espaço único ($d=32$) | 4 subespaços menores ($d_k=8$) |
| **Custo Paramétrico** | Idêntico | Idêntico |

---

## E6. Comprimento da Sequência de Entrada ($T$) e Complexidade $O(T^2)$
Medição empírica do tempo de execução variando o contexto $T$ para $d_{\text{out}}=32$ e $h=4$.

| Tamanho da Sequência ($T$) | Elementos na Matriz ($T \times T$) | Tempo de Execução (ms) | Fator de Escala (vs. $T=4$) |
|:---:|:---:|:---:|:---:|
| **4** | 16 | 0.165 | 1.0x |
| **8** | 64 | 0.198 | 1.2x |
| **16** | 256 | 0.285 | 1.7x |
| **32** | 1,024 | 0.540 | 3.3x |
| **64** | 4,096 | 1.280 | 7.8x |
| **128** | 16,384 | 3.920 | 23.8x |
| **256** | 65,536 | 14.150 | 85.8x |

> **Conclusão:** O número de operações da atenção escala quadraticamente ($O(T^2)$), evidenciado pelo aumento no tempo de execução conforme $T$ ultrapassa 64 tokens.
