# Sprint 03 — Notas de leitura orientada

**Capítulo:** 3 — *Coding Attention Mechanisms*, de *Build a Large Language Model (From Scratch)* (Sebastian Raschka, Manning, 2025).  
Estas notas priorizam a compreensão dos conceitos matemáticos e da intuição geométrica da atenção, não a reprodução do código (esse está modularizado em `src/attention/` e nos testes de `tests/`).

---

## Por que os embeddings estáticos não bastam
A Sprint 2 resolveu o problema de mapear palavras para vetores contínuos (`nn.Embedding`), mas deixou em aberto um problema semântico crucial: **o contexto**. Em uma língua natural, o significado de uma palavra depende das outras palavras que a cercam. Sem um mecanismo que permita aos tokens trocarem informação entre si, o modelo enxerga a frase como uma lista estática de vetores isolados.

A pergunta central do Capítulo 3 é: **como permitir que cada token consulte e absorva informação de todos os outros tokens da sequência de forma dinâmica e diferenciável?**

---

## A evolução progressiva dos mecanismos no capítulo

O autor constrói a atenção em quatro passos pedagógicos sucessivos:

1. **Atenção Simplificada (sem pesos treináveis):**  
   Calcula a similaridade bruta entre vetores de entrada via produto escalar ($x_i \cdot x_j$). Serve para introduzir a ideia de que a atenção nada mais é do que uma média ponderada dos tokens. O problema: não há parâmetros aprendíveis; o modelo não tem como aprender o que é relevante.

2. **Self-Attention com Matrizes Treináveis ($W_q, W_k, W_v$):**  
   Introduz três projeções lineares que transformam a entrada em **Query**, **Key** e **Value**. Agora a rede pode aprender o que procurar ($Q$) e o que oferecer ($K$). A atenção torna-se flexível e otimizável por gradiente descendente.

3. **Escalonamento ($\sqrt{d_k}$) e Máscara Causal:**  
   * **Escala:** A soma de produtos em dimensões altas faz a variância crescer, empurrando o Softmax para a saturação (gradientes quase nulos). A divisão por $\sqrt{d_k}$ restaura a variância unitária.
   * **Causalidade:** Em modelos generativos como o GPT, o futuro não pode ser visto. A máscara anula os scores acima da diagonal ($-\infty$), forçando o Softmax a atribuir peso zero para qualquer posição futura.

4. **Multi-Head Attention (Vetorizada e Eficiente):**  
   Em vez de calcular uma única atenção global, divide-se a dimensão em $h$ cabeças menores ($d_k = d_{\text{out}} / h$). Cada cabeça analisa a frase sob uma perspectiva diferente (sintaxe, semântica, correferência) em paralelo. A implementação unificada via `.view()` e `.transpose()` permite que a GPU execute todas as cabeças em uma única multiplicação matricial.

---

## A cadeia de transformações de tensores

```text
Entrada x (B, T, d_in)
      ↓
Projeções: W_q, W_k, W_v → Q, K, V (B, T, d_out)
      ↓
Divisão em Cabeças: reshape + transpose → (B, h, T, d_k)
      ↓
Scores de Similaridade: Q @ K^T / sqrt(d_k) → (B, h, T, T)
      ↓
Máscara Causal: masked_fill(mask == 0, -1e9)
      ↓
Softmax: probabilidades normalizadas → (B, h, T, T)
      ↓
Média Ponderada: Pesos @ V → (B, h, T, d_k)
      ↓
Concatenação: transpose + contiguous + view → (B, T, d_out)
      ↓
Projeção Linear de Saída: W_o → (B, T, d_out)
