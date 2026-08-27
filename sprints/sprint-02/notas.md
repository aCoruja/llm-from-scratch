# Sprint 02 — Notas de leitura orientada

**Capítulo:** 2 — *Working with Text Data*, de *Build a Large Language Model (From Scratch)* (Raschka, 2025).

Estas notas priorizam a relação entre as etapas do pipeline, não a reprodução do código do livro (esse está em `src/` e nos testes de `tests/`).

## Por que texto bruto não serve de entrada

Uma rede neural opera sobre tensores de números reais. Uma *string* não tem soma, produto ou norma definidos entre suas palavras. Todo o capítulo é, no fundo, a resposta a uma única pergunta: **como transformar uma sequência de caracteres em uma estrutura numérica que preserve tanto conteúdo quanto ordem?**

## A cadeia de transformações

```
texto → tokens → Token IDs → sequências (entrada/alvo) → embeddings → + posição → entrada do modelo
```

Cada seta é uma redução de forma de representação, e cada uma resolve exatamente o problema que a anterior deixa em aberto:

1. **Tokenização.** Divide o texto em unidades discretas (tokens) — palavras, subpalavras ou pontuação. Sem isso não há nada para contar, indexar ou vetorizar. O capítulo mostra primeiro uma versão simples por expressão regular (didática) e depois o algoritmo real usado pelo GPT: **Byte Pair Encoding**, que constrói um vocabulário de subpalavras mesclando iterativamente os pares mais frequentes de um corpus enorme. A diferença prática entre as duas é que o BPE nunca "trava" diante de uma palavra desconhecida — ele quebra em pedaços menores até reconhecer algo, no limite até bytes individuais.

2. **Vocabulário e Token IDs.** Um token sozinho ainda é texto. O vocabulário é o dicionário finito que atribui um número inteiro único a cada token possível — é esse número, o Token ID, que a rede efetivamente recebe. A relação é sempre relativa a um vocabulário específico: o mesmo token pode ter IDs diferentes em vocabulários diferentes. Tokens especiais (`<|unk|>`, `<|endoftext|>`) resolvem dois problemas que a tokenização por si não resolve: palavras fora do vocabulário e a fronteira entre documentos concatenados.

3. **Por que o Token ID não basta.** Um ID é um índice de posição em uma tabela, atribuído por convenção (ordem alfabética, frequência de mesclagem). Não existe relação matemática pretendida entre a proximidade de dois IDs e a proximidade de significado dos tokens que representam. Usar o ID diretamente como entrada de uma camada faria a rede tratar "304 está perto de 305" como um fato semântico — o que é arbitrário.

4. **Sequências de treinamento (janela deslizante).** O corpus tokenizado é uma longa sequência de Token IDs. Uma janela de tamanho fixo (`max_length`) percorre essa sequência, avançando `stride` posições a cada passo, gerando pares (entrada, alvo) onde o alvo é a entrada deslocada em uma posição. Essa é a auto-supervisão: o próprio texto fornece o rótulo de "qual é o próximo token", sem qualquer anotação manual.

5. **Embeddings.** Cada Token ID é substituído por um vetor denso e treinável (`nn.Embedding`, uma tabela de consulta `vocab_size × output_dim`). É só a partir daqui que operações como produto interno ou multiplicação de matrizes fazem sentido geométrico entre representações de tokens.

6. **Positional embeddings.** A camada de embedding, sozinha, é indiferente à posição: o mesmo token gera sempre o mesmo vetor, não importa onde apareça na sequência. Sem correção, o modelo veria a entrada como um conjunto, não como uma sequência ordenada. Um segundo vetor, indexado pela posição (não pelo conteúdo), é somado ao token embedding — essa soma é o *input embedding* que efetivamente entra no modelo.

7. **DataLoader.** Agrupa as sequências (entrada, alvo) em lotes (`batch_size`), embaralha a cada época e descarta o último lote incompleto. É a peça que conecta os dados preparados ao laço de treinamento das sprints seguintes.

## O que fica para a Sprint 3

O *input embedding* — tensor `(batch_size, context_length, output_dim)` — é exatamente o que o mecanismo de *self-attention* do Capítulo 3 recebe como entrada. A dimensão de embedding escolhida aqui vira a dimensão das matrizes de projeção *query/key/value*; o tamanho de contexto escolhido aqui define o tamanho da matriz de atenção e da máscara causal.
