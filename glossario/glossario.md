# Glossário

Glossário cumulativo do semestre. Cada entrada apresenta: termo original, equivalente em
português, definição, função no modelo de linguagem, relações com outros conceitos e um
exemplo conceitual ou computacional — conforme os campos definidos no [README](../README.md#glossário-técnico).

## Capítulo 1 — Understanding Large Language Models

Ver `docs/` e `sprints/sprint-01/` para o material da Sprint 1. Entradas em formato de
tabela para este capítulo serão migradas para cá em uma sprint futura.

## Capítulo 2 — Working with Text Data

> Fonte: Sprint 2. PDF compilado com o mesmo conteúdo em
> [`docs/sprint-02/Glossario_Capitulo2.pdf`](../docs/sprint-02/Glossario_Capitulo2.pdf).

### Tokenization (Tokenização)

| Campo | Conteúdo |
|:--|:--|
| Definição | Processo de dividir uma sequência contínua de texto em unidades menores e discretas — os tokens — que podem ser palavras, partes de palavras, sinais de pontuação ou até bytes individuais. |
| Função | Primeira etapa do pipeline: transforma uma *string* (que a rede não sabe processar) em uma sequência de unidades discretas, mapeáveis para números. |
| Relações | Precede a construção do [Vocabulary](#vocabulary-vocabulário) e a conversão para [Token ID](#token-id-identificador-do-token). No GPT, é feita por [Byte Pair Encoding (BPE)](#byte-pair-encoding-bpe), não pela versão simples baseada em regex. |
| Exemplo | `"Hello, do you like tea?"` → `["Hello", ",", "do", "you", "like", "tea", "?"]` (regex). |

### Token

| Campo | Conteúdo |
|:--|:--|
| Definição | Unidade mínima produzida pela tokenização — palavra inteira, subpalavra ou sinal de pontuação. Nunca um conceito semântico completo por si só. |
| Função | Unidade de trabalho de todo o pipeline: o texto passa a ser manipulado como sequência de tokens, cada um associado a um Token ID e depois a um vetor de embedding. |
| Relações | Um token não carrega significado por si — isso só surge quando seu Token ID é convertido em vetor pela camada de embedding. |
| Exemplo | Em `"the sunlit terraces"`, o BPE do GPT-2 produz `["the", " sun", "lit", " terr", "aces"]` — `"sunlit"` e `"terraces"` viram subpalavras. |

### Corpus

| Campo | Conteúdo |
|:--|:--|
| Definição | Conjunto de texto bruto usado como fonte de dados, para construir o vocabulário, extrair sequências de treinamento ou pré-treinar o modelo. |
| Função | Matéria-prima da Sprint: sem corpus não há tokens, vocabulário nem sequências de treinamento. |
| Relações | Alimenta a [Tokenization](#tokenization-tokenização); seu tamanho em tokens determina quantas amostras de [Input–target pair](#inputtarget-pair-par-entradaalvo-sequência-de-treinamento) podem ser extraídas para um dado contexto. |
| Exemplo | Corpus usado nesta sprint: *The Verdict* (Edith Wharton, domínio público) — 20.479 caracteres, 5.145 tokens BPE. Corpus de domínio próprio (eletrônica, em português) ainda pendente — ver nota em `data/README.md`. |

### Vocabulary (Vocabulário)

| Campo | Conteúdo |
|:--|:--|
| Definição | Conjunto de todos os tokens distintos reconhecidos por um tokenizador, organizado como dicionário token → Token ID único. |
| Função | Define o espaço discreto sobre o qual o modelo opera: todo token de entrada precisa existir no vocabulário (ou cair em token especial); todo token de saída é, na verdade, um índice desse vocabulário. |
| Relações | Construído a partir dos tokens únicos de um [Corpus](#corpus); define o tamanho da tabela de [Token embedding](#embedding-representação-vetorial-densa) e do vetor de *logits* da camada final. |
| Exemplo | No corpus *The Verdict*: 1.130 entradas (1.132 com os tokens especiais). No vocabulário BPE do GPT-2, fixo: 50.257 entradas. |

### Token ID (identificador do token)

| Campo | Conteúdo |
|:--|:--|
| Definição | Número inteiro que representa um token dentro do vocabulário — o índice da linha correspondente na tabela de vocabulário. |
| Função | Única representação numérica bruta do texto antes da camada de embedding; índice de consulta (*lookup*) na tabela de token embeddings. |
| Relações | Obtido a partir de um [Token](#token) via o [Vocabulary](#vocabulary-vocabulário); não deve ser confundido com representação semântica — IDs próximos não implicam tokens relacionados. |
| Exemplo | `"Hello"` → Token ID `15496` no vocabulário BPE do GPT-2 (`tiktoken`, `encoding="gpt2"`). |

### Special tokens (Tokens especiais)

| Campo | Conteúdo |
|:--|:--|
| Definição | Tokens artificiais adicionados ao vocabulário para tratar casos que a tokenização normal não resolve: `<\|unk\|>` (palavra fora do vocabulário) e `<\|endoftext\|>` (fronteira entre documentos concatenados). |
| Função | `<\|unk\|>` evita falha diante de palavra nunca vista; `<\|endoftext\|>` evita que o modelo aprenda associações espúrias entre o fim de um texto e o início do próximo. |
| Relações | Entradas adicionais no [Vocabulary](#vocabulary-vocabulário); o [BPE](#byte-pair-encoding-bpe), por lidar com qualquer string, normalmente dispensa `<\|unk\|>`, mas mantém `<\|endoftext\|>`. |
| Exemplo | `"Hello, do you like tea?"` contra vocabulário que não conhece `"Hello"` → `["<\|unk\|>", ",", "do", "you", "like", "tea", "?"]`. |

### Byte Pair Encoding (BPE) — Codificação por Pares de Bytes

| Campo | Conteúdo |
|:--|:--|
| Definição | Algoritmo de tokenização em nível de subpalavra que constrói seu vocabulário mesclando iterativamente os pares de caracteres/bytes mais frequentes de um corpus de treinamento. |
| Função | Tokenizador efetivamente usado pelo GPT. Por operar em subpalavras (e, no limite, bytes), representa qualquer string de entrada sem nunca precisar de `<\|unk\|>`. |
| Relações | Substitui, na prática, a [Tokenization](#tokenization-tokenização) + [Vocabulary](#vocabulary-vocabulário) caseiras; implementado via `tiktoken` (`encoding="gpt2"`), vocabulário fixo de 50.257 tokens. |
| Exemplo | `"Akwirw ier"` → `["Ak", "w", "ir", "w", " ", "ier"]`, seis subpalavras que recombinadas reconstroem a string original. |

### Sliding window (Janela deslizante)

| Campo | Conteúdo |
|:--|:--|
| Definição | Técnica de amostragem que percorre a sequência de Token IDs com uma janela de tamanho fixo (`max_length`), avançando `stride` posições a cada passo, para extrair múltiplas subsequências de treinamento. |
| Função | Transforma um corpus (sequência longa de tokens) em um conjunto de exemplos de tamanho fixo, prontos para compor lotes. |
| Relações | Produz os pares de [Input–target pair](#inputtarget-pair-par-entradaalvo-sequência-de-treinamento); `stride` controla a sobreposição entre janelas e, junto com o [Context length](#context-length-tamanho-do-contexto), determina o total de amostras. |
| Exemplo | Com `max_length=4` e `stride=4`, o corpus de teste (5.145 tokens BPE) produz 1.286 amostras; com `stride=1`, cerca de 5.141. |

### Input–target pair (Par entrada/alvo, sequência de treinamento)

| Campo | Conteúdo |
|:--|:--|
| Definição | Par de sequências de mesmo tamanho em que o alvo é a entrada deslocada em uma posição para a direita. |
| Função | Implementa, de forma automática, a tarefa de auto-supervisão do LLM ("prever o próximo token"), sem rotulação manual — o próprio texto fornece o rótulo. |
| Relações | Gerado pela [Sliding window](#sliding-window-janela-deslizante); consumido pelo [DataLoader](#dataloader-carregador-de-dados), que agrupa em [Batch](#batch-lote). |
| Exemplo | IDs `[40, 367, 2885, 1464]`: entrada = `[40, 367, 2885]`, alvo = `[367, 2885, 1464]`. Em texto: entrada = `"I HAD always"`, alvo = `" HAD always thought"`. |

### Context length (Tamanho do contexto)

| Campo | Conteúdo |
|:--|:--|
| Definição | Número máximo de tokens que uma amostra de entrada contém simultaneamente, e por extensão, quantos tokens anteriores o modelo consegue considerar ao prever o próximo. |
| Função | Determina a dimensão da sequência de entrada e da tabela de [Positional embedding](#positional-embedding-embedding-posicional). |
| Relações | Parâmetro central da [Sliding window](#sliding-window-janela-deslizante); contexto maior → menos amostras para um corpus fixo (com `stride` acompanhando o contexto). |
| Exemplo | Contexto 4 → 1.286 amostras; contexto 128 → 40 amostras, no mesmo corpus. |

### Dataset (`torch.utils.data.Dataset`)

| Campo | Conteúdo |
|:--|:--|
| Definição | Estrutura que encapsula o acesso a um conjunto de amostras — aqui, `GPTDatasetV1`, que aplica a janela deslizante sobre o corpus tokenizado e expõe cada par (entrada, alvo) por índice. |
| Função | Padroniza o armazenamento/acesso aos dados, permitindo que o `DataLoader` os agrupe em lotes de forma genérica. |
| Relações | Implementa a interface exigida pelo [DataLoader](#dataloader-carregador-de-dados) (`__len__`, `__getitem__`). |
| Exemplo | `GPTDatasetV1(text, tokenizer, max_length=8, stride=8)` produz `len(dataset) == 643` para o corpus de teste. |

### DataLoader (carregador de dados)

| Campo | Conteúdo |
|:--|:--|
| Definição | Componente do PyTorch (`torch.utils.data.DataLoader`) que envolve um `Dataset` e organiza suas amostras em lotes, podendo embaralhar a ordem a cada época. |
| Função | Interface final entre os dados preparados e o laço de treinamento: entrega, a cada iteração, um lote `(entrada, alvo)` pronto para a rede. |
| Relações | Consome um [Dataset](#dataset-torchutilsdatadataset); `batch_size` define o tamanho do [Batch](#batch-lote); `drop_last` controla o descarte do último lote incompleto. |
| Exemplo | `create_dataloader_v1(text, batch_size=8, max_length=4, stride=4)` produz lotes com `inputs.shape == (8, 4)`. |

### Batch (Lote)

| Campo | Conteúdo |
|:--|:--|
| Definição | Conjunto de `batch_size` amostras de entrada (e seus alvos) agrupadas em um único tensor, processadas em uma passagem da rede. |
| Função | Explora paralelismo de hardware e estabiliza a estimativa do gradiente no treinamento. |
| Relações | Produzido pelo [DataLoader](#dataloader-carregador-de-dados); é o primeiro eixo dos tensores de [Embedding](#embedding-representação-vetorial-densa), na forma `(batch_size, context_length, output_dim)`. |
| Exemplo | `batch_size=8`, `max_length=4` → lote de entrada `(8, 4)`; após embedding com `output_dim=256` → `(8, 4, 256)`. |

### Embedding (Representação vetorial densa)

| Campo | Conteúdo |
|:--|:--|
| Definição | Vetor de números reais, de dimensão fixa (`output_dim`), que representa um token em um espaço vetorial contínuo. Ao contrário do Token ID, seus valores são aprendidos no treinamento. |
| Função | Permite que a rede opere sobre representações contínuas e diferenciáveis, sobre as quais operações vetoriais fazem sentido geométrico. |
| Relações | Produzido a partir de um [Token ID](#token-id-identificador-do-token) por uma [Embedding layer](#embedding-layer-camada-de-embedding-nnembedding); somado ao [Positional embedding](#positional-embedding-embedding-posicional) forma o [Input embedding](#input-embedding-embedding-de-entrada-token--posição). |
| Exemplo | Com `output_dim=256`, cada Token ID vira um vetor de 256 números reais em vez do escalar original. |

### Embedding layer (Camada de embedding, `nn.Embedding`)

| Campo | Conteúdo |
|:--|:--|
| Definição | Camada que funciona como tabela de consulta de dimensão `(vocab_size, output_dim)`: dado um Token ID, retorna a linha correspondente. Valores treináveis, inicializados aleatoriamente. |
| Função | Ponte entre o mundo discreto dos Token IDs e o mundo contínuo em que a rede opera. Equivalente a multiplicar um vetor *one-hot* pela matriz de pesos, mas implementado como indexação direta. |
| Relações | Instanciada com `vocab_size` igual ao tamanho do [Vocabulary](#vocabulary-vocabulário); sua saída, somada à de uma segunda `nn.Embedding` indexada por posição, forma o [Input embedding](#input-embedding-embedding-de-entrada-token--posição). |
| Exemplo | `nn.Embedding(50257, 256)` → 50.257 × 256 = 12.865.792 parâmetros treináveis. |

### Positional embedding (Embedding posicional)

| Campo | Conteúdo |
|:--|:--|
| Definição | Vetor, de mesma dimensão do token embedding, que codifica exclusivamente a posição do token na sequência (0, 1, 2, …). Neste projeto: posicionamento absoluto e aprendido, via segunda `nn.Embedding` indexada por posição. |
| Função | Resolve a limitação da camada de embedding pura, que gera o mesmo vetor para o mesmo token não importa a posição. Sem isso, a sequência seria vista como um conjunto (*bag of tokens*), não ordenada. |
| Relações | Somado ao [Token embedding](#embedding-representação-vetorial-densa) para formar o [Input embedding](#input-embedding-embedding-de-entrada-token--posição); tabela de tamanho `(context_length, output_dim)`. |
| Exemplo | Token ID 40 na posição 0 e na posição 2 recebem, após a soma, vetores diferentes (`vetor_pos0 != vetor_pos2`), verificado em `src/embeddings/embeddings.py`. |

### Input embedding (Embedding de entrada, token + posição)

| Campo | Conteúdo |
|:--|:--|
| Definição | Soma elemento a elemento entre o token embedding de cada posição e o positional embedding correspondente. Vetor final que representa cada token ao entrar nos blocos do Transformer. |
| Função | Entrada literal da arquitetura GPT — ponto de saída desta Sprint e ponto de entrada da Sprint 3 (mecanismo de atenção). |
| Relações | Resultado de Token embedding + Positional embedding; forma `(batch_size, context_length, output_dim)`, a mesma esperada pela primeira camada de *self-attention*. |
| Exemplo | Para `batch_size=8`, `context_length=4`, `output_dim=256`: input embedding tem forma `(8, 4, 256)`. |
