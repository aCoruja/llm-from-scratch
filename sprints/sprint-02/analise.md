# Sprint 02 — Análise dos resultados

**Capítulo:** 2 — *Working with Text Data*. Versão em PDF com o mesmo conteúdo,
diagramada: [`docs/sprint-02/Analise_Sprint2.pdf`](../../docs/sprint-02/Analise_Sprint2.pdf).

Todos os valores citados abaixo vêm da execução registrada em
[`experimentos/resultados/sprint-02-experimentos.md`](../../experimentos/resultados/sprint-02-experimentos.md),
sobre o corpus de referência do capítulo (*The Verdict*, Edith Wharton — 20.479
caracteres, 5.145 tokens BPE; ver nota sobre o corpus de domínio próprio em
[`data/README.md`](../../data/README.md)).

## 1. Por que um LLM não pode trabalhar diretamente com o texto bruto?

Uma rede neural é uma composição de operações matemáticas sobre tensores de números
reais. Uma *string* não tem soma ou produto definido entre suas palavras — é
necessário um pipeline de conversão (tokenização → Token IDs → embeddings) antes de
qualquer camada poder processar o texto.

## 2. Qual é a função do vocabulário?

Define todos os tokens que o modelo reconhece, associando cada um a um Token ID
único. Na entrada, permite converter texto em inteiros; na saída, define a dimensão do
vetor de *logits* da última camada (prever o próximo token = prever uma distribuição
sobre todo o vocabulário). No corpus de teste, o vocabulário próprio tem 1.130 entradas
(1.132 com tokens especiais); o vocabulário BPE do GPT-2, fixo, tem 50.257 — mais de
40× maior, por ser genérico e não depender do corpus.

## 3. Qual é a diferença entre um token e um Token ID?

O **token** é a unidade textual (`"the"`, `"sun"`, `","`) produzida pela tokenização.
O **Token ID** é o inteiro que o representa dentro de um vocabulário específico — só
faz sentido em relação a esse vocabulário. O mesmo token pode ter IDs diferentes em
vocabulários diferentes (BPE do GPT-2 vs. vocabulário construído a partir do corpus
local).

## 4. Por que os Token IDs não são utilizados diretamente como representação semântica?

Um Token ID é um índice arbitrário (ordem alfabética ou frequência de mesclagem no
BPE). Não há relação matemática pretendida entre a proximidade de dois IDs e a
proximidade semântica dos tokens. Usar o ID diretamente forçaria a rede a tratar
"304 está perto de 305" como um fato semântico, o que é falso. A camada de embedding
resolve isso trocando o índice por um vetor treinável, cujas distâncias *podem* ser
aprendidas para refletir relações reais de uso.

## 5. Qual é a função dos embeddings?

Trocam o índice inteiro por um vetor denso de dimensão fixa (`output_dim`), treinável.
Com `output_dim=256`, a tabela de embeddings do GPT-2 tem 50.257 × 256 = 12.865.792
parâmetros — mais de 12,8 milhões só para mapear tokens em vetores, antes de qualquer
bloco de atenção. O número escala linearmente com `output_dim`: 804.112 parâmetros com
16 dimensões, 38.597.376 com 768.

## 6. Por que é necessário representar a posição dos tokens?

A camada de embedding é uma tabela indexada por Token ID: o mesmo token gera sempre o
mesmo vetor, não importa a posição. Sem correção, a entrada seria vista como um
conjunto (*bag of tokens*), não uma sequência ordenada. O positional embedding soma um
vetor dependente só da posição. Verificado experimentalmente
(`tests/test_embeddings.py::test_mesmo_token_id_em_posicoes_diferentes_gera_vetores_diferentes`):
o mesmo Token ID (40) somado ao positional embedding da posição 0 produz um vetor
diferente do mesmo ID somado ao da posição 2.

## 7. Qual é a relação entre tamanho do contexto e quantidade de amostras de treinamento?

Para um corpus fixo, quanto maior o contexto (`max_length`), menos amostras a janela
deslizante extrai (com `stride = max_length`): amostras ≈ (N − ctx) / ctx.

| Contexto (`max_length`) | 4 | 32 | 128 |
|:--:|:--:|:--:|:--:|
| Amostras geradas | 1.286 | 160 | 40 |

Multiplicar o contexto por 32 dividiu as amostras por ~32. O `stride` modula isso à
parte: contexto fixo em 32, `stride=32` → 160 amostras; `stride=8` (75% de
sobreposição) → 640 amostras — quatro vezes mais, à custa de redundância entre
amostras vizinhas.

## 8. Qual é o impacto da dimensão do embedding sobre as estruturas utilizadas pelo modelo?

`output_dim` é o tamanho do último eixo de todo tensor a partir da etapa de embedding:
`(batch_size, context_length, output_dim)`. Aumentar `output_dim` não muda quantos
tokens são processados, mas aumenta parâmetros treináveis (linear em `vocab_size ×
output_dim`), tempo de montagem do *input embedding* (0,138 ms → 1,388 ms de 16 para
768 dimensões, um fator de ~10× contra 48× no número de dimensões — sugerindo que,
nessa escala pequena, custos fixos de indexação ainda dominam) e memória por
representação.

## 9. Qual é a função do DataLoader no pipeline?

Camada de organização entre o `Dataset` (produz um par entrada/alvo por índice) e o
laço de treinamento (consome lotes): agrupa `batch_size` amostras em um tensor
`(batch_size, context_length)`, embaralha a ordem a cada época (`shuffle=True`) e
descarta o último lote incompleto (`drop_last=True`), garantindo lotes de formato
uniforme.

## 10. Quais informações produzidas nesta Sprint serão utilizadas pelo mecanismo de atenção da Sprint seguinte?

O artefato final é o *input embedding* — tensor `(batch_size, context_length,
output_dim)`, soma de token embedding e positional embedding. O Capítulo 3 parte
exatamente dele: a *self-attention* projeta cada vetor em *query*, *key* e *value* por
matrizes treináveis sobre esse mesmo `output_dim`; o `context_length` escolhido aqui
define o tamanho da matriz de atenção (`context_length × context_length`) e da máscara
causal. A Sprint 2 entrega tanto o **conteúdo** (token embedding) quanto o **contexto
posicional** (positional embedding) que a atenção vai ponderar dinamicamente.
