# Resultados dos Experimentos - Sprint 2

## E1 + E6. Tamanho do contexto x quantidade de amostras

| max_length (contexto) | amostras geradas (stride=ctx) | formula (N-ctx)//ctx | cobertura do corpus |
|---|---|---|---|
| 4 | 1286 | 1285 | 100.0% |
| 8 | 643 | 642 | 100.0% |
| 16 | 321 | 320 | 99.8% |
| 32 | 160 | 159 | 99.5% |
| 64 | 80 | 79 | 99.5% |
| 128 | 40 | 39 | 99.5% |

## E5 (extra). Sobreposicao entre janelas (stride) x quantidade de amostras

| stride | sobreposicao entre janelas (ctx-stride) | amostras geradas |
|---|---|---|
| 8 | 24 | 640 |
| 16 | 16 | 320 |
| 32 | 0 | 160 |
| 64 | 0 | 80 |

## E2. Tamanho do lote x forma dos tensores

| batch_size | num. de lotes por epoca | forma do lote (entrada) | forma do lote (alvo) |
|---|---|---|---|
| 1 | 160 | (1, 32) | (1, 32) |
| 2 | 80 | (2, 32) | (2, 32) |
| 4 | 40 | (4, 32) | (4, 32) |
| 8 | 20 | (8, 32) | (8, 32) |
| 16 | 10 | (16, 32) | (16, 32) |
| 32 | 5 | (32, 32) | (32, 32) |

## E3 + E7. Dimensao do embedding x estruturas produzidas

| output_dim | forma da entrada do modelo | params tabela de tokens | params tabela posicional | tempo (ms) |
|---|---|---|---|---|
| 16 | (4, 8, 16) | 804,112 | 128 | 0.138 |
| 50 | (4, 8, 50) | 2,512,850 | 400 | 0.189 |
| 128 | (4, 8, 128) | 6,432,896 | 1,024 | 0.322 |
| 256 | (4, 8, 256) | 12,865,792 | 2,048 | 0.701 |
| 768 | (4, 8, 768) | 38,597,376 | 6,144 | 1.388 |

## E4 + E5. Quantidade de tokens para textos diferentes

| texto | num. caracteres | tokens (regex) | tokens (BPE/GPT-2) | tokens/caractere (BPE) |
|---|---|---|---|---|
| frase curta (PT) | 50 | 9 | 15 | 0.300 |
| frase curta (EN) | 44 | 7 | 7 | 0.159 |
| frase com palavra rara/inventada | 46 | 4 | 18 | 0.391 |
| paragrafo tecnico | 232 | 43 | 72 | 0.310 |
| the-verdict.txt (corpus completo) | 20479 | 4690 | 5145 | 0.251 |
