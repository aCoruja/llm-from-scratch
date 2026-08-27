# Sprint 02

**Capítulo:** 2 — *Working with Text Data*.
**Foco:** tokenização, vocabulário, Token IDs, sequências de treinamento, embeddings,
positional embeddings e DataLoader.

## O que foi entregue

- `src/tokenizer/tokenizer.py` — tokenização por regex (didática) e BPE (`tiktoken`,
  vocabulário do GPT-2).
- `src/tokenizer/vocabulary.py` — construção do vocabulário e conversão
  Token ↔ Token ID ↔ Vocabulário, com tokens especiais `<|unk|>` e `<|endoftext|>`.
- `src/tokenizer/dataset.py` — `GPTDatasetV1` (janela deslizante) e
  `create_dataloader_v1`.
- `src/embeddings/embeddings.py` — token embeddings e positional embeddings.
- `src/pipeline.py` — integra as etapas acima no fluxo completo (texto → entrada do
  modelo), reaproveitado pelas sprints seguintes.
- `tests/test_tokenizer.py`, `test_vocabulary.py`, `test_dataset.py`,
  `test_embeddings.py` — testes unitários por componente.
- `experimentos/notebooks/sprint-02-experimentos.ipynb` — notebook autocontido
  (roda direto no Google Colab) com todo o pipeline e os experimentos, incluindo
  gráficos.
- `experimentos/resultados/sprint-02-experimentos.md` — saída bruta dos experimentos.
- `glossario/glossario.md` — seção "Capítulo 2" acrescentada ao glossário cumulativo.
- `docs/sprint-02/` — versões em PDF (compiladas de LaTeX) do glossário e da análise,
  com o mesmo conteúdo de `glossario/glossario.md` e `analise.md`.
- `notas.md`, `analise.md` — leitura orientada e análise técnica desta sprint.

## Corpus

Usa-se *The Verdict* (Edith Wharton, domínio público), o mesmo corpus de exemplo do
capítulo — ainda **não** o corpus de domínio próprio (eletrônica, em português)
descrito no README raiz. Ver [`data/README.md`](../../data/README.md). Trocar de
corpus não exige alterar `src/`: basta apontar `run_pipeline`/`create_dataloader_v1`
para o novo arquivo de texto.

## Como reproduzir

```bash
# a partir da raiz do repositório
source .venv/bin/activate   # ou: python3.11 -m venv .venv && pip install -r requirements.txt

# módulos individuais (cada um roda uma demonstração ao ser executado diretamente)
python3 -m src.tokenizer.tokenizer
python3 -m src.tokenizer.vocabulary
python3 -m src.tokenizer.dataset
python3 -m src.embeddings.embeddings

# pipeline completo: texto -> entrada do modelo
python3 -m src.pipeline

# testes unitários desta sprint
pytest tests/test_tokenizer.py tests/test_vocabulary.py tests/test_dataset.py tests/test_embeddings.py -v
```

O notebook em `experimentos/notebooks/sprint-02-experimentos.ipynb` pode ser aberto
diretamente no Google Colab (é autocontido: instala `tiktoken`, baixa o corpus e roda
tudo, sem depender de `src/`).
