# data/

## `the-verdict.txt`

Corpus de exemplo do Capítulo 2 do livro de referência (*The Verdict*, Edith Wharton,
1908 — domínio público). Usado na Sprint 2 para validar o pipeline de tokenização,
vocabulário, sequências de treinamento e embeddings.

**Ainda não é o corpus final do projeto.** O README raiz define o domínio escolhido
para o modelo como *texto técnico de eletrônica em português* — esse corpus ainda
precisa ser reunido (apostilas, datasheets, material didático próprio) e adicionado
aqui. Quando isso acontecer, os componentes de `src/tokenizer/` e `src/embeddings/`
não precisam mudar: basta apontar o pipeline (`src/pipeline.py`,
`create_dataloader_v1`) para o novo arquivo de texto.
