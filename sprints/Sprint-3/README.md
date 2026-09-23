# Sprint 03

**Capítulo:** 3 — *Coding Attention Mechanisms*.  
**Foco:** scaled dot-product attention, self-attention com pesos treináveis, causal attention (máscara autoregressiva), multi-head attention e análises comparativas.

## O que foi entregue
- `src/attention/scaled_dot_product.py` — função matemática base da atenção com escalonamento por $\sqrt{d_k}$, aplicação de máscara e dropout.
- `src/attention/self_attention.py` — classe de auto-atenção simples (cabeça única) com matrizes de projeção linear aprendidas ($W_q, W_k, W_v, W_o$).
- `src/attention/causal_attention.py` — atenção autoregressiva com buffer de máscara triangular inferior e dropout, impedindo o vazamento de informação do futuro.
- `src/attention/multi_head_attention.py` — implementação eficiente e vetorizada em paralelo de múltiplas cabeças de atenção ($h$ cabeças com dimensão $d_k = d_{\text{out}} / h$).
- `src/attention/__init__.py` — exportação centralizada dos quatro módulos de atenção da sprint.
- `tests/test_attention.py` — testes unitários automatizados validando formatos dos tensores, propriedades do Softmax ($\sum = 1.0$) e anulamento estrito de posições futuras.
- `experimentos/notebooks/sprint-03-experimentos.ipynb` — notebook autocontido (roda no Google Colab) com todos os testes práticos e mapas de calor (*heatmaps*).
- `experimentos/resultados/sprint-03-experimentos.md` — saída bruta dos experimentos em tabelas comparativas.
- `glossario/glossario.md` — seção "Capítulo 3" acrescentada ao glossário cumulativo.
- `docs/sprint-03/` — versões em PDF (compiladas de LaTeX) do glossário e da análise desta sprint.
- `notas.md`, `analise.md` — leitura orientada e análise técnica detalhada dos resultados.

## Entrada e Dados
Os módulos de atenção recebem os tensores no formato `(batch_size, context_length, output_dim)` produzidos pelo pipeline de embeddings da Sprint 2 (a partir do corpus *The Verdict* e entradas de teste controladas). Esses mesmos blocos serão reutilizados diretamente na Sprint 4 para a construção dos blocos do Transformer.

## Como reproduzir

```bash
# a partir da raiz do repositório
source .venv/bin/activate # ou: python3.11 -m venv .venv && pip install -r requirements.txt

# módulos individuais (cada um roda uma demonstração ao ser executado diretamente)
python3 -m src.attention.scaled_dot_product
python3 -m src.attention.self_attention
python3 -m src.attention.causal_attention
python3 -m src.attention.multi_head_attention

# testes unitários desta sprint
pytest tests/test_attention.py -v
