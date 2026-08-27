<div align="center">

# Projeto LLM

### Construção de um *Large Language Model* From Scratch

**Universidade do Oeste de Santa Catarina — Campus de Joaçaba**
Área das Ciências Exatas e Tecnológicas

![Status](https://img.shields.io/badge/status-Sprint%202-blue)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Entrega](https://img.shields.io/badge/entrega%20final-03%2F12%2F2026-orange)

</div>

---

<div align="center">

| | |
|:--|:--|
| **Curso** | Engenharia de Computação |
| **Componente Curricular** | Inteligência Artificial e Sistemas Inteligentes |
| **Professor** | Kleyton Hoffmann |
| **Período Letivo** | 2026/2 |
| **Acadêmicos** | Vitoria Aparecida Vendausen<br>Matheus Dapper Alves Leite |

</div>

---

## Sumário

| | |
|:--|:--|
| [Sobre o projeto](#sobre-o-projeto) | [Cronograma e progresso](#cronograma-e-progresso) |
| [Objetivo](#objetivo) | [Glossário técnico](#glossário-técnico) |
| [Corpus e recorte](#corpus-e-recorte) | [Experimentos](#experimentos) |
| [Além do livro](#além-do-livro) | [Aplicação: em Desenvolvimento](#aplicação-em-desenvolvimento) |
| [Escopo e delimitações](#escopo-e-delimitações) | [Reprodutibilidade](#reprodutibilidade) |
| [Pipeline](#pipeline) | [Uso de IA generativa](#uso-de-ia-generativa) |
| [Estrutura do repositório](#estrutura-do-repositório) | [Referência principal](#referência-principal) |
| [Metodologia por sprint](#metodologia-por-sprint) | |

---

## Sobre o projeto

Este repositório documenta a construção incremental de um modelo de linguagem baseado na arquitetura *Transformer*, implementado a partir de seus componentes fundamentais.

O princípio orientador é a **implementação progressiva**: cada componente é construído, testado e compreendido individualmente antes de ser integrado ao conjunto. O projeto não recorre a modelos previamente treinados nem a serviços de inferência via API — o objetivo é compreender os mecanismos internos, e não consumir uma abstração pronta.

O desenvolvimento acompanha, capítulo a capítulo, a obra de referência de Sebastian Raschka, em Python com PyTorch.

## Objetivo

> **Geral.** Compreender os fundamentos computacionais dos modelos de linguagem por meio da implementação incremental dos principais componentes de uma arquitetura baseada em *Transformer*.

**Específicos:**

- Implementar mecanismos de tokenização e preparação de dados textuais
- Compreender e implementar representações vetoriais e *embeddings*
- Implementar mecanismos de atenção e *self-attention*
- Construir os componentes de uma arquitetura GPT
- Treinar o modelo, analisando funções de perda e comportamento do treinamento
- Realizar geração de texto com o modelo desenvolvido
- Aplicar estratégias de *fine-tuning*
- Analisar criticamente resultados, limitações e comportamento do modelo

## Corpus e recorte

A arquitetura é definida pela obra de referência; o **corpus é escolha deste projeto**, e é ele que lhe confere identidade.

| | |
|:--|:--|
| **Domínio escolhido** | Texto técnico de eletrônica em português |
| **Justificativa** | Vocabulário restrito e terminologia estável favorecem um modelo de dimensões modestas; permite avaliação qualitativa por conhecimento de domínio |
| **Origem** | *(preencher: apostilas, datasheets, material didático próprio)* |
| **Volume** | *(preencher após consolidação)* |

Esta decisão atravessa todo o projeto. O tamanho de vocabulário do tokenizador, o comportamento das *embeddings* e a qualidade do texto gerado são consequências diretas dela — e cada sprint registra o impacto observado.

A escolha também estabelece continuidade com a Sprint 6, dedicada a *fine-tuning* e adaptação a tarefas específicas.

## Além do livro

A obra de referência fornece a arquitetura e o roteiro de implementação. Este repositório acrescenta:

| Contribuição | Onde se registra |
|:--|:--|
| **Corpus próprio** e análise de seu impacto sobre o modelo | `data/` e `sprints/*/analise.md` |
| **Experimentos comparativos** não previstos na obra | `experimentos/` |
| **Glossário técnico cumulativo**, com relações entre conceitos | `glossario/` |
| **Testes unitários** por componente | `tests/` |
| **Análise crítica** de resultados, limitações e falhas observadas | `sprints/*/analise.md` |
| **Domínio de aplicação** para o *fine-tuning* | `src/finetuning/` |

A reprodução integral de código disponível na obra ou em repositórios públicos não constitui, por si, desenvolvimento da atividade. O trabalho está no que se decide, se mede e se interpreta.

## Escopo e delimitações

**Implementação integralmente em software.** Todo o projeto é desenvolvido em Python com PyTorch, conforme a linguagem adotada pela obra de referência. **Não há aceleração em hardware dedicado neste escopo** — a implementação de componentes em FPGA foi avaliada e deixada fora do ciclo inicial.

A razão é de prioridade, não de viabilidade. O projeto é incremental e semanal: cada sprint depende da anterior, e a compreensão arquitetural tem precedência sobre otimização de desempenho. Recursos investidos em descrição de hardware seriam subtraídos das semanas dedicadas aos mecanismos de atenção, núcleo conceitual da arquitetura.

**Reavaliação prevista.** Concluídas as Sprints 1 a 5 dentro do cronograma, a implementação de um bloco isolado em Verilog poderá ser incorporada como **experimento comparativo de custo computacional**, contrastando latência e vazão contra a execução em CPU — extensão opcional sobre componente já funcional, não requisito.

**Escala do modelo.** Dimensões compatíveis com os recursos computacionais disponíveis. O objetivo não é competir com modelos comerciais de larga escala, mas compreender experimentalmente os fundamentos de sua construção.

**Sem modelos pré-treinados nem APIs.** O recurso a pesos pré-treinados restringe-se ao que a obra de referência prevê no contexto de *fine-tuning*, nos capítulos finais.

## Pipeline

```
   Texto
     │
     ├─▶ Tokenização ──────────────▶ Token IDs
     │                                   │
     │                                   ▼
     │                              Embeddings
     │                                   │
     │                                   ▼
     │                        Positional Embeddings
     │                                   │
     │                                   ▼
     │                        Multi-Head Attention
     │                                   │
     │                                   ▼
     │                          Transformer Blocks
     │                                   │
     │                                   ▼
     │                                  GPT
     │                                   │
     │                    ┌──────────────┴──────────────┐
     │                    ▼                             ▼
     └───────────▶  Treinamento              Geração de Texto
```

Cada etapa corresponde a um módulo independente do repositório, com testes próprios e registro de experimentos.

## Estrutura do repositório

```
projeto-llm/
├── src/
│   ├── tokenizer/          # Cap. 2 — tokenização, vocabulário, Token IDs
│   ├── embeddings/         # Cap. 2 — embeddings e positional embeddings
│   ├── attention/          # Cap. 3 — self-attention, causal, multi-head
│   ├── transformer/        # Cap. 4 — layer norm, feed-forward, residuais
│   ├── gpt/                # Cap. 4 — modelo GPT integrado
│   ├── training/           # Cap. 5 — laço de treinamento, perda, otimizadores
│   ├── generation/         # Cap. 5 — estratégias de geração de texto
│   └── finetuning/         # Cap. 6 e 7 — adaptação a tarefas específicas
├── sprints/
│   ├── sprint-01/
│   │   ├── notas.md        # Notas de leitura orientada
│   │   ├── analise.md      # Análise dos resultados da sprint
│   │   └── README.md       # Como reproduzir os experimentos
│   └── ...
├── glossario/
│   └── glossario.md        # Glossário cumulativo do semestre
├── experimentos/
│   ├── notebooks/
│   └── resultados/         # Métricas, curvas de perda, comparativos
├── data/                   # Corpus de treinamento
├── tests/                  # Testes unitários por componente
├── docs/                   # Relatório técnico e diagramas
└── requirements.txt
```

Cada diretório em `src/` mantém documentação própria explicando o componente, suas escolhas de implementação e os experimentos realizados sobre ele.

## Metodologia por sprint

<div align="center">

**Leitura → Glossário → Quiz → Implementação → Experimentação → Análise**

</div>

| Etapa | Registro no repositório |
|:--|:--|
| **Leitura orientada** | Notas em `sprints/sprint-NN/notas.md` |
| **Glossário técnico** | Entradas acrescidas a `glossario/glossario.md` |
| **Quiz** | Avaliação individual, sem artefato no repositório |
| **Implementação** | Código em `src/`, com testes em `tests/` |
| **Experimentação** | *Notebooks* e resultados em `experimentos/` |
| **Análise** | Discussão em `sprints/sprint-NN/analise.md` |

Os componentes implementados em sprints anteriores são reutilizados e integrados nas seguintes, de modo que o repositório reflita a evolução acumulada do modelo.

## Cronograma e progresso

| Data | Sprint | Cap. | Foco | Estado |
|:--|:--|:--|:--|:--:|
| 30/07 | 0 | — | Ambiente, Git, PyTorch, estrutura do repositório | ✅ |
| 06/08 | 1 | 1 | Introdução aos LLMs, glossário, mapa conceitual GPT | ✅ |
| 13/08 | 1 | 1 | Discussão técnica e **entrega da Sprint 1** | ✅ |
| 20/08 | 2 | 2 | Tokenização, vocabulário, Token IDs | ✅ |
| 27/08 | 2 | 2 | Embeddings, positional embeddings, DataLoader e **entrega da Sprint 2** | ✅ |
| 03/09 | 3 | 3 | Mecanismos de atenção, *self-attention* | ▢ |
| 10/09 | 3 | 3 | *Scaled dot-product* e *causal attention* | ▢ |
| 17/09 | 3 | 3 | *Multi-head attention* e comparativos | ▢ |
| 24/09 | 3 | 3 | Integração da atenção e **entrega da Sprint 3** | ▢ |
| 15/10 | 4 | 4 | *Transformer block*, *layer norm*, *feed-forward* | ▢ |
| 22/10 | 4 | 4 | Conexões residuais e construção do GPT | ▢ |
| 29/10 | 4 | 4 | Inferência, geração e **entrega da Sprint 4** | ▢ |
| 05/11 | 5 | 5 | Treinamento, função de perda, otimizadores | ▢ |
| 12/11 | 5 | 5 | Avaliação do treinamento e hiperparâmetros | ▢ |
| 19/11 | 6 | 6 e 7 | *Fine-tuning* e integração final | ▢ |
| **03/12** | — | — | **Entrega final**: repositório, glossário, relatório | ▢ |

## Glossário técnico

O glossário é **cumulativo** e mantido em `glossario/glossario.md`. Cada entrada apresenta:

| Campo | Conteúdo |
|:--|:--|
| Termo original | Denominação em inglês |
| Equivalente | Tradução ou termo técnico em português |
| Definição | Descrição concisa do conceito |
| Função | Papel desempenhado no modelo de linguagem |
| Relações | Conexões com outros conceitos da arquitetura |
| Exemplo | Ilustração conceitual, matemática ou computacional |

O propósito não é traduzir terminologia, mas registrar o **papel funcional** de cada conceito dentro da arquitetura.

## Experimentos

Cada componente implementado é acompanhado de experimentação sobre parâmetros, configurações ou dados, observando o impacto sobre comportamento, desempenho ou custo computacional.

Os resultados são registrados em `experimentos/resultados/` e **discutidos** — gráficos e valores numéricos sem interpretação não constituem análise.

| Componente | Variável investigada | Sprint |
|:--|:--|:--:|
| Tokenizador | Tamanho de vocabulário e taxa de compressão | 2 |
| Tokenizador | Segmentação por caractere *versus* BPE no corpus adotado | 2 |
| Embeddings | Dimensionalidade e efeito sobre a perda | 2 |
| Atenção | Número de cabeças e custo computacional | 3 |
| Transformer | Profundidade e estabilidade do treinamento | 4 |
| Treinamento | Taxa de aprendizado, tamanho de lote, agendamento | 5 |
| Geração | Temperatura, *top-k* e diversidade do texto gerado | 5 |

## Aplicação: em Desenvolvimento

### Sprint 6 — Fine-tuning e Adaptação a Tarefas Específicas

Na Sprint 6, está sendo estudada a possibilidade de realizar o *fine-tuning* e a adaptação do modelo de linguagem para tarefas ou domínios específicos.

Nesta etapa, ainda estão sendo avaliadas diferentes possibilidades de aplicação para o modelo, considerando quais áreas poderiam se beneficiar de uma especialização e quais conjuntos de dados poderiam ser utilizados para esse processo. A definição do domínio, das tarefas e da estratégia de *fine-tuning* ainda está em desenvolvimento.

O objetivo desta etapa é identificar uma aplicação que permita avaliar, de forma prática, os benefícios da especialização do modelo em comparação ao seu comportamento geral. Entre os critérios considerados estão a disponibilidade e qualidade dos dados, a possibilidade de avaliar objetivamente as respostas e a compatibilidade do domínio escolhido com as capacidades e limitações do modelo.

Dessa forma, o *fine-tuning* ainda não é considerado uma etapa concluída, permanecendo como uma possibilidade em investigação dentro do desenvolvimento do projeto. A definição do domínio de aplicação e da metodologia de treinamento será realizada conforme os resultados das análises e experimentações das etapas anteriores.

## Reprodutibilidade

```bash
git clone <url-do-repositorio>
cd projeto-llm

python3.11 -m venv .venv            # requer Python 3.11 (tensorflow ainda não suporta versões mais novas)
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pytest tests/                       # verifica os componentes implementados
```

Cada sprint possui instruções próprias em `sprints/sprint-NN/README.md`, permitindo reproduzir isoladamente os experimentos correspondentes.

## Uso de IA generativa

Ferramentas de IA generativa foram empregadas como apoio ao desenvolvimento — interpretação de erros, esclarecimento conceitual e revisão de documentação — conforme facultado pelo plano da disciplina.

Todo código, análise e resultado presente neste repositório foi compreendido e é passível de explicação, alteração e extensão por seu autor, em conformidade com os critérios de arguição estabelecidos.

## Referência principal

RASCHKA, Sebastian. *Build a Large Language Model (From Scratch)*. Manning Publications.

---

<div align="center">
<sub>Unoesc · Campus de Joaçaba · 2026/2</sub>
</div>
