# AILS-NTUA at SemEval-2026 Task 8: MTRAGEval

[![arXiv](https://img.shields.io/badge/arXiv-2603.10524-b31b1b.svg)](https://arxiv.org/abs/2603.10524)
[![SemEval 2026](https://img.shields.io/badge/SemEval-2026%20Task%208-blue)](https://semeval.github.io/SemEval2026/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

> **🏆 1st / Task A** (nDCG@5 0.5776, +20.5% over baseline) · **2nd / Task B** (HM 0.7698)

Official code for the AILS-NTUA system submitted to **SemEval-2026 Task 8: MTRAGEval** —
evaluation of Multi-Turn Retrieval-Augmented Generation systems across three subtasks.

**Authors:**
[Dimosthenis Athanasiou](https://github.com/dimosathan),
Maria Lymperaiou,
Giorgos Filandrianos,
Athanasios Voulodimos,
Giorgos Stamou —
*NTUA AILS Lab*

---

## System Overview

Our system is built around two core ideas:

**1. Query Diversity over Retriever Diversity (Task A)**
Instead of combining multiple retrievers, we generate five complementary LLM rewrites of each
conversational query — Minimal, Corpus-Specific, HyDE, Chain-of-Thought, and Anchor-Keywords —
and fuse the results with a novel *Nested RRF* strategy. This yields +25.7% Recall@5 over
no-rewriting baselines while relying on a single ELSER v1 index.

**2. Agentic Evidence Commitment (Task B)**
A multi-stage generation pipeline commits to evidence before generating:
span extraction → dual-candidate generation → multi-judge selection with extractiveness shaping
→ micro-adjustment. This yields +8.9 HM over single greedy generation.

<p align="center">
  <img src="paper/figures/architecture.png" alt="System Architecture" width="720"/>
</p>

---

## Results

| Task | Description | Metric | Score | Rank |
|------|-------------|--------|-------|------|
| **A** | Multi-Turn Retrieval | nDCG@5 | 0.5776 | **1 / 38** |
| **B** | Grounded Generation | HM (Extractiveness × Correctness) | 0.7698 | **2 / 26** |
| **C** | End-to-End RAG | HM | 0.5409 | 11 / 29 |

---

## Repository Structure

```
semeval26-task8/
├── configs/                   # YAML configs — model names, hyperparameters, fusion weights
│   ├── retrieval.yaml         #   Nested RRF weights, rewriter models, ELSER settings
│   ├── generation.yaml        #   Generator models, temperatures, judge thresholds
│   └── task_c.yaml            #   End-to-end pipeline overrides
│
├── src/                       # Core reusable library (importable via `from src.X import Y`)
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── rewriting.py       #   5 query rewrite strategies (Minimal, CS, HyDE, CoT, AnchorKW)
│   │   ├── elastic.py         #   ELSER v1 retrieval interface
│   │   ├── reranking.py       #   Cohere Rerank v4 + weighted scoring
│   │   └── fusion.py          #   Nested RRF (main contribution)
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── answerability.py   #   Answerability classification (DeepSeek-V3.2)
│   │   ├── extraction.py      #   Evidence span extraction
│   │   ├── candidates.py      #   Dual-candidate generation (GPT-4o)
│   │   ├── judging.py         #   Technical + User Satisfaction judge
│   │   └── adjustment.py      #   Post-generation micro-adjustment pass
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py         #   nDCG@k, Recall@k, HM, extractiveness
│       └── answerability.py   #   Answerability classification metrics
│
├── scripts/                   # Runnable entry points (CLI, call from terminal)
│   ├── run_task_a.py          #   python scripts/run_task_a.py --config configs/retrieval.yaml
│   ├── run_task_b.py          #   python scripts/run_task_b.py --config configs/generation.yaml
│   └── run_task_c.py          #   python scripts/run_task_c.py
│
├── utils/
│   ├── __init__.py
│   ├── io_utils.py            #   load/save jsonl, config parsing, passage loading
│   ├── prompt_utils.py        #   XML wrappers, history formatter, prompt builders
│   └── general.py             #   batching, timing, logging helpers
│
├── notebooks/                 # Experiments and analysis (import from src/, never define logic here)
│   ├── 01_data_exploration.ipynb
│   ├── 02_task_a_ablation.ipynb      #   Table 3: Nested RRF ablation
│   ├── 03_task_b_analysis.ipynb      #   Table 4: Generation pipeline ablation
│   ├── 04_task_c_error_analysis.ipynb
│   └── 99_scratch.ipynb
│
├── data/                      # See data/README.md for download instructions
│   ├── README.md              #   ← Download links for MTRAG corpora
│   ├── raw/                   #   Original MTRAG files (git-ignored)
│   ├── processed/             #   Chunked passages, dev/test splits (git-ignored)
│   └── external/              #   ClapNQ, FiQA, Cloud, Govt corpora (git-ignored)
│
├── outputs/                   # Generated artifacts (git-ignored except final tables/figures)
│   ├── predictions/           #   Task A/B/C output jsonl files
│   ├── logs/                  #   Run logs
│   ├── figures/               #   Generated plots
│   └── tables/                #   LaTeX / CSV result tables
│
├── paper/                     # Camera-ready assets
│   ├── figures/
│   ├── tables/
│   └── notes/
│
├── requirements.txt
└── .gitignore
```

> **Design philosophy (borrowed from [llm-unlearning](https://github.com/iraklis07/llm-unlearning)):**
> Experiments live in `notebooks/`, reusable logic lives in `src/` and `utils/`.
> `scripts/` are the CLI entry points that wire everything together.
> If a function appears in more than one notebook → it belongs in `src/`.

---

## Data

Our system is evaluated on the **MTRAG benchmark** ([IBM/mt-rag-benchmark](https://github.com/IBM/mt-rag-benchmark)),
which spans four document corpora:

| Corpus | Domain | # Passages |
|--------|--------|-----------|
| ClapNQ | Wikipedia | 183,408 |
| Cloud | Technical Documentation | 61,022 |
| FiQA | Finance | 49,607 |
| Govt | Government | 72,422 |

Download instructions and paths are in [`data/README.md`](data/README.md).

---

## Setup

```bash
git clone https://github.com/dimosathan/semeval26-task8.git
cd semeval26-task8
pip install -r requirements.txt
```

Set your API keys:

```bash
export OPENAI_API_KEY="sk-..."
export DEEPSEEK_API_KEY="..."
export COHERE_API_KEY="..."
export ELASTICSEARCH_URL="http://localhost:9200"
```

---

## Running the System

### Task A — Multi-Turn Retrieval

```bash
python scripts/run_task_a.py \\
    --config configs/retrieval.yaml \\
    --input  data/processed/dev_conversations.jsonl \\
    --output outputs/predictions/task_a_dev.jsonl
```

Or programmatically:

```python
from src.retrieval.fusion import nested_rrf
from src.retrieval.rewriting import rewrite_all
from utils.io_utils import load_config

config = load_config("configs/retrieval.yaml")
rewrites = rewrite_all(conversation, config)
passages = nested_rrf(rewrites, es_client, cohere_client, config)
```

### Task B — Grounded Generation

```bash
python scripts/run_task_b.py \\
    --config  configs/generation.yaml \\
    --input   outputs/predictions/task_a_dev.jsonl \\
    --output  outputs/predictions/task_b_dev.jsonl
```

### Task C — End-to-End RAG

```bash
python scripts/run_task_c.py \\
    --retrieval-config  configs/retrieval.yaml \\
    --generation-config configs/generation.yaml \\
    --input  data/processed/dev_conversations.jsonl \\
    --output outputs/predictions/task_c_dev.jsonl
```

---

## Models

| Stage | Model | Temp |
|-------|-------|------|
| Query rewriting | DeepSeek-V3.2 | 0.0 |
| Retrieval | ELSER v1 (Elasticsearch 8.10) | — |
| Reranking | Cohere Rerank v4 | — |
| Span extraction | DeepSeek-V3.2 | 0.0 |
| Candidate generation | GPT-4o | 0.0 / 0.1 |
| Technical judge | DeepSeek-V3.2 | 0.0 |
| User satisfaction judge | GPT-4o-mini | 0.0 |
| Micro-adjustment | GPT-4o-mini | 0.0 |

---

## Key Findings

- **Nested RRF** consistently outperforms flat RRF across all corpora and metrics
- **Query rewriting diversity** matters more than retriever diversity for conversational RAG
- **Answerability classification** is the single most impactful component for Task C
- **Extractiveness shaping** via judge selection improves HM without sacrificing correctness

See the ablation notebooks for full details: `notebooks/02_task_a_ablation.ipynb`, `notebooks/03_task_b_analysis.ipynb`.

---

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{athanasiou-etal-2026-ails,
  title     = "{AILS-NTUA} at {S}em{E}val-2026 Task 8: Query Diversity via Nested {RRF}
               and Agentic Evidence Grounding for Multi-Turn {RAG}",
  author    = "Athanasiou, Dimosthenis and Lymperaiou, Maria and Filandrianos, Giorgos
               and Voulodimos, Athanasios and Stamou, Giorgos",
  booktitle = "Proceedings of the 20th International Workshop on Semantic Evaluation (SemEval-2026)",
  year      = "2026",
  address   = "San Diego, California",
  publisher = "Association for Computational Linguistics",
  url       = "https://arxiv.org/abs/2603.10524",
}
```

If you use the MTRAG benchmark data, please also cite:

```bibtex
@article{katsis2025mtrag,
  title   = "{MTRAG}: A Multi-Turn Conversational Benchmark for Evaluating
             Retrieval-Augmented Generation Systems",
  author  = "Katsis, Yannis and Rosenthal, Sara and Fadnis, Kshitij and
             Gunasekara, Chulaka and Lee, Young-Suk and Popa, Lucian and
             Shah, Vraj and Zhu, Huaiyu and Contractor, Danish and Danilevsky, Marina",
  journal = "Transactions of the Association for Computational Linguistics",
  volume  = "13",
  pages   = "784--808",
  year    = "2025",
  doi     = "10.1162/TACL.a.19",
}
```

---

## Acknowledgements

We thank the MTRAG benchmark creators at IBM Research for the dataset and evaluation framework.
This work was conducted at the [AILS Lab](https://ails.ece.ntua.gr/), NTUA.
