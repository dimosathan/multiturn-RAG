# AILS-NTUA at SemEval-2026 Task 8: MTRAGEval

[![arXiv](https://img.shields.io/badge/arXiv-2603.10524-b31b1b.svg)](https://arxiv.org/abs/2603.10524)
[![SemEval 2026](https://img.shields.io/badge/SemEval-2026%20Task%208-blue)](https://semeval.github.io/SemEval2026/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

> 🏆 **1st / Task A** (nDCG@5 0.5776, +20.5%) · 🥈 **2nd / Task B** (HM 0.7698)

Official implementation of the **AILS-NTUA** system for  
**SemEval-2026 Task 8: MTRAGEval** — evaluation of Multi-Turn RAG systems.

---

## 📄 Paper

👉 [AILS-NTUA at SemEval-2026 Task 8: Query Diversity via Nested RRF and Agentic Evidence Grounding for Multi-Turn RAG](https://arxiv.org/abs/2603.10524)

---

## 🧠 System Overview

Our system is built around two key ideas:

### 1. Query Diversity over Retriever Diversity (Task A)

We generate **five complementary LLM rewrites** per query:
- Minimal
- Corpus-Specific
- HyDE
- Chain-of-Thought
- Anchor Keywords

and combine them using **Nested RRF**, instead of using multiple retrievers.

👉 Result: +25.7% Recall@5 vs no rewriting

---

### 2. Agentic Evidence Commitment (Task B)

A structured generation pipeline:


Span Extraction → Dual Candidates → Judge Selection → Micro-adjustment


Key components:
- Evidence span extraction (DeepSeek-V3.2)
- Dual candidate generation (GPT-4o)
- Technical + user satisfaction judges
- Extractiveness shaping

👉 Result: +8.9 HM over greedy generation

---

<p align="center">
  <img src="paper/figures/architecture.png" width="720"/>
</p>

---

## 📊 Results

| Task | Metric | Score | Rank |
|------|--------|-------|------|
| Task A | nDCG@5 | 0.5776 | 🥇 1 / 38 |
| Task B | HM | 0.7698 | 🥈 2 / 26 |
| Task C | HM | 0.5409 | 11 / 29 |

---

## 📂 Repository Structure


configs/ # experiment configs
src/ # core implementation (reusable code)
scripts/ # runnable entry points
utils/ # helpers (I/O, prompts, misc)
notebooks/ # experiments & analysis
data/ # dataset (not included)
outputs/ # predictions & logs
paper/ # paper & figures


---

## ⚙️ Setup

```bash
git clone https://github.com/dimosathan/semeval26-task8.git
cd semeval26-task8
pip install -r requirements.txt

Set API keys:

export OPENAI_API_KEY="..."
export DEEPSEEK_API_KEY="..."
export COHERE_API_KEY="..."
export ELASTICSEARCH_URL="http://localhost:9200"
🚀 Running
Task A
python scripts/run_task_a.py \
  --config configs/retrieval.yaml
Task B
python scripts/run_task_b.py \
  --config configs/generation.yaml
Task C
python scripts/run_task_c.py
```
##📊 Dataset

We use the MTRAG benchmark:

👉 https://github.com/IBM/mt-rag-benchmark

All data are in English.

Setup instructions:

👉 see data/README.md

##🔬 Key Findings
Query diversity > retriever diversity
Nested RRF > flat RRF
Answerability is the main bottleneck
Extractiveness shaping improves faithfulness
##📚 Citation
@inproceedings{athanasiou-etal-2026-ails,
  title={AILS-NTUA at SemEval-2026 Task 8},
  author={Athanasiou, Dimosthenis and others},
  booktitle={SemEval-2026},
  year={2026}
}
##🙌 Acknowledgements

We thank the MTRAG benchmark authors (IBM Research)
and the SemEval-2026 Task 8 organizers.

AILS Lab — NTUA


