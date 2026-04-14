# AILS-NTUA at SemEval-2026 Task 8

**Query Diversity and Evidence-Guided Agentic Generation for Multi-Turn RAG**

🏆 **1st place in Subtask A**  
🥈 **2nd place in Subtask B**

---

## 📄 Paper

- 📌 Camera-ready (to appear): *link εδώ*
- 📌 ArXiv: *link εδώ (αν έχεις)*

---

## 🧠 Overview

We present our system for SemEval-2026 Task 8 on evaluating multi-turn RAG conversations.

Our approach focuses on:

- **Query Diversity over Retriever Diversity** for robust retrieval
- **Nested Reciprocal Rank Fusion (Nested RRF)** for combining multiple query strategies
- **Agentic Generation Pipeline**, including:
  - Answerability classification
  - Evidence span extraction
  - Dual candidate generation
  - LLM-based judge selection
- **Extractiveness shaping** to balance faithfulness and naturalness

---

## 🏗️ System Architecture

The system consists of two main components:

### Retrieval (Task A)
- Multi-strategy query rewriting:
  - Minimal rewrite
  - Corpus-specific rewrite
  - HyDE
  - Chain-of-thought rewrite
  - Anchor keyword rewrite
- Fusion via Nested RRF

### Generation (Task B & C)
- Answerability detection
- Evidence span extraction
- Dual candidate generation
- LLM-based selection (user / system judge)
- Post-processing with extractiveness shaping

---

## 📂 Repository Structure

```text
configs/        # experiment configurations
methods/        # core system implementation
utils/          # helper functions
notebooks/      # experiments and analysis
data/           # (not included)
outputs/        # predictions and logs
paper/          # paper, figures, tables
