# Implementation notes

This document maps the components described in the paper (arXiv:2603.10524)
and in the diploma thesis to the code of this repository and to the reference
notebooks (`notebooks/reference/`), and records implementation details that
are needed to run the system but are not spelled out in the text.

## Component map

| Paper / thesis | Code | Reference notebook |
|---|---|---|
| Five rewriting strategies, prompts (App. B.13) | `mtrag/retrieval/rewriters.py`, `prompts.py` | `TEST_SET_FINAL` cells 5, 7, 9, 11 |
| ELSER v1, top-100 per rewrite | `mtrag/retrieval/elser.py` | cells 5–11 |
| Hybrid reranking, Cohere Rerank v4, weighted RRF (Eq. 1, k = 60, α = 0.5) | `mtrag/retrieval/rerank.py` | cell 13; `DevFinal` cells 26–31 |
| Nested RRF (Eq. 2, Table 2, k_internal = 40) | `mtrag/retrieval/fusion.py` | cell 15 |
| Task B pipeline (App. C, Eqs. 4–6, Tables 30, 32) | `mtrag/generation/` | cell 33 |
| Task C multi-judge gate + arbiter, τ = 0.7, top-3 (§3, App. D) | `mtrag/answerability/multi_judge.py`, `mtrag/rag.py` | `4.ipynb` cells 72–78 |
| Single-judge baseline (Table 39) | `mtrag/answerability/single.py` | cell 25 |
| Retrieval metrics, paired bootstrap (10,000 resamples) | `mtrag/evaluation.py`, `scripts/evaluate_retrieval.py` | — |
| Development-set input, no-rewrite baseline | `scripts/prepare_dev_tasks.py`, `run_task_a.py --strategies none` | `DevFinal` cell 7 (`load_queries`) |
| ELSER indexing | `scripts/build_elser_index.py` | `DevFinal` cell 37 |
| Dataset statistics (App. A) | `analysis/eda_*.py`, `analysis/corpus_stats.py` | `4.ipynb` cells 79–83 |
| Per-turn and standalone analysis (App. B.11–B.12) | `analysis/task_a_supplementary.py` | `4.ipynb` cell 84 |

## Implementation details

**Rewriting.** Prompts are byte-identical to the reference notebooks
(`tests/test_rewrite_prompts.py`). First turns are returned unchanged by all
strategies, and any parsing failure falls back to the original query. The
XML history keeps the tag order of the reference implementation
(`format_history_xml(..., legacy_tag_order=True)`), so the rewrites can be
reproduced exactly.

**Development-set input.** `scripts/prepare_dev_tasks.py` converts the
retrieval query files (`<corpus>_questions.jsonl`, the input of the
development notebooks) into task rows. Since these files hold the user turns
of each conversation, the assistant-turn window of the rewriters only takes
effect with `--source generation` (conversations from `reference.jsonl`) or
on the test set, whose `input` contains both speakers.

**Indexing.** `scripts/build_elser_index.py` reproduces the index layout of
the reference notebooks (`doc_id` keyword, `text`, `text_embedding`
`sparse_vector` written by an ingest pipeline that runs the ELSER model on the
passage text).

**Run provenance.** Every script writes the resolved configuration, model
routing, input checksums and git commit next to its outputs
(`mtrag.utils.write_run_metadata`).

**Reranking.** The rerank query is the strategy's own rewrite (for HyDE, the
rewrite plus the hypothetical passage). Passages are sent as
`title + " " + text`, truncated to 2,000 characters. The default fusion is
weighted RRF over the top-60 candidates. The score-interpolation variant
explored in the development notebooks, `(1−α)·minmax(ELSER) + α·minmax(Cohere)`,
is available as `rerank.method: minmax`.

**Generation.** The user-satisfaction judge is sampled with a per-task seeded
draw (`seed:task_id`, rate 0.6). This keeps the judged subset identical across
runs regardless of thread scheduling. Model routing is in `configs/models.yaml`.

**Answerability.** The arbiter returns a label and a confidence. A turn is
refused when the label is UNANSWERABLE and the confidence is at least
`refusal_threshold` (0.70). The new-entity guard of v2.6 pre-sets the document
and span judges to UNANSWERABLE with `guard_confidence = 0.90` (configurable).

**Determinism and cost.** All LLM stages call hosted models, so outputs can
change with API versions even at τ = 0. Every stage caches its outputs under
`outputs/<run>/`, and every script prints a token/cost report.
