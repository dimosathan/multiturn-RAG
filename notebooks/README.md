# Notebooks

`reference/` contains the two Colab notebooks that document the experiments,
kept **for provenance only** (all outputs removed, credentials and endpoints
replaced by `os.environ[...]`). They still contain absolute Google Drive paths
and are not meant to be re-run; use `scripts/` and the `mtrag` package instead.

| Notebook | Contents | Ported to |
|---|---|---|
| `reference/TEST_SET_FINAL.ipynb` | Test-set pipeline: rewriting (cells 5–11), Cohere reranking (13), nested RRF (15), top-10 file (19), Task C generation (23–30), Task B generation (32–34), format checks | `mtrag.retrieval`, `mtrag.generation`, `mtrag.answerability` |
| `reference/DevFinal.ipynb` | Development-set experiments: rewriter variants, Cohere-weight grid search, per-retriever pipelines (ELSER v1/v2, Cohere, BM25, SPLADE), RRF tuning, multi-retriever ensembles | ablation scripts (`scripts/fuse_runs.py`, `scripts/evaluate_retrieval.py`) |

Other notebooks on Google Drive (not included): `3.ipynb` (generation
versions v4.5–v5.10, Task B variants), `4.ipynb` (answerability experiments
incl. the multi-judge "Supreme Court" v2.0–v2.6, EDA and paper figures — the
EDA scripts are in `analysis/`), and `Task A/Best/*.ipynb` (per-corpus
retriever/rewriter studies).

Before committing a notebook, strip its outputs (`nbstripout <file>`) and
check that no key is hard-coded.
