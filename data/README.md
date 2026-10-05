# Data

No data are distributed with this repository. All files come from the
[IBM MT-RAG benchmark](https://github.com/IBM/mt-rag-benchmark) (development
set, Katsis et al., 2025) and from the SemEval-2026 Task 8 organisers (test
set). Place them as follows — these are the default paths used by `configs/`
and `analysis/`:

```
data/
├── corpora/                 # passage-level corpora (512 tokens, stride 100)
│   ├── clapnq.jsonl         # mt-rag-benchmark/corpora/passage_level/clapnq.jsonl.zip
│   ├── fiqa.jsonl
│   ├── govt.jsonl
│   └── cloud.jsonl
├── retrieval/               # dev retrieval tasks (Task A, 777 answerable/partial queries)
│   └── <corpus>/
│       ├── dev.tsv          # qrels: query-id  corpus-id  score
│       └── <corpus>_lastturn.jsonl / <corpus>_questions.jsonl
├── generation/              # dev generation tasks (842 turns)
│   ├── reference.jsonl      # Task B setting (gold passages)
│   ├── reference+RAG.jsonl
│   └── RAG.jsonl            # Task C setting (retrieved passages)
└── test/                    # SemEval-2026 Task 8 test set (507 tasks)
    ├── rag_taskAC.jsonl     # Tasks A and C input
    ├── reference_taskB.jsonl
    └── reference_final.jsonl  # released after the evaluation phase (with labels)
```

```bash
git clone https://github.com/IBM/mt-rag-benchmark
cd mt-rag-benchmark/corpora/passage_level && for c in clapnq cloud fiqa govt; do unzip $c.jsonl.zip; done
```

## Formats

* **Corpus line:** `{"_id": "551325-0-398", "title": "...", "text": "..."}` — the
  passage id is `<document-id>-<start>-<end>`.
* **Task line (`input`):** list of turns `{"speaker": "user"|"agent", "text": ...}`;
  the last element is the current user question. The collection is given by
  `Collection` (`clapnq`, `fiqa`, `govt`, `ibmcloud` in the test set; the full
  `mt-rag-*-elser-512-100-*` identifiers in the dev set — both are accepted).
* **Retrieval prediction:** `{"task_id", "Collection", "contexts": [{"document_id", "score"}, ...]}`.
* **Generation prediction:** the task line plus `"predictions": [{"text": "..."}]`.

## Corpus statistics

Recount the statistics from the files rather than copying them:

```bash
python analysis/corpus_stats.py data/corpora/{clapnq,fiqa,govt,cloud}.jsonl
```



## Indexing for ELSER v1

Task A queries Elasticsearch indices (one per corpus, names in
`configs/task_a.yaml`) whose documents carry the passage id in `doc_id` and a
`text_expansion` field `text_embedding` produced by an ingest pipeline with the
`.elser_model_1` model. See the Elastic documentation on
[ELSER](https://www.elastic.co/guide/en/machine-learning/current/ml-nlp-elser.html).

## Evaluation scripts

Official scoring uses the organisers' scripts in
[`mt-rag-benchmark/scripts/evaluation/`](https://github.com/IBM/mt-rag-benchmark/tree/main/scripts/evaluation)
(`run_retrieval_eval.py`, `run_generation_eval.py`, `format_checker.py`).
`scripts/evaluate_retrieval.py` in this repository is for fast iteration and
significance tests only.

## Citation

```bibtex
@article{katsis2025mtrag,
  title   = {MTRAG: A Multi-Turn Conversational Benchmark for Evaluating Retrieval-Augmented Generation Systems},
  author  = {Katsis, Yannis and Rosenthal, Sara and Fadnis, Kshitij and Gunasekara, Chulaka and Lee, Young-Suk and
             Popa, Lucian and Shah, Vraj and Zhu, Huaiyu and Contractor, Danish and Danilevsky, Marina},
  journal = {Transactions of the Association for Computational Linguistics},
  volume  = {13}, pages = {784--808}, year = {2025}, doi = {10.1162/TACL.a.19}
}
```
