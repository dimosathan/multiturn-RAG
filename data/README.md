# Data

This directory is intentionally empty (except for this README).
All data used in our experiments comes from the official
[IBM MT-RAG Benchmark](https://github.com/IBM/mt-rag-benchmark) repository,
which is also the official dataset for **SemEval 2026 Task 8 (MTRAGEval)**.

---

## Setup

### 1. Clone the benchmark repository

```bash
git clone https://github.com/IBM/mt-rag-benchmark
```

### 2. Decompress the corpora

```bash
cd mt-rag-benchmark/corpora/passage_level/
unzip clapnq.jsonl.zip
unzip cloud.jsonl.zip
unzip fiqa.jsonl.zip
unzip govt.jsonl.zip
```

### 3. Place the files as follows

Create the following structure inside the `data/` directory of this repo:
data/
├── corpora/
│ ├── clapnq.jsonl # from mt-rag-benchmark/corpora/passage_level/clapnq.jsonl.zip
│ ├── cloud.jsonl # from mt-rag-benchmark/corpora/passage_level/cloud.jsonl.zip
│ ├── fiqa.jsonl # from mt-rag-benchmark/corpora/passage_level/fiqa.jsonl.zip
│ └── govt.jsonl # from mt-rag-benchmark/corpora/passage_level/govt.jsonl.zip
│
├── retrieval/
│ ├── clapnq/
│ │ ├── dev.tsv # qrels (relevance judgments)
│ │ ├── clapnq_lastturn.jsonl # queries: last turn only
│ │ └── clapnq_questions.jsonl # queries: full question form
│ ├── cloud/
│ │ ├── dev.tsv
│ │ ├── cloud_lastturn.jsonl
│ │ └── cloud_questions.jsonl
│ ├── fiqa/
│ │ ├── dev.tsv
│ │ ├── fiqa_lastturn.jsonl
│ │ └── fiqa_questions.jsonl
│ └── govt/
│ ├── dev.tsv
│ ├── govt_lastturn.jsonl
│ └── govt_questions.jsonl
│
└── generation/
├── reference.jsonl # Generation using reference passages (842 tasks)
├── reference+RAG.jsonl # Reference passages kept in top-5 (436 tasks)
└── RAG.jsonl # Full RAG: top-5 retrieved passages (842 tasks)

text

> **Note:** We use **passage-level** corpora (not document-level) throughout
> all experiments, consistent with our Elasticsearch ELSER v1 index setup.

---

## Corpora Details

| Corpus | Domain | # Documents | # Passages |
|--------|--------|------------|-----------|
| ClapNQ | Wikipedia | 4,293 | 183,408 |
| Cloud  | Technical Documentation | 57,638 | 61,022 |
| FiQA   | Finance | 7,661 | 49,607 |
| Govt   | Government | 8,578 | 72,422 |

---

## Retrieval Tasks

The benchmark provides queries in 3 forms per corpus:

| File | Description |
|------|-------------|
| `dev.tsv` | Qrels (relevance judgments): `query_id`, `doc_id`, `relevance_score` |
| `*_lastturn.jsonl` | Last user turn only — used as input in our retrieval pipeline |
| `*_questions.jsonl` | Full standalone question form |

---

## Generation Tasks

The 110 multi-turn conversations are converted into **842 evaluation tasks**.
Each task corresponds to a conversation turn $k$, containing all previous turns
plus the current user question.

Generation tasks are provided under three retrieval settings:

| Setting | Description | File | # Tasks |
|---------|-------------|------|---------|
| Reference | Generation using gold reference passages | `reference.jsonl` | 842 |
| Reference + RAG | Retrieved passages with reference kept in top-5 (tasks with ≤2 contexts) | `reference+RAG.jsonl` | 436 |
| Full RAG | Top-5 retrieved passages only | `RAG.jsonl` | 842 |

### Task Format

Each line in the generation JSONL files follows this schema:

```json
{
  "task_id": "unique_id_as_string",
  "conversation_id": "unique_id_as_string",
  "task_type": "rag",
  "turn": 3,
  "collection": "name_of_domain_corpus",
  "dataset": "MT-RAG Authors (Internal)",
  "contexts": [
    {
      "document_id": "id_1",
      "text": "document text for model consumption",
      "title": "string (optional)",
      "url": "string (optional)",
      "score": 0.92,
      "feedback": {
        "relevant": {
          "author_XYZ": { "value": "yes|no", "timestamp": 1700000000 }
        }
      },
      "query": { "...query metadata and string..." }
    }
  ],
  "input": [
    { "...previous conversation turns..." },
    {
      "speaker": "user",
      "text": "query text",
      "metadata": {
        "author_type": "human",
        "author_id": "author_XYZ",
        "created_at": 1700000000
      },
      "enrichments": {
        "answerability": ["ANSWERABLE|UNANSWERABLE|CONVERSATIONAL|PARTIAL"],
        "Question Type": ["Factoid|Composite|Keyword|Opinion|..."],
        "Multi-Turn": ["Follow-up|Clarification"]
      }
    }
  ],
  "targets": [
    {
      "speaker": "agent",
      "text": "response text",
      "metadata": {
        "author_type": "model",
        "author_id": "mixtral-8x7b-instruct-v01",
        "created_at": 1700000000
      }
    }
  ],
  "answerability": ["ANSWERABLE|UNANSWERABLE|CONVERSATIONAL|PARTIAL"],
  "Question Type": ["Factoid|Composite|Keyword|Opinion|..."],
  "Multi-Turn": ["Follow-up|Clarification"]
}
```

---

## Evaluation Scripts

All evaluation scripts are provided by the task organizers at:
[`mt-rag-benchmark/scripts/evaluation/`](https://github.com/IBM/mt-rag-benchmark/tree/main/scripts/evaluation)

### Scripts overview

| Script | Purpose |
|--------|---------|
| `run_retrieval_eval.py` | Evaluates retrieval predictions (nDCG@10, Recall@k) |
| `run_generation_eval.py` | Evaluates generation predictions (answerability, extractiveness, fluency) |
| `run_algorithmic.py` | Runs algorithmic (non-LLM) generation metrics |
| `format_checker.py` | Validates prediction file format before submission |
| `judge_wrapper.py` | LLM-as-a-judge wrapper used by generation eval |
| `judge_utils.py` | Utilities for the judge pipeline |
| `azure_openai_client.py` | Azure OpenAI client used by the judge |
| `huggingface_client.py` | HuggingFace client used by the judge |
| `config.yaml` | Configuration for evaluation (model, thresholds, etc.) |
| `constraints.txt` | Dependencies for the evaluation scripts |

### Running retrieval evaluation

```bash
cp mt-rag-benchmark/scripts/evaluation/run_retrieval_eval.py .

python run_retrieval_eval.py \
  --inputfile predictions/your_predictions.jsonl \
  --outputfile results/your_results.jsonl
```

Expected prediction format (one JSON per line):

```json
{"taskid": "q001", "Collection": "mt-rag-clapnq-elser-512-100-20240503", "contexts": [{"documentid": "doc123", "score": 0.95}, ...]}
```

### Running generation evaluation

```bash
cp mt-rag-benchmark/scripts/evaluation/run_generation_eval.py .
cp mt-rag-benchmark/scripts/evaluation/config.yaml .

python run_generation_eval.py \
  --inputfile predictions/your_generation_predictions.jsonl \
  --outputfile results/your_generation_results.jsonl
```

> For full details on metrics, formats, and judge configuration, see the
> organizers' documentation at
> [`mt-rag-benchmark/scripts/evaluation/README.md`](https://github.com/IBM/mt-rag-benchmark/tree/main/scripts/evaluation).

---

## Citation

If you use this data, please cite the original MTRAG paper:

```bibtex
@article{katsis2025mtrag,
  title   = {MTRAG: A Multi-Turn Conversational Benchmark for Evaluating
             Retrieval-Augmented Generation Systems},
  author  = {Yannis Katsis and Sara Rosenthal and Kshitij Fadnis and
             Chulaka Gunasekara and Young-Suk Lee and Lucian Popa and
             Vraj Shah and Huaiyu Zhu and Danish Contractor and
             Marina Danilevsky},
  journal = {Transactions of the Association for Computational Linguistics},
  volume  = {13},
  pages   = {784--808},
  year    = {2025},
  doi     = {10.1162/TACL.a.19},
  url     = {https://doi.org/10.1162/TACL.a.19}
}
