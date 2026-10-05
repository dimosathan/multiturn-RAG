# Results

## Official test-set results (SemEval-2026 Task 8)

| Task | Metric | Score | Rank | Best other system | Strongest organiser baseline |
|---|---|---|---|---|---|
| A — Retrieval | nDCG@5 | **0.5776** | **1 / 38** | — | 0.4795 (ELSER + GPT-OSS-20b rewrite) |
| B — Generation (reference passages) | HM | 0.7698 | 2 / 26 | 0.7827 | 0.6390 (GPT-OSS-120b) |
| C — End-to-end RAG | HM | 0.5409 | 11 / 29 | 0.5861 | 0.5366 (Qwen-30B-A3B-Thinking) |

HM = harmonic mean of RB_alg, RL_F and RB_llm.
Task B components: RB_alg 0.633 / RL_F 0.897 / RB_llm 0.832.
Task C components: RB_alg 0.400 / RL_F 0.729 / RB_llm 0.598.
Source: paper Table 13. "+20.5 %" in the paper is the relative gain over the
organiser baseline (0.5776 / 0.4795 − 1).


## Development-set results

Development numbers (Tables 4–6, 14–27, 33–39 of the paper) should be
regenerated with the scripts in this repository and stored here together with
the exact config used, e.g.

```
results/dev/<run_name>/config.yaml
results/dev/<run_name>/metrics.json
```
