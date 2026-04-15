from src.retrieval import run_retrieval_pipeline
from src.generation import run_generation_pipeline


def run_task_a(data, config):
    results = []

    for item in data:
        passages = run_retrieval_pipeline(item, config)

        results.append({
            "taskid": item["id"],
            "contexts": passages
        })

    return results


def run_task_b(data, config):
    results = []

    for item in data:
        answer = run_generation_pipeline(item, config)

        results.append({
            "taskid": item["taskid"],
            "response": answer
        })

    return results


def run_task_c(data, retrieval_config, generation_config):
    results = []

    for item in data:
        passages = run_retrieval_pipeline(item, retrieval_config)
        item["contexts"] = passages

        answer = run_generation_pipeline(item, generation_config)

        results.append({
            "taskid": item["id"],
            "response": answer
        })

    return results
