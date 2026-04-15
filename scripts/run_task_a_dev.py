from utils.io_utils import load_jsonl, save_jsonl, load_config
from src.retrieval import run_retrieval_pipeline


def main():
    config = load_config("configs/retrieval.yaml")

    data = load_jsonl("data/processed/dev_conversations.jsonl")

    results = []
    for item in data:
        passages = run_retrieval_pipeline(item, config)

        results.append({
            "taskid": item["id"],
            "contexts": passages
        })

    save_jsonl(results, "outputs/predictions/task_a_dev.jsonl")


if __name__ == "__main__":
    main()
