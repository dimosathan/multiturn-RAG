from utils.io_utils import load_jsonl, save_jsonl, load_config
from src.retrieval import run_retrieval_pipeline
from src.generation import run_generation_pipeline


def main():
    retrieval_config = load_config("configs/retrieval.yaml")
    generation_config = load_config("configs/generation.yaml")

    data = load_jsonl("data/processed/dev_conversations.jsonl")

    results = []

    for item in data:
        passages = run_retrieval_pipeline(item, retrieval_config)
        item["contexts"] = passages

        answer = run_generation_pipeline(item, generation_config)

        results.append({
            "taskid": item["id"],
            "response": answer
        })

    save_jsonl(results, "outputs/predictions/task_c_dev.jsonl")


if __name__ == "__main__":
    main()
