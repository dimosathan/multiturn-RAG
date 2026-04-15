from utils.io_utils import load_jsonl, save_jsonl, load_config
from src.generation import run_generation_pipeline


def main():
    config = load_config("configs/generation.yaml")

    data = load_jsonl("outputs/predictions/task_a_dev.jsonl")

    results = []
    for item in data:
        answer = run_generation_pipeline(item, config)

        results.append({
            "taskid": item["taskid"],
            "response": answer
        })

    save_jsonl(results, "outputs/predictions/task_b_dev.jsonl")


if __name__ == "__main__":
    main()
