
import json
import sys
import time
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from backend.retrieval.retriever import ResearchRetriever

load_dotenv()

DATASET_PATH = PROJECT_ROOT / "data" / "evaluation" / "eval_questions.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "retrieval_results.json"


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def calculate_metrics(results, relevant_paper_ids, k=5):
    relevant = set(relevant_paper_ids)

    retrieved_papers = [
        item["paper_id"] for item in results[:k]
    ]

    unique_retrieved = set(retrieved_papers)

    recall = (
        len(unique_retrieved & relevant) / len(relevant)
        if relevant else 0.0
    )

    reciprocal_rank = 0.0
    for rank, paper_id in enumerate(retrieved_papers, start=1):
        if paper_id in relevant:
            reciprocal_rank = 1.0 / rank
            break

    return recall, reciprocal_rank


def evaluate_method(retriever, dataset, adaptive=False):
    recalls = []
    reciprocal_ranks = []
    latencies = []
    question_results = []

    for index, item in enumerate(dataset, start=1):
        question = item["question"]
        relevant_ids = item["relevant_paper_ids"]

        start = time.perf_counter()

        if adaptive:
            results = retriever.search_adaptive(question, top_k=5)
        else:
            results = retriever.search(question, top_k=5)

        latency = time.perf_counter() - start

        recall, reciprocal_rank = calculate_metrics(
            results, relevant_ids, k=5
        )

        recalls.append(recall)
        reciprocal_ranks.append(reciprocal_rank)
        latencies.append(latency)

        question_results.append({
            "question": question,
            "relevant_paper_ids": relevant_ids,
            "retrieved_paper_ids": [
                result["paper_id"] for result in results
            ],
            "recall_at_5": recall,
            "reciprocal_rank": reciprocal_rank,
            "latency_seconds": round(latency, 4),
        })

        print(
            f"[{index}/{len(dataset)}] "
            f"{'Adaptive' if adaptive else 'Baseline'}: "
            f"Recall@5={recall:.2f}, "
            f"RR={reciprocal_rank:.2f}, "
            f"Latency={latency:.3f}s"
        )

    return {
        "recall_at_5": mean(recalls) if recalls else 0.0,
        "mrr": mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "average_latency_seconds": mean(latencies) if latencies else 0.0,
        "question_results": question_results,
    }


def main():
    dataset = load_dataset()

    if not dataset:
        raise ValueError("Evaluation dataset is empty.")

    print(f"Loaded {len(dataset)} evaluation questions.")

    retriever = ResearchRetriever()

    print("\nRunning baseline retrieval...")
    baseline = evaluate_method(retriever, dataset, adaptive=False)

    print("\nRunning adaptive retrieval...")
    adaptive = evaluate_method(retriever, dataset, adaptive=True)

    output = {
        "dataset_size": len(dataset),
        "evaluation_level": "paper_id",
        "baseline": baseline,
        "adaptive": adaptive,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 55)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 55)

    print(
        f"Baseline Recall@5: "
        f"{baseline['recall_at_5']:.4f}"
    )
    print(
        f"Adaptive Recall@5: "
        f"{adaptive['recall_at_5']:.4f}"
    )

    print(f"\nBaseline MRR: {baseline['mrr']:.4f}")
    print(f"Adaptive MRR: {adaptive['mrr']:.4f}")

    print(
        f"\nBaseline average latency: "
        f"{baseline['average_latency_seconds']:.4f}s"
    )
    print(
        f"Adaptive average latency: "
        f"{adaptive['average_latency_seconds']:.4f}s"
    )

    print(f"\nDetailed results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
