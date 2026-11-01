
import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from backend.retrieval.retriever import ResearchRetriever


def display_results(title, results):
    print(f"\n{title}")
    print("=" * 85)

    for rank, item in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print(f"Paper: {item['paper_title']}")
        print(f"Section: {item['section']}")
        print(f"Pages: {item['page_start']}-{item['page_end']}")
        print(f"Chunk ID: {item['chunk_id']}")
        print(f"Original similarity: {item.get('original_score', item['score']):.4f}")
        print(f"Feedback adjustment: {item.get('feedback_adjustment', 0):+.4f}")
        print(f"Adjusted score: {item.get('adjusted_score', item['score']):.4f}")
        print(f"Feedback evidence: {item.get('feedback_evidence_count', 0)}")
        print(f"Text: {item['text'][:350]}...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    retriever = ResearchRetriever()

    baseline = retriever.search(
        query=args.query,
        top_k=args.top_k,
    )

    adaptive = retriever.search_adaptive(
        query=args.query,
        top_k=args.top_k,
    )

    display_results("BASELINE RETRIEVAL", baseline)
    display_results("ADAPTIVE RETRIEVAL", adaptive)


if __name__ == "__main__":
    main()
