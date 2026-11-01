
import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from backend.retrieval.retriever import ResearchRetriever


def main():
    parser = argparse.ArgumentParser(
        description="Test semantic retrieval from research papers."
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Question to search for.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve.",
    )

    parser.add_argument(
        "--paper",
        default=None,
        help="Optional paper ID filter, e.g. dpr.",
    )

    args = parser.parse_args()

    retriever = ResearchRetriever()

    results = retriever.search(
        query=args.query,
        top_k=args.top_k,
        paper_id=args.paper,
    )

    if not results:
        print("No results found.")
        return

    print(f"\nQuery: {args.query}")
    print(f"Retrieved: {len(results)} chunks\n")

    for index, result in enumerate(results, start=1):
        print("=" * 80)
        print(f"RESULT {index}")
        print(f"Similarity score: {result['score']:.4f}")
        print(f"Paper: {result['paper_title']}")
        print(f"Paper ID: {result['paper_id']}")
        print(f"Section: {result['section']}")
        print(f"Subsection: {result['subsection']}")
        print(
            f"Pages: {result['page_start']}-{result['page_end']}"
        )
        print("\nText:")
        print(result["text"])
        print()


if __name__ == "__main__":
    main()
