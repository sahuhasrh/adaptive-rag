
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from backend.services.rag_service import RAGService


def main():
    rag = RAGService()

    print("\nAdaptiveRAG Research Assistant")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            continue

        try:
            result = rag.answer(question)

            print("\nAssistant:\n")
            print(result["answer"])

            print("\nRetrieved sources:")
            for source in result["sources"]:
                print(
                    f"[{source['citation_id']}] "
                    f"{source['paper_title']} | "
                    f"{source['section']} | "
                    f"Pages {source['page_start']}-{source['page_end']} | "
                    f"Score: {source['score']:.4f}"
                )

            print("\n" + "-" * 70)

        except Exception as error:
            print(f"\nError: {error}\n")


if __name__ == "__main__":
    main()
