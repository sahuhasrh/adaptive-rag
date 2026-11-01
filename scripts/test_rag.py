
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from backend.services.rag_service import RAGService
from backend.database.feedback_store import (
    save_feedback,
    get_feedback_summary,
)


REASONS = {
    "1": "irrelevant_retrieval",
    "2": "missing_information",
    "3": "answer_incorrect",
}


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

            print("\nAnswer ID:", result["answer_id"])

            print("\nRetrieved sources:")
            for source in result["sources"]:
                print(
                    f"[{source['citation_id']}] "
                    f"{source['paper_title']} | "
                    f"{source['section']} | "
                    f"Pages {source['page_start']}-"
                    f"{source['page_end']} | "
                    f"Score: {source['score']:.4f}"
                )

            print("\nWas this answer helpful?")
            print("1. Yes (thumbs up)")
            print("2. No (thumbs down)")

            choice = input("Your feedback (1/2/Enter to skip): ").strip()

            if choice == "1":
                save_feedback(result["answer_id"], rating=1)
                print("Positive feedback saved.")

            elif choice == "2":
                print("\nWhy was this answer not helpful?")
                print("1. Retrieved documents were irrelevant")
                print("2. Important information was missing")
                print("3. Documents were relevant, but answer was incorrect")

                reason_choice = input("Choose reason (1/2/3): ").strip()

                if reason_choice not in REASONS:
                    print("Invalid reason. Feedback was not saved.")
                else:
                    save_feedback(
                        result["answer_id"],
                        rating=-1,
                        reason=REASONS[reason_choice],
                    )
                    print("Negative feedback saved.")

            print("\nFeedback summary:")
            print(get_feedback_summary())

            print("\n" + "-" * 70)

        except Exception as error:
            print(f"\nError: {error}\n")


if __name__ == "__main__":
    main()
