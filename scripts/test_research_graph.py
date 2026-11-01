import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


import argparse
from dotenv import load_dotenv

from backend.graph.research_graph import ResearchGraph
from backend.database.feedback_store import save_feedback, get_feedback_summary

load_dotenv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    args = parser.parse_args()

    research_graph = ResearchGraph()
    result = research_graph.ask(args.query)

    print("\n" + "=" * 65)
    print("ANSWER")
    print("=" * 65)
    print(result["answer"])

    print("\nSOURCES")
    for source in result["sources"]:
        print(
            f"{source['citation']} {source['paper_title']} | "
            f"{source['section']} | Pages "
            f"{source['page_start']}-{source['page_end']}"
        )

    print(f"\nCorrective retries: {result['retry_count']}")
    print(f"Context assessment: {result['selected_grade']}")

    print("\nWas this answer helpful?")
    print("1. Yes")
    print("2. No")

    choice = input("Enter 1 or 2 (or press Enter to skip): ").strip()

    if choice == "1":
        save_feedback(result["answer_id"], 1)
        print("Positive feedback saved.")

    elif choice == "2":
        print("\nWhy was it not helpful?")
        print("1. Retrieved passages were irrelevant")
        print("2. Important information was missing")
        print("3. Answer was incorrect")

        reasons = {
            "1": "irrelevant_retrieval",
            "2": "missing_information",
            "3": "answer_incorrect",
        }

        reason_choice = input("Choose 1, 2, or 3: ").strip()
        reason = reasons.get(reason_choice)

        if reason:
            save_feedback(result["answer_id"], -1, reason)
            print("Negative feedback saved.")
        else:
            print("Invalid choice. Feedback was not saved.")

    print("\nFeedback summary:")
    print(get_feedback_summary())


if __name__ == "__main__":
    main()
