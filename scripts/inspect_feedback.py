
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from backend.database.feedback_store import (
    initialize_database,
    get_connection,
    get_feedback_summary,
)


def main():
    initialize_database()

    with get_connection() as connection:
        rows = connection.execute("""
            SELECT
                a.answer_id,
                a.original_query,
                a.retrieval_strategy,
                f.rating,
                f.reason,
                f.created_at
            FROM answers a
            LEFT JOIN feedback f
                ON a.answer_id = f.answer_id
            ORDER BY a.created_at DESC
        """).fetchall()

    print("\nStored answers and feedback:\n")

    for row in rows:
        print("-" * 65)
        print("Answer ID:", row["answer_id"])
        print("Query:", row["original_query"])
        print("Strategy:", row["retrieval_strategy"])
        print("Rating:", row["rating"])
        print("Reason:", row["reason"])

    print("\nSummary:")
    print(get_feedback_summary())


if __name__ == "__main__":
    main()
