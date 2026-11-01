
import json
import os
import sqlite3
from pathlib import Path
from datetime import datetime, timezone


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_database_path() -> Path:
    """Resolve the SQLite database path from DATABASE_URL."""
    database_url = os.getenv(
        "DATABASE_URL",
        "sqlite:///./adaptive_rag.db",
    )

    if not database_url.startswith("sqlite:///"):
        raise ValueError("Only SQLite DATABASE_URL is supported.")

    raw_path = database_url[len("sqlite:///"):].replace("/", os.sep)
    path = Path(raw_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_connection():
    connection = sqlite3.connect(get_database_path())
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database():
    """Create feedback-memory tables if they do not exist."""
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS answers (
                answer_id TEXT PRIMARY KEY,
                original_query TEXT NOT NULL,
                normalized_query TEXT NOT NULL,
                retrieval_strategy TEXT NOT NULL,
                retrieved_chunks_json TEXT NOT NULL,
                relevance_scores_json TEXT NOT NULL,
                answer_text TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                answer_id TEXT NOT NULL UNIQUE,
                rating INTEGER NOT NULL CHECK (rating IN (-1, 1)),
                reason TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (answer_id)
                    REFERENCES answers(answer_id)
                    ON DELETE CASCADE,
                CHECK (
                    (rating = 1 AND reason IS NULL)
                    OR
                    (rating = -1 AND reason IN (
                        'irrelevant_retrieval',
                        'missing_information',
                        'answer_incorrect'
                    ))
                )
            )
        """)


def normalize_query(query: str) -> str:
    """Normalize query text for future feedback matching."""
    return " ".join(query.lower().strip().split())


def record_answer(
    answer_id: str,
    query: str,
    answer_text: str,
    retrieved_chunks: list[dict],
    retrieval_strategy: str = "baseline_vector",
):
    """Store the answer and its retrieval results for future feedback."""
    chunk_ids = [
        chunk.get("chunk_id")
        for chunk in retrieved_chunks
    ]

    scores = [
        {
            "chunk_id": chunk.get("chunk_id"),
            "similarity_score": chunk.get("score"),
        }
        for chunk in retrieved_chunks
    ]

    with get_connection() as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO answers (
                answer_id,
                original_query,
                normalized_query,
                retrieval_strategy,
                retrieved_chunks_json,
                relevance_scores_json,
                answer_text,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                answer_id,
                query,
                normalize_query(query),
                retrieval_strategy,
                json.dumps(chunk_ids),
                json.dumps(scores),
                answer_text,
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def save_feedback(
    answer_id: str,
    rating: int,
    reason: str | None = None,
):
    """Save or update feedback for a previously generated answer."""
    if rating not in (-1, 1):
        raise ValueError("Rating must be 1 or -1.")

    if rating == 1:
        reason = None
    elif reason not in {
        "irrelevant_retrieval",
        "missing_information",
        "answer_incorrect",
    }:
        raise ValueError("A valid reason is required for negative feedback.")

    with get_connection() as connection:
        cursor = connection.execute(
            "SELECT answer_id FROM answers WHERE answer_id = ?",
            (answer_id,),
        )

        if cursor.fetchone() is None:
            raise ValueError(f"Answer not found: {answer_id}")

        connection.execute(
            """
            INSERT INTO feedback (
                answer_id,
                rating,
                reason,
                created_at
            )
            VALUES (?, ?, ?, ?)
            ON CONFLICT(answer_id) DO UPDATE SET
                rating = excluded.rating,
                reason = excluded.reason,
                created_at = excluded.created_at
            """,
            (
                answer_id,
                rating,
                reason,
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def get_feedback_summary() -> dict:
    """Return simple feedback statistics."""
    with get_connection() as connection:
        total_answers = connection.execute(
            "SELECT COUNT(*) FROM answers"
        ).fetchone()[0]

        total_feedback = connection.execute(
            "SELECT COUNT(*) FROM feedback"
        ).fetchone()[0]

        positive = connection.execute(
            "SELECT COUNT(*) FROM feedback WHERE rating = 1"
        ).fetchone()[0]

        negative = connection.execute(
            "SELECT COUNT(*) FROM feedback WHERE rating = -1"
        ).fetchone()[0]

        reasons = connection.execute("""
            SELECT reason, COUNT(*) AS count
            FROM feedback
            WHERE rating = -1
            GROUP BY reason
        """).fetchall()

    return {
        "total_answers": total_answers,
        "total_feedback": total_feedback,
        "positive_feedback": positive,
        "negative_feedback": negative,
        "negative_reasons": {
            row["reason"]: row["count"] for row in reasons
        },
    }
