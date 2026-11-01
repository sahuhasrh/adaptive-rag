
import json
import os
import re
from collections import defaultdict

from backend.database.feedback_store import (
    get_connection,
    initialize_database,
)

MAX_ADJUSTMENT = float(os.getenv("FEEDBACK_MAX_ADJUSTMENT", "0.08"))
MIN_EVIDENCE = int(os.getenv("FEEDBACK_MIN_EVIDENCE", "2"))
MIN_QUERY_SIMILARITY = float(
    os.getenv("FEEDBACK_MIN_QUERY_SIMILARITY", "0.25")
)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do",
    "does", "for", "from", "how", "i", "in", "is", "it",
    "of", "on", "or", "the", "to", "was", "what", "when",
    "where", "which", "why", "with",
}


def query_similarity(query_a: str, query_b: str) -> float:
    """Calculate transparent token Jaccard similarity between two queries."""
    tokens_a = {
        token for token in re.findall(r"[a-z0-9]+", query_a.lower())
        if token not in STOPWORDS
    }
    tokens_b = {
        token for token in re.findall(r"[a-z0-9]+", query_b.lower())
        if token not in STOPWORDS
    }

    if not tokens_a or not tokens_b:
        return 0.0

    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def calculate_feedback_adjustments(query: str) -> dict[str, dict]:
    """
    Calculate per-chunk adjustments from feedback on similar queries.

    Positive feedback is a positive signal.
    Only 'irrelevant_retrieval' negative feedback is a negative
    retrieval signal. Other negative reasons are generation or
    coverage problems and do not penalize retrieved chunks.
    """
    initialize_database()

    evidence = defaultdict(
        lambda: {
            "weighted_signal": 0.0,
            "total_weight": 0.0,
            "count": 0,
            "positive": 0,
            "negative": 0,
        }
    )

    with get_connection() as connection:
        rows = connection.execute("""
            SELECT
                a.normalized_query,
                a.retrieved_chunks_json,
                f.rating,
                f.reason
            FROM answers AS a
            INNER JOIN feedback AS f
                ON a.answer_id = f.answer_id
        """).fetchall()

    for row in rows:
        similarity = query_similarity(
            query,
            row["normalized_query"],
        )

        if similarity < MIN_QUERY_SIMILARITY:
            continue

        rating = row["rating"]
        reason = row["reason"]

        if rating == 1:
            signal = 1.0
        elif rating == -1 and reason == "irrelevant_retrieval":
            signal = -1.0
        else:
            # Missing information and incorrect generation do not
            # establish that the retrieved chunks were irrelevant.
            continue

        try:
            chunk_ids = json.loads(row["retrieved_chunks_json"])
        except (TypeError, json.JSONDecodeError):
            continue

        for chunk_id in set(chunk_ids):
            item = evidence[chunk_id]
            item["weighted_signal"] += similarity * signal
            item["total_weight"] += similarity
            item["count"] += 1

            if signal > 0:
                item["positive"] += 1
            else:
                item["negative"] += 1

    adjustments = {}

    for chunk_id, item in evidence.items():
        if item["count"] < MIN_EVIDENCE:
            continue

        if item["total_weight"] == 0:
            continue

        average_signal = (
            item["weighted_signal"] / item["total_weight"]
        )

        adjustment = MAX_ADJUSTMENT * average_signal

        # Defensive bound: feedback can never dominate similarity.
        adjustment = max(
            -MAX_ADJUSTMENT,
            min(MAX_ADJUSTMENT, adjustment),
        )

        adjustments[chunk_id] = {
            "adjustment": adjustment,
            "evidence_count": item["count"],
            "positive_count": item["positive"],
            "negative_count": item["negative"],
        }

    return adjustments


def apply_feedback_ranking(
    query: str,
    results: list[dict],
) -> list[dict]:
    """Add explainable feedback adjustments and rerank candidates."""
    adjustments = calculate_feedback_adjustments(query)

    ranked_results = []

    for result in results:
        item = dict(result)
        original_score = float(item["score"])

        feedback = adjustments.get(
            item["chunk_id"],
            {
                "adjustment": 0.0,
                "evidence_count": 0,
                "positive_count": 0,
                "negative_count": 0,
            },
        )

        item["original_score"] = original_score
        item["feedback_adjustment"] = feedback["adjustment"]
        item["adjusted_score"] = (
            original_score + feedback["adjustment"]
        )
        item["feedback_evidence_count"] = feedback["evidence_count"]
        item["score"] = item["adjusted_score"]

        ranked_results.append(item)

    ranked_results.sort(
        key=lambda result: result["adjusted_score"],
        reverse=True,
    )

    return ranked_results
