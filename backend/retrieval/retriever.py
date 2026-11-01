
from qdrant_client import models

from backend.embeddings.embedder import (
    load_embedding_model,
    embed_texts,
)
from backend.retrieval.qdrant_store import (
    COLLECTION_NAME,
    create_qdrant_client,
)
from backend.retrieval.feedback_ranker import apply_feedback_ranking


QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)


class ResearchRetriever:
    def __init__(self):
        self.model = load_embedding_model()
        self.client = create_qdrant_client()

    def _retrieve_candidates(
        self,
        query: str,
        candidate_k: int,
    ) -> list[dict]:
        if not query.strip():
            return []

        query_vector = embed_texts(
            self.model,
            [QUERY_PREFIX + query.strip()],
        )[0]

        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=candidate_k,
            with_payload=True,
        )

        results = []

        for point in response.points:
            payload = point.payload or {}

            results.append({
                "chunk_id": payload.get("chunk_id"),
                "paper_id": payload.get("paper_id"),
                "paper_title": payload.get("paper_title"),
                "section": payload.get("section"),
                "subsection": payload.get("subsection"),
                "page_start": payload.get("page_start"),
                "page_end": payload.get("page_end"),
                "element_type": payload.get("element_type"),
                "token_count": payload.get("token_count"),
                "text": payload.get("text"),
                "score": float(point.score),
            })

        return results

    def search(
        self,
        query: str,
        top_k: int = 5,
        paper_id: str | None = None,
    ) -> list[dict]:
        """Baseline vector search, without feedback adjustments."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        candidates = self._retrieve_candidates(
            query=query,
            candidate_k=top_k,
        )

        # Paper filtering is intentionally not applied here because
        # the current Qdrant collection needs its payload index fixed.
        if paper_id:
            candidates = [
                item for item in candidates
                if item["paper_id"] == paper_id
            ]

        return candidates[:top_k]

    def search_adaptive(
        self,
        query: str,
        top_k: int = 5,
        candidate_multiplier: int = 4,
    ) -> list[dict]:
        """Retrieve a wider candidate set, then apply feedback ranking."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        candidate_k = min(
            100,
            max(top_k, top_k * candidate_multiplier),
        )

        candidates = self._retrieve_candidates(
            query=query,
            candidate_k=candidate_k,
        )

        ranked = apply_feedback_ranking(
            query=query,
            results=candidates,
        )

        return ranked[:top_k]
