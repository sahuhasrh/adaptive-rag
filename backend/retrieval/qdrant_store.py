
import os
import uuid

from qdrant_client import QdrantClient, models

from backend.embeddings.embedder import VECTOR_SIZE

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "adaptive_rag")


def create_qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url or not api_key:
        raise ValueError(
            "QDRANT_URL and QDRANT_API_KEY must be configured in .env"
        )

    return QdrantClient(
        url=url,
        api_key=api_key,
        timeout=60,
    )



def ensure_collection(client: QdrantClient) -> None:
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )
        print(f"Created Qdrant collection: {COLLECTION_NAME}")

    info = client.get_collection(COLLECTION_NAME)
    actual_size = info.config.params.vectors.size

    if actual_size != VECTOR_SIZE:
        raise ValueError(
            f"Collection has vector size {actual_size}, "
            f"but the embedding model produces {VECTOR_SIZE}. "
            "Use a new collection name or recreate the collection."
        )

    # Create the keyword index required for paper-specific filtering.
    payload_schema = info.payload_schema or {}

    if "paper_id" not in payload_schema:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="paper_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )
        print("Created payload index: paper_id")

    print(f"Using Qdrant collection: {COLLECTION_NAME}")



def upsert_chunks(
    client: QdrantClient,
    chunks: list[dict],
    vectors: list[list[float]],
) -> None:
    if len(chunks) != len(vectors):
        raise ValueError("Chunk and vector counts do not match.")

    points = []

    for chunk, vector in zip(chunks, vectors):
        point_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, chunk["chunk_id"])
        )

        payload = {
            "chunk_id": chunk["chunk_id"],
            "paper_id": chunk["paper_id"],
            "paper_title": chunk["paper_title"],
            "section": chunk["section"],
            "subsection": chunk["subsection"],
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "element_type": chunk["element_type"],
            "token_count": chunk["token_count"],
            "text": chunk["text"],
        }

        points.append(
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )
