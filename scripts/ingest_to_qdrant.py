
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from backend.embeddings.embedder import (
    load_embedding_model,
    embed_texts,
)
from backend.retrieval.qdrant_store import (
    create_qdrant_client,
    ensure_collection,
    upsert_chunks,
    COLLECTION_NAME,
)

CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
BATCH_SIZE = 32


def read_batches():
    batch = []

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            batch.append(json.loads(line))

            if len(batch) >= BATCH_SIZE:
                yield batch
                batch = []

    if batch:
        yield batch


def main():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {CHUNKS_FILE}"
        )

    model = load_embedding_model()
    client = create_qdrant_client()

    print("Checking Qdrant collection...")
    ensure_collection(client)

    total = 0

    for batch_number, chunks in enumerate(read_batches(), start=1):
        # Embed the original chunk text. Metadata remains in the payload.
        texts = [chunk["text"] for chunk in chunks]
        vectors = embed_texts(model, texts)

        upsert_chunks(client, chunks, vectors)

        total += len(chunks)
        print(
            f"Batch {batch_number}: uploaded {len(chunks)} chunks "
            f"(total: {total})"
        )

    collection_info = client.get_collection(COLLECTION_NAME)

    print("\nIngestion completed.")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Points in Qdrant: {collection_info.points_count}")


if __name__ == "__main__":
    main()
