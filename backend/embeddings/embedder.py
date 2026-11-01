
from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384


def load_embedding_model() -> TextEmbedding:
    """Load the local FastEmbed model."""
    print(f"Loading embedding model: {MODEL_NAME}")
    model = TextEmbedding(model_name=MODEL_NAME)
    print("Embedding model ready.")
    return model


def embed_texts(model: TextEmbedding, texts: list[str]) -> list[list[float]]:
    """Generate dense embeddings for a batch of texts."""
    vectors = model.embed(texts, batch_size=32)
    return [vector.tolist() for vector in vectors]
