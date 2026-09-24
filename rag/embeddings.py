"""
Embedding generation using BGEM3FlagModel.

Usage:
    from rag.embeddings import get_embedder, embed_texts
    
    embedder = get_embedder()
    vectors = embed_texts(embedder, ["Hello world", "hello python"])
"""

from FlagEmbedding import BGEM3FlagModel

from app.config import settings
from utils.log import get_logger

logger = get_logger(__name__)

_embedding_cache: BGEM3FlagModel | None = None

def get_embedder() -> BGEM3FlagModel:
    """
    Loads the embedding model once and reuses it. Loading is slow,
    so this is cached at the module level -- call this once
    at app startup, not per-request.
    """
    global _embedding_cache

    if _embedding_cache is not None:
        return _embedding_cache

    logger.info("Loading embedding model", model=settings.embedding_model)
    _embedding_cache = BGEM3FlagModel(settings.embedding_model, use_fp16=True)
    logger.info("Embedding model loaded")

    return _embedding_cache

def embed_texts(embedder: BGEM3FlagModel, texts: list[str]) -> list[list[float]]:
    """
    Embeds a batch of texts and returns a dense vectors (one per text).
    """

    if not texts:
        logger.warning("embed_texts called with an empty list")
        return []

    logger.info("Embedding texts", num_texts=len(texts))

    output = embedder.encode(
        texts, 
        batch_size=12, 
        max_length=8192, 
        return_dense=True, 
        return_sparse=False, 
        return_colbert_vecs=False
    )

    vectors = output["dense_vectors"].tolist()

    logger.info("Embedding complete", num_vectors=len(vectors), dim=len(vectors[0]))

    return vectors

def embed_query(embedder: BGEM3FlagModel, query: str) -> list[float]:
    """
    Embeds single query string
    """
    return embed_texts(embedder, [query])[0]