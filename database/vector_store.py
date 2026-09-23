"""
Qdrant vector store: collection setup, upserting embedded chunks, and search.

Usage:
    from database.vector_store import get_qdrant_client, ensure_collection, upsert_chunks
    client = get_qdrant_client()
    ensure_collection(client)
    upsert_chunks(client, chunks, embeddings)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from app.settings_config import settings
from utils.log import get_logger
from loguru import logger

def get_qdrant_client() -> QdrantClient:
    """
    Creates a Qdrant client from settings. Call this once and reuse this client 
    rather than creating a new one per request.
    """
    logger.info("connecting to Qdrant", url=settings.qdrant_url)

    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key
    )

    return client

def ensure_collection(client: QdrantClient) -> None:
    """
    Creates a collection if it doesn't already exist. Safe to call on every app
    startup -- it is a no-op if the collection is already there.
    """
    collection_name = settings.qdrant_collection
    existing = [c.name for c in client.get_collections().collections]

    if collection_name in existing:
        logger.info("collection already exists", collection=collection_name) 
        return

    logger.info(
        "Creating collection",
        collection=collection_name,
        dim=settings.embedding_dim
    )

    client.create_collection(
        collection_name=collection_name, 
        vectors_config=qmodels.VectorParams(
            size=settings.embedding_dim,
            distance=qmodels.Distance.COSINE
        )
    )

def upsert_chunks(
        client: QdrantClient, 
        chunks: list, 
        embeddings: list[list[float]]
) -> None:
    """
    Writes chunks and their embeddings into Qdrant vector store as points.
    chunks: list of Chunk objects from rag/chunker.py
    embeddings: list of vectors, same length and order as chunks.
    """
    if len(chunks) != len(embeddings):
        logger.error(
            "Chunks/embedding size mismatch occured",
            num_chunks=len(chunks), 
            num_embeeddings=len(embeddings)
        )

        raise ValueError("chunks and embeddings must the same length")

    points = []
    for chunk, vector in zip(chunks, embeddings):
        points.append(
            qmodels.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector, 
                payload={
                    "text": chunk.text, 
                    "source": chunk.source, 
                    "page": chunk.page, 
                    "chunk_id": chunk.chunk_id
                }
            )
        )

    client.upsert(
        collection_name=settings.qdrant_collection, 
        points=points
    )

    logger.info(
        "Upserted chunks into Qdrant", 
        collection=settings.qdrant_collection, 
        num_points=len(points)
    )

def search(
    client: QdrantClient, 
    query_vector: list[float],
    top_k: int = 20) -> list[dict]: 
    """
    Runs a dense vectors search and returns matches as plain dicts
    (text, source, page, score) for every downstream use
    """
    logger.info("Running vector search", top_k=top_k)

    results = client.search(
        collection_name=settings.qdrant_collection,
        query_vector=query_vector, 
        top_k=top_k
    )

    matches = [
        {
            "text": r.payload["text"],
            "source": r.payload["source"],
            "page": r.payload["page"],
            "score": r.payload["score"]
        }
        for r in results
    ]

    logger.info("Search complete", num_matches=len(matches))
    return matches