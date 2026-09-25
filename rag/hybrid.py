"""
Hybrid retrieval: combines both dense (semantic) and sparse (keyword) search.

BGE-M3 produces both a dense vector and a sparse vector from one encode call, 
so we store both in Qdrant and blend their scores at query time.

Usage:
    from rag.hybrid import embed_hybrid, hybrid_search

    dense_vecs, sparse_vecs = embed_hybird(embedder, texts)
    results = hybrid_search(client, embedder, query="expiry date of the products")
"""

from qdrant_client.http import models as qmodels

from app.config import settings
from utils.log import get_logger

logger = get_logger(__name__)

def embed_hybrid(embedder, texts: list[str]) -> tuple[list[list[float]], list[dict]]:
    """
    Encodes texts and return both representations at once:
        - dense_vecs: standart vectors for semantic clarity, 
        - sparse_vecs: token-weight dicts for keyword-style matching.
    """
    if not texts:
        logger.warning("embed_hybrid called with at empty list of texts.")
        return [], []

    logger.info("Embedding texts (hybrid):", num_texts=len(texts))

    output = embedder.encode(
        texts, 
        batch_size=12, 
        max_length=8192,
        return_dense=True, 
        return_sparse=True, 
        return_colbert_vects=False
    )

    dense_vecs=output["dense_vectors"].tolist()
    sparse_vects=output["lexical_weights"]

    logger.info("Hybrid embedding complete", num_dense=len(dense_vecs), num_sparse=len(sparse_vects))

    return dense_vecs, sparse_vects

def _to_qdrant_sparse(sparse_weights: dict) -> qmodels.SparseVector:
    """
    Converts BGE-M3's {"token_id": weight} dict into indices/values
    format Qdrant's sparse vector type expects.
    """

    indices = [int(token_id) for token_id in sparse_weights.keys()]
    values = [int(weight) for weight in sparse_weights.values()]
    return qmodels.SparseVector(indices=indices, values=values)

def hybrid_search(
        client, 
        embedder, 
        query:str, 
        top_k: int = 20, 
        alpha: float = None
) -> list[dict]:
    """
    Runs dense and sparce vectors seperately, then blends the scores.
    alpha: weight given to the dense score (0.0 = pure sparce/keyword,
    1.0 = pure dense/semantic). Defaults to settings.hybrid_alpha
    """
    alpha = alpha if alpha is not None else settings.hybrid_alpha

    dense_vecs, sparse_vects = embed_hybrid(embedder, [query])
    query_dense=dense_vecs[0]
    query_sparse=_to_qdrant_sparse(sparse_vects[0])

    logger.info("Running hybrid search", top_k=top_k, alpha=alpha)

    dense_hits = client.search(
        collection_name=settings.qdrant_collection, 
        query_vector=("dense", query_dense), 
        limit=top_k
    )

    sparse_hits = client.search(
        collection_name=settings.qdrant_collection, 
        query_vector=("sparse", query_sparse), 
        limit=top_k
    )

    combined_scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}

    for hit in dense_hits:
        combined_scores[hit.id] = combined_scores.get(hit.id, 0.0) + (1-alpha) * hit.score
        payloads[hit.id] = hit.payload
    for hit in sparse_hits:
        combined_scores[hit.id] = combined_scores.get(hit.id, 0.0) + (1-alpha) * hit.score
        payloads.setdefault(hit.id, hit.payload)

    ranked_ids = sorted(combined_scores, key=combined_scores.get, reverse=True)[:top_k]

    results = [
        {
            "text": payloads[point_id]["text"],
            "source": payloads[point_id]["source"],
            "page": payloads[point_id]["page"],
            "score": payloads[point_id]
        }
        for point_id in ranked_ids
    ]

    logger.info("Ranked search complete", num_results=len(results))
    return results