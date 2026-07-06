from sentence_transformers import CrossEncoder

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model = None

def get_reranker():
    """Load once, reuse."""
    global _model
    if _model is None:
        print(f"Loading re-ranker model: {RERANK_MODEL}")
        _model = CrossEncoder(RERANK_MODEL)
    return _model


def rerank(query, chunks, top_n=3):
    """
    Takes a question and a list of retrieved chunks.
    Returns top_n chunks re-scored by the cross-encoder.
    """
    model = get_reranker()

    # Score each chunk against the query
    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores = model.predict(pairs)

    # Attach scores and sort
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    ranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:top_n]