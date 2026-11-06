from search import keyword_search
from vector import vector_search

def rrf_merge(kw_ids, vec_ids, k=60):
    scores = {}
    for rank, id_ in enumerate(kw_ids, start=1):
        scores[id_] = scores.get(id_, 0) + 1 / (k + rank)
    for rank, id_ in enumerate(vec_ids, start=1):
        scores[id_] = scores.get(id_, 0) + 1 / (k + rank)
    return sorted(scores, key=lambda i: scores[i], reverse=True)

def hybrid_search(question, notes, top_k=3):
    kw = keyword_search(question, notes, top_k=10)
    vec = vector_search(question, top_k=10)
    return rrf_merge(kw, vec)[:top_k]