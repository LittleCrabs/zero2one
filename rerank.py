import json
import os
import urllib.request

from dotenv import load_dotenv

from vector import vector_search

load_dotenv()

RERANK_URL = "https://api.siliconflow.cn/v1/rerank"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


def rerank(question: str, documents: list[str], top_n: int = 3) -> list[tuple[int, float]]:
    payload = json.dumps({
        "model": RERANK_MODEL,
        "query": question,
        "documents": documents,
        "top_n": top_n,
    }).encode()

    req = urllib.request.Request(
        RERANK_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + os.environ["SILICONFLOW_API_KEY"],
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        res = json.loads(r.read().decode())

    return [(item["index"], item["relevance_score"]) for item in res["results"]]


def vector_search_rerank(question: str, top_k: int = 3, recall_k: int = 20) -> list[int]:
    recall_ids = vector_search(question, top_k=recall_k)

    notes = json.load(open("notes.json", encoding="utf-8"))
    note_map = {n["id"]: n["text"] for n in notes}
    documents = [note_map[i] for i in recall_ids]

    ranked = rerank(question, documents, top_n=top_k)
    return [recall_ids[idx] for idx, _ in ranked]


if __name__ == "__main__":
    q = "怎么让自己更有行动力？"
    recall_ids = vector_search(q, top_k=20)
    notes = json.load(open("notes.json", encoding="utf-8"))
    note_map = {n["id"]: n["text"] for n in notes}
    docs = [note_map[i] for i in recall_ids]

    ranked = rerank(q, docs, top_n=20)
    for idx, score in ranked:
        rid = recall_ids[idx]
        mark = " ← gold" if rid in {1, 2, 31, 32} else ""
        print(f"id={rid} score={score:.6f}{mark}")