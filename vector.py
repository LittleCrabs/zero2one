import json
import os
import urllib.request

from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.siliconflow.cn/v1"
MODEL = "BAAI/bge-m3"
API_KEY = os.environ["SILICONFLOW_API_KEY"]

NOTES_FILE = "notes.json"
VECTORS_FILE = "vectors.json"


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def vector_search(question: str, top_k: int = 20) -> list[int]:
    """把问题转成向量，和 vectors.json 里每条算余弦，返回 top_k 的 id。"""
    vectors = json.load(open(VECTORS_FILE, encoding="utf-8"))
    q_vec = embed_texts([question])[0]

    top_k_score = []
    for vector in vectors:
        score = cosine(q_vec, vector["vec"])

        top_k_score.append({
            "id": vector["id"],
            "score":score,
        })
        # print(f"id:{vector["id"]}---score:{score}")

    top_k_score.sort(key=lambda x: x["score"], reverse=True)
    return [item["id"] for item in top_k_score[:top_k]]





def embed_texts(texts: list[str]) -> list[list[float]]:
    """调硅基流动 /embeddings，返回与输入顺序一致的向量列表。"""
    payload = json.dumps({
        "model": MODEL,
        "input": texts,
        "encoding_format": "float",
    }).encode()

    req = urllib.request.Request(
        BASE_URL + "/embeddings",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + API_KEY,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        res = json.loads(r.read().decode())

    items = sorted(res["data"], key=lambda d: d["index"])
    return [it["embedding"] for it in items]


def build_vectors():
    """把 notes.json 里所有 text 转成向量，存到 vectors.json。"""
    notes = json.load(open(NOTES_FILE, encoding="utf-8"))
    texts = [n["text"] for n in notes]
    vectors = embed_texts(texts)

    out = []
    for note, vec in zip(notes, vectors):
        out.append({"id": note["id"], "vec": vec})

    json.dump(out, open(VECTORS_FILE, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"已写入 {len(out)} 条向量，维度 {len(out[0]['vec'])}")




if __name__ == "__main__":
    build_vectors()

    # print(vector_search("怎么才能不拖延？"))
    # 关键词失手：id=7（写东西卡住）。问题与 gold 文本字面零重叠，
    # 关键词检索只能召回碰巧共享"时候/东西"的无关条目；向量靠语义召回 [9,10]。
    # 结论：字面重叠类查询两种方法都行；换说法类查询只有向量能救。
