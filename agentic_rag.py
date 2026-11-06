import json
import math
import os
import re
import time
import urllib.request

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from vector import embed_texts, cosine

load_dotenv()

# ---------- 1. 初始化模型和常量 ----------
model = init_chat_model(
    model="deepseek-flash",
    extra_body={"thinking": {"type": "disabled"}}
)

RERANK_URL = "https://api.siliconflow.cn/v1/rerank"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# ---------- 2. 加载数据 ----------
print("加载数据中...")
chunks = json.load(open("chunks_dump.json", encoding="utf-8"))
embeddings = json.load(open("embeddings_dump.json", encoding="utf-8"))
questions = [q for q in json.load(open("eval_set_labeled.json", encoding="utf-8")) if q.get("kind") == "chunk"]

chunk_map = {c["chunk_id"]: c for c in chunks}
emb_map = {e["chunk_id"]: e["vec"] for e in embeddings}


# ---------- 3. 基础检索函数 ----------
def clean_words(text):
    return re.sub(r"[^\u4e00-\u9fa5a-z0-9]", "", text.lower())


def bigrams(s):
    return [s[i:i + 2] for i in range(len(s) - 1)]


STOP = {"怎么", "什么", "可以", "能不", "不能", "一下", "的话", "我们", "你们",
        "请问", "多少", "哪些", "这个", "那个", "有没", "没有", "是不", "不是",
        "如何", "怎样", "需要", "如果", "但是", "因为", "所以", "然后"}


def text_search(q, k=20):
    qg = set(bigrams(clean_words(q))) - STOP
    if not qg:
        return []
    scored = []
    for c in chunks:
        body = set(bigrams(clean_words(c["text"])))
        head = set(bigrams(clean_words((c["title"] or "") + " " + (c["heading"] or ""))))
        hit = len(qg & body) + len(qg & head) * 2
        if hit > 0:
            scored.append((hit, c["chunk_id"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [cid for _, cid in scored[:k]]


def vector_search(q, k=20):
    q_vec = embed_texts([q])[0]
    scored = []
    for cid, vec in emb_map.items():
        scored.append((cosine(q_vec, vec), cid))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [cid for _, cid in scored[:k]]


def rrf_merge(kw_ids, vec_ids, k=60):
    scores = {}
    for rank, id_ in enumerate(kw_ids, start=1):
        scores[id_] = scores.get(id_, 0) + 1 / (k + rank)
    for rank, id_ in enumerate(vec_ids, start=1):
        scores[id_] = scores.get(id_, 0) + 1 / (k + rank)
    return sorted(scores, key=lambda i: scores[i], reverse=True)


# ---------- 4. 核心链路：改写 -> 混合 -> 重排 ----------
def rewrite_query(original_q: str) -> str:
    """带业务语境的 Query Rewrite"""
    prompt = f"""你是一个通信行业客服助手。用户正在咨询流量卡、套餐、返费等相关业务。
如果用户的问题存在口语化或业务歧义（比如把“返费”叫成“到账”），请帮他把问题补全成一句完整、自然、适合在知识库检索的话。
注意：只针对当前问题改写，绝对不要添加不相关的关键词。直接输出改写后的一句话。

原问题：{original_q}
改写后："""
    response = model.invoke(prompt)
    return response.content.strip()


def rerank_candidates(query: str, candidate_ids: list[str], top_k: int = 6) -> list[str]:
    """对候选 chunk 进行 Rerank"""
    docs = [chunk_map[cid]["text"] for cid in candidate_ids]

    payload = json.dumps({
        "model": RERANK_MODEL,
        "query": query,
        "documents": docs,
        "top_n": top_k
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

    return [candidate_ids[item["index"]] for item in res["results"]]


def agentic_search(q: str) -> list[str]:
    """完整链路：改写 -> 混合召回 -> 精排"""
    # 1. 改写
    rewritten_q = rewrite_query(q)

    # 2. 混合召回 (取前 20 条作为粗筛候选)
    kw = text_search(rewritten_q, k=20)
    vec = vector_search(rewritten_q, k=20)
    candidates = rrf_merge(kw, vec)[:20]

    # 3. 重排 (返回最终 top-6)
    final_ids = rerank_candidates(rewritten_q, candidates, top_k=6)

    # 打印中间过程，方便观察
    print(f"    ↳ 改写后: {rewritten_q}")

    return final_ids


# ---------- 5. 评估 ----------
def evaluate():
    hits = 0
    fails = []

    for i, q in enumerate(questions):
        print(f"  正在处理 {i + 1}/{len(questions)}: {q['q']}")

        gold = q["gold_chunk_id"]
        got = agentic_search(q["q"])

        if gold in got:
            hits += 1
        else:
            fails.append((q["q"], gold, got[:3]))

        time.sleep(0.5)  # 防止 API 限流

    print(f"\n=== Agentic RAG (改写+混合+Rerank) ===")
    print(f"命中: {hits}/{len(questions)}")
    for f in fails:
        print(f"  失手 [{f[0]}] gold={f[1]} got={f[2]}")


if __name__ == "__main__":
    evaluate()