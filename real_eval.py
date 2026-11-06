import json
import math
import os
import re
import time
import urllib

# ---------- 1. 导入你已有的 API 封装和余弦计算 ----------
# 假设你的 vector.py 里定义了 embed_texts 和 cosine
from vector import embed_texts, cosine

# ---------- 2. 加载真实数据 ----------
print("加载数据中...")
chunks = json.load(open("chunks_dump.json", encoding="utf-8"))
embeddings = json.load(open("embeddings_dump.json", encoding="utf-8"))
questions = [q for q in json.load(open("eval_set_labeled.json", encoding="utf-8")) if q.get("kind") == "chunk"]

chunk_map = {c["chunk_id"]: c for c in chunks}
emb_map = {e["chunk_id"]: e["vec"] for e in embeddings}


# ---------- 3. 关键词检索 (简化版二元组) ----------
def clean_words(text):
    return re.sub(r"[^\u4e00-\u9fa5a-z0-9]", "", text.lower())


def bigrams(s):
    return [s[i:i + 2] for i in range(len(s) - 1)]


STOP = {"怎么", "什么", "可以", "能不", "不能", "一下", "的话", "我们", "你们",
        "请问", "多少", "哪些", "这个", "那个", "有没", "没有", "是不", "不是",
        "如何", "怎样", "需要", "如果", "但是", "因为", "所以", "然后"}


def text_search(q, k=6):
    qg = set(bigrams(clean_words(q))) - STOP
    if not qg:
        return []
    scored = []
    for c in chunks:
        body = set(bigrams(clean_words(c["text"])))
        head = set(bigrams(clean_words((c["title"] or "") + " " + (c["heading"] or ""))))
        # 算交集中的字词数量
        hit = len(qg & body) + len(qg & head) * 2
        if hit > 0:
            scored.append((hit, c["chunk_id"]))
    # 按分数降序排，取前 k
    scored.sort(key=lambda x: x[0], reverse=True)
    return [cid for _, cid in scored[:k]]


# ---------- 4. 向量检索 ----------
def vector_search(q, k=6):
    # 调 API 拿问题向量
    q_vec = embed_texts([q])[0]
    scored = []
    for cid, vec in emb_map.items():
        scored.append((cosine(q_vec, vec), cid))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [cid for _, cid in scored[:k]]

# ---------- 4.5 混合检索 (RRF) ----------
def rrf_merge(kw_ids, vec_ids, k=60):
    scores = {}
    for rank, id_ in enumerate(kw_ids, start=1):
        scores[id_] = scores.get(id_, 0) + 1 / (k + rank)
    for rank, id_ in enumerate(vec_ids, start=1):
        scores[id_] = scores.get(id_, 0) + 1 / (k + rank)
    return sorted(scores, key=lambda i: scores[i], reverse=True)

def hybrid_search(q, k=6):
    # 每路多取一点候选，给融合留空间
    kw = text_search(q, k)
    vec = vector_search(q, k)
    return rrf_merge(kw, vec)[:k]


RERANK_URL = "https://api.siliconflow.cn/v1/rerank"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


def rerank_search(q, recall_k=20, top_k=6):
    # 1. 粗筛：用混合检索（或向量）拿 recall_k 个 id
    # 这里我们直接用 hybrid_search 的逻辑，但只取候选，不截断到 6
    kw = text_search(q, recall_k)
    vec = vector_search(q, recall_k)
    recall_ids = rrf_merge(kw, vec)[:recall_k]

    # 2. 取出对应的文本
    docs = [chunk_map[cid]["text"] for cid in recall_ids]

    # 3. 调 API 精排
    payload = json.dumps({
        "model": RERANK_MODEL,
        "query": q,
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

    # 4. 把 rerank 返回的下标映射回 chunk_id
    ranked_results = res["results"]
    return [recall_ids[item["index"]] for item in ranked_results]


# ---------- 5. 评测 ----------
def evaluate():
    print("\n=== 测试关键词检索 ===")
    text_hits = 0
    text_fails = []
    for q in questions:
        gold = q["gold_chunk_id"]
        got = text_search(q["q"])
        if gold in got:
            text_hits += 1
        else:
            text_fails.append((q["q"], gold, got[:3]))
    print(f"text: {text_hits}/{len(questions)}")
    for f in text_fails:
        print(f"  失手 [{f[0]}] gold={f[1]} got={f[2]}")

    print("\n=== 测试向量检索 ===")
    vec_hits = 0
    vec_fails = []
    for i, q in enumerate(questions):
        print(f"  正在检索 {i+1}/{len(questions)}: {q['q']}")
        gold = q["gold_chunk_id"]
        got = vector_search(q["q"])
        if gold in got:
            vec_hits += 1
        else:
            vec_fails.append((q["q"], gold, got[:3]))
        time.sleep(0.3)
    print(f"\nvector: {vec_hits}/{len(questions)}")
    for f in vec_fails:
        print(f"  失手 [{f[0]}] gold={f[1]} got={f[2]}")

    print("\n=== 测试混合检索 (复用向量结果) ===")
    # 注意：hybrid_search 内部会再调一次 vector_search，为了不浪费 API，这里我们直接测
    # 但为了代码简单，先让它调吧，24 题再等一次
    hy_hits = 0
    hy_fails = []
    for i, q in enumerate(questions):
        print(f"  正在检索 {i+1}/{len(questions)}: {q['q']}")
        gold = q["gold_chunk_id"]
        got = hybrid_search(q["q"])
        if gold in got:
            hy_hits += 1
        else:
            hy_fails.append((q["q"], gold, got[:3]))
        time.sleep(0.3)
    print(f"\nhybrid: {hy_hits}/{len(questions)}")
    for f in hy_fails:
        print(f"  失手 [{f[0]}] gold={f[1]} got={f[2]}")

    print("\n=== 测试 Rerank 检索 ===")
    rr_hits = 0
    rr_fails = []
    for i, q in enumerate(questions):
        print(f"  正在重排 {i + 1}/{len(questions)}: {q['q']}")
        gold = q["gold_chunk_id"]
        got = rerank_search(q["q"])
        if gold in got:
            rr_hits += 1
        else:
            rr_fails.append((q["q"], gold, got[:3]))
        time.sleep(0.3)

    print(f"\nrerank: {rr_hits}/{len(questions)}")
    for f in rr_fails:
        print(f"  失手 [{f[0]}] gold={f[1]} got={f[2]}")


if __name__ == "__main__":
    # evaluate()
    from langchain.chat_models import init_chat_model

    # 初始化大模型（复用你之前项目里的配置）
    model = init_chat_model(
        model="deepseek-flash",
        extra_body={"thinking": {"type": "disabled"}}
    )


    def rewrite_query(original_q: str) -> str:
        prompt = f"""你是一个查询改写助手。用户正在咨询通信业务。
    如果用户的问题存在业务黑话或歧义（比如把“返费”叫成“到账”），请帮他把问题补全成一句完整、自然、适合在知识库检索的话。
    注意：只针对当前问题改写，绝对不要添加不相关的关键词。直接输出改写后的一句话。

    原问题：{original_q}
    改写后："""
        response = model.invoke(prompt)
        return response.content.strip()


    # 在 evaluate() 里，只挑原来失手的那两道题来测试
    if __name__ == "__main__":

        test_qs = ["套餐什么时候到账", "联通飞豹卡合约期"]

        for q_text in test_qs:
            print(f"\n--- 测试问题: {q_text} ---")

            # 1. 原始检索
            print("【原始向量检索】")
            vec_orig = vector_search(q_text)
            print(f"got: {vec_orig[:3]}")

            # 2. 改写 + 检索
            rewritten = rewrite_query(q_text)
            print(f"\n【改写后查询】: {rewritten}")

            print("【改写后向量检索】")
            vec_new = vector_search(rewritten)
            print(f"got: {vec_new[:3]}")

            print("【改写后 Rerank】")
            rr_new = rerank_search(rewritten)
            print(f"got: {rr_new[:3]}")

            time.sleep(1)