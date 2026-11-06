import json

from hybrid import hybrid_search
from search import keyword_search
from vector import vector_search

questions = json.load(open("questions.json", encoding="utf-8"))
notes = json.load(open("notes.json", encoding="utf-8"))

kw_hits = 0
vec_hits = 0
hybrid_hits = 0

kw_fails = []
vec_fails = []
hybrid_fails = []

for q in questions:
    gold = set(q["gold_ids"])

    kw = keyword_search(q["question"], notes)
    if gold & set(kw):
        kw_hits += 1
    else:
        kw_fails.append((q, kw))

    vec = vector_search(q["question"])
    if gold & set(vec):
        vec_hits += 1
    else:
        vec_fails.append((q, vec))

    hy = hybrid_search(q["question"], notes)
    if gold & set(hy):
        hybrid_hits += 1
    else:
        hybrid_fails.append((q, hy))

print(f"关键词命中: {kw_hits}/{len(questions)}")
print(f"向量命中: {vec_hits}/{len(questions)}")
print(f"混合命中: {hybrid_hits}/{len(questions)}")

for name, fails in [("关键词", kw_fails), ("向量", vec_fails), ("混合", hybrid_fails)]:
    print(f"\n=== {name}失手的题 ===")
    for q, got in fails:
        print(f"[{q['id']}] {q['question']}")
        print(f"  gold={sorted(q['gold_ids'])}  got={got}")

for q in questions:
    if q["id"] == 31:
        print("31 混合结果:", hybrid_search(q["question"], notes))