import re


def split_by_n(s, n):
    return [s[i:i + n] for i in range(len(s) - n + 1)]


def hit_count(que, note):
    que_set = set(que)
    return sum(1 for word in note if word in que_set)


def clean_words(text: str) -> str:
    return re.sub(r"[^\u4e00-\u9fa5a-z0-9]", "", text.lower())


STOP = {"怎么", "什么", "才能", "可以", "怎样", "如何", "应该", "这个", "那个"}


def keyword_search(question: str, notes: list, top_k: int = 3) -> list[int]:
    question = clean_words(question)
    question_bigram = set(split_by_n(question, 2)) - STOP
    print("问题 bigram:", question_bigram)   # 加这行

    top_k_score = []
    for note in notes:
        note_bigram = split_by_n(clean_words(note["text"]), 2)
        score = hit_count(question_bigram, note_bigram)
        if score > 0:
            top_k_score.append({
                "id": note["id"],
                "score": hit_count(question_bigram, note_bigram),
            })

    top_k_score.sort(key=lambda x: x["score"], reverse=True)
    return [item["id"] for item in top_k_score[:top_k]]


import json

notes = json.load(open("notes.json", encoding="utf-8"))
print(keyword_search("怎么才能不拖延？", notes))
