import json

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from vector import vector_search

load_dotenv()

notes = json.load(open("notes.json", encoding="utf-8"))
note_map = {n["id"]: n["text"] for n in notes}


@tool
def search_notes(query: str) -> str:
    """从书摘库中检索与 query 最相关的 3 条书摘。输入是用户的问题或关键词。"""
    ids = vector_search(query, top_k=3)
    if not ids:
        return "没有找到相关书摘。"
    lines = [f"[id={i}] {note_map[i]}" for i in ids]
    return "\n".join(lines)


model = init_chat_model(
    model="deepseek-flash",
    extra_body={"thinking": {"type": "disabled"}},
)

agent = create_agent(
    model=model,
    tools=[search_notes],
    system_prompt=(
        "你是一个书摘助手。用户问问题时，先调用 search_notes 检索书摘，"
        "再基于检索到的内容回答。如果检索结果和问题不相关，"
        "可以换一个说法再检索一次。回答时注明用了哪几条书摘的 id。"
    ),
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "怎么让自己更有行动力？"}]
})

for m in result["messages"]:
    m.pretty_print()