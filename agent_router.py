import json
import sqlite3

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate

from agentic_rag import agentic_search
from vector import embed_texts, cosine

load_dotenv()

model = init_chat_model(model="deepseek-flash", extra_body={"thinking": {"type": "disabled"}})


def classify_and_extract(q: str):
    """让 LLM 强制提取实体，不依赖它的意图分类"""
    prompt = f"""你是一个信息提取器。请从用户的提问中提取以下信息。只输出 JSON，不要解释。
如果用户提到了某个地方（如：北京、济南），提取到 city。
如果提到了运营商（如：联通、电信），提取到 operator。
如果提到了价格上限，提取到 max_fee。
如果提到了流量下限，提取到 min_data。

用户提问：{q}

严格按照以下 JSON 格式输出：
{{"city": "", "operator": "", "max_fee": null, "min_data": null}}
"""
    res = model.invoke(prompt)
    try:
        clean_res = res.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_res)
        # 在代码里我们自己定路由规则：
        # 如果提取到了 city 且问题里包含“能/发/办”等字眼，就走商品查询
        if data.get("city") or data.get("operator") or data.get("max_fee"):
            if any(word in q for word in ["能", "发", "办", "买", "卡", "宽"]):
                data["intent"] = "query_product"
            else:
                data["intent"] = "query_kb"
        else:
            data["intent"] = "query_kb"
        return data
    except Exception as e:
        print("解析失败:", e, res.content)
        return {"intent": "query_kb", "city": "", "operator": "", "max_fee": None, "min_data": None}

def search_products_by_sql(city: str, operator: str, max_fee: int, min_data: int):
    """执行 SQL 硬过滤"""
    conn = sqlite3.connect("haoka.db")
    conn.row_factory = sqlite3.Row
    where, args = [], []

    if operator:
        where.append("operator = ?")
        args.append(operator)
    if max_fee is not None:
        where.append("monthly_fee <= ?")
        args.append(max_fee)
    if min_data is not None:
        where.append("data_gb >= ?")
        args.append(min_data)
    if city:
        # 核心逻辑：如果城市出现在禁发地（disable_area）里，直接过滤掉！
        # 注意：这里的 LIKE 只是简单演示，真实情况需要处理“北京市”和“北京”的匹配
        where.append("(disable_area IS NULL OR disable_area NOT LIKE ?)")
        args.append(f"%{city}%")

    sql = "SELECT * FROM products WHERE " + (" AND ".join(where) if where else "1=1") + " LIMIT 5"
    print(f"执行 SQL: {sql} | 参数: {args}")

    rows = conn.execute(sql, args).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def smart_rag(query):
    # 1. 先做意图识别和实体抽取
    data = classify_and_extract(query)

    # 2. 如果走商品线，先走 SQL
    if data["intent"] == "query_product":
        products = search_products_by_sql(
            city=data["city"],
            operator=data["operator"],
            max_fee=data["max_fee"],
            min_data=data["min_data"]
        )

        # 如果 SQL 查完为空，直接返回兜底话术
        if not products:
            return "抱歉，没有找到符合您条件（或可发货到该地区）的商品。"

        # 如果不为空，可以继续用你的 RAG 在这几个商品里做语义精排
        # 或者直接把商品信息返回给用户

    # 3. 如果走知识库线，走你的完美 RAG 链路
    elif data["intent"] == "query_kb":
        chunks = agentic_search(query)
        # 之后可以把 chunks 送给大模型生成回答
        return chunks


# 测试一下
print(smart_rag("我在海南定安县，能办那个联通卡吗？"))




# # 测试一下
# if __name__ == "__main__":
#     test_q = "我在北京，能办江西电信单宽200M600元/年吗？"
#     print(f"用户提问: {test_q}")
#
#     intent_data = classify_and_extract(test_q)
#     print(f"LLM 提取结果: {intent_data}")
#
#     if intent_data["intent"] == "query_product":
#         products = search_products_by_sql(
#             city=intent_data["city"],
#             operator=intent_data["operator"],
#             max_fee=intent_data["max_fee"],
#             min_data=intent_data["min_data"]
#         )
#         print(f"SQL 过滤后剩下的商品: {products}")