import json
import sqlite3

# 假设你把 haoka.db 复制到了当前目录，如果没有，请调整路径
conn = sqlite3.connect("haoka.db")
conn.row_factory = sqlite3.Row

# 找几个有禁发地区的商品
rows = conn.execute("SELECT product_id, product_name, area, disable_area FROM products WHERE disable_area IS NOT NULL AND disable_area != '' LIMIT 3").fetchall()

for r in rows:
    print(f"商品: {r['product_name']} | 归属地: {r['area']} | 禁发地: {r['disable_area']}")