import sqlite3
from pathlib import Path
from pprint import pprint

from utils import load_vectorstore


def inspect_pdf_database():
    db_path = Path(__file__).resolve().parent / "pdf_store.db"

    # 只读打开，避免路径错误时新建空数据库
    with sqlite3.connect(
        db_path.as_uri() + "?mode=ro",
        uri=True,
    ) as conn:
        conn.row_factory = sqlite3.Row

        rows = conn.execute(
            "SELECT * FROM pdf_documents ORDER BY id"
        ).fetchall()

        print(f"\n===== PDF 数据库：{len(rows)} 条记录 =====")

        for row in rows:
            pprint(dict(row))


def inspect_vector_database():
    vectorstore = load_vectorstore()

    # 先查看前 5 个片段，不打印很长的向量
    data = vectorstore.get(
        limit=5,
        include=["documents", "metadatas"],
    )

    print("\n===== Chroma：前 5 个文本片段 =====")

    for i, chunk_id in enumerate(data["ids"]):
        print(f"\n片段 ID：{chunk_id}")

        print("元数据：")
        pprint(data["metadatas"][i])

        print("正文：")
        print(data["documents"][i])


if __name__ == "__main__":
    inspect_pdf_database()
    inspect_vector_database()