# from eval.LangsmithVal import target
#
# questions = [
#     "采购制度指定买哪个品牌的研发电脑？",
#     "线上出了最高等级故障，临时改数据的权限能开多久？",
#     "HR-2026-026的年度上限是多少？",
#     "FIN-2026-024的付款起算条件是什么？",
#     "SEC-2026-019的授权期限是多少？",
# ]
#
# for question in questions:
#     result = target({"question": question})
#
#     print("\n问题：", question)
#     print("回答：", result["answer"])
#
#     for rank, context in enumerate(result["contexts"], 1):
#         print(f"\n片段 {rank}：\n{context}")

from backend.RAGSystem_graph_enp import load_bm25_documents
from backend.vector_bm25_store import _get_or_build_bm25
from utils import load_vectorstore

store = load_vectorstore()
documents = load_bm25_documents(store)
bm25 = _get_or_build_bm25(documents, k=5)

question = "FIN-2026-024的付款起算条件是什么？"

tokens = bm25.preprocess_func(question)
scores = bm25.vectorizer.get_scores(tokens)

print("问题分词：", tokens)
print(
    "库中是否存在该条款：",
    any("FIN-2026-024" in doc.page_content for doc in bm25.docs),
)

for index in sorted(
    range(len(scores)),
    key=lambda i: float(scores[i]),
    reverse=True,
)[:5]:
    doc = bm25.docs[index]

    print("\n分数：", float(scores[index]))
    print("页码：", doc.metadata.get("page"))
    print("正文：", doc.page_content[:300])