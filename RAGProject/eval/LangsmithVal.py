"""
RAG 评估指标说明
===============================================================================
指标                 | 评估意义                     | 分数低时优先排查
---------------------|------------------------------|---------------------------
Context Precision    | 检索片段是否有用，             | 1. 是否召回了关键词相似但
上下文精确性             | 有用的片段是否排在前面。     |    无法回答问题的片段。
                     |                              | 2. BM25 中文分词是否合理。
                     | 当前使用有参考答案的版本：     | 3. 融合、重排是否排序不佳。
                     | 对照问题和参考答案，           | 4. 是否存在重复、过时片段。
                     | 判断各片段是否有帮助，         | 5. 评估的是否为最终上下文，
                     | 再结合排名计算分数。           |    而非重排前的全部候选。
---------------------|------------------------------|---------------------------
Context Recall       | 检索上下文是否覆盖             | 1. 原始文档是否包含答案。
上下文召回率         | 参考答案所需的事实信息。          | 2. PDF 解析、OCR 是否漏字。
                     |                              | 3. 切块是否拆散关键证据。
                     | 当前使用 LLM 版本：            | 4. 检索模型、分词、查询表达
                     | 检查参考答案中的各条事实，      |    是否导致漏检。
                     | 是否能从检索片段得到支持。      | 5. Top-K、阈值、重排或截断
                     |                              |    是否过滤掉必要证据。
---------------------|------------------------------|---------------------------
Faithfulness         | 最终回答中的事实是否            | 1. 是否传入了生成回答时
忠实度               | 得到检索上下文支持。            |    实际使用的上下文。
                     |                              | 2. 模型是否自行补充信息。
                     | 关注“回答有没有依据”，         | 3. 数字、主体、条件和结论
                     | 不直接判断现实中是否正确，      |    是否被模型改写错误。
                     | 也不衡量回答是否完整。          | 4. 上下文不足时，是否仍要求
                     |                              |    模型给出确定答案。
---------------------|------------------------------|---------------------------
Factual Correctness  | 最终回答与参考答案中的          | 1. 参考答案是否正确、完整。
事实正确性           | 事实是否一致。                  | 2. 检索证据是否缺失、过时
                     |                              |    或互相矛盾。
                     | mode="f1" 时综合考虑          | 3. 证据充分时，生成模型是否
                     | 错误信息和遗漏信息。           |    理解、计算或归纳错误。
                     |                              | 4. 回答是否遗漏关键条件。
---------------------|------------------------------|---------------------------
Answer Relevancy     | 最终回答是否切合用户问题。      | 1. 是否答非所问、内容冗余。
回答相关性           |                              | 2. 是否只回答了部分问题。
                     | 通过 LLM 根据回答反推问题，    | 3. 多轮对话是否丢失指代。
                     | 再用 Embedding 比较反推问题   | 4. 查询改写是否改变原意。
                     | 与原问题的相似程度。           | 5. 反推问题质量、Embedding
                     | 不直接判断事实是否正确。        |    模型是否适合当前语言。
===============================================================================

使用注意：
1. 上述 Context Precision / Recall 指当前使用的有参考答案、LLM 评分版本。
2. retrieved_contexts 应保留原始排名；评估最终 RAG 时传入实际用于生成的片段。
3. 不同阶段可以分别评估：初步召回、融合后、重排后，避免混淆问题来源。
4. 低分是排查线索，不直接证明某个组件有问题；需查看具体样本与评分依据。
5. 高忠实度不保证答案正确；高相关性也不保证答案正确或完整。
6. 无答案题需单独评估拒答。正确拒答的 Answer Relevancy 也可能较低。
7. 评分异常、NaN、输出截断应记录为评估失败，不要直接当作 0 分。
8. 比较版本时固定测试集、知识库、评分模型及指标配置，并记录耗时和失败率。
9. FactualCorrectness 与旧接口的 AnswerCorrectness 不是同一个指标。
"""
import os

from backend.Agent import ask_agent

from backend.RAGSystem_graph_enp import load_bm25_documents, build_retrieval_graph
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from openai import api_key

from utils import load_vectorstore

from config import key, url

from eval.create_judge_eval import faithfulness_eval, answerrelevance_eval, precision_eval, recall_eval

embed_store = load_vectorstore()
chunks = load_bm25_documents(embed_store)
builder = build_retrieval_graph(embed_store,chunks)

#pdf_path = "./RAG知识测试语料.pdf"
    #pages = parse_pdf(pdf_path)
    #chunks = chunk_documents(pages,100,10)
load_dotenv(override=True)
key = os.getenv("DEEPSEEK_FLASH_KEY")
url = os.getenv("DEEPSEEK_FLASH_URL")
llm_val = init_chat_model(
    model = 'deepseek-flash',
    model_provider='openai',
    api_key = key,
    base_url = url,
)

CONTEXT_TOP_K = 5

def generate_answer(question, docs):
    # 两组统一取前5条
    docs = list(docs)[:CONTEXT_TOP_K]

    contexts = [doc.page_content for doc in docs]
    context_text = "\n\n---\n\n".join(contexts)

    response = llm_val.invoke(
        f"""请只根据资料回答问题。
证据不足时明确说明。回答涉及金额、时间、审批人或适用条件时，准确保留相关信息；未询问的内容不必展开。

资料：
{context_text}

问题：
{question}
"""
    )

    return {
        "answer": response.content,
        "contexts": contexts,
        "question": question,
    }
def ask_agent_answer(question):
    response = builder.invoke({
        "original_query": question,
        "rewritten_queries": [],
        "vector_results": [],
        "bm25_results": [],
        "merged_results": [],
        "reranked_results": [],
        "relevance_score": 0,
        "context": "",
        "answer": "",
        "sources": [],
        "needs_fallback": False,
    })
    #answer = ask_agent(question)
    # answer = llm_val.invoke(question)
    answer = llm_val.invoke(
        f"""请只根据资料回答问题。
    证据不足时明确说明，保留金额、时间、审批人及适用条件。

    资料：
    {response['merged_results']}

    问题：
    {question}
    """
    )
    response['answer'] = answer.content
    return response

embed_store = load_vectorstore()

# 先召回20条，再取前5条作为最终上下文
VECTOR_TOP_K = 10
CONTEXT_TOP_K = 5

def target_vector(inputs: dict) -> dict:
    question = inputs["question"]

    docs = embed_store.similarity_search(
        question,
        k=10,
    )

    return generate_answer(question, docs)

def target(inputs: dict) -> dict:
    question = inputs["question"]

    result = builder.invoke({
        "original_query": question,
        "rewritten_queries": [],
        "vector_results": [],
        "bm25_results": [],
        "merged_results": [],
        "reranked_results": [],
        "relevance_score": 0,
        "context": "",
        "answer": "",
        "sources": [],
        "needs_fallback": False,
    })
    # 测RRF
    docs = result["merged_results"]

    # 单测BM25，且只有一个query
    #rankings = result["bm25_results"]
    #docs = rankings[0] if rankings else []

    return generate_answer(
        question,
        docs,
    )

# from langsmith import Client
#
# from eval.create_judge_eval import faithfulness_eval, precision_eval, answerrelevance_eval, recall_eval
#
# client = Client()
#
# results = client.evaluate(
#     target,  # 调用你的 Agent
#     data="qs_hybrid_dev",
#     evaluators=[faithfulness_eval,answerrelevance_eval,precision_eval,recall_eval],  # Judge 评分函数
#     experiment_prefix="bm25_top5",  # 本次实验名称前缀
#     max_concurrency=1,
# )


import json
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def main():
    dataset_path = (
        BASE_DIR /
         "混合检索_测试集20题.jsonl"
    )

    output_path = BASE_DIR /"eval" / "rrf_test_results.jsonl"

    # 测RRF用target；测纯向量改成target_vector
    target_fn = target

    evaluators = [
        faithfulness_eval,
        answerrelevance_eval,
        precision_eval,
        recall_eval,
    ]

    scores_by_metric = {}

    with (
        dataset_path.open("r", encoding="utf-8-sig") as source,
        output_path.open("w", encoding="utf-8") as output_file,
    ):
        for line in source:
            if not line.strip():
                continue

            example = json.loads(line)
            inputs = example["inputs"]
            reference = example["outputs"]

            record = {
                "inputs": inputs,
                "reference_outputs": reference,
                "metadata": example.get("metadata", {}),
                "scores": {},
                "errors": {},
            }

            print("\n问题：", inputs["question"])

            # 单独记录被测RAG耗时，不包含评分耗时
            start = time.perf_counter()

            try:
                outputs = target_fn(inputs)

                assert isinstance(outputs["answer"], str)
                assert isinstance(outputs["contexts"], list)
                assert all(
                    isinstance(text, str)
                    for text in outputs["contexts"]
                )

                record["outputs"] = outputs
                print("回答：", outputs["answer"])

            except Exception as exc:
                record["errors"]["target"] = repr(exc)
                print("运行失败：", exc)
                outputs = None

            record["latency_seconds"] = time.perf_counter() - start

            if outputs is not None:
                for evaluator in evaluators:
                    try:
                        # 按你现有评分函数的参数形式调用
                        if evaluator in (precision_eval, recall_eval):
                            result = evaluator(
                                inputs=inputs,
                                outputs=outputs,
                                reference_outputs=reference,
                            )
                        else:
                            result = evaluator(
                                inputs=inputs,
                                outputs=outputs,
                            )

                        metric = result["key"]
                        score = result["score"]

                        record["scores"][metric] = score
                        scores_by_metric.setdefault(metric, []).append(score)

                        print(f"{metric}：{score}")

                    except Exception as exc:
                        record["errors"][evaluator.__name__] = repr(exc)
                        print(f"{evaluator.__name__} 评分失败：{exc}")

            # 每题立即保存，评分失败也留下记录
            output_file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
            output_file.flush()

    print("\n各指标平均分（仅成功评分的样本）：")

    for metric, values in scores_by_metric.items():
        print(
            f"{metric}：{sum(values) / len(values):.4f}"
            f"，成功评分 {len(values)} 题"
        )

    print("结果保存到：", output_path)


if __name__ == "__main__":
    main()

