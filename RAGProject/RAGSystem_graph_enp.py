import hashlib
from pathlib import Path
import re

from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from create_judge_eval import relevance2
from graph_state import RetrievalState
from utils import retrieval_docs, content_hash, get_reranker, format_sources, load_vectorstore
from vector_bm25_store import _get_or_build_bm25
from config import embed_model_ch, dir_chroma, embed_collection_name, bm25_key, RELEVANCE_THRESHOLD, key, url, \
    embed_search_key


def get_deepseek_llm():
    model = init_chat_model(
        model = "deepseek-flash",
        model_provider="openai",
        api_key = key,
        base_url = url,
    )
    return model

def get_qwen1_7b_llm():
    from transformers import pipeline
    from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

    model_path = r"C:\huggingface_model\Qwen3_1.7B\Qwen--Qwen3-1.7B"

    pipe = pipeline(
        "text-generation",
        model=model_path,
        tokenizer=model_path,
        device_map="auto",
        torch_dtype="auto",
        max_new_tokens=128,
        return_full_text=False,
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    model = ChatHuggingFace(llm=llm)
    return model
def get_qwen1_7b_llm1():
    from transformers import pipeline, AutoTokenizer
    from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

    model_path = r"C:\huggingface_model\Qwen3_1.7B\Qwen--Qwen3-1.7B"

    # 单独加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )

    # 包装 chat template，固定关闭 Qwen3 thinking
    original_apply_chat_template = tokenizer.apply_chat_template

    def apply_chat_template_no_thinking(*args, **kwargs):
        kwargs["enable_thinking"] = False
        return original_apply_chat_template(*args, **kwargs)

    tokenizer.apply_chat_template = apply_chat_template_no_thinking

    pipe = pipeline(
        "text-generation",
        model=model_path,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype="auto",
        max_new_tokens=64,
        return_full_text=False,
        do_sample=False,
    )

    llm = HuggingFacePipeline(
        pipeline=pipe
    )

    model = ChatHuggingFace(
        llm=llm
    )

    return model

def bm25_search(
    queries: list[str],
    documents: list[Document],
    k: int = bm25_key,
) -> list[Document]:
    """
    BM25 关键词检索：基于词频的精确匹配检索

    BM25 不需要 Embedding，适合精确关键词匹配场景。
    需要传入完整文档集合来构建 BM25 索引。

    优化：相同文档列表的 BM25 索引会被缓存（基于 id() + len() 双重校验），
    重复调用不会反复重建倒排索引。

    参数:
        queries: 查询列表
        documents: 用于构建 BM25 索引的文档集合
        k: 每个查询召回的文档数

    返回:
        去重后的文档列表
    """
    if not documents:
        return []

    bm25 = _get_or_build_bm25(documents, k=k)

    seen_hashes = set()
    results = []

    for q in queries:
        docs = bm25.invoke(q)
        for doc in docs:
            h = hash(doc.page_content)
            if h not in seen_hashes:
                seen_hashes.add(h)
                results.append(doc)

    return results

def vector_search(
    queries: list[str],
    vectorstore: Chroma,
    k: int = embed_search_key,
) -> list[Document]:
    """
    向量语义检索：对每个查询变体执行 similarity_search，合并结果

    参数:
        queries: 查询列表（原始 + 改写变体）
        vectorstore: ChromaDB 向量库实例
        k: 每个查询召回的文档数

    返回:
        去重后的文档列表
    """
    seen_hashes = set()
    results = []

    for q in queries:
        docs = vectorstore.similarity_search(q, k=k)
        for doc in docs:
            h = hash(doc.page_content)
            if h not in seen_hashes:
                seen_hashes.add(h)
                results.append(doc)

    return results

def parse_pdf(file_path: str):
    loader = PyPDFLoader(file_path=file_path)
    pages = loader.load()
    #print(type(pages))
    #print(pages)
    return pages

def chunk_documents(documents: list[Document],
                    chunk_size: int = 500,
                    chunk_overlap: int = 50,) -> list[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""],

    )
    chunks = splitter.split_documents(documents)
    # print(type(chunks))
    # print(type(chunks[0]))
    # print(chunks)
    return chunks

def rewrite_query(query: str, llm=None) -> list[str]:
    """
    使用 LLM 改写查询为多个变体，提升召回率

    改写策略：从不同角度表达同一问题，覆盖更多语义空间。
    例如 "Python 性能优化" 可能改写为：
    - "如何提升 Python 代码执行速度"
    - "Python 性能瓶颈和解决方案"
    - "Python performance optimization best practices"

    参数:
        query: 用户原始查询
        llm: LLM 实例（可选，默认用 config.get_llm()）

    返回:
        包含原始查询 + 改写变体的列表
    """
    if llm is None:
        llm = get_deepseek_llm()

    prompt = f"""你是一个查询改写助手。请将用户的查询改写为 2 个不同角度的变体，
保持原始查询的核心意图不变，但用不同的表达方式。

原始查询：{query}

请直接输出改写后的查询，每行一个，不要编号，不要额外解释。"""

    response = llm.invoke(prompt)
    variants = [
        line.strip()
        for line in response.content.strip().split("\n")
        if line.strip()
    ]

    # 始终包含原始查询
    all_queries = [query] + variants[:2]
    return all_queries

def _rewrite_node(state: RetrievalState) -> dict:
    queries = rewrite_query(state["original_query"])
    return {
        "rewritten_queries":queries
    }


def _merge_node(state: RetrievalState) -> dict:
    vector_results = state.get("vector_results",[])
    bm25_results = state.get("bm25_results",[])

    seen = set()
    merged = []

    for doc in vector_results:
        h = content_hash(doc.page_content)
        if h not in seen:
            seen.add(h)
            merged.append(doc)
    for doc in bm25_results:
        h = content_hash(doc.page_content)
        if h not in seen:
            seen.add(h)
            merged.append(doc)

    return {
        "merged_results":merged
    }

def _rerank_node(state: RetrievalState) -> dict:
    queries = state.get("original_query",[])
    doc = state.get("merged_results",[])

    reranker_model = get_reranker()
    print(f"queries:{queries}")
    print(f"doc:{doc}")
    reranked = reranker_model.compress_documents(
        documents=doc,
        query=queries,
    )
    return {
        "reranked_results":reranked
    }

def _llm_grage_node(state: RetrievalState) -> dict:
    ori_queries = state.get("original_query",[])
    reranked_res = state.get("reranked_results",[])

    if not reranked_res:
        return {
            "relevance_score": 0.0,
            "needs_fallback": True,
        }

    model = get_deepseek_llm()
    # 拼接文档内容作为评估依据
    docs_text = "\n\n---\n\n".join(
        f"[文档 {i + 1}]\n{doc.page_content[:300]}"
        for i, doc in enumerate(reranked_res[:5])  # 最多评估 5 个文档
    )

    prompt = f"""请评估以下检索到的文档是否能充分回答用户的问题。

    用户问题：{ori_queries}

    检索到的文档：
    {docs_text}

    请只输出一个 0.0 到 1.0 之间的数字评分：
    - 1.0 = 完全可以回答
    - 0.7 = 基本可以回答，但可能缺少细节
    - 0.4 = 有部分相关信息
    - 0.0 = 完全无关

    只输出数字，不要其他内容。"""

    response = model.invoke(prompt)
    score = _parse_score(response.content)
    print(score)
    return {
        "relevance_score":score,
        "needs_fallback":score < RELEVANCE_THRESHOLD
    }

def _route_by_score(state: RetrievalState) -> dict:
    if state.get("needs_fallback",False):
        return "fallback"
    return END

# def _generate_node(state: RetrievalState) -> dict:
#     ori_queries = state.get("original_query",[])
#     reranked_res = state.get("reranked_results",[])
#     print(reranked_res)
#     llm = get_deepseek_llm()
#     context = "\n\n---\n\n".join(
#         f"[来源 {i + 1}: {doc.metadata.get('source', 'unknown')}, "
#         f"第 {doc.metadata.get('page', '?')} 页]\n{doc.page_content}"
#         for i, doc in enumerate(reranked_res)
#     )
#     prompt = f"""请根据以下检索到的文档内容回答用户问题。
#     要求：
#     1. 只基于提供的文档内容回答，不要编造信息
#     2. 如果文档中没有相关信息，明确说明
#     3. 在回答中标注信息来源（如"根据文档1..."）
#
#     检索到的文档：
#     {context}
#
#     用户问题：{ori_queries}"""
#
#     response = llm.invoke(prompt)
#     sources = format_sources(reranked_res)
#
#     return {"answer":response.content, "sources":sources}

def _fallback_node(state: RetrievalState) -> dict:
    """
    检索图节点：外部检索兜底（stub）

    在生产环境中，这里可以接入 Tavily/Bing 等网络搜索。
    学习仓库中仅演示模式，不实际调用外部搜索。
    """
    return {
        "answer": (
            f"抱歉，知识库中没有找到足够的信息来回答「{state['original_query']}」。"
            "建议查阅其他资料或尝试换个问法。"
        ),
        "sources": [],
        "needs_fallback": True,
    }

def _parse_score(raw: str) -> float:
    """
    从 LLM 输出中鲁棒地提取评分数字

    LLM 有时不听话，可能输出 "0.85"、"评分：0.9"、"0.7/1.0" 等格式。
    用正则提取第一个 0.0~1.0 范围内的浮点数，比 float() 直接转换更鲁棒。
    """
    if not raw:
        return 0.5

    # 匹配 0.0 ~ 1.0 之间的数字（支持 0, 0.5, .8, 1, 1.0 等格式）
    matches = re.findall(r'\b([01](?:\.\d+)?|\.\d+)\b', raw.strip())
    for m in matches:
        try:
            val = float(m)
            if 0.0 <= val <= 1.0:
                return val
        except ValueError:
            continue

    return 0.5  # 所有方法都失败时给中间分，既不过于乐观也不过于悲观

def load_bm25_documents(vectorstore: Chroma) -> list[Document]:
    data = vectorstore.get(
        include=["documents", "metadatas"],
    )

    ids = data["ids"]
    texts = data.get("documents") or []
    metadatas = data.get("metadatas")

    documents = []

    for i, text in enumerate(texts):
        if not text or not text.strip():
            continue

        metadata = (
            (metadatas[i] or {})
            if metadatas is not None
            else {}
        )

        documents.append(
            Document(
                id=ids[i],
                page_content=text,
                metadata=metadata,
            )
        )

    return documents

def build_retrieval_graph(vectorstore: Chroma,
                          documents: list[Document]
                          ) -> StateGraph:
    # 通过闭包捕获 vectorstore 和 documents
    def _vector_node(state: RetrievalState) -> dict:
        queries = state.get("rewritten_queries", [state["original_query"]])
        results = vector_search(queries,vectorstore)
        return {"vector_results": results}

    def _bm25_node(state: RetrievalState) -> dict:
        queries = state.get("rewritten_queries", [state["original_query"]])
        results = bm25_search(queries, documents)
        return {"bm25_results": results}

    builder = StateGraph(RetrievalState)

    builder.add_node("rewrite", _rewrite_node)
    builder.add_node("vector_search", _vector_node)
    builder.add_node("bm25_search", _bm25_node)
    builder.add_node("merge",_merge_node)
    builder.add_node("rerank",_rerank_node)
    builder.add_node("llm_grade",_llm_grage_node)
    #builder.add_node("generate",_generate_node)
    builder.add_node("fallback",_fallback_node)

    builder.add_edge(START,"rewrite")
    builder.add_edge("rewrite","vector_search")
    builder.add_edge("rewrite","bm25_search")

    builder.add_edge("vector_search","merge")
    builder.add_edge("bm25_search","merge")
    builder.add_edge("merge","rerank")
    builder.add_edge("rerank","llm_grade")
    builder.add_conditional_edges("llm_grade",_route_by_score)
    #builder.add_edge("generate",END)
    builder.add_edge("fallback",END)

    return builder.compile()


def answerRAG(builder,question) -> dict:
    response = builder.invoke({
        "original_query":question,
        "rewritten_queries":[],
        "vector_results":[],
        "bm25_results":[],
        "merged_results":[],
        "reranked_results":[],
        "relevance_score":0,
        "context":"",
        "answer":"",
        "sources":[],
        "needs_fallback":False,
    })
    return response



def main():
    #pdf_path = "./RAG知识测试语料.pdf"
    #pages = parse_pdf(pdf_path)
    #chunks = chunk_documents(pages,100,10)

    embed_store = load_vectorstore()
    chunks = load_bm25_documents(embed_store)
    builder = build_retrieval_graph(embed_store,chunks)

    def target(inputs: dict) -> dict:
        question = inputs["question"]
        result = answerRAG(
            builder=builder,
            question=inputs["question"],
        )

        return {'answer':answerRAG(
            builder=builder,
            question=question)['answer'],
            'contexts':[doc.page_content for doc in result["reranked_results"]],
        }

    from langsmith import Client

    from create_judge_eval import relevance


    client = Client()

    results = client.evaluate(
        target,  # 调用你的 Agent
        data="rag_datasets",  # 改成你网站上的数据集名称
        evaluators=[relevance2],  # Judge 评分函数
        experiment_prefix="rag-baseline",  # 本次实验名称前缀
        max_concurrency=1,
    )
    """
    | 参数                    | 要传的类型               | 示例            |
|
    | `user_input`           | 字符串 `str`            | `"小明几岁？"` |
    | `response`             | 字符串 `str`            | `"小明20岁。"` |
    | `retrieved_contexts`   | 字符串列表 `list[str]`   | `["小明今年20岁。", "小明住在北京。"]` |
    """


if __name__ == "__main__":
    main()







