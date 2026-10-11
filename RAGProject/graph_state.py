import operator
from typing import TypedDict, Annotated

from langchain_chroma import Chroma
from langchain_core.documents import Document


class IngestionState(TypedDict):
    file_path: str
    raw_document: list[Document]
    text_chunks: Annotated[list[Document],operator.add]
    table_chunks: Annotated[list[Document],operator.add]
    chunk_count: int
    store_vs: Annotated[list[Chroma],operator.add]
    status: str
    errors: Annotated[list[str],operator.add]


# ============================================================================
# 检索生成图状态
# ============================================================================

class RetrievalState(TypedDict):
    """
    检索+生成流水线的共享状态

    original_query: 用户原始提问
    rewritten_queries: LLM 改写后的查询变体列表
    vector_results: 向量检索结果（reducer: 并行追加）
    bm25_results: BM25 检索结果（reducer: 并行追加）
    merged_results: 去重合并后的文档列表
    reranked_results: 重排序后的文档列表
    relevance_score: 文档充分性评分（0-1）
    context: 拼接给 LLM 的最终上下文
    answer: LLM 生成的回答
    sources: 来源引用列表
    needs_fallback: 是否需要外部检索兜底
    """
    original_query: str
    rewritten_queries: list[str]
    vector_results: Annotated[list[Document], operator.add]
    bm25_results: Annotated[list[Document], operator.add]
    merged_results: list[Document]
    reranked_results: list[Document]
    relevance_score: float
    context: str
    answer: str
    sources: list[dict]
    needs_fallback: bool
