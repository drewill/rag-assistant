import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core import documents
from langchain_core.documents import Document

from config import bm25_key
from config import dir_chroma, embed_collection_name

vectorstore_out = None
def build_vectorstore(collection_name:str):
    embeddings = HuggingFaceEmbeddings(model_name = "")
    vectorstore = Chroma(embeddings = embeddings,
                         collection_name=collection_name,
                         persist_directory=dir_chroma)
    vectorstore_out = vectorstore
    return vectorstore

def insert_vectorstore(chunks:documents):
    vectorstore = build_vectorstore(collection_name=embed_collection_name)

    ids = []
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)
        content_hash = hashlib.md5(
            chunk.page_content.encode("utf-8")
        ).hexdigest()[:8]
        ids.append(f"{Path(source).stem}_p{page}_{content_hash}_{i}")

    vectorstore.add_documents(documents=chunks, ids=ids)

import re
import jieba


def bm25_tokenize(text: str) -> list[str]:
    parts = re.findall(
        r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*|[\u4e00-\u9fff]+",
        text,
    )

    tokens = []

    for part in parts:
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            tokens.extend(
                word for word in jieba.lcut(part)
                if word.strip()
            )
        else:
            tokens.append(part.lower())

    return tokens
_BM25_CACHE: dict[tuple[int, int], object] = {}
def _get_or_build_bm25(documents: list[Document], k: int = bm25_key):
    """
        获取（或构建）BM25 检索器

        BM25Retriever.from_documents() 内部会做分词、构建倒排索引，
        对大文档集来说不便宜。这里用模块级 dict 缓存索引实例。
        """
    from langchain_community.retrievers import BM25Retriever

    cache_key = (id(documents), len(documents))
    bm25 = _BM25_CACHE.get(cache_key)
    if bm25 is None:
        bm25 = BM25Retriever.from_documents(documents,preprocess_func=bm25_tokenize, k=k)
        _BM25_CACHE[cache_key] = bm25
    else:
        # 命中缓存时同步 k 值（k 是检索时参数，不影响索引）
        bm25.k = k
    return bm25

def clear_bm25_cache():
    _BM25_CACHE.clear()
