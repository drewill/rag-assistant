import hashlib
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from config import embed_model_ch, dir_chroma
from graph_state import IngestionState
from utils import create_sample_documents



def build_RAG_graph():
    file_path = create_sample_documents()
    if not os.path.exists(file_path):
        print("样本 PDF 不存在，跳过此示例")
        return None


    builder = StateGraph(IngestionState)

    builder.add_node("parse", _parse_pdf)
    builder.add_node("chunk", _chunk_documents)
    builder.add_node("store",_embed_and_store)

    builder.add_edge(START,"parse")
    builder.add_edge("parse", "chunk")
    builder.add_edge("chunk", "store")
    builder.add_edge("store", END)

    build = builder.compile()

    results = build.invoke({
        "file_path": file_path,
        "raw_document": [],
        "text_chunks":[],
        "table_chunks":[],
        "store_vs":[],
        "chunk_count": 0,
        "status": "初始化",
        "errors": [],
    })
    print(f"状态：{results['status']}")
    print(f"切片数：{results['chunk_count']}")
    if results.get('errors'):
        for err in results["errors"]:
            print(err)


    return results["store_vs"][0]



def _parse_pdf(state: IngestionState)-> dict:
    file_path = state["file_path"]
    try:
        loader = PyPDFLoader(file_path=file_path)
        pages = loader.load()
        print(type(pages))
        print(pages)
        return {
            "raw_document": pages,
            "status": f"已解析{len(pages)}页"
        }
    except Exception as e:
        return {
            "raw_document": [],
            "errors":[f"解析文档失败：{e}"],
            "status": "解析失败",
        }

chunk_size=500
chunk_overlap=50

def _chunk_documents(state: IngestionState) -> dict:
    load = state["raw_document"]
    if not load:
        return {
            "text_chunks":[],
            "status":"无文档可切分"
        }
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""],

    )
    chunks = splitter.split_documents(load)
    return {
        "text_chunks":chunks,
        "chunk_count": len(chunks),
        "status":f"已切分成功：{len(chunks)}快"
    }

collection_name = "rag_collection"
def _embed_and_store(state: IngestionState) -> dict:
    chunks = state["text_chunks"]
    try:
        embeddings = HuggingFaceEmbeddings(model_name=embed_model_ch)

        # client = chromadb.PersistentClient(path=str(dir_chroma))
        # reset_collection(collection_name)
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(dir_chroma),
        )
        ids = []
        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", 0)
            content_hash = hashlib.md5(
                chunk.page_content.encode("utf-8")
            ).hexdigest()[:8]
            ids.append(f"{Path(source).stem}_p{page}_{content_hash}_{i}")

        vectorstore.add_documents(documents=chunks, ids=ids)
        return {
            "store_vs":[vectorstore]
        }
    except Exception as e:
        return{
            "store_vs":[],
            "errors":[f"存入错误：{e}"]
        }