import hashlib
import operator
from typing import TypedDict,Annotated

import chromadb
from dotenv import load_dotenv
import os


from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from pathlib import Path


from langgraph.graph import StateGraph, START, END

from config import dir_chroma, embed_model_ch, embed_collection_name
#from RAGProject.graph_state import IngestionState




def build_RAG(pdf_path):
    # build pdf file

    pdf_path1 = r"C:\Users\qxx\Desktop\Langchain1.0-Study-main\MyProject\RAGProject\test.pdf"
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF 文件不存在：{pdf_path}")

    # step 1: load pdf
    print("Step 1: 解析 PDF=======================》")
    pages = parse_pdf(pdf_path)

    if not pages or not any(
            page.page_content.strip() for page in pages
    ):
        raise ValueError("PDF 未提取到文字，请检查文件是否为扫描件")

    # step 2: 文本切片
    print("文本切片===================》")
    chunks = chunk_documents(pages)
    if not chunks:
        raise ValueError("PDF 切分结果为空")

    # step 3:embedding + chromaDB 入库
    print("Step 3: Embedding + 入库")
    vectorstore = embed_and_store(chunks)

    return vectorstore




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

def embed_and_store(chunks: list[Document],
                    collection_name: str = "rag_collection",) -> Chroma:
    """
    将文档切片向量化并存入 ChromaDB

    流程：chunks → Embedding → ChromaDB.upsert
    ChromaDB 自动持久化到 config.CHROMA_DIR。

    参数:
        chunks: 文档切片列表
        collection_name: ChromaDB 集合名

    返回:
        ChromaDB 实例
    """

    embeddings = HuggingFaceEmbeddings(model_name = embed_model_ch)

    dir_chroma.mkdir(exist_ok=True, parents=True)

    collection_name = embed_collection_name
    #client = chromadb.PersistentClient(path=str(dir_chroma))
    #reset_collection(collection_name)
    vectorstore = Chroma(
        collection_name = collection_name,
        embedding_function=embeddings,
        persist_directory=str(dir_chroma),
    )
    # vectorstore = Chroma(
    #     collection_name = collection_name,
    #     embedding_function=embeddings,
    #     client=client,
    # )

   # vectorstore.delete_collection()
    # 为每个 chunk 生成唯一 ID（基于内容哈希 + 来源），避免重复入库
    ids = []
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source","unknown")
        page = chunk.metadata.get("page",0)
        content_hash = hashlib.md5(
            chunk.page_content.encode("utf-8")
        ).hexdigest()[:8]
        ids.append(f"{Path(source).stem}_p{page}_{content_hash}_{i}")

    vectorstore.add_documents(documents = chunks,ids = ids)
    return vectorstore





def main():

    build_RAG("val_data/RAG知识测试语料.pdf")
    #vectorstore = build_RAG()

    # if vectorstore is None:
    #     print("RAG构建失败")
    #     return
    # out = retrieval_docs(vectorstore,"RAG")
    # for i in out:
    #     print(i)
    # print("hello world")





if __name__ == "__main__":
    main()