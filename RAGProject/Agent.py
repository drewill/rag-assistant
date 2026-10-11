from pathlib import Path

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os
from rich import print as rprint
from langchain_core.tools import tool

from PDFDatabase import get_all_pdfs
from vector_bm25_store import clear_bm25_cache
from config import config_memory
from RAGSystem_graph_enp import answerRAG, load_bm25_documents, build_retrieval_graph
from utils import retrieval_docs, load_vectorstore

from rich import print as rprint

from langgraph.checkpoint.memory import InMemorySaver

load_dotenv(override=True)
key = os.getenv("DEEPSEEK_FLASH_KEY")
url = os.getenv("DEEPSEEK_FLASH_URL")

#vectorstore = build_RAG_graph()
vectorstore = load_vectorstore()
memory = InMemorySaver()





@tool
def get_weather(city: str):
    """
    这是一个查询天气的工具

    args:
        city: 城市的名称
    """
    return f"{city}天气很差，雾霾很重！"

#@tool(parse_docstring=True)
# def local_RAG_vector_knowledge(query: str)->str :
#     """
#     搜索本地知识库中的 PDF、论文和技术文档。
#     当用户的问题需要依据知识库资料回答时使用该工具。
#
#     Args:
#         query: 这是用户提出的问题
#
#     """
#     out = retrieval_docs(vectorstore,query)
#
#     #build_retrieval_graph(vectorstore,)
#     if not out:
#         print("知识库中没有检索到相关内容。")
#
#     results = []
#
#     for i, doc in enumerate(out):
#
#         pdf_id = doc.metadata.get(
#             "pdf_id",
#             "unknown"
#         )
#
#         pdf_name = doc.metadata.get(
#             "pdf_name",
#             "unknown"
#         )
#
#         page = doc.metadata.get(
#             "page",
#             0
#         )
#
#         try:
#             page = int(page) + 1
#         except:
#             pass
#
#         results.append(
#             f"""
#     【资料 {i + 1}】
#
#     PDF ID：
#     {pdf_id}
#
#     文件：
#     {pdf_name}
#
#     页码：
#     {page}
#
#     内容：
#     {doc.page_content}
#     """
#         )
#
#     return "\n".join(results)
#
#     return results
@tool(parse_docstring=True)
def local_RAG_vectorAndBm25_knowledge(query: str)->str:
    """
    搜索本地知识库中的 PDF、论文和技术文档。
    当用户的问题需要依据知识库资料回答时使用该工具。

    Args:
        query: 这是用户提出的问题

    """
    reranked_res = ask_retrieval_agent(question=query)
    if not reranked_res:
        print("知识库中没有检索到相关内容。")
    results = []
    for i,doc in enumerate(reranked_res):
        scorce = doc.metadata.get("source","")
        page = doc.metadata.get("page",0)
        results.append(
            f"资料{i+1}:"
            f"文件来源：{scorce},"
            f"文件页码：{page},"
            f"文件内容：{doc.page_content}"
        )




    return "\n".join(results)




@tool
def list_pdf_documents() -> str:
    """
    查看当前知识库中所有PDF文件。
    当用户询问知识库有哪些文件、有哪些论文时使用。
    """

    pdfs = get_all_pdfs()

    if not pdfs:
        return "当前知识库没有PDF文件。"

    results = []

    for pdf in pdfs:

        size_mb = (
            pdf.file_size /
            1024 /
            1024
        )

        results.append(
            f"""
                PDF ID：{pdf.id}
                文件名：{pdf.original_name}
                大小：{size_mb:.2f} MB
                入库状态：{pdf.ingest_status}
                上传时间：{pdf.created_at}
            """
        )

    return "\n".join(results)


retrieval_graph = None
def refresh_retrieval():
    """知识库增删后，更新 BM25 和检索图。"""
    global retrieval_graph

    # 读取删除后剩余的文档
    documents = load_bm25_documents(vectorstore)

    # 清除原来的 BM25
    clear_bm25_cache()

    # 新图捕获新的 documents
    new_graph = build_retrieval_graph(
        vectorstore=vectorstore,
        documents=documents,
    )

    # 替换旧图
    retrieval_graph = new_graph


prompt ="""
        你是一个本地知识库智能助手。
        
        你拥有两个工具：
        
        1. local_RAG_knowledge
        用于查询PDF正文中的知识。
        
        2. list_pdf_documents
        用于查看知识库有哪些PDF。
        
        如果用户询问文档内容，
        使用local_RAG_knowledge。
        
        如果用户询问知识库有哪些文件，
        使用list_pdf_documents。
        """

prompt1 = """
You are a knowledge base assistant.

When answering questions based on the knowledge base,
always answer in English.

Use the retrieved documents as the factual basis for your answer.
Do not translate the answer into Chinese.
"""

model = init_chat_model(
    model = "deepseek-flash",
    model_provider="openai",
    api_key = key,
    base_url = url,
)

agent = create_agent(
    model=model,
    tools=[get_weather, local_RAG_vectorAndBm25_knowledge,list_pdf_documents],
    system_prompt=prompt,
    checkpointer=memory
)

def ask_agent(question:str)->str:
    response = agent.invoke({
        "messages": [
            {"role": "user", "content": question}
        ]
    },
    config = config_memory)
    final_message = response['messages'][-1]
    content = final_message.content
    if isinstance(content, str):
        return content
    print(final_message)
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)

    return "\n".join(parts)

def ask_retrieval_agent(question:str)->str:
    if retrieval_graph is None:
        refresh_retrieval()

    results = answerRAG(builder=builder_rag(),question=question)
    reranked_res = results['reranked_results']

    return reranked_res


def builder_rag():
    embed_store = load_vectorstore()
    chunks = load_bm25_documents(embed_store)
    builder = build_retrieval_graph(embed_store, chunks)
    return builder


def main():
    answer = ask_agent("RAG是什么")
    rprint(answer)
if __name__ == "__main__":
    main()