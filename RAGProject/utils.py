import hashlib

from langchain_chroma import Chroma

from langchain_huggingface import HuggingFaceEmbeddings

from config import RERANKER_MODEL
from config import dir_chroma, embed_model_ch, embed_collection_name


def create_sample_documents():
    try:
        from fpdf import FPDF
    except ImportError:
        print("fpdf2 未安装，跳过样本 PDF 生成")
        print("安装方式：pip install fpdf2")
        return

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "2024 AI Technology Report", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0,6,
        "This report provides an overview of major AI developments in 2024. "
        "Large Language Models (LLMs) have seen significant improvements in "
        "reasoning capabilities, multilingual understanding, and tool use."
    )
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Key Metrics Comparison", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # 表格
    pdf.set_font("Helvetica", size=10)
    headers = ["Model", "Parameters", "Context Length", "MMLU Score"]
    col_widths = [40, 35, 40, 35]

    # 表头
    pdf.set_fill_color(200, 220, 255)
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 8, h, border=1, fill=True)
    pdf.ln()

    rows = [
        ["GPT-4o", "1.8T (est.)", "128K", "88.7%"],
        ["Claude 3.5", "Unknown", "200K", "88.3%"],
        ["Gemini Ultra", "Unknown", "1M", "90.0%"],
        ["Llama 3.1", "405B", "128K", "85.2%"],
    ]
    for row in rows:
        for i,cell in enumerate(row):
            pdf.cell(col_widths[i], 7, cell, border=1)
        pdf.ln()

    pdf.ln(5)
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 6,
                   "The table above shows that Gemini Ultra achieves the highest MMLU score "
                   "at 90.0%, while Claude 3.5 offers the longest context window at 200K tokens. "
                   "Open-source models like Llama 3.1 are closing the gap with proprietary models."
                   )
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Retrieval-Augmented Generation", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 6,
                   "RAG (Retrieval-Augmented Generation) combines information retrieval with "
                   "text generation. The system first retrieves relevant documents from a knowledge "
                   "base, then uses an LLM to generate answers grounded in those documents.\n\n"
                   "Key RAG improvements in 2024:\n"
                   "- Hybrid search combining BM25 and vector retrieval\n"
                   "- Cross-encoder reranking for higher precision\n"
                   "- Multi-modal RAG handling tables, images, and structured data\n"
                   "- Query rewriting and expansion techniques"
                   )
    pdf_path = "./sample_report.pdf"

    pdf.output(pdf_path)
    return pdf_path

def reset_collection(collection_name: str = "rag_collection"):
    """
    清空指定集合（用于重复运行示例时避免重复数据）

    参数:
        collection_name: 要清空的集合名
    """
    import chromadb


    dir = dir_chroma
    client = chromadb.PersistentClient(path=str(dir))
    try:
        client.delete_collection(collection_name)
        print(f"  [OK] 已清空集合: {collection_name}")
    except Exception:
        pass  # 集合不存在时忽略


def retrieval_docs(vectorstore,content: str,key: int=3):
    #docs = vectorstore.similarity_search("2024 AI Report", k=3)

    # retrieval = vectorstore.as_retriever(
    #     search_type = "mmr", # search_type="similarity"
    #     search_kwargs = {
    #         "k":3, "fetch_k":8,"lambda_mult":0.5
    #     }
    # )
    retrieval = vectorstore.as_retriever(
        search_type = "similarity",
        search_kwargs = {
            "k":key
        }
    )
    docs= retrieval.invoke(content)
    return docs

def load_vectorstore():
    """连接已有的向量库，不生成或导入示例文档。"""
    embeddings = HuggingFaceEmbeddings(
        model_name=embed_model_ch
    )
    dir_chroma.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=embed_collection_name,
        embedding_function=embeddings,
        persist_directory=str(dir_chroma),
    )

def content_hash(text: str) -> str:
    """计算文本内容的哈希值，用于去重"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]

def get_reranker(model_name:str = RERANKER_MODEL):
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
    cross_encoder = HuggingFaceCrossEncoder(model_name=model_name)
    return CrossEncoderReranker(model=cross_encoder)

def format_sources(documents) -> list[dict]:
    """
    从文档列表中提取来源信息

    返回:
        来源字典列表，每个含 source、page、preview 字段
    """
    sources = []
    for doc in documents:
        meta = doc.metadata
        sources.append({
            "source": meta.get("source", "unknown"),
            "page": meta.get("page", "?"),
            "preview": truncate_text(doc.page_content, 80),
        })
    return sources

def truncate_text(text: str, max_len: int = 100) -> str:
    """截断文本用于安全预览"""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."