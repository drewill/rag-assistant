import logging
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.Agent import refresh_retrieval

BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "data" / "upload_temp"

logger = logging.getLogger(__name__)

# 本地学习版：串行执行入库和问答，避免同时操作
operation_lock = Lock()
"""
用户上传 PDF
      ↓
main.py
      ↓
临时保存
      ↓
检查是不是PDF
      ↓
ingest_pdf()
      ↓
save_pdf()
      │
      ├─ 计算hash
      ├─ 查重
      ├─ shutil.copy2()
      ├─ 正式保存到 data/pdfs
      └─ SQLite写记录
      ↓
状态："处理中"
      ↓
build_RAG(stored_path)
      │
      ├─ PDF读取
      ├─ 文本切块
      ├─ Embedding
      └─ Chroma入库
      ↓
成功
      ↓
状态："已入库"
"""


"""
init_database()
    ↓
导入 Agent
    ↓
保存 ask_agent
    ↓
yield
════════════════════
服务器一直运行
════════════════════
    ↓
Ctrl+C
    ↓
yield后面的代码
    ↓
程序结束
"""
@asynccontextmanager
async def lifespan(app: FastAPI):
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    from backend.PDFDatabase import init_database

    init_database()

    # 启动时导入，加载一次模型和向量库
    from backend.Agent import ask_agent

    #app.state.ask_agent = ask_agent
    app.state.ask_agent = ask_agent

    yield
    print("网站结束访问。")


app = FastAPI(
    title="PDF 知识库",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


@app.get("/")
def homepage():
    """返回前端页面。"""
    return FileResponse(
        BASE_DIR / "static" / "index1.html",
        media_type="text/html",
    )


@app.get("/api/documents")
def list_documents():
    """读取 PDF 数据库，返回文档列表。"""
    from backend.PDFDatabase import get_all_pdfs

    pdfs = get_all_pdfs()

    return [
        {
            "id": pdf.id,
            "name": pdf.original_name,
            "status": pdf.ingest_status,
            "size_kb": round(pdf.file_size / 1024, 1),
        }
        for pdf in pdfs
    ]


@app.post("/api/upload")
def upload_pdf(file: UploadFile = File(...)):
    """接收 PDF，保存后调用现有入库函数。"""
    from backend.PDFDatabase import ingest_pdf

    try:
        # 去除上传文件名中可能存在的目录部分
        filename = (file.filename or "").replace("\\", "/")
        filename = filename.rsplit("/", 1)[-1]

        if not filename or Path(filename).suffix.lower() != ".pdf":
            raise HTTPException(
                status_code=400,
                detail="请选择 PDF 文件",
            )

        max_size = 20 * 1024 * 1024

        # 每次请求使用独立目录，保留原始文件名
        with TemporaryDirectory(dir=TEMP_DIR) as temp_dir:
            temp_path = Path(temp_dir) / filename
            total_size = 0

            with temp_path.open("wb") as output:
                while chunk := file.file.read(1024 * 1024):
                    total_size += len(chunk)

                    if total_size > max_size:
                        raise HTTPException(
                            status_code=413,
                            detail="文件不能超过 20 MB",
                        )

                    output.write(chunk)

            with temp_path.open("rb") as uploaded:
                header = uploaded.read(1024)

            if b"%PDF-" not in header:
                raise HTTPException(
                    status_code=400,
                    detail="文件内容不是有效的 PDF 格式",
                )

            with operation_lock:
                # 你的 ingest_pdf 会把文件复制到 data/pdfs
                # 完成后才能清理这里的临时文件
                store = ingest_pdf(str(temp_path))

                if store is None:
                    raise RuntimeError("入库未返回向量库")

        return {
            "message": "上传并入库成功",
            "filename": filename,
        }

    except HTTPException:
        raise
    except Exception:
        logger.exception("PDF 入库失败")
        raise HTTPException(
            status_code=500,
            detail="PDF 入库失败，请查看 PyCharm 终端中的错误",
        )
    finally:
        file.file.close()

@app.delete("/api/documents/{pdf_id}")
def delete_document(pdf_id: int):
    from backend.PDFDatabase import delete_pdf
    from utils import load_vectorstore

    try:
        with operation_lock:
            vectorstore = load_vectorstore()

            result = delete_pdf(
                pdf_id=pdf_id,
                vectorstore=vectorstore,
            )

            # 在这里接入你的 BM25 / 检索图刷新函数。
            # 上传、删除后都应该刷新。
            # refresh_retrieval()
            refresh_retrieval()
        return result

    except LookupError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("删除 PDF 失败")
        raise HTTPException(
            status_code=500,
            detail="删除未完成，请查看终端日志后重试",
        ) from exc
@app.post("/api/chat")
def chat(body: ChatRequest):
    """将网页问题交给 Agent。"""
    question = body.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="问题不能为空",
        )

    try:
        with operation_lock:
            answer = app.state.ask_agent(question)

        return {"answer": answer}

    except Exception:
        logger.exception("问答失败")
        raise HTTPException(
            status_code=500,
            detail="问答失败，请查看 PyCharm 终端中的错误",
        )