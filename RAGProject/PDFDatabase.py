from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, Integer, String, DateTime
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase, sessionmaker
import hashlib
import shutil
import uuid
from RAGSystem import build_RAG

BASE_DIR = Path(__file__).resolve().parent
print(type(BASE_DIR))
DB_PATH = BASE_DIR / "pdf_store.db"

PDF_DIR = BASE_DIR / "data" / "pdfs"

PDF_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DATABASE_URL = f"sqlite:///{DB_PATH}"

# ============================================================
# 2. 创建数据库引擎
# ============================================================
engine = create_engine(
    DATABASE_URL,
    echo=False
)

# ============================================================
# 3. Session
# ============================================================

SessionLocal = sessionmaker(
    bind=engine,
    expire_on_commit=False
)


# ============================================================
# 4. SQLAlchemy 基类
# ============================================================

class Base(DeclarativeBase):
    pass


# ============================================================
# 5. PDF 表
# ============================================================

class PDFDocument(Base):

    __tablename__ = "pdf_documents"

    # 主键
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    # 用户原始文件名
    original_name: Mapped[str] = mapped_column(
        String(255)
    )

    # 实际保存文件名
    stored_name: Mapped[str] = mapped_column(
        String(255)
    )

    # PDF保存位置
    file_path: Mapped[str] = mapped_column(
        String(500)
    )

    # 文件hash
    file_hash: Mapped[str] = mapped_column(
        String(64),
        unique=True
    )

    # 文件大小
    file_size: Mapped[int] = mapped_column(
        Integer
    )

    # RAG入库状态
    ingest_status: Mapped[str] = mapped_column(
        String(50),
        default="未入库"
    )

    # 创建时间
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now
    )


# ============================================================
# 6. 创建数据库表
# ============================================================

def init_database():
    Base.metadata.create_all(engine)
    print("[OK] PDF数据库初始化完成")


def calculate_file_hash(
    file_path: str
) -> str:

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:

        while True:

            data = f.read(1024 * 1024)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()

def save_pdf(file_path: str):

    source_path = Path(file_path)

    # --------------------------
    # 检查文件
    # --------------------------

    if not source_path.exists():
        raise FileNotFoundError(
            f"文件不存在：{file_path}"
        )

    if source_path.suffix.lower() != ".pdf":
        raise ValueError(
            "只能存储 PDF 文件"
        )

    # --------------------------
    # 计算hash
    # --------------------------

    file_hash = calculate_file_hash(
        str(source_path)
    )

    # --------------------------
    # 查询数据库是否已经存在
    # --------------------------

    with SessionLocal() as session:

        old_pdf = (
            session.query(PDFDocument)
            .filter(
                PDFDocument.file_hash == file_hash
            )
            .first()
        )

        if old_pdf:

            print(

                f"[INFO] PDF 已存在："
                f"{old_pdf.original_name}"
            )

            return old_pdf

        # --------------------------
        # 生成新的保存文件名
        # --------------------------

        stored_name = (
            f"{uuid.uuid4().hex}_"
            f"{source_path.name}"
        )

        target_path = (
            PDF_DIR /
            stored_name
        )

        # --------------------------
        # 复制PDF
        # --------------------------

        shutil.copy2(
            source_path,
            target_path
        )

        # --------------------------
        # 获取文件大小
        # --------------------------

        file_size = target_path.stat().st_size

        # --------------------------
        # 写入数据库
        # --------------------------

        pdf = PDFDocument(
            original_name=source_path.name,
            stored_name=stored_name,
            file_path=str(target_path),
            file_hash=file_hash,
            file_size=file_size,
            ingest_status="未入库"
        )

        session.add(pdf)

        session.commit()

        session.refresh(pdf)

        print(
            f"[OK] PDF 已保存，ID={pdf.id}"
        )

        return pdf

def get_all_pdfs():

    with SessionLocal() as session:

        pdfs = (
            session.query(PDFDocument)
            .order_by(PDFDocument.id)
            .all()
        )

        return pdfs

def get_pdf_by_id(pdf_id: int):

    with SessionLocal() as session:

        pdf = session.get(
            PDFDocument,
            pdf_id
        )

        return pdf

def update_pdf_status(
    pdf_id: int,
    status: str
):

    with SessionLocal() as session:

        pdf = session.get(
            PDFDocument,
            pdf_id
        )

        if not pdf:
            return False

        pdf.ingest_status = status

        session.commit()

        return True

def ingest_pdf(file_path: str):

    # ==========================================
    # 1. 保存PDF + 数据库记录
    # ==========================================

    pdf = save_pdf(file_path)

    pdf_id = pdf.id

    stored_path = pdf.file_path

    print(
        f"PDF ID：{pdf_id}"
    )

    # ==========================================
    # 2. 修改状态
    # ==========================================

    update_pdf_status(
        pdf_id,
        "处理中"
    )

    try:
        vectorstore = build_RAG(stored_path)

        update_pdf_status(
            pdf_id,
            "已入库"
        )
        print(
            f"[OK] PDF入库完成："
            f"{pdf.original_name}"
        )
        return vectorstore


    except Exception as e:

        update_pdf_status(
            pdf_id,
            "入库失败"
        )

        raise e

def delete_pdf(pdf_id: int, vectorstore) -> dict:
    """删除指定 PDF 的向量片段、实体文件和数据库记录。"""

    with SessionLocal() as session:
        pdf = session.get(PDFDocument, pdf_id)

        if pdf is None:
            raise LookupError("文档不存在")

        filename = pdf.original_name
        stored_path = pdf.file_path

        # 只允许删除 PDF 保存目录内的文件
        file_path = Path(stored_path).resolve()
        pdf_root = PDF_DIR.resolve()

        if not file_path.is_relative_to(pdf_root):
            raise ValueError("文件路径不在 PDF 保存目录内")

        # 必须与入库时 metadata["source"] 的值一致
        source_filter = {"source": stored_path}

        result = vectorstore.get(
            where=source_filter,
            include=["metadatas"],
        )
        chunk_ids = result["ids"]

        # 已入库却匹配不到片段，先停止，避免误报删除成功。
        # “删除中”允许上一次部分完成后再次重试。
        if not chunk_ids and pdf.ingest_status == "已入库":
            raise ValueError(
                "没有匹配到向量片段，请检查 metadata 中的 "
                "source 是否与数据库 file_path 一致"
            )

        # 在跨存储删除前保留可重试的状态
        pdf.ingest_status = "删除中"
        session.commit()

        # 1. 删除这份 PDF 的全部向量片段
        if chunk_ids:
            vectorstore.delete(ids=chunk_ids)

        # 验证删除结果，失败时保留数据库记录
        remaining = vectorstore.get(
            where=source_filter,
            include=["metadatas"],
        )
        if remaining["ids"]:
            raise RuntimeError("仍有向量片段未删除，请重试")

        # 2. 删除正式保存的 PDF；已不存在则跳过
        file_path.unlink(missing_ok=True)

        # 3. 最后删除 SQLite 记录
        session.delete(pdf)
        session.commit()

        return {
            "message": "删除成功",
            "id": pdf_id,
            "filename": filename,
            "deleted_chunks": len(chunk_ids),
        }

if __name__ == "__main__":
    init_database()



    ingest_pdf(r"C:\Users\qxx\Desktop\sample_report.pdf")
    ingest_pdf(r"C:\Users\qxx\Desktop\MyProject\RAGProject\咖啡烘焙与风味基础.pdf")
    ingest_pdf(r"C:\Users\qxx\Desktop\MyProject\RAGProject\城市公园生态观察记录.pdf")
    print(get_pdf_by_id(1).ingest_status)