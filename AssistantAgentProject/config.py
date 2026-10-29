from dotenv import load_dotenv
import os
from pathlib import Path
from langchain.chat_models import init_chat_model
#from modelscope.models.cv.tinynas_detection.damo.apis.detector_evaluater import mkdir

BASE_DIR = Path(__file__).resolve().parent

#agent model :deepseek-flash=======================================
load_dotenv(override=True)
key = os.getenv("DEEPSEEK_FLASH_KEY")
url = os.getenv("DEEPSEEK_FLASH_URL")
#==================================================================


# chroma config ===================================================
dir_chroma = BASE_DIR / "chroma_store"


embed_collection_name = "rag_collection"
embed_models_url = os.getenv("EMBED_MODEL")
embed_model_ch = os.getenv("EMBED_MODEL_ch")
embed_search_key = 10
# ==================================================================


RERANKER_MODEL = "C:\huggingface_model\cross-encoderModel\ms-marco-MiniLM-L-6-v2" # 你的cross_encoder模型的路径
RELEVANCE_THRESHOLD = 0.3  # 文档充分性评分阈值（低于此值触发 fallback）
ENSEMBLE_WEIGHTS = [0.4, 0.6]  # [BM25 权重, Vector 权重]
RERANK_TOP_N = 6           # 重排序后保留的文档数
RETRIEVAL_K = 5            # 每路召回的文档数
# =================================================================

# bm25=====================================================================
bm25_key = 10
# =========================================================================

# rrf ============================================================
k = 60
vector_weight = 0.5
bm25_weight = 1.0
# ===============================================================

# query_rewrite model =====================================================

QWEN3_1_7B_MODEL = r"C:\huggingface_model\Qwen3_1.7B"

# ===========================================================================


config_memory = {
    "configurable":{
        "thread_id": "user_001"
    }
}
