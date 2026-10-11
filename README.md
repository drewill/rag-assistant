# 企业级知识库查询项目
# PDF 知识库问答系统

基于 FastAPI、LangChain 和 LangGraph 构建的 PDF 知识库问答项目。

用户可以通过网页上传 PDF、查看文档列表并针对文档内容提问。系统结合关键词检索和向量检索，经过重排与证据充分性判断，生成带来源提示的回答。

本项目用于学习和实践 RAG、Agent 编排、问答评估及 Web 服务开发，目前以本地单机运行为主。

## 功能

- **PDF 文档管理**：上传、列表查询、入库状态展示和删除。
- **文档入库**：PDF 文本提取、文本切分、Embedding 和 Chroma 存储。
- **混合检索**：结合 BM25 关键词检索与向量语义检索。
- **查询改写**：生成查询变体，扩展检索范围。
- **结果重排**：使用 Cross-Encoder 对候选片段重新排序。
- **证据判断**：判断检索内容是否足以回答问题，信息不足时返回提示。
- **网页问答**：通过 FastAPI 提供文档管理与问答接口。
- **评估与追踪**：接入 LangSmith，并使用 Ragas 指标评估回答。

## 技术栈

| 模块 | 技术 |
|---|---|
| 开发语言 | Python |
| Web 后端 | FastAPI、Uvicorn |
| 前端 | HTML、CSS、JavaScript |
| LLM 应用框架 | LangChain |
| 工作流编排 | LangGraph |
| 文档元数据存储 | SQLite、SQLAlchemy |
| 向量存储 | Chroma |
| 关键词检索 | BM25 |
| 向量模型 | HuggingFace Embedding 模型 |
| 重排模型 | Cross-Encoder |
| 调用追踪与实验管理 | LangSmith |
| RAG 评估 | Ragas |

生成模型通过兼容 OpenAI API 的接口调用，模型名称、服务地址和密钥由项目配置提供。

## 工作流程

### 文档入库

1. 用户上传 PDF。
2. 检查文件大小、后缀和 PDF 标记。
3. 计算文件哈希，保存文件并登记数据库记录。
4. 提取文本并切分为片段。
5. 计算文本向量并写入 Chroma。
6. 更新文档入库状态。

### 知识库问答

1. 接收用户问题。
2. 改写问题，生成多个检索查询。
3. 分别执行 BM25 和向量检索。
4. 合并候选结果并去重。
5. 使用重排模型筛选相关片段。
6. 判断证据是否充分。
7. 基于上下文生成回答；证据不足时返回提示。

> 当前候选合并方式应以代码为准。简单合并去重与 RRF 排名融合是不同实现。

## 数据存储

| 数据 | 保存位置或方式 |
|---|---|
| 上传后的 PDF | `data/pdfs/` |
| PDF 名称、路径、大小和状态 | SQLite |
| 文本片段、来源信息和向量 | Chroma |
| BM25 检索器 | 进程内存 |

BM25 可根据 Chroma 中保存的文本片段重建。知识库新增、删除或更新文档后，需要同步刷新 BM25 和运行中的检索图。

SQLite 与 Chroma 是独立存储，不会自动同步。删除文档时，需要分别处理文件、数据库记录和向量片段。

## 项目结构

以下列出主要文件，具体以仓库为准：

```text
RAGProject/
├── main.py                  # FastAPI 入口及 HTTP 接口
├── Agent.py                 # Agent 与问答入口
├── RAGSystem.py             # PDF 解析、切块和向量入库
├── RAGSystem_graph_enp.py    # LangGraph 检索与生成流程
├── PDFDatabase.py           # PDF 元数据与文件管理
├── vector_bm25_store.py      # BM25 构建与缓存
├── graph_state.py           # 图状态定义
├── utils.py                 # 向量库加载、重排等工具
├── config.py                # 配置读取与参数
├── create_judge_eval.py     # 评估器定义
├── langsmithEVAL.py         # 批量评估入口
├── static/
│   └── index.html           # 网页界面
└── data/
    ├── pdfs/                # 正式保存的 PDF
    └── upload_temp/         # 上传临时目录
```

## 本地运行

### 1. 准备环境

当前开发环境为 Windows、Conda 和 PyCharm。

已有开发环境可执行：

```powershell
conda activate project02
```

首次部署需要安装项目依赖。仓库提供 `requirements.txt` 后，可执行：

```powershell
python -m pip install -r requirements.txt
```

### 2. 配置模型与存储

按照 `config.py` 和环境变量读取逻辑配置：

- LLM API Key。
- LLM 服务地址与模型名称。
- 本地 Embedding 模型路径。
- 本地重排模型路径。
- Chroma 持久化目录与集合名称。
- 检索数量、切块参数和证据判断阈值。

注意区分：

- **模型目录**：保存模型权重、配置和分词器。
- **向量库目录**：保存 Chroma 数据。
- **集合名称**：向量库中使用的 collection 名称。

不要将真实 API Key 或密码提交到仓库。

### 3. 启动服务

进入包含 `main.py` 的目录后运行：

```powershell
python -m uvicorn main:app
```

开发时如需修改代码后自动重启：

```powershell
python -m uvicorn main:app --reload
```

访问地址：

- 网页：`http://127.0.0.1:8000/`
- API 文档：`http://127.0.0.1:8000/docs`

首次启动需要加载模型，可能需要等待一段时间。

### 4. 使用系统

1. 打开网页并上传 PDF。
2. 等待入库完成。
3. 在文档列表中检查入库状态。
4. 输入与文档内容相关的问题。
5. 查看回答并核对来源。

## API 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 返回网页 |
| GET | `/api/documents` | 获取文档列表 |
| POST | `/api/upload` | 上传并入库 PDF |
| POST | `/api/chat` | 提交问题 |
| DELETE | `/api/documents/{pdf_id}` | 删除指定 PDF 及关联数据 |

问答请求示例：

```json
{
  "question": "什么是 RAG？"
}
```

响应示例：

```json
{
  "answer": "RAG 是检索增强生成……"
}
```

## 评估

使用 LangSmith 管理测试数据集与实验结果，使用 Ragas 或自定义评估器进行评分。

主要评估维度：

| 指标 | 目的 |
|---|---|
| 答案正确性 | 对比参考答案，判断回答是否正确 |
| Faithfulness | 判断回答是否得到检索上下文支持 |
| Answer Relevancy | 判断回答是否切题 |
| 正确拒答率 | 判断无答案问题是否被合理拒答 |

评估时区分“文档有答案”和“文档没有答案”的样本。

明确表示无法作答的回答可能获得较低的 Answer Relevancy 分数，因此不能仅凭该指标判断系统质量。

评估数据中的参考答案不应直接加入检索知识库，以免发生数据泄漏。

## 当前限制

- 当前主要面向本地单机学习场景。
- 入库与问答使用进程内锁串行执行，并发能力有限。
- 上传接口等待入库完成后返回，尚未实现独立后台任务队列。
- 问答接口返回完整答案，尚未实现流式输出。
- 扫描 PDF 的识别效果依赖后续 OCR 能力。
- SQLite、Chroma 和 BM25 之间的一致性需要应用代码维护。
- 自动评估分数不能替代人工检查。

## 后续计划

- [ ] 完善文档增删后的 BM25 与检索图刷新。
- [ ] 建立文档和片段的稳定 ID 关联。
- [ ] 增加入库失败重试与数据一致性检查。
- [ ] 实现 SSE 流式回答。
- [ ] 增加后台入库任务与进度查询。
- [ ] 增加用户登录与知识库权限隔离。
- [ ] 完善回归评估、并发测试和延迟统计。
- [ ] 提供 Docker 部署方式。

RAGAS主要评估指标
指标	                             通俗理解	                                    主要比较什么
Faithfulness                忠实度	回答有没有资料依据？	                    实际回答与检索资料
Response Relevancy          回答相关性	回答有没有切题？	                    实际回答与用户问题
Factual Correctness         事实正确性	相对参考答案，答对了多少、遗漏了多少？	    实际回答与参考答案
Context Precision           上下文精确性	有用资料有没有排在前面？	            检索结果的相关性及排序
Context Recall              上下文召回率	回答所需的证据找全了吗？	            检索资料与参考答案或标注
Semantic Similarity         语义相似度	实际答案和参考答案意思接近吗？	        两个答案的语义表示