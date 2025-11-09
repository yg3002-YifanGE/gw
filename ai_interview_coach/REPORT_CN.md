AI Interview Coach 项目说明（内部报告）

一、项目概述
- 目标：提供一个交互式“AI 面试教练”，支持按岗位与主题检索面试题，收集作答并给出结构化反馈（含 STAR 结构），可统计进步并导出历史。
- 技术要点：
  - RAG 检索：对 Kaggle 面试题做 TF‑IDF 向量化 + 余弦相似度检索。
  - 评估模块：无 Key 使用启发式量表（内容相关性、技术正确性、表达清晰度、STAR 结构），提供改进建议；如配置 `OPENAI_API_KEY`，可启用 LLM 增强反馈。
  - 服务形态：FastAPI 提供 REST API；内置 `/app/` 简易前端页面；Docker 打包可一键运行。

二、数据与索引
- 数据目录：默认 `../kaggle_data`（与项目同级）
- 已支持文件：
  - `deeplearning_questions.csv`（Deep Learning 问题，处理 BOM）
  - `1. Machine Learning Interview Questions`（文本问题列表）
  - `2. Deep Learning Interview Questions`（文本问题列表）
- 索引：`rag/ingest.py` 构建 TF‑IDF，产出 `vectorizer.pkl / matrix.npy / meta.json`；运行时首次缺失会自动构建，也可 `POST /api/index/build` 触发。
- 元数据：构建索引时为每题打上启发式标签：
  - `topic`（Machine Learning / Deep Learning）
  - `qtype`（ml / dl / technical / behavioral）
  - `difficulty`（easy / medium / hard）

三、系统架构
- `app/main.py`：FastAPI 入口与路由（会话、问答、总结、导出、前端静态文件挂载）
- `app/models.py`：请求/响应模型（Profile、SessionConfig、QuestionResponse、Feedback 等）
- `services/retriever.py`：TF‑IDF 构建与检索、元数据选项查询
- `services/sessions.py`：会话存储（JSON），包含 pending question、mock 进度与配置
- `services/eval.py`：启发式评分 + 建议生成
- `services/llm.py`：可选 OpenAI 反馈（`OPENAI_API_KEY` 生效）
- `web/`：简易前端（HTML/CSS/JS），通过 REST API 交互

四、功能与使用
1) 运行
- Docker：
  - 构建：`docker build -t ai-interview-coach .`
  - 一键运行：`bash docker_run.sh`
  - 打开：`http://localhost:8000/app/`（前端），`http://localhost:8000/docs`（API 文档）
- 本地：
  - `pip install -r requirements.txt`
  - `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`

2) 前端页面（/app/）
- 启动会话：选择岗位与筛选条件（topic / qtype / difficulty），选择模式（自由/模拟），点击“启动”。
- 获取问题：点击“获取问题”，可用“换一题（force_new）”。
- 提交答案：建议使用 STAR 结构，提交后返回评分与建议；模拟模式会显示 `剩余题数`。
- 进度与总结：查看累计题数/平均分与每题记录；查看 `summary` 显示各维度均分与优势/改进点 Top 列表；可导出 JSON。

3) 主要 API（节选）
- `GET /health` 健康检查
- `POST /api/index/build` 重建索引
- `GET /api/meta/options` 获取可选 `topics/qtypes/difficulties`
- `POST /api/session/start` `{"profile":{"role":"Data Scientist"},"config":{"filters":{"qtype":"ml","difficulty":"medium"}}}`
- `POST /api/session/start_mock?role=Data%20Scientist&total_questions=5&qtype=ml&difficulty=medium`
- `GET /api/session/{id}/question?force_new=false`
- `POST /api/session/{id}/answer` `{"answer_text":"..."}`
- `GET /api/session/{id}/progress`
- `GET /api/session/{id}/summary`
- `GET /api/session/{id}/export`

五、模型与算法说明
1) 检索（RAG）
- 使用 `TfidfVectorizer(stop_words="english")` 对问题文本向量化，`cosine_similarity` 计算相似度。
- 查询构造：结合 `role` + 可选 `topic` + 最近问答的上下文；可再叠加 filters（topic/qtype/difficulty）。

2) 启发式评估
- 维度：
  - 内容相关性（关键词覆盖）
  - 技术正确性（以相关性为 proxy，附加常数平滑）
  - 表达清晰度（长度阈值的简化指标）
  - STAR 结构（检测 Situation/Task/Action/Result 关键词出现）
- 输出：1..5 分的四维分数、总体分（加权平均）、优势/改进/提示（三组文本）与关键词证据。
- 说明：启发式具有可解释性、零依赖。可通过 `OPENAI_API_KEY` 启用 LLM 增强，获得更细致的点评。

六、数据与存储
- 会话：`data/sessions.json`，存储 `profile/config/history/pending_question`，便于重启后保留进度。
- 索引：`data/index/` 下保存向量器和矩阵（由运行或 `POST /api/index/build` 生成）。
- .gitignore 已忽略 `data/` 与 `.env`，避免将本地数据与密钥提交仓库。

七、部署与提交建议
- Docker：确保数据目录挂载及 `DATA_DIR` 指向 `/data/kaggle_data`；提交前再跑一遍 `docker build` 验证。
- 端口：缺省使用 `8000`；如端口冲突可映射到宿主其它端口。
- 复现实验：推荐提供一份导出的 session JSON（脱敏）随报告提交。

八、局限与改进方向
- 启发式评分无法深度理解答案语义；可通过 LLM 增强或加入知识点检测提升准确性。
- 检索使用 TF‑IDF，语义召回有限；可升级为嵌入向量（FAISS）并使用多路召回。
- 目前题库标签依赖启发式；可人工校准更精细的标签（难度、题型、主题）。
- 可增加多语言支持、更多岗位题库、以及前端的作答模板与可视化报告。

附：常见问题
- `/app/` 打不开：请确保使用最新镜像（`docker build -t ai-interview-coach .`）或通过 `docker_run.sh` 挂载 `web/`。
- 无法检索：首次运行会自动构建索引，或手动调用 `POST /api/index/build`。
- 没有 OpenAI Key：系统会使用启发式评分，功能完整可用。

