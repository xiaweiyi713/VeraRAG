# VeraRAG 开发进度

> 最后更新：2026-05-27

## 项目概况

| 指标 | 数值 |
|------|------|
| 源码行数 | ~8,000 行 (src/) |
| 测试行数 | ~3,000 行 (tests/) |
| Web UI 行数 | ~2,200 行 (web/) |
| 测试数量 | 81 passed + 3 skipped (optional E2E) / 13 test files |
| 语料文档 | 42 篇 / 8 主题 |
| 标注问题 | 102 道 / 6 类型 × 17 道 |
| 冲突类型 | 8 种 |
| LLM Provider | 6 种 |
| Lint/Type | ruff ✓ mypy ✓ |

---

## 已完成功能

### 1. 核心 Pipeline（10 阶段）
- [x] Task Analyzer — 规则 + LLM 任务分析，复杂度/跳数估计
- [x] Decomposition Planner — 子问题分解 + 不确定性驱动的 refine_plan
- [x] Dynamic Retrieval Agent — 多轮检索，query 变体生成，反证检索，覆盖度评估
- [x] Evidence Normalizer — 语义去重，可信度/时效性评分，质量过滤
- [x] Conflict Graph Builder — 8 种冲突检测（数值/时间/实体/范围/因果/粒度/定义/来源），severity + resolver_strategy
- [x] Uncertainty Controller — 5 维不确定性估计，6 种决策动作
- [x] Reasoning Agent — LLM 结构化推理（answer + claims + steps），fallback 合成
- [x] Verifier Agent — NLI + LLM claim 级验证，冲突忽略检测
- [x] Repair Agent — 过度自信降级，不支持声明修复，冲突注释添加
- [x] VeraRAG Pipeline Orchestrator — SSE streaming callback，config 驱动的 ablation 开关

### 2. 检索系统
- [x] BM25Retriever — rank_bm25 + jieba 中文分词
- [x] DenseRetriever — sentence-transformers + cosine similarity
- [x] FAISSRetriever — IndexFlatIP 向量检索
- [x] HybridRetriever — RRF 融合 (sparse + dense)
- [x] EvidenceAwareReranker — CrossEncoder + 可信度/时效性多信号

### 3. 评估框架（5 模块 ~20 指标）
- [x] AnswerMetrics — EM, F1
- [x] EvidenceMetrics — precision/recall, supporting fact, citation, joint EM
- [x] ConflictMetrics — detection F1, type accuracy, resolution accuracy
- [x] CalibrationMetrics — ECE, Brier Score, AUROC
- [x] HallucinationMetrics — unsupported claim rate, entity/numerical hallucination

### 4. VeraBench 基准测试集
- [x] 42 篇语料文档（AI政策/科技公司/气候/量子计算/新能源/半导体/AI医疗/LLM历史）
- [x] 102 道标注问题（6 类型 × 17 道）：single_evidence, multi_evidence, conflict, temporal, unanswerable, misleading
- [x] Ground truth claims + evidence refs + expected conflicts + expected behavior
- [x] VeraBenchLoader — schema 验证（问题类型/证据类别/冲突类型/行为标注）
- [x] VeraBenchEvaluator — demo/pipeline/baseline 模式，按类型/难度分组
- [x] difficulty 分级验证脚本 (`scripts/validate_difficulty.py`)
- [x] Calibration 校准曲线生成 (`experiments/calibration_curve.py`)

### 5. 文档导入管线
- [x] DocumentLoader — JSONL/JSON/TXT/Markdown/PDF
- [x] TextChunker — fixed/sentence/paragraph/heading 4 种策略
- [x] IngestionPipeline — load → chunk → build retriever index (BM25/Dense/FAISS/Hybrid)
- [x] CLI 工具 `experiments/build_index.py`
- [x] Web UI 文件上传（PDF/TXT/MD 拖拽上传 + 自动入库）

### 6. Web UI
- [x] 首页 — 查询输入 + 6 阶段进度动画 + SSE/WebSocket 流式
- [x] 结果面板 — 置信度条 + 5 个 Tab：
  - 证据列表（来源/标题/分数）
  - Claim Ledger（状态图标 ●✕◐ + 支持证据/冲突证据 ID）
  - 冲突图（SVG 节点 + 支持线/反驳线）
  - 不确定性瀑布图（SVG 堆叠横条 + 5 维权重）
  - 修复记录（原文删除线 + 修复后绿色高亮 + 置信度变化）
- [x] 历史页面 — 查询列表 + 置信度条
- [x] 详情页面 — Claim Ledger + 冲突边 + 不确定性分解 + 修复 diff + Markdown 导出
- [x] 设置弹窗 — 6 种 LLM Provider 配置
- [x] `/query/retrieval-demo` — 真实 BM25 检索 + 模拟推理（无需 LLM）
- [x] SSE 心跳 + 网络断开提示 + 可配置超时
- [x] WebSocket 双向通信（优先 WS，自动降级 SSE）
- [x] 亮色/暗色主题切换（CSS 变量 + localStorage 持久化）
- [x] 移动端响应式适配（nav / 输入区 / Tab 栏 / 卡片）

### 7. 实验脚本
- [x] `run_verabench.py` — VeraBench 评测（demo/full 模式）
- [x] `run_ablation.py` — 7 组消融实验
- [x] `run_baselines.py` — 3 种基线对比（Vanilla RAG / Hybrid RAG / Self-RAG）
- [x] `run_hotpotqa.py` / `run_fever.py` / `run_ckt_conflict.py` — 外部数据集评测（含 demo 模式）
- [x] `build_index.py` — 索引构建 CLI
- [x] `calibration_curve.py` — 校准曲线 SVG 生成

### 8. 基线实现
- [x] Vanilla RAG — 单轮 BM25 + LLM 生成
- [x] Hybrid RAG — BM25 + 重排序 + LLM 生成
- [x] Self-RAG — 检索 → 生成 → 自反思 → 重生成
- [x] 每种基线都有 Mock 版本用于无 LLM 测试

### 9. 测试覆盖
- [x] `test_data_structures.py` — 核心数据结构
- [x] `test_evaluation_metrics.py` — 5 个评估模块
- [x] `test_entity_conflict.py` — 实体/否定冲突
- [x] `test_query_variants.py` — 查询变体生成
- [x] `test_refine_plan.py` — 计划修正
- [x] `test_semantic_dedup.py` — 语义去重
- [x] `test_benchmark.py` — VeraBench loader/evaluator
- [x] `test_web_api.py` — FastAPI 端点
- [x] `test_web_db.py` — SQLite 操作 + API Key 加密验证
- [x] `test_e2e_real_llm.py` — 真实 LLM 端到端测试（optional, RUN_REAL_LLM_TESTS=1）

### 10. 工程化
- [x] 统一 LLMClient（6 provider: OpenAI/Anthropic/Ollama/DashScope/智谱/DeepSeek）
- [x] pyproject.toml（可 pip install -e .）
- [x] MIT LICENSE
- [x] requirements.txt（完整依赖列表）
- [x] GitHub Actions CI（Python 3.10/3.11/3.12）+ 覆盖率徽章
- [x] Dockerfile
- [x] Makefile（test/lint/format/run/coverage/docker）
- [x] 结构化日志（logging 替换 print）
- [x] ruff（替代 flake8/black）— lint + format
- [x] mypy — 类型检查通过
- [x] conda 环境配置（environment.yml）
- [x] API Key 加密存储（Fernet）
- [x] 配置热更新（无需重启服务）
- [x] 论文素材：架构图 / 消融实验表 / 冲突类型表 / VeraBench 概览表

---

## 待开发

### 研究方向
- [ ] **Dense Retriever 集成** — 当前因 sentence-transformers 未装而跳过，需实测 BGE 向量检索效果
- [ ] **Cross-Encoder Reranker 集成** — 同上，实测 bge-reranker 效果
- [ ] **不确定性校准实验** — Temperature Scaling + ECE 优化前后对比
- [ ] **多轮对话支持** — 追问、澄清、上下文跟踪
- [ ] **多语言支持** — 当前中文为主，扩展英文 QA 能力
- [ ] **Agent 路由优化** — 根据任务类型自动选择最优 pipeline 配置

### 产品方向
- [ ] **PDF 导出** — weasyprint 依赖较重，当前仅支持 Markdown 导出
- [ ] **API Key 管理 UI** — 多 key 轮换、过期提醒

### 论文方向
- [ ] **LaTeX 论文源码** — 基于已有素材撰写完整论文
- [ ] **更多 baseline 对比** — CRUD-RAG, RECALL, ALCE 等最新方法
- [ ] **Human Evaluation** — 人工评估答案质量
- [ ] **Case Study** — 精选典型案例分析
- [ ] **Limitation 讨论** — 系统局限性与未来工作
