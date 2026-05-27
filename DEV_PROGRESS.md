# VeraRAG 开发进度

> 最后更新：2026-05-27

## 项目概况

| 指标 | 数值 |
|------|------|
| 源码行数 | ~7,400 行 (src/) |
| 测试行数 | ~2,600 行 (tests/) |
| Web UI 行数 | ~1,800 行 (web/) |
| 测试数量 | 172 passed + 3 skipped (real LLM) / 19 test files |
| 语料文档 | 42 篇 / 8 主题 |
| 标注问题 | 102 道 / 6 类型 |
| 冲突类型 | 8 种 |
| LLM Provider | 6 种 |

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
- [x] 48 道标注问题（6 类型 × 8 道）：single_evidence, multi_evidence, conflict, temporal, unanswerable, misleading
- [x] Ground truth claims + evidence refs + expected conflicts + expected behavior
- [x] VeraBenchLoader — schema 验证（问题类型/证据类别/冲突类型/行为标注）
- [x] VeraBenchEvaluator — demo/pipeline/baseline 模式，按类型/难度分组

### 5. 文档导入管线
- [x] DocumentLoader — JSONL/JSON/TXT/Markdown/PDF
- [x] TextChunker — fixed/sentence/paragraph/heading 4 种策略
- [x] IngestionPipeline — load → chunk → build retriever index (BM25/Dense/FAISS/Hybrid)
- [x] CLI 工具 `experiments/build_index.py`

### 6. Web UI
- [x] 首页 — 查询输入 + 6 阶段进度动画 + SSE 流式
- [x] 结果面板 — 置信度条 + 5 个 Tab：
  - 证据列表（来源/标题/分数）
  - Claim Ledger（状态图标 ●✕◐ + 支持证据/冲突证据 ID）
  - 冲突图（SVG 节点 + 支持线/反驳线）
  - 不确定性瀑布图（SVG 堆叠横条 + 5 维权重）
  - 修复记录（原文删除线 + 修复后绿色高亮 + 置信度变化）
- [x] 历史页面 — 查询列表 + 置信度条
- [x] 详情页面 — Claim Ledger + 冲突边 + 不确定性分解 + 修复 diff
- [x] 设置弹窗 — 6 种 LLM Provider 配置
- [x] `/query/retrieval-demo` — 真实 BM25 检索 + 模拟推理（无需 LLM）

### 7. 实验脚本
- [x] `run_verabench.py` — VeraBench 评测（demo/full 模式）
- [x] `run_ablation.py` — 7 组消融实验（full/no_conflict/no_uncertainty/no_verification/no_repair/minimal/single_round）
- [x] `run_baselines.py` — 3 种基线对比（Vanilla RAG / Hybrid RAG / Self-RAG）
- [x] `run_hotpotqa.py` / `run_fever.py` / `run_ckt_conflict.py` — 外部数据集评测
- [x] `build_index.py` — 索引构建 CLI

### 8. 基线实现
- [x] Vanilla RAG — 单轮 BM25 + LLM 生成
- [x] Hybrid RAG — BM25 + 重排序 + LLM 生成
- [x] Self-RAG — 检索 → 生成 → 自反思 → 重生成
- [x] 每种基线都有 Mock 版本用于无 LLM 测试

### 9. 测试覆盖（172 tests + 3 real LLM）
- [x] `test_data_structures.py` — 核心数据结构
- [x] `test_evaluation_metrics.py` — 5 个评估模块
- [x] `test_conflict_detection.py` — 8 种冲突检测器
- [x] `test_entity_conflict.py` — 实体/否定冲突
- [x] `test_query_variants.py` — 查询变体生成
- [x] `test_refine_plan.py` — 计划修正
- [x] `test_semantic_dedup.py` — 语义去重
- [x] `test_ingestion.py` — 文档加载/分块/管线
- [x] `test_benchmark.py` — VeraBench loader/evaluator
- [x] `test_web_api.py` — FastAPI 端点
- [x] `test_web_db.py` — SQLite 操作
- [x] `test_agents.py` — TaskAnalyzer/ReasoningAgent/VerifierAgent/RepairAgent
- [x] `test_retrievers.py` — BM25/Reranker
- [x] `test_evidence_modules.py` — EvidenceExtractor/EvidenceScorer
- [x] `test_uncertainty_modules.py` — Estimator/Calibrator/Controller
- [x] `test_integration.py` — Mock LLM 端到端 pipeline
- [x] `test_e2e_real_llm.py` — 真实 LLM E2E 测试（可选，RUN_REAL_LLM_TESTS=1）
- [x] `conftest.py` — real_llm marker 注册与自动 skip

### 10. P0 修复
- [x] **verifier_agent.py** — `conflict.conflict` → `conflict.confidence`（属性引用 bug）
- [x] **Pipeline LLM 配置传递** — `api_key` 优先从配置读取，fallback 到环境变量
- [x] **base_url passthrough** — Pipeline 传递 `base_url` 到 LLMClient

### 11. 工程化
- [x] 统一 LLMClient（6 provider: OpenAI/Anthropic/Ollama/DashScope/智谱/DeepSeek）
- [x] pyproject.toml（可 pip install -e .）
- [x] MIT LICENSE
- [x] requirements.txt（完整依赖列表）
- [x] GitHub Actions CI（Python 3.10/3.11/3.12）
- [x] Dockerfile
- [x] Makefile（test/lint/format/run/coverage/docker）
- [x] 结构化日志（logging 替换 print）
- [x] 论文素材：架构图 / 消融实验表 / 冲突类型表 / VeraBench 概览表

---

## 待完善

### P1 — 核心功能增强
- [x] **Pipeline 集成真实 LLM 端到端测试** — 已添加 test_e2e_real_llm.py（3 个真实 LLM 测试 + 3 个配置传递测试）
- [x] **P0: verifier bug 修复** — conflict.conflict → conflict.confidence
- [x] **P0: Web API key 配置传递** — Pipeline 优先读取 config 中的 api_key，fallback 到环境变量
- [ ] **外部数据集自动下载验证** — `download_datasets.py` 已实现但未测试 HotpotQA/FEVER 下载是否成功
- [x] **配置热更新** — Web UI 修改 LLM 配置后每次查询创建新 Pipeline 实例，已生效
- [x] **Web UI 错误处理增强** — SSE 重试(3次)、断网 banner、错误分类、BM25 线程锁

### P1 — 工程与产品（已完成）
- [x] **Evidence ID 对齐** — Pipeline 使用 doc_id 作为稳定 ID，不再生成 UUID
- [x] **BaseRetriever save_index/load_index** — 改为 raise NotImplementedError
- [x] **API Key Fernet 加密存储** — web/db.py CryptoEngine，明文 key 不再落盘
- [x] **亮色/暗色主题切换** — CSS 变量双主题 + localStorage 持久化 + 切换按钮
- [x] **Web UI 文件上传** — PDF/TXT/MD 拖拽上传 + 自动 chunk + 合并 BM25 索引
- [x] **结果导出（Markdown）** — /history/{id}/export?format=md 端点 + detail.html 导出按钮
- [x] **配置优先级统一** — Web DB > config yaml > defaults，/api/status 返回生效配置
- [x] **移动端响应式适配** — nav 栏/输入区/Tab/卡片 响应式布局
- [x] **实验可复现性** — 结果 JSON 包含 git commit/model/temperature/timestamp 等元数据
- [x] **IngestionPipeline.ingest_and_index_from_docs** — 支持从文档列表直接构建索引

### P2 — 数据与评测
- [ ] **VeraBench 扩充到 100+ 问题** — 当前已有 102 道（已扩充）
- [ ] **添加 difficulty 分级验证** — easy/medium/hard 的标注质量检查
- [ ] **Calibration 校准曲线生成** — 可视化置信度 vs 实际准确率
- [ ] **Baseline 在真实 LLM 下的分数** — 当前只有 demo 模式的模拟分数
- [ ] **HotpotQA/FEVER 实际评测结果** — 运行 `run_hotpotqa.py` / `run_fever.py` 获取真实分数

### P3 — 工程完善
- [x] **`save_index`/`load_index` 基类方法** — 已改为 raise NotImplementedError
- [ ] **覆盖率徽章** — 在 README 添加 coverage badge
- [ ] **Type checking 通过** — mypy 当前未正式运行
- [ ] **Lint 通过** — ruff 替代 flake8/black
- [ ] **conda 环境配置** — 添加 `environment.yml`

---

## 待开发

### 研究方向
- [ ] **Dense Retriever 集成** — 当前因 sentence-transformers 未装而跳过，需实测 BGE 向量检索效果
- [ ] **Cross-Encoder Reranker 集成** — 同上，实测 bge-reranker 效果
- [ ] **不确定性校准实验** — Temperature Scaling + ECE 优化前后对比
- [ ] **多轮对话支持** — 追问、澄清、上下文跟踪
- [ ] **多语言支持** — 当前中文为主，扩展英文 QA 能力
- [ ] **Agent 路由优化** — 根据任务类型自动选择最优 pipeline 配置
- [ ] **冲突检测升级** — 规则 + NLI + LLM adjudication 三层
- [ ] **Claim extraction schema 强化** — subject/predicate/object/time/scope/number/polarity/modality

### 产品方向
- [x] **PDF 文件上传** — 已完成
- [x] **API Key 加密存储** — 已完成
- [ ] **WebSocket 模式** — 替代 SSE，支持双向通信
- [x] **暗色/亮色主题切换** — 已完成
- [ ] **移动端适配** — 响应式布局优化
- [ ] **结果导出** — PDF/Markdown 格式导出查询结果

### 论文方向
- [ ] **LaTeX 论文源码** — 基于已有素材撰写完整论文
- [ ] **更多 baseline 对比** — CRUD-RAG, RECALL, ALCE 等最新方法
- [ ] **Human Evaluation** — 人工评估答案质量
- [ ] **Case Study** — 精选典型案例分析
- [ ] **Limitation 讨论** — 系统局限性与未来工作
