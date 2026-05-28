# VeraRAG 开发进度

> 最后更新：2026-05-28

## 项目概况

| 指标 | 数值 |
|------|------|
| 仓库 | https://github.com/xiaweiyi713/VeraRAG |
| 源码行数 | ~8,200 行 (src/) |
| 测试 | 182 passed + 3 skipped (real LLM) |
| Web UI | ~1,800 行 (web/) |
| VeraBench 题库 | 152 题 / 6 类型 |
| VeraBench 语料 | 57 篇 / 13 主题 |
| 冲突检测 | 三层架构（规则10+NLI+LLM） |
| LLM Provider | 6 种 |

---

## 已完成功能

### 1. 核心 Pipeline（10 阶段）
- [x] Task Analyzer — 规则 + LLM 任务分析，复杂度/跳数估计
- [x] Decomposition Planner — 子问题分解 + 不确定性驱动的 refine_plan
- [x] Dynamic Retrieval Agent — 多轮检索，query 变体生成，反证检索，覆盖度评估
- [x] Evidence Normalizer — 语义去重，可信度/时效性评分，质量过滤
- [x] Conflict Graph Builder — **三层架构**: 规则层(10个检测器) + NLI层(CrossEncoder) + LLM裁决层
- [x] Uncertainty Controller — 5 维不确定性估计，6 种决策动作，**不确定性驱动检索**
- [x] Reasoning Agent — LLM 结构化推理，Claim 新增 verifiable/support_type/claim_type
- [x] Verifier Agent — NLI + LLM claim 级验证，冲突忽略检测
- [x] Repair Agent — 过度自信降级，不支持声明修复，冲突注释添加
- [x] VeraRAG Pipeline Orchestrator — SSE streaming callback，config 驱动的 ablation 开关

### 2. 检索系统
- [x] BM25Retriever — rank_bm25 + jieba 中文分词
- [x] DenseRetriever — sentence-transformers + cosine similarity
- [x] FAISSRetriever — IndexFlatIP 向量检索
- [x] HybridRetriever — RRF 融合 (sparse + dense)，**优雅降级到 BM25**
- [x] EvidenceAwareReranker — CrossEncoder + 可信度/时效性多信号

### 3. 冲突检测（三层架构，本次升级）
- [x] **规则层** — 10 个检测器（数值/实体/时间/范围/因果/粒度/定义/来源/支持/语义矛盾）
- [x] **NLI 层** — CrossEncoder NLI 模型判断 entailment/contradiction（可选，自动降级）
- [x] **LLM 裁决层** — 规则和 NLI 都无法判定时的 LLM 兜底
- [x] 年份过滤 — 4 位数字(1800-2099)不参与数值冲突检测
- [x] 动态阈值 — 无共享实体时数值冲突阈值提高到 1.3
- [x] 语义相似度 — Jaccard bigram + SequenceMatcher 双算法

### 4. Claim Schema 强化（本次新增）
- [x] `Claim.verifiable` — bool，判断 claim 是否可验证
- [x] `Claim.support_type` — direct/indirect/none
- [x] `AnswerClaim.claim_type` — factual/inference/prediction
- [x] `AnswerClaim.verifiable` — bool
- [x] `AnswerClaim.support_type` — direct/indirect/none
- [x] EvidenceExtractor 自动判断 verifiable 和 support_type（规则+LLM）

### 5. 评估框架
- [x] AnswerMetrics — EM, F1, **Soft F1（关键词/数字重叠，适合中文）**
- [x] EvidenceMetrics — precision/recall, supporting fact, citation, joint EM
- [x] ConflictMetrics — detection F1, type accuracy, resolution accuracy
- [x] CalibrationMetrics — ECE, Brier Score, AUROC
- [x] HallucinationMetrics — unsupported claim rate, entity/numerical hallucination
- [x] **评估报告新增 ECE + Brier Score 校准指标**

### 6. VeraBench 基准测试集（本次扩展）
- [x] 57 篇语料（AI政策/科技公司/气候/量子计算/新能源/半导体/AI医疗/LLM历史/生物医药/航天/核聚变）
- [x] 152 道标注问题（6 类型均衡分布）
- [x] Ground truth claims + evidence refs + expected conflicts + expected behavior
- [x] VeraBenchLoader — schema 验证
- [x] VeraBenchEvaluator — demo/pipeline/baseline 模式，按类型/难度分组，ECE/Brier

### 7. 不确定性驱动检索（本次实现）
- [x] Pipeline 根据 uncertainty action 动态调整检索策略
- [x] `RESOLVE_CONFLICTS` → seek_counter=True, budget=30（优先检索反证）
- [x] `CONTINUE_RETRIEVAL` → budget=80（扩大检索范围）
- [x] `PROCEED` → 提前退出检索循环
- [x] `prev_decision` 在检索轮次间传递

### 8. 文档导入管线
- [x] DocumentLoader — JSONL/JSON/TXT/Markdown/PDF
- [x] TextChunker — fixed/sentence/paragraph/heading 4 种策略
- [x] IngestionPipeline — load → chunk → build retriever index

### 9. Web UI
- [x] 首页 — 查询输入 + 6 阶段进度动画 + SSE 流式
- [x] 结果面板 — 置信度条 + 5 个 Tab（证据/Claims/冲突图/不确定性/修复）
- [x] 历史页面 — 查询列表 + 置信度条
- [x] 详情页面 — Claim Ledger + 冲突边 + 不确定性分解 + 修复 diff
- [x] 设置弹窗 — 6 种 LLM Provider 配置
- [x] 亮色/暗色主题切换 — CSS 变量双主题
- [x] 文件上传 — PDF/TXT/MD 拖拽上传
- [x] 结果导出 — Markdown 格式
- [x] SSE 重试(3次) + 断网 banner + BM25 线程锁
- [x] WebSocket 模式 — 优先 WS 降级 SSE
- [x] 移动端响应式适配

### 10. 实验脚本
- [x] `run_verabench.py` — VeraBench 评测（demo/full 模式），自动索引语料，输出 ECE/Brier
- [x] `run_ablation.py` — 7 组消融实验
- [x] `run_baselines.py` — 3 种基线对比
- [x] `build_index.py` — 索引构建 CLI

### 11. 测试覆盖（182 tests）
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
- [x] `test_web_db.py` — SQLite 操作 + API Key 加密测试
- [x] `test_agents.py` — **含冲突检测三层架构 9 个新测试**
- [x] `test_retrievers.py` — BM25/Reranker
- [x] `test_evidence_modules.py` — EvidenceExtractor/EvidenceScorer
- [x] `test_uncertainty_modules.py` — Estimator/Calibrator/Controller
- [x] `test_integration.py` — Mock LLM 端到端 pipeline（5 个测试全部通过）
- [x] `test_e2e_real_llm.py` — 真实 LLM E2E 测试（可选）
- [x] `conftest.py` — real_llm marker 注册与自动 skip

### 12. 工程化
- [x] 统一 LLMClient（6 provider: OpenAI/Anthropic/Ollama/DashScope/智谱/DeepSeek）
- [x] pyproject.toml（可 pip install -e .）
- [x] MIT LICENSE
- [x] requirements.txt
- [x] GitHub Actions CI（Python 3.10/3.11/3.12）
- [x] Dockerfile
- [x] Makefile（test/lint/format/run/coverage/docker）
- [x] API Key Fernet 加密存储
- [x] ruff 替代 flake8/black
- [x] mypy 配置并通过
- [x] conda 环境配置（environment.yml）
- [x] 覆盖率徽章（CI coverage upload）
- [x] 校准曲线生成脚本
- [x] difficulty 分级验证脚本
- [x] 外部数据集下载脚本修复（HotpotQA/FEVER fallback URL + MD5 校验）
- [x] 已推送到 GitHub: https://github.com/xiaweiyi713/VeraRAG

---

## 待开发事项

### P0 — 需要立即完成

- [ ] **跑完整 152 题 VeraBench 真实评测**
  - 代码已就绪，需要有效的 API key
  - 命令: `DEEPSEEK_API_KEY=<key> python experiments/run_verabench.py --config configs/model.yaml --output results/verabench_full.json`
  - 上次用 DeepSeek key 跑时 key 过期（401），需要新 key

### P1 — 核心功能增强

- [ ] **VeraBench 继续扩展到 300+ 题**
  - 当前 152 题，还需新增 ~150 题
  - 优先补充 hard 难度题目（当前偏少）
  - 新增领域：自动驾驶、机器人、脑机接口、核能、金融科技等
  - 每新增 50 题 需配套 10-15 篇新语料

- [ ] **冲突检测 NLI 层实测**
  - 安装 sentence-transformers 后实测 CrossEncoder NLI 效果
  - 对比：纯规则 vs 规则+NLI vs 规则+NLI+LLM 的 F1 差异
  - 调整 `nli_threshold` 最优值

- [ ] **Claim 提取质量验证**
  - 用 LLM 提取 vs 规则提取的对比实验
  - 验证 `verifiable`/`support_type` 自动分类的准确率
  - 在 verifier agent 中利用新 schema 改进验证策略

- [ ] **不确定性校准实验**
  - Temperature Scaling 优化前后 ECE 对比
  - 校准曲线可视化（confidence vs accuracy 分 bin 图）
  - 5 个不确定性维度的权重优化（当前是手动设定）

- [ ] **动态检索效果验证**
  - 对比：固定策略 vs 不确定性驱动策略的检索质量
  - 验证 RESOLVE_CONFLICTS 动作是否真的提高了冲突题的 F1
  - 调整 budget 参数（当前 30/50/80）

### P2 — 数据与评测

- [ ] **Baseline 在真实 LLM 下的分数**
  - Vanilla RAG / Hybrid RAG / Self-RAG 三种基线的真实评测
  - 与 VeraRAG full pipeline 的对比表格
  - `run_baselines.py --config configs/model.yaml`

- [ ] **消融实验（真实 LLM）**
  - 7 组消融：full/no_conflict/no_uncertainty/no_verification/no_repair/minimal/single_round
  - 量化各组件的贡献
  - `run_ablation.py --config configs/model.yaml`

- [ ] **HotpotQA/FEVER 外部数据集评测**
  - 验证 `download_datasets.sh` 是否能成功下载数据
  - 运行 `run_hotpotqa.py` / `run_fever.py` 获取跨数据集泛化能力

- [ ] **VeraBench difficulty 分级验证**
  - 检查 easy/medium/hard 标注的合理性
  - 确认 hard 题确实需要更多推理步骤

### P3 — 研究方向

- [ ] **Dense Retriever 集成实测**
  - 安装 sentence-transformers 后实测 BGE 向量检索
  - BM25-only vs Hybrid(BM25+Dense) 的 retrieval recall 对比

- [ ] **Cross-Encoder Reranker 实测**
  - bge-reranker 效果评估
  - HybridRetriever + Reranker 的级联效果

- [ ] **多轮对话支持**
  - 追问、澄清、上下文跟踪
  - Web UI 对话模式

- [ ] **多语言支持**
  - 当前中文为主，扩展英文 QA 能力
  - English VeraBench 子集

- [ ] **Agent 路由优化**
  - 根据任务类型自动选择最优 pipeline 配置
  - simple question → 跳过 decomposition/conflict graph

### P4 — 论文方向

- [ ] **LaTeX 论文撰写**
  - 基于已有素材（架构图/消融表/冲突表/概览表）
  - 核心贡献：三层冲突检测 + 不确定性驱动检索 + VeraBench 基准

- [ ] **更多 baseline 对比**
  - CRUD-RAG, RECALL, ALLE 等最新方法
  - 在 VeraBench 上的统一对比

- [ ] **Human Evaluation**
  - 人工评估答案质量（流畅性/准确性/完整性）

- [ ] **Case Study**
  - 精选典型案例分析（冲突检测、不确定性驱动检索、repair 效果）

### P5 — 工程完善

- [x] **覆盖率徽章** — README 添加 coverage badge
- [x] **Type checking** — mypy 通过
- [x] **Lint** — ruff 替代 flake8/black
- [x] **conda 环境配置** — 添加 environment.yml
- [x] **WebSocket 模式** — 替代 SSE，支持双向通信

---

## 本次开发改动摘要（2026-05-28）

合并 `worktree-verarag-enhancement` 分支到 main，整合 Phase 1-4 增强功能。

| 合并方向 | 内容 |
|---------|------|
| main → 合并后 | 三层冲突检测、Claim Schema 强化、152 题 VeraBench、不确定性驱动检索 |
| 增强 → 合并后 | ruff/mypy/conda、WebSocket、亮暗主题、移动端、PDF 上传、API Key 加密、校准曲线、difficulty 验证 |

## 历史改动摘要（2026-05-27）

| 文件 | 改动说明 |
|------|---------|
| `src/evidence/conflict_graph.py` | 三层架构: NLI层 + SUPPORT检测 + 语义矛盾 + 年份过滤 + 动态阈值 |
| `src/utils/data_structures.py` | Claim/AnswerClaim 新增 verifiable/support_type/claim_type |
| `src/evidence/extractor.py` | 自动判断 verifiable/support_type（规则+LLM） |
| `src/agents/reasoning_agent.py` | prompt 输出 claim_type/verifiable/support_type，解析更新 |
| `src/pipeline/verarag.py` | 不确定性驱动检索策略（prev_decision → budget/seek_counter） |
| `src/retriever/hybrid.py` | DenseRetriever 可选，_dense_available 标志优雅降级 |
| `src/retriever/bm25.py` | 修复重复行（语法错误） |
| `src/evaluation/answer_metrics.py` | soft_f1_score（关键词/数字重叠，适合中文） |
| `src/benchmark/evaluator.py` | BenchmarkReport 新增 ECE + Brier Score |
| `experiments/run_verabench.py` | 自动索引语料 + ECE/Brier 报告展示 + 可复现元数据 |
| `tests/test_agents.py` | +9 冲突检测三层架构测试 |
| `tests/test_integration.py` | MockLLM 关键词冲突修复 + 语料扩展至 5 篇 |
| `data/verabench/corpus.jsonl` | +15 篇新语料 (D075-D089) |
| `data/verabench/questions.jsonl` | +50 道新题 (V103-V152) |
