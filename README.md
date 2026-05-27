# VeraRAG: Verifiable Agentic Retrieval-Augmented Reasoning

面向复杂知识任务的可验证 Agentic RAG 推理系统

## 项目简介

VeraRAG 是一个能够在复杂、多跳、证据冲突、信息不完整场景下，动态检索、规划推理、检测冲突、估计不确定性、自反思修正，并输出可验证答案的 Agentic RAG 系统。

### 核心特点

- **动态检索规划**：根据问题分解结果自适应检索，而非一次性 top-k
- **证据冲突图谱**：显式建模文档间的支持、反驳、时间冲突和数值冲突关系
- **不确定性控制**：基于多维度不确定性估计决定继续检索、冲突仲裁、答案修复或拒答
- **结构化验证**：Claim 级别的证据验证，确保答案的每个断言都有据可依

## 当前状态

| 项目 | 状态 |
|------|------|
| 核心管道（10阶段） | 已完成 |
| Web UI（SSE 流式） | 已完成（演示模式 + 真实 LLM 模式） |
| VeraBench 基准测试 | 102 道标注问题 / 42 篇语料 |
| 测试 | 172 passed + 3 real LLM (optional) |
| Baseline 对比 | Demo 模式可用，真实 LLM 待跑 |
| 消融实验 | Demo 模式可用，真实 LLM 待跑 |

### What Works
- 完整的 10 阶段 Pipeline：任务分析 → 问题分解 → 动态检索 → 证据归一化 → 冲突图构建 → 不确定性评估 → 推理 → 验证 → 修复 → 输出
- 6 种 LLM 后端：OpenAI / Anthropic / Ollama / 通义千问 / 智谱 / DeepSeek
- Web UI：实时 SSE 流式推理展示、查询历史、结果详情、设置管理
- VeraBench 评测框架 + 3 种 Baseline 对比
- BM25 / Dense / FAISS / Hybrid 四种检索 + CrossEncoder 重排序

### Demo Mode
未配置 LLM 时，Web UI 自动进入演示模式：
- 模拟 6 阶段推理流程（含进度动画）
- 真实 BM25 检索（基于 VeraBench 语料）
- 模拟推理结果展示

### Real LLM Mode
配置 LLM 后（设置页面或环境变量），使用真实 Pipeline：
- 任务分析 → 问题分解 → 多轮检索 → 冲突检测 → 推理 → 验证 → 修复
- Claim-level verification + conflict-aware reasoning
- SSE 流式实时输出

### Known Limitations
- Demo 模式的 baseline 分数和消融结果是模拟的，不具备研究参考价值
- 证据 ID 在 Pipeline 运行时是动态生成的（UUID），与 VeraBench gold ID 对齐需要额外工作
- Dense Retriever 和 CrossEncoder Reranker 需要安装 sentence-transformers（当前 BM25 可独立运行）
- 冲突检测当前以规则启发式为主，NLI + LLM adjudication 待集成

## 项目结构

```
VeraRAG/
├── src/                 # 源代码（~7,400 行）
│   ├── agents/         # 6 个 Agent（分析/分解/检索/推理/验证/修复）
│   ├── retriever/      # 5 种检索器（BM25/Dense/FAISS/Hybrid/Reranker）
│   ├── evidence/       # 证据处理（提取/归一化/冲突图/评分）
│   ├── uncertainty/    # 不确定性控制（估计/校准/控制器）
│   ├── evaluation/     # 评估指标（5 模块 ~20 指标）
│   ├── ingestion/      # 文档导入（加载/分块/索引）
│   ├── benchmark/      # VeraBench 基准测试
│   └── pipeline/       # 主流程编排（SSE streaming）
├── web/                # Web UI（~1,800 行）
│   ├── api.py          # SSE + 演示 + BM25 检索端点
│   ├── app.py          # FastAPI 应用
│   ├── db.py           # SQLite 查询历史
│   └── templates/      # Jinja2 模板
├── data/verabench/     # VeraBench 基准测试集
│   ├── corpus.jsonl    # 42 篇语料 / 8 主题
│   └── questions.jsonl # 102 道标注问题 / 6 类型
├── experiments/        # 实验脚本
│   ├── run_verabench.py
│   ├── run_ablation.py
│   ├── run_baselines.py
│   └── baselines/      # Vanilla RAG / Hybrid RAG / Self-RAG
├── tests/              # 测试（~2,600 行）
├── configs/            # 配置文件
└── paper/              # 论文素材
```

## 快速开始

### 安装

```bash
# 克隆项目
git clone https://github.com/yourusername/VeraRAG.git
cd VeraRAG

# 安装依赖（使用国内镜像加速）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 启动 Web UI

```bash
python run_web.py --port 8000
```

打开 `http://localhost:8000`，即可使用交互式问答界面。

- 未配置 LLM 时自动进入**演示模式**，可预览完整推理流程
- 点击导航栏「设置」配置 LLM 提供商和 API Key

### Python API

```python
from src.pipeline.verarag import VeraRAG

# 方式 1: 直接传入 API key
pipeline = VeraRAG({
    "llm": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_key": "sk-xxx",
    }
})

# 方式 2: 通过环境变量
# export OPENAI_API_KEY="sk-xxx"
pipeline = VeraRAG({"llm": {"provider": "openai"}})

# 查询
result = pipeline.query("量子计算的主要挑战是什么？")
print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence}")
print(f"Evidence: {len(result.evidence)} items")
print(f"Conflicts: {result.metadata['num_conflicts']}")
```

### 运行测试

```bash
# 全量测试（不需要 LLM API key）
python -m pytest tests/ -q

# 真实 LLM E2E 测试（需要 API key）
OPENAI_API_KEY=sk-xxx RUN_REAL_LLM_TESTS=1 python -m pytest tests/test_e2e_real_llm.py -v
```

## 核心模块

### Pipeline 流程

```
Question → Task Analysis → Decomposition → Dynamic Retrieval →
Evidence Normalization → Conflict Graph → Uncertainty Assessment →
Reasoning → Verification → Repair → Output
```

### 冲突检测（8 种类型）

| 类型 | 说明 | 示例 |
|------|------|------|
| Numeric | 数值差异 | "错误率 15%" vs "错误率 5%" |
| Temporal | 时间线矛盾 | "2023年发布" vs "2024年发布" |
| Entity | 实体不匹配 | "谷歌" vs "IBM" |
| Source | 来源可信度冲突 | 权威机构 vs 个人博客 |
| Scope | 范围差异 | 全球数据 vs 区域数据 |
| Causal | 因果关系分歧 | 原因 A vs 原因 B |
| Definitional | 定义冲突 | 不同定义体系 |
| Granularity | 粒度差异 | 概览 vs 详细数据 |

## 评估指标

### 答案质量
- Exact Match (EM), F1 Score, Joint EM

### 证据质量
- Evidence Precision/Recall/F1, Citation Precision/Recall

### 冲突检测
- Conflict Detection F1, Type Accuracy

### 不确定性校准
- ECE, Brier Score, AUROC

### 幻觉率
- Unsupported Claim Rate, Entity/Numerical Hallucination Rate

## 配置说明

### LLM 配置

支持 6 种后端，优先级：Web UI 配置 > config yaml > 环境变量 > 默认值

```yaml
llm:
  provider: "openai"  # openai / anthropic / ollama / dashscope / zhipuai / deepseek
  model: "gpt-4o"
  api_key: "sk-xxx"   # 也可通过环境变量 OPENAI_API_KEY 设置
  temperature: 0.7
  max_tokens: 2000
```

### Pipeline 配置

```yaml
pipeline:
  max_retrieval_rounds: 5
  max_subquestions: 10
  enable_conflict_graph: true
  enable_uncertainty: true
  enable_verification: true
  enable_repair: true
```

## 贡献

欢迎贡献代码、报告问题或提出建议！

## 许可证

MIT License
