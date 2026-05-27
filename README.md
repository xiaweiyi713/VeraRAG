# VeraRAG: Verifiable Agentic Retrieval-Augmented Reasoning

![Coverage](https://img.shields.io/badge/coverage-87%25-green)

面向复杂知识任务的可验证 Agentic RAG 推理系统

## 项目简介

VeraRAG 是一个能够在复杂、多跳、证据冲突、信息不完整场景下，动态检索、规划推理、检测冲突、估计不确定性、自反思修正，并输出可验证答案的 Agentic RAG 系统。

### 核心特点

- **动态检索规划**：根据问题分解结果自适应检索，而非一次性 top-k
- **证据冲突图谱**：显式建模文档间的支持、反驳、时间冲突和数值冲突关系
- **不确定性控制**：基于多维度不确定性估计决定继续检索、冲突仲裁、答案修复或拒答
- **结构化验证**：Claim 级别的证据验证，确保答案的每个断言都有据可依

## 项目结构

```
VeraRAG/
├── README.md
├── configs/              # 配置文件
│   ├── model.yaml       # 模型配置
│   ├── hotpotqa.yaml    # HotpotQA 数据集配置
│   ├── fever.yaml       # FEVER 数据集配置
│   └── ckt_conflict.yaml # CKT-Conflict 数据集配置
├── data/                # 数据目录
│   ├── raw/            # 原始数据
│   ├── processed/      # 处理后数据
│   ├── indexes/        # 检索索引
│   └── ckt_conflict/   # CKT-Conflict 数据集
├── src/                 # 源代码
│   ├── retriever/      # 检索模块
│   ├── agents/         # Agent 模块
│   ├── evidence/       # 证据处理模块
│   ├── uncertainty/    # 不确定性控制模块
│   ├── evaluation/     # 评估模块
│   └── pipeline/       # 主流程
├── experiments/         # 实验脚本
├── scripts/            # 辅助脚本
├── web/                # Web UI 模块
│   ├── api.py          # API 端点（SSE 流式）
│   ├── app.py          # FastAPI 应用入口
│   ├── db.py           # SQLite 数据库
│   ├── templates/      # Jinja2 模板
│   └── static/         # JS/CSS
├── run_web.py          # Web UI 启动脚本
└── tests/              # 测试文件
```

## 安装

### 环境要求

- Python 3.8+
- CUDA (可选，用于 GPU 加速)

### 依赖安装

```bash
# 基础依赖
pip install -r requirements.txt

# 检索相关
pip install rank-bm25 sentence-transformers faiss-gpu

# LLM 相关
pip install openai anthropic

# 评估相关
pip install scikit-learn numpy
```

### 配置 API Key

设置环境变量：

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

## 快速开始

### 启动 Web UI

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python run_web.py --port 8000

# 或指定配置文件
python run_web.py --config configs/model.yaml --port 8000
```

打开浏览器访问 `http://localhost:8000`，即可使用交互式问答界面。

- 未配置 LLM 时自动进入**演示模式**，可预览完整推理流程
- 点击导航栏「设置」配置 LLM 提供商和 API Key

### Python API

```python
from src.pipeline.verarag import create_verarag

# 创建 VeraRAG 实例
verarag = create_verarag(config_path="configs/model.yaml")

# 准备文档
documents = [
    {"id": "doc1", "title": "Document 1", "text": "..."},
    {"id": "doc2", "title": "Document 2", "text": "..."},
]
verarag.index_documents(documents)

# 查询
result = verarag.query("What is the relationship between X and Y?")

# 查看结果
print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence}")
print(f"Evidence: {len(result.evidence)} items")
print(f"Conflicts: {result.metadata['num_conflicts']}")
```

### 运行实验

```bash
# HotpotQA
python experiments/run_hotpotqa.py \
    --data-path data/raw/hotpotqa/hotpot_dev_distractor_v1.json \
    --num-samples 100 \
    --output results/hotpotqa_results.json

# FEVER
python experiments/run_fever.py \
    --data-path data/raw/fever/fever_data.jsonl \
    --num-samples 100 \
    --output results/fever_results.json

# CKT-Conflict
python experiments/run_ckt_conflict.py \
    --data-path data/ckt_conflict/test.json \
    --num-samples 100 \
    --output results/ckt_conflict_results.json
```

## 核心模块

### 1. Task Analyzer

分析问题类型和复杂度，判断需要的推理能力。

```python
from src.agents.task_analyzer import TaskAnalyzer

analyzer = TaskAnalyzer()
task_analysis = analyzer.analyze("What caused the 2008 financial crisis?")
print(task_analysis.task_type)  # CAUSAL_REASONING
print(task_analysis.complexity)  # HIGH
```

### 2. Decomposition Planner

将复杂问题分解为可独立回答的子问题。

```python
from src.agents.planner import DecompositionPlanner

planner = DecompositionPlanner()
subquestions = planner.decompose(question, task_analysis)
```

### 3. Dynamic Retrieval Agent

多轮自适应检索，根据子问题动态调整检索策略。

```python
from src.agents.retrieval_agent import DynamicRetrievalAgent

retrieval_agent = DynamicRetrievalAgent(retriever=retriever)
evidence_pool = retrieval_agent.dynamic_retrieve(subquestions, evidence_pool)
```

### 4. Conflict Graph Builder

构建证据冲突图谱，检测支持、反驳、数值冲突等关系。

```python
from src.evidence.conflict_graph import ConflictGraphBuilder

builder = ConflictGraphBuilder()
conflict_graph = builder.build_graph(evidence_list)
conflicts = conflict_graph.get_conflicts()
```

### 5. Uncertainty Controller

估计不确定性并决定下一步行动。

```python
from src.uncertainty.controller import UncertaintyController

controller = UncertaintyController()
decision = controller.assess(subquestions, evidence_pool, conflict_graph)
# decision.action: CONTINUE_RETRIEVAL, RESOLVE_CONFLICTS, PROCEED, etc.
```

## 评估指标

### 答案质量
- Exact Match (EM)
- F1 Score
- Joint EM (answer + supporting facts)

### 证据质量
- Evidence Precision/Recall/F1
- Supporting Fact F1
- Citation Precision/Recall

### 冲突检测
- Conflict Detection F1
- Conflict Type Accuracy
- False Conflict Rate

### 不确定性校准
- Expected Calibration Error (ECE)
- Brier Score
- AUROC for Abstention

### 幻觉率
- Unsupported Claim Rate
- Entity Hallucination Rate
- Numerical Hallucination Rate

## 配置说明

### 模型配置 (`configs/model.yaml`)

```yaml
llm:
  provider: "openai"  # openai, anthropic, local
  model: "gpt-4o"
  temperature: 0.7
  max_tokens: 2000

retriever:
  sparse:
    enabled: true
    backend: "rank_bm25"
  dense:
    enabled: true
    model: "BAAI/bge-base-en-v1.5"
  hybrid:
    sparse_weight: 0.3
    dense_weight: 0.7
```

## 实验结果预期

基于项目企划，在以下数据集上的预期结果：

| 数据集 | 指标 | Vanilla RAG | VeraRAG |
|--------|------|-------------|---------|
| HotpotQA | Answer F1 | 61.2 | 74.8 |
| 2WikiMultiHopQA | Answer F1 | 58.4 | 71.9 |
| MuSiQue | Answer F1 | 43.1 | 59.6 |
| FEVER | Accuracy | 68.5 | 80.3 |
| CKT-Conflict | Primary Score | 41.3 | 68.9 |

## 贡献

欢迎贡献代码、报告问题或提出建议！

## 引用

如果本项目对您的研究有帮助，请考虑引用：

```bibtex
@misc{verarag2025,
  title={VeraRAG: Verifiable Agentic Retrieval-Augmented Reasoning for Complex Knowledge Tasks},
  author={VeraRAG Team},
  year={2025},
  url={https://github.com/yourusername/VeraRAG}
}
```

## 许可证

MIT License
