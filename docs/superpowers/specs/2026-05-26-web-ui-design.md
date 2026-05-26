# VeraRAG Web UI/API 设计文档

## 概述

为 VeraRAG 添加 Web 界面和 REST API 服务，使其成为可部署的交互式问答系统。采用 FastAPI + Jinja2 SSR + SSE 流式输出方案。

## 架构

```
用户浏览器
    │
    ├── /                  → 首页（问答界面）
    ├── /query (SSE)       → 提交问题，流式接收推理过程
    ├── /history           → 历史查询列表
    ├── /history/{id}      → 查看某次查询的详细结果
    └── /api/status        → 系统状态（LLM 是否可用等）
        │
    FastAPI (Python)
        │
        ├── Jinja2 模板渲染 HTML
        ├── SSE 端点推送推理中间状态
        └── 调用 VeraRAG Pipeline
            │
            ├── VeraRAG.query_stream()  → 流式推理流水线
            └── SQLite (本地存储查询历史)
```

### 关键决策

- **SSE 而非 WebSocket**：单向推送足够，SSE 更简单、HTTP 原生支持
- **SQLite 存历史**：零配置、单文件、对研究项目足够
- **Jinja2 SSR**：一体化部署，uvicorn 启动即用
- **Tailwind CSS via CDN**：不需要前端构建工具

## API 端点

### SSE 流式查询 `POST /query`

请求体：
```json
{
  "question": "RAG 能否完全解决法律领域的幻觉问题？",
  "max_rounds": 5,
  "config_overrides": {}
}
```

SSE 事件类型：

| event | data | 说明 |
|-------|------|------|
| `stage` | `{"stage": "task_analysis", "status": "started"}` | 阶段开始/完成 |
| `task_analysis` | `{"task_type": "...", "complexity": "...", "keywords": [...]}` | 任务分析结果 |
| `decomposition` | `{"subquestions": [...]}` | 子问题分解结果 |
| `evidence` | `{"round": 1, "new_evidence": [...], "total": 5}` | 证据检索进度 |
| `conflict` | `{"conflicts": 2, "conflict_score": 0.35}` | 冲突检测结果 |
| `uncertainty` | `{"overall": 0.42, "action": "continue_retrieval"}` | 不确定性评估 |
| `reasoning` | `{"answer": "...", "claims": [...], "confidence": 0.78}` | 推理结果 |
| `verification` | `{"overall_status": "supported", "issues": []}` | 验证结果 |
| `complete` | `{"result_id": "abc123", "confidence": 0.82, "elapsed_time": 12.3}` | 完成 |

### 其他端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 首页问答界面 |
| `/history` | GET | 查询历史列表 |
| `/history/{id}` | GET | 单次查询详情 |
| `/api/status` | GET | 系统状态 |
| `/api/config` | GET/PUT | 运行时配置 |

## 数据存储

SQLite 表 `queries`：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | TEXT (PK) | UUID |
| `question` | TEXT | 用户问题 |
| `answer` | TEXT | 系统答案 |
| `confidence` | REAL | 置信度 |
| `result_json` | TEXT | 完整 VeraRAGOutput JSON |
| `created_at` | DATETIME | 创建时间 |

## 页面设计

### 首页 `/`

- 顶部导航栏：Logo + LLM 状态指示 + 历史记录链接
- 问题输入框 + 提交按钮
- 推理过程面板（SSE 实时更新）：
  - 各阶段状态：✓ 已完成 / ⟳ 进行中 / ○ 待处理
  - 证据列表实时追加
  - 不确定性进度条
- 结果面板：
  - 答案文本
  - 置信度进度条
  - Tab 切换：证据 / 冲突 / 不确定性分解

### 历史记录页 `/history`

- 列表展示：问题、答案摘要、置信度、时间
- 点击进入详情

### 详情页 `/history/{id}`

- 完整展示：答案、声明、证据列表、冲突图、不确定性分解、推理链、验证报告

## 文件结构

```
VeraRAG/
├── web/                      # Web 模块（新增）
│   ├── __init__.py
│   ├── app.py                # FastAPI 应用入口
│   ├── api.py                # API 端点
│   ├── db.py                 # SQLite 操作
│   ├── templates/
│   │   ├── base.html         # 基础布局
│   │   ├── index.html        # 首页
│   │   ├── history.html      # 历史列表
│   │   └── detail.html       # 查询详情
│   └── static/
│       └── app.js            # SSE 客户端
├── run_web.py                # 启动脚本
└── requirements.txt          # 新增依赖
```

## 依赖变更

```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
jinja2>=3.1.0
aiosqlite>=0.19.0
sse-starlette>=1.8.0
```

## 实现要点

### 1. 流式流水线

在 VeraRAG 核心流水线中添加回调钩子，每个阶段完成后调用回调函数，用于 SSE 推送：

```python
class VeraRAG:
    def query_stream(self, question, max_rounds=None, callback=None):
        # Stage 1
        if callback:
            callback("stage", {"stage": "task_analysis", "status": "started"})
        task_analysis = self.task_analyzer.analyze(question)
        if callback:
            callback("task_analysis", task_analysis.to_dict())
        # ... 依次推送各阶段 ...
```

### 2. SSE 端点

使用 sse-starlette 的 EventSourceResponse：

```python
@router.post("/query")
async def stream_query(request: QueryRequest):
    async def event_generator():
        def callback(event_type, data):
            queue.put_nowait((event_type, data))
        # 在后台线程运行流水线
        await run_in_executor(pipeline.query_stream, question, max_rounds, callback)
        # 从队列读取并 yield SSE 事件
    return EventSourceResponse(event_generator())
```

### 3. 前端 SSE 客户端

使用 EventSource 或 fetch + ReadableStream 消费 SSE 事件，根据事件类型更新 DOM。

### 4. 启动脚本

```bash
python run_web.py --host 0.0.0.0 --port 8000 --config configs/model.yaml
```
