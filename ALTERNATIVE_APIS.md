# VeraRAG 本地模型配置指南

## 方案对比

| 方案 | 优点 | 缺点 | 推荐度 |
|------|------|------|--------|
| Ollama + 本地模型 | 完全免费、隐私保护 | 需要GPU、速度较慢 | ⭐⭐⭐⭐⭐ |
| 通义千问 (阿里云) | 免费额度、中文友好 | 需要手机号注册 | ⭐⭐⭐⭐ |
| 智谱 AI (GLM) | 免费额度、模型强 | 需要手机号注册 | ⭐⭐⭐⭐ |
| DeepSeek | 免费、性能强 | 需要注册 | ⭐⭐⭐⭐ |
| 纯规则模式 | 无需API | 功能受限 | ⭐⭐⭐ |

---

## 方案 1: Ollama + 本地模型 (推荐)

### 安装 Ollama

```bash
# macOS
brew install ollama

# 启动服务
ollama serve
```

### 拉取模型

```bash
# 轻量级中文模型 (推荐)
ollama pull qwen2.5:7b

# 或更小的模型
ollama pull qwen2.5:3b

# 英文模型
ollama pull llama3.1:8b
```

### 配置使用

```bash
# 设置环境变量
export OLLAMA_BASE_URL="http://localhost:11434"
```

---

## 方案 2: 通义千问 (阿里云)

### 注册获取 API Key

1. 访问 https://dashscope.aliyun.com/
2. 注册账号
3. 创建 API Key

### 免费额度

- 新用户赠送 100 万 tokens
- Qwen-Long、Qwen-Turbo 等模型

### 配置

```bash
export DASHSCOPE_API_KEY="your-api-key"
```

---

## 方案 3: 智谱 AI (GLM)

### 注册获取 API Key

1. 访问 https://open.bigmodel.cn/
2. 注册账号
3. 创建 API Key

### 免费额度

- 新用户赠送 25 元
- GLM-4-Flash 模型免费

### 配置

```bash
export ZHIPUAI_API_KEY="your-api-key"
```

---

## 方案 4: DeepSeek

### 注册获取 API Key

1. 访问 https://platform.deepseek.com/
2. 注册账号
3. 创建 API Key

### 价格

- 非常便宜: 输入 1 元/百万 tokens, 输出 2 元/百万 tokens
- 新用户有免费额度

### 配置

```bash
export DEEPSEEK_API_KEY="your-api-key"
```

---

## 立即开始：纯规则模式

如果暂时不想配置任何 API，可以使用纯规则模式运行部分功能：

```bash
python demo.py  # 演示使用规则模式，无需 API
```

---

## 快速配置示例

### 使用 Ollama (推荐)

```bash
# 1. 安装 Ollama
brew install ollama

# 2. 启动服务
ollama serve

# 新终端窗口：拉取模型
ollama pull qwen2.5:7b

# 3. 运行实验
python demo.py  # 自动检测 Ollama
```
