# VeraRAG 未完成功能补全设计

## 概述

补全 VeraRAG 项目中 4 个已有空壳/简陋实现的功能，采用增量补全策略，不改变现有架构。

## 1. 实体冲突检测 `_check_entity_conflict()`

**文件**: `src/evidence/conflict_graph.py`

**目标**: 检测两段证据中关于同一实体的矛盾陈述。

**实现方式**: 规则 + 轻量 NLP，无需 LLM API。

具体规则：
- **人名/地名/机构名提取**: 用正则提取大写开头连续词组，比较同一位置的实体是否不同
- **数字实体**: 提取数值+单位，比较数值差异是否超过阈值
- **否定冲突**: 检测 "X is Y" vs "X is not Y" 模式

**返回**: `ConflictEdge` 或 `None`，与现有 `_check_numerical_conflict()` / `_check_temporal_conflict()` 接口一致。

## 2. 计划精炼 `refine_plan()`

**文件**: `src/agents/planner.py`

**目标**: 根据不确定性反馈调整子问题列表和检索策略。

**输入**: `subquestions`, `uncertainty_breakdown`, `conflict_graph`, `evidence_pool`

**逻辑**:
- 检索不确定性高 -> 为覆盖度低的子问题提升优先级，生成补充查询
- 冲突不确定性高 -> 标记冲突涉及的子问题，添加"解决冲突"类型新子问题
- 来源不确定性高 -> 添加要求高可信度来源的子问题
- 过滤覆盖度 > 0.8 的已充分回答子问题
- 限制最大子问题数量（复用 `max_subquestions` 配置）

**返回**: 精炼后的 `List[SubQuestion]`

## 3. 子问题精炼 + 查询变体增强

**文件**: `src/agents/retrieval_agent.py`

### 3a. `_refine_subquestion()`

**目标**: 覆盖度不足时精炼子问题。

**逻辑**:
- 分析未命中的关键词
- 用同义词替换关键实体/动作词生成精炼版本
- 保留原子问题的上下文（原问题、关联子问题）

### 3b. `_generate_query_variants()` 增强

**目标**: 为每个子问题生成多个查询变体以提高检索召回率。

**变体类型**:
1. 原问题 + 关键词组合（现有逻辑保留）
2. 去掉停用词的精简版
3. 同义词替换版（基于小型同义词词典）
4. 实体聚焦版（提取核心实体作为查询）

每个子问题生成 3-5 个变体。

## 4. 语义去重 `deduplicate()`

**文件**: `src/evidence/normalizer.py`

**目标**: 过滤语义高度相似的证据，避免冗余。

**实现方式**: 复用 `DenseRetriever` 的 `SentenceTransformer` 模型。

**逻辑**:
- 用 sentence-transformers 编码所有证据文本
- 计算余弦相似度
- 默认阈值 0.92（可通过 config 的 `similarity_threshold` 调整）
- 贪心策略：按综合分从高到低排序，依次保留，与已保留集合相似度超阈值则丢弃
- 无 sentence-transformers 时退化为精确匹配去重

**额外修复**: `current_year = 2025` 硬编码改为 `datetime.now().year`

## 测试计划

- 实体冲突检测: 构造已知冲突/不冲突的实体对，验证检测准确性
- 计划精炼: 构造不同不确定性场景，验证子问题调整正确性
- 查询变体: 验证变体生成数量和多样性
- 语义去重: 构造语义相似但文本不同的证据对，验证去重效果

## 约束

- 不引入新的外部依赖（实体冲突检测纯规则，语义去重复用已有 sentence-transformers）
- 保持与现有接口一致
- 所有功能有 fallback 机制（缺少模型/数据时降级为规则方式）
