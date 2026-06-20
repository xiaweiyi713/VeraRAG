# VeraRAG 长期研究与开发路线图

> 本文是 VeraRAG 的长期开发计划，按"先变现已有基建、再补齐核心短板、最后扩展与发表"的逻辑组织。
> 每个阶段都给出：**目标 / 为什么 / 具体任务 / 完成标准(DoD) / 依赖**。
> 维护约定：每完成一项勾选 `[x]`，并在 `CHANGELOG.md` 记录；阶段完成后更新"北极星指标"表。

---

## 0. 研究主张（North Star）

VeraRAG 的差异化不是"答得更准"，而是**"可验证的诚实行为"**：
1. **证据可溯源**（每条断言都能定位到原文 span）；
2. **该拒答时拒答**（证据不足不硬答）；
3. **冲突显式标注 + 误导前提纠正**；
4. **不确定性可校准**（置信度能反映真实正确率，支持选择性预测）。

所有研究方向最终都服务于一个目标：**用一个可信、可复现、带置信区间的证据，证明"可验证 Agentic RAG"在这些维度上确实优于传统 RAG。**

### 北极星指标（current → target）

| 指标 | 当前（v1.0/v1.1 旧数据） | 近期目标 | 远期目标 |
|------|------|------|------|
| Behavior Accuracy（总体） | 0.763 | ≥ 0.80 | ≥ 0.88 |
| Conflict Detection F1 | ~0.007 | ≥ 0.30 | ≥ 0.50 |
| ECE（校准，越低越好） | 0.416 | ≤ 0.10 | ≤ 0.05 |
| Evidence Recall / Precision | 0.80 / 0.16 | 0.80 / ≥ 0.30 | ≥ 0.85 / ≥ 0.40 |
| 多证据 / 时序 Behavior | 0.60 / 0.76 | ≥ 0.80 / ≥ 0.85 | ≥ 0.88 |
| 权威结果版本 | 无（仅历史 v1.0/v1.1） | **v1.1.2 带 CI** | 多模型 × 多数据集 |

> ⚠️ 现状提醒：评测/发布/统计基建已达发表级，但**模型成果（conflict、calibration）与权威结果表尚未兑现**。本路线图的前三个阶段就是"把已造好的工具变现"。

---

## 阶段 0 —— 兑现严谨度：产出 v1.1.2 权威结果（最高优先，1–2 周）

**目标**：用已建好的严谨管线，在 VeraBench **v1.1.2** 上跑出一版**带 bootstrap 置信区间、带 benchmark 指纹与运行签名**的权威结果表，取代 README/RESULTS 中过时的 v1.0/v1.1 数字。

**为什么**：这是当前投入产出比最高的一步——基建全已就位，只差"按下运行键"。没有这版权威结果，之前所有统计/溯源/指纹工作都无法"被看见"。

**具体任务**：
- [x] 固定"canonical run"配置（模型、温度、检索、轮数、随机种子），写入 `configs/` 并在 RESULTS 中标注。
- [ ] 用 `run_verabench` 在 v1.1.2 全量跑，产出带 stratified bootstrap CI + 共享证据簇敏感区间的报告。
- [ ] 用 `validate_results` / `compare_verabench_reports` 校验并归档（带指纹、run signature）。
- [ ] 更新 README「实测结果」与 `docs/RESULTS.md` 为 v1.1.2 数字 + CI，加版本兼容性声明。
- [ ] 至少补 1 个对照模型（如 Qwen 或 GPT-4o-mini）做交叉验证，证明结论不绑定单一模型。

**DoD**：仓库里存在唯一一版指纹化、带 CI 的 v1.1.2 旗舰结果；README 数字与之一致；可一键复现。

**依赖**：无（可立即开始）。

---

## 阶段 1 —— 攻克冲突检测（最大短板，Conflict-F1 ≈ 0，3–5 周）

**目标**：把 Conflict Detection F1 从 ~0.01 提升到有意义的水平（≥ 0.30），并用外部集验证泛化。

**为什么**：这是 VeraBench 上最刺眼的短板，也是"可验证 RAG"卖点之一。训练管线已建好（`train_conflict_cross_encoder` 等），就差训练 + 接入 + 验证。

**具体任务**：
- [x] **先诊断再训练**：用 `compare_conflict_detectors` + `audit_conflict_model` 搞清当前 F1≈0 是"漏报"还是"误报+对齐错"（当前 v1.1.2 gold-evidence rules-only 已非 F1≈0；self-pair 漏报与 V017 额外 pair 已修到 15/0/0 TP/FP/FN）。
- [ ] **训练 cross-encoder**：在 GPU 上跑 `train_conflict_cross_encoder`（Windows 脚本已备），用 `build_conflict_training_data` 的 pairwise 数据。
- [ ] **接入三层检测器**：把训练好的模型作为 NLI 层替换/补充，端到端重测。
- [ ] **消融**：规则 / +NLI(原) / +训练模型 / +LLM 各层对 F1 的贡献（`run_conflict_ablation`）。
- [ ] **降误报**：冲突图过度检测导致 precision 低；引入共享实体/语义阈值、配对裁剪。
- [ ] **外部验证**：在 `conflict_mini_v1` 上验证，证明不是过拟合 VeraBench。

**DoD**：Conflict-F1 ≥ 0.30，外部集上不显著下降；消融表清晰显示训练层的边际贡献；冲突题 Behavior Acc 同步提升。

**依赖**：阶段 0（需要权威 baseline 作对比基准）；GPU 资源。

---

## 阶段 2 —— 不确定性与校准（ECE 0.416，2–4 周）

**目标**：让置信度**真正可用**（当前各类型均匀 ~0.16，无区分力），ECE 降到 ≤ 0.10，并交付"选择性预测"能力。

**为什么**：校准是"可验证"主张的量化兑现。当前置信度既不校准也无区分力，是个真实的系统缺陷。选择性预测（risk-coverage）是这个项目天然的研究亮点。

**具体任务**：
- [x] **诊断置信度无区分力的根因**（estimator 输出几乎常数）：检查 5 维不确定性的权重与归一化。
- [x] **重建置信信号**：让它与"证据支持度 / 验证通过率 / 冲突程度"真实相关。
- [x] **后验校准**：在 held-out split 上做 temperature scaling / Platt scaling。
- [x] **分行为校准**：abstain / answer / conflict_note 分别校准（它们的"正确"含义不同）。
- [x] **选择性预测**：画 risk–coverage 曲线，证明"愿意拒答 → 覆盖率换准确率"可控；报告 AURC、coverage@accuracy。
- [ ] 把校准结果纳入权威报告（CI 化）。

**DoD**：ECE ≤ 0.10；置信度对正确/错误有区分力（AUROC 明显 > 0.5）；一条 risk-coverage 曲线进入 RESULTS。

**依赖**：阶段 0；与阶段 1 可并行。

---

## 阶段 3 —— 检索质量（Precision 0.16、多证据/时序召回 ~0.66，2–4 周）

**目标**：在不牺牲召回的前提下提升证据精确率（≥ 0.30），并回补多证据/时序类的"部分证据→作答"能力。

**为什么**：检索是 RAG 地基。当前 top-k 带入过多非 gold chunk（precision 0.16），且多跳/时序题召回偏低、易过度拒答。

**具体任务**：
- [ ] **检索消融**：BM25 / Dense / Hybrid / +Reranker 在 VeraBench 上的 P/R 对比，确定最优组合（已新增离线 `evaluate_retrieval --matrix`，支持 `bm25_rerank` / `dense_rerank` / `hybrid_rerank` 本地缓存模型消融，以及 `--matrix-dense-models` 中文/多语 checkpoint 轴；当前 downloaded-model top-3 + `complexity_adaptive` 前沿为 `bm25_rerank`，macro P/R/F1 = 0.4456/0.9320/0.5893；多语 Hybrid top-3 adaptive = 0.4376/0.9150/0.5785；多语 Dense top-3 adaptive = 0.4240/0.8912/0.5612；英文 BGE 仍明显落后；仍需端到端行为验证）。
- [ ] **提精确率**：reranker 阈值、证据去冗、动态 top-k（按问题复杂度调 k；离线 `bm25_rerank` top-3 + `complexity_adaptive` 已让 precision 到 0.4456、recall 0.9320；pipeline 已支持可选 `bm25_rerank` + `retriever.retrieval_top_k` + `retriever.top_k_policy`，并新增 `configs/verabench_v112_retrieval_rerank_top3.yaml` 端到端候选配置；canonical 默认仍 fixed/depth-10，仍需端到端验证）。
- [ ] **提多跳/时序召回**：子问题分解驱动的迭代检索、反证检索、时序感知检索（按日期/版本）。
- [ ] **修过度拒答**：进一步细化"部分覆盖→作答 vs 无关→拒答"边界（已在 reasoning prompt 起步，需数据验证）。
- [x] 评估 Citation Precision/Recall、Supporting-Fact 指标（已进入
  VeraBench question/report 字段，支持 `[E1]` 与 pipeline chunk citation
  映射；仍需在端到端 Stage-3 A/B 中产出正式数值）。

**DoD**：Evidence Precision ≥ 0.30 且 Recall 不降；多证据/时序 Behavior 回升至 ≥ 0.80。

**依赖**：阶段 0；与阶段 1/2 可并行。

---

## 阶段 4 —— 基线、消融与多模型（科学论证，3–4 周）

**目标**：用统计显著性证明"VeraRAG > 传统 RAG"，并量化每个组件的贡献。

**为什么**：没有基线对比和消融，"系统更好"只是断言。McNemar/配对 CI 工具已就位，正好用上。

**具体任务**：
- [ ] **真实 LLM 基线**：Vanilla RAG / Hybrid RAG / Self-RAG，外加 1 个现代强基线（CRAG / RQ-RAG / 朴素 long-context）。
- [ ] **全组件消融**：依次关闭 冲突图 / 不确定性 / 验证 / 修复，量化各自贡献（`run_ablation`）。
- [ ] **多模型鲁棒性**：在 2–3 个 LLM（DeepSeek / Qwen / GPT-4o-mini）上复现主结论。
- [ ] **统计对比**：`compare_verabench_reports` 出 delta CI、改进概率、McNemar p 值。
- [ ] 形成"VeraRAG vs baselines"主表 + 消融表。

**DoD**：主表显示 VeraRAG 在 Behavior/Conflict 上显著优于基线（p<0.05）；消融表证明每个组件正贡献或给出"何时该开"的结论。

**依赖**：阶段 0–3（需要稳定的系统与权威 baseline）。

---

## 阶段 5 —— Benchmark 扩展与外部泛化（4–8 周）

**目标**：把 VeraBench 扩到 300+ 题、补领域/难度/语言，并在外部数据集上验证泛化。

**为什么**：152 题、单语言、单构造来源是发表时最易被质疑的点。泛化是"benchmark 不是过拟合"的关键证据。

**具体任务**：
- [ ] **扩 VeraBench v1.2**：→ 300+ 题，补 hard 难度、新领域（自动驾驶/脑机接口/金融科技等），每 50 题配 10–15 篇语料；沿用迁移+校验+指纹流程。
- [ ] **英文子集**：构造 English VeraBench 小集，验证方法不绑定中文。
- [ ] **外部数据集**：HotpotQA（多跳）、FEVER（验证/反驳）、一个冲突/歧义集（如 CKT/AmbigQA）——证明 VeraRAG 行为能力可迁移。
- [ ] **人工评估**：抽样做答案质量 + 行为正确性的人评，报告标注者间一致性（IAA）。
- [ ] **污染审计**：跨模型训练截止日做 contamination check（工具已备）。

**DoD**：VeraBench v1.2（300+，带卡片更新）+ 至少 2 个外部数据集结果 + 一份人评报告（含 IAA）。

**依赖**：阶段 4（稳定系统后再扩，避免反复重跑）。

---

## 阶段 6 —— 系统/Agent 进阶（研究新意 + 工程，持续）

**目标**：在准确率不降的前提下降低成本/延迟，并引入自适应能力。

**为什么**：当前推理模型 ~60–80s/题、每题 10+ 次 LLM 调用，成本是规模化与复现的障碍；自适应路由是潜在的研究贡献点。

**具体任务**：
- [ ] **自适应流水线路由**：按任务类型跳过阶段（简单事实题跳过分解/冲突图），出"成本-准确率"前沿。
- [ ] **降本**：减少冲突裁决的 O(n²) LLM 调用（前置过滤 / 训练模型替代 LLM 层）、prompt 缓存、批处理。
- [ ] **多轮对话/澄清**：追问、上下文跟踪、不确定时主动澄清。
- [ ] **更强 claim 验证**：claim 分解 + NLI 微调，提升验证精度。
- [ ] **蒸馏**：用大模型的行为标注微调小模型，降低推理成本。

**DoD**：自适应路由在同等准确率下显著降低平均 LLM 调用数/延迟；多轮 demo 可用。

**依赖**：阶段 1–4（先有稳定、可度量的系统）。

---

## 阶段 7 —— 论文与正式发布（收口）

**目标**：把整套工作收口为一篇可投稿论文 + 一个正式 release + 公开 leaderboard。

**核心贡献（论文叙事）**：
1. **VeraBench**——一个强调"可验证行为"（拒答/冲突/纠正前提/溯源）的中文 benchmark，带 provenance 指纹、bootstrap CI、污染审计、数据集卡片；
2. **可验证 Agentic RAG 流水线**——动态检索 + 三层冲突检测 + 不确定性驱动 + Claim 级验证/修复；
3. **行为对齐发现**——量化"传统 RAG 的倾向作答偏差"及其修复（0.53→0.76+），及各组件消融。

**具体任务**：
- [ ] 基于 `paper/` 素材写 LaTeX 论文（架构图/消融表/冲突表/校准曲线/risk-coverage）。
- [ ] 目标会议：**NeurIPS Datasets & Benchmarks**（benchmark 主打）/ ACL/EMNLP（方法）/ SIGIR（检索）。
- [ ] v1.0 release：PyPI 发布（release.yml 已备）、复现 artifact、tagged release、公开 leaderboard。
- [ ] 撰写 reproducibility checklist（数据/代码/seed/指纹全链路）。

**DoD**：投稿就绪的论文 + PyPI v1.0 + 公开可提交的 leaderboard。

**依赖**：阶段 0–5（核心成果齐备）。

---

## 排期与依赖总览

```
阶段0(权威结果) ──┬─> 阶段1(冲突)  ─┐
                  ├─> 阶段2(校准)  ─┼─> 阶段4(基线/消融/多模型) ─> 阶段5(扩展/泛化) ─> 阶段7(论文/发布)
                  └─> 阶段3(检索)  ─┘                                   ↑
                                          阶段6(系统进阶/降本) ──────────┘(持续，喂养4/5)
```
- **先做阶段 0**（1–2 周，解锁一切对比）。
- 阶段 1/2/3 **可并行**（不同子系统），是核心攻坚（约 1–2 个月）。
- 阶段 4 依赖 1–3 稳定后启动。
- 阶段 5/6 在系统稳定后扩展。
- 阶段 7 收口。

---

## 跨阶段原则与风险

**原则**：
- **基建已足够，重心转向出成果**：除非阻塞研究，否则暂缓新增 validator/workflow（边际价值已低）。
- **每个结论都要带 CI 和指纹**：沿用已建的统计/provenance 纪律。
- **改动即提交**：避免再出现上百文件长期未提交的风险。
- **先小样本验证，再全量重跑**：每次改 prompt/模型先在分层小样本上验证方向，再花 ~2.5h 跑全量。

**风险登记**：
| 风险 | 影响 | 缓解 |
|------|------|------|
| 冲突训练需 GPU，资源不稳 | 阻塞阶段 1 | Windows GPU 脚本已备；可先用 dry-run + 小模型验证链路 |
| 全量评测慢（~2.5h/次）+ API 成本 | 迭代慢 | 断点续跑已备；小样本先行；多模型分批 |
| 单一构造来源/单语言 → 泛化质疑 | 影响发表 | 阶段 5 外部数据集 + 英文子集 + 人评 |
| 基建持续膨胀挤占研究 | 投入产出失衡 | 本路线图明确"转向出成果"；冻结非必要基建 |
| 置信度重建可能牵动行为 | 回退风险 | held-out 校准、分行为评估、回归测试守门 |

---

## 立即可做的三件事（本周）

1. **阶段 0**：固定 canonical 配置，跑 v1.1.2 全量，产出带 CI 的权威结果，更新 README/RESULTS。
2. **阶段 1 诊断**：先用 `compare_conflict_detectors` 把 Conflict-F1≈0 的根因定位清楚（漏报 vs 误报）。
3. **阶段 2 诊断**：定位置信度无区分力的根因（estimator 输出近常数）。

> 这三件都是"诊断/变现"，不需要新基建，且直接喂养后续所有阶段。
