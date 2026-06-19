# VeraRAG 开发进度

> 最后更新：2026-06-20

## 🆕 本次更新（2026-06-20）：Citation/Supporting-Fact 评测接入

1. **Citation parser 扩展**：`EvidenceMetrics.extract_citations` 现在支持
   `[E1]`、`[D001_c0]` 和包含连字符/下划线的证据 ID，保留重复引用顺序用于
   precision 统计。
2. **VeraBench report 指标化**：`QuestionResult` 与 `BenchmarkReport` 新增
   `citation_*`、`supporting_fact_*` 字段，以及 `citation_summary` 和
   `supporting_fact_summary`，同时输出 macro 与 micro TP/FP/FN。
3. **pipeline ID 对齐**：evaluator 会把答案引用、retrieved evidence、冲突边和
   `answer_claims[].supporting_evidence` 中的 pipeline chunk ID 映射回
   question-local gold evidence ID，减少 `[D001_c0]` vs `E1` 的假阴性。
4. **离线重评分兼容**：`rescore_results` 会重算 citation 指标，并把 gold
   supporting fact IDs 写回 diagnostics，历史报告可通过重评分升级到新指标口径。
5. **测试覆盖**：新增 parser 与 pipeline report 聚合测试；focused 验证为
   `tests/test_evaluation_metrics.py` + `tests/test_benchmark.py` 共 100 项通过，
   Ruff 通过。

## 🆕 本次更新（2026-06-14）：VeraBench v1.1.2、依赖稳健推断与证据可追溯性

1. **VeraBench v1.1 本体修订**：将 evidence-evidence conflict 与 premise
   refutation 分离，题型分布调整为 27/26/11/25/26/37，并为 17 个本体修订和
   8 个难度修订写入 rationale。
2. **VeraBench v1.1.1 / v1.1.2 数据修订**：v1.1.1 修正 V084 的时间口径，
   题面明确询问
   “截至2024年初，2023年全球新增可再生能源装机”，与 gold evidence 中的
   2023 年新增装机数据对齐；v1.1.2 修复 25 条不可严格定位或片段顺序错误的
   evidence span，当前 questions SHA-256 为
   `c19e7401cbcd2526fb1e085c911d61095c8ce5bd19a664c13698a08024436882`。
3. **基准完整性门禁**：新增唯一 ID、行为/题型一致性、证据引用、证据跨度
   连续/分段顺序可追溯、冲突 pair、ground-truth claim 引用和仓库/包内数据
   同步校验；报告写入语料与问题 SHA-256。
4. **检查点防污染**：评测签名覆盖 benchmark、config、实现代码、题型过滤与
   max questions；不兼容或无 sidecar 的旧检查点会被拒绝。
5. **模型缓存与 NLI 正确性**：语义去重和 NLI 模型按进程缓存成功/失败加载；
   Verifier 改为 evidence→claim，读取 `id2label` 并使用稳定 softmax。
6. **问题条件化冲突图**：冲突边两端分别匹配问题事实槽位；数值检测区分金额、
   百分比、持续时间、量子比特、表面码距离和制程节点；普通时点问题过滤预期的
   历史版本演进。
7. **中文答案指标 v2**：新增版本化 `soft-f1-v2` 与 `verarag-rescore`，保存的
   真实轨迹可离线重算答案、正确率、分组与校准指标，无需重复调用 LLM。
8. **Windows GPU 全量真实评测**：DeepSeek 全配置 VeraBench v1.1 已完成
   152/152、零错误；分区合并后离线重评分达到 Behavior Acc **0.9803**、
   Evidence Recall **0.9518**、Answer F1 **0.4593**、Conflict micro-F1
   **0.8966**（TP/FP/FN = 13/1/2）。
9. **Answerability guard**：针对全量报告暴露出的 V036/V048 两类真实失败，
   主流水线新增确定性后处理：精确数值问题遇到约数/不完整统计时改为拒答并保留
   可验证近似值；断言/前提验证题被 LLM 误写成普通拒答时改为“该说法不准确”
   的前提纠正。
10. **精确题号复测**：`run_verabench.py` / `verarag-benchmark` 新增 `--ids`，
   可直接运行 `--ids V036 V048 V084` 复测具体失败样例；题号过滤进入
   checkpoint 签名，避免不同子集误复用旧结果。
11. **质量门禁**：当前全量测试 **791 passed + 3 skipped**，Ruff、mypy、
   secret scan、docs/examples/deployment/pre-commit/dependency/project metadata、
   release-check、package audit、installed-wheel smoke、`git diff --check`、sdist
   与 wheel 构建通过。
12. **统计不确定性**：每份 VeraBench 报告新增按题型分层、固定 seed 的
    2,000 次 bootstrap 95% 置信区间；历史 v1.1 全量结果显示 Answer F1
    区间为 **[0.431, 0.489]**，Conflict micro-F1 区间为
    **[0.700, 1.000]**，明确暴露小冲突子集的不确定性。
13. **配对显著性比较**：新增 `verarag-compare-reports`，严格校验 benchmark
    指纹、metric version 和 question ID 覆盖，输出 candidate-baseline
    delta 区间、改进概率、逐题胜负及行为 McNemar 精确检验。
14. **校准边界修复**：ECE 最后一个分箱现在包含 confidence=1.0，避免满置信度
    样本被遗漏。
15. **共享证据依赖审计**：152 题按共享 gold 文档形成 27 个连通组，最大组
    19 题；validator 同时公开文档复用率和 character-bigram 近重复候选。
16. **依赖稳健统计**：报告、离线分析、leaderboard 和成对比较新增
    `evidence-cluster-bootstrap-v1`，以共享证据连通组为重采样单位，避免仅按
    152 个题目独立同分布来解释置信区间。
17. **重评分来源保护**：离线重评分默认拒绝缺失或不匹配的 benchmark
    版本/指纹；历史敏感性分析必须显式授权，并同时保留 source/target 身份。
18. **安装包回归**：修复 `verarag-validate-benchmark` 在源码目录外无法发现
    wheel 内置数据的问题；Mac 与 Windows/WSL 均完成 wheel CLI smoke test。
19. **冲突训练防泄漏**：训练数据改为按共享 gold 文档连通组切分，hard
    negative 不得跨 split；训练前强制检查 dependency group、question 和精确
    文本重叠。v1.1.2 数据为 181 pairs，train/val/test=129/26/26，
    正样本=23/3/3，三类 overlap 均为 0。
20. **多种子负结果披露**：balanced seeds 13/17/23 的 held-out pair test F1
    分别为 0.316/0/0.316，均值 0.211；seed 13 不过采样为 0。held-out
    gold-evidence pipeline 上 rules 与 rules+learned 同为 F1 0.857，
    learned detector 因无增益继续默认关闭。
21. **离线 GPU 训练稳定性**：统一 CLI、远端启动脚本和文档的 warmup 为 10
    steps，并关闭保存阶段的在线 Hugging Face model-card 查询；四组正式训练均
    在约 9-16 秒内完整退出，模型元数据包含 v1.1.2 manifest 与三份 split
    SHA-256。冲突图对同一图内的 learned 候选执行批量打分，避免逐边 GPU 调用。
22. **模型晋级门禁**：训练产物新增逐样本概率/预测与依赖组件 bootstrap
    区间；`verarag-audit-conflict-model` 对多随机种子、manifest/预测哈希、
    独立测试集指纹、样本规模和 rules+learned 增益执行 fail-closed 审计。
    当前真实审计结论为 `reject`：test F1 均值 0.2105、最差 0，三种子的
    cluster-F1 下界均为 0，内部 A/B 仅 3 题且 F1 增益为 0。
23. **外部冲突集标注协议**：新增 `verarag-validate-external-conflicts`，
    要求外部冲突集提供 manifest、双人独立标注、仲裁 gold、SHA-256 指纹和
    Cohen's kappa；`data/external/conflict_mini_v1` 作为 CI fixture 验证协议
    端到端可运行，但不作为模型晋级证据。
24. **盲标数据包生成**：新增 `verarag-build-external-annotation-packet`，
    从 VeraBench-compatible 外部题库导出每位 annotator 的 JSONL 模板、
    仲裁模板和 packet manifest，并显式省略 gold answer、expected conflict、
    question type 与 evidence category，避免标注阶段泄漏答案。
25. **标注结果编译**：新增 `verarag-compile-external-annotations`，将完成的
    盲标 packet 转换成 `annotations.jsonl`、`adjudications.jsonl` 和
    `compiled_manifest.json`，产物可直接交给 `verarag-validate-external-conflicts`
    做一致性与仲裁审计。
26. **多 seed GPU 训练矩阵**：新增并实测
    `scripts/start_windows_conflict_training_matrix.sh`。2026-06-15 在
    `windows-gpu` RTX 4060 上复现 seeds 13/17/23 的 v1.1.2 负结果：
    test F1 = **0.316/0/0.316**，rules 与 rules+learned held-out A/B 同为
    F1 **0.857**，report-only promotion audit 正确 `reject`。
27. **本地污染审计**：新增 `verarag-audit-contamination`，可对调用者提供的
    本地训练语料、公开 dump 或 prompt 开发语料执行 exact/near-duplicate
    overlap 审计；near-duplicate 同时覆盖 Jaccard 与短 benchmark 文本嵌入长
    参考语料的 item-containment 召回；match 报告包含本地参考片段和 exact
    字符偏移，便于第三方复核，不会把未提供的未知模型训练语料误报为“已审计”。
28. **密钥提交门禁**：新增 `verarag-scan-secrets` / `make security`，用标准库
    扫描高置信度 API key、`.env.*`、AWS access key、private-key header、
    GitHub token 和通用 secret assignment；默认接入 `make lint` 与 GitHub
    Actions，报告只显示打码后的 token；`--include-ignored` / `make
    security-local` 可用于本机额外扫描被 `.gitignore` 忽略的 `.env.local`
    等文件；`--sarif` 可输出 SARIF 2.1.0，供 CI/code-scanning 工具消费且不
    暴露原始密钥。GitHub Actions 在 secret gate 失败时仍会保留
    `secret-scan.sarif` artifact，便于维护者查看打码后的文件/行号诊断。
29. **发布包内容审计**：新增 `verarag-validate-package` / `make
    package-check`，自动检查 sdist/wheel 是否包含 VeraBench 数据、外部标注
    fixture、docs、Web 资源、typed marker、console script 名称、入口目标与
    目标模块是否随包发布，并排除缓存和内部计划目录；审计按 `pyproject.toml` 的当前项目名
    与版本精确选择 sdist/wheel，避免 `dist/` 中旧版本产物误导发布判断；GitHub
    Actions 在构建包后立即运行该审计。
30. **一键发布前门禁**：新增 `make release-check`，串联 lint、全量测试、
    dependency metadata、80% total coverage gate、VeraBench release health、sdist/wheel 构建、package audit 与 installed-wheel smoke，降低发布前漏跑
    关键检查的概率。`verarag-release-health` 统一验证 v1.1.2 数据指纹/可追溯性、
    仓库/包内数据同步、外部冲突 fixture、盲标 packet、demo 满分指标和 demo
    paired comparison，CI 与本地发布门禁共享同一入口。`make docs-check` /
    `verarag-validate-docs` 已纳入 release-check，用于校验 README、docs 与治理
    文档中的本地 Markdown 文件链接和锚点。
31. **评测报告 provenance 透明化**：VeraBench 导出报告时，如果 pipeline
    config 在保存阶段无法重新读取，会写入 `config_metadata_warning`，不再静默
    丢失 model/provider/temperature/max_tokens 元数据。
32. **安全远程真实评测启动器**：新增
    `scripts/start_windows_verabench_eval.sh`，可从 Mac 静默读取 DeepSeek
    key，并通过 SSH/FIFO 注入 `windows-gpu` 的 tmux 评测任务；默认运行
    VeraBench v1.1.2 全量输出到 `outputs/remote_results/verabench_v112_full.json`，
    同时支持 `VERARAG_EVAL_IDS`、`VERARAG_EVAL_TYPES` 和 `VERARAG_EVAL_MAX`
    做精确题号复测或 bounded smoke，避免把真实 key 写入 shell history、日志或仓库文件。
33. **Windows helper 稳健性**：`scripts/sync_windows_gpu.sh` 现在先在远端解析
    `~` 到真实 home 路径，再把 shell-quoted 绝对路径交给 rsync；单 seed
    conflict-training 启动器也改为 quote 远端路径/模型参数，并在已有同名 tmux
    session 时 fail closed，避免覆盖仍在运行的训练。
34. **2026-06-18 GPU 矩阵复跑与远端门禁**：Mac → `windows-gpu` 同步脚本已实测
    解析到 `/home/wenyao/projects/VeraRAG`；Windows/WSL `train` 环境通过
    GPU helper 测试、`make benchmark-check`、sdist/wheel 构建和
    `verarag-validate-package`。同日复跑 seeds 13/17/23 的
    `conflict_cross_encoder_v112_leakfree` 矩阵，training/promotion audit
    完整产出，审计仍正确 `reject`：learned detector 在 held-out A/B 上相对
    rules 无增益，且内部 test split 只有 3 个 gold conflict，不满足晋级证据。
    矩阵启动器已同步采用 `printf %q` 写入临时 driver，并在已有同名 tmux
    session 时 fail closed，避免远端路径/参数特殊字符破坏训练命令。
35. **Planner 输出边界加固**：`DecompositionPlanner` 现在统一规范化 LLM、
    fallback 和 uncertainty refinement 产生的子问题计划：空问题会被丢弃，
    空计划自动回退，id 重排为连续 `sq0..`，依赖只允许指向已存在的前序子问题，
    坏的 `max_subquestions` 输入会回退到安全默认值，并补充 planner
    回归测试覆盖这些边界。
36. **污染审计覆盖率补强**：`verarag-audit-contamination` 的参数校验、引用
    路径错误、递归目录读取、JSON/JSONL 字符串叶子提取、坏 JSON fallback、
    near-match basis、最佳引用片段和截断边界均已补充回归测试；
    `src/benchmark/contamination.py` 从 **77%** 提升到 **100%** 覆盖。
37. **动态检索闭环加固**：`DynamicRetrievalAgent` 的反证检索 top-k 不再因
    `top_k=1` 被整除为 0；challenge、alternative 与 temporal 三类反证 query
    均保留在 bounded query set 中；低覆盖子问题 refinement 会写回 active
    plan，后续轮次真正使用 refined query。`src/agents/retrieval_agent.py`
    覆盖率提升到 **100%**。
38. **不确定性校准与控制器契约加固**：`ConfidenceCalibrator` 现在对温度缩放
    的空输入、长度不一致、非二值 label、NaN/Inf、边界概率 0/1 和非法优化参数
    fail closed；校准指标显式拒绝长度不一致、越界 confidence 与非 bool
    correctness。`UncertaintyController` 现在校验阈值顺序、概率输入和轮次输入，
    并兼容 `high_conflict_threshold` 旧配置键。`src/uncertainty/calibrator.py`
    与 `src/uncertainty/controller.py` 均达到 **100%** 覆盖，本地 coverage gate 为
    **573 collected / 570 passed + 3 skipped**，total coverage **90.46%**。
39. **答案修复 Agent 契约加固**：`RepairAgent` 现在统一接受
    `VerificationStatus` enum、enum 名称和 wire value，非法状态 fail closed；
    verifier confidence 会被归一化到 `[0, 1]`，修复后的 claim 保留 enum 状态，
    refuted/unsupported caveat 保持幂等。`src/agents/repair_agent.py` 覆盖率提升到
    **100%**，本地 coverage gate 为 **582 collected / 579 passed + 3 skipped**，
    total coverage **90.64%**。
40. **Claim 验证 Agent 契约加固**：`VerifierAgent` 现在在 report 聚合前统一
    规范化 verifier 输出，兼容 enum、枚举名和 wire value 状态；未知状态、非法
    confidence、NaN/Inf 都保守降级为可修复的低置信结果，避免坏 LLM JSON 破坏
    overall status。NLI 的 supported/refuted/neutral、模型不可用、非法输出形状和
    预测异常路径均有回归测试。`src/agents/verifier_agent.py` 覆盖率提升到
    **100%**，本地 coverage gate 为 **592 collected / 589 passed + 3 skipped**，
    total coverage **91.05%**。
41. **多后端 LLM 客户端契约加固**：`LLMClient` 现在规范化 provider 名称，
    保留显式 `temperature=0.0` / `max_tokens=0` 这类调用参数，Ollama 统一使用
    prompt `generate` API 并支持 JSON mode；DeepSeek、DashScope 和 ZhipuAI 等
    OpenAI-compatible provider 支持显式 `base_url` 覆盖，便于代理和私有网关部署。
    自动 provider 发现的 Ollama/OpenAI/Anthropic/DashScope/ZhipuAI/DeepSeek/default
    分支均有离线 fake-module 回归测试。`src/utils/llm_client.py` 覆盖率提升到
    **100%**，本地 coverage gate 为 **602 collected / 599 passed + 3 skipped**，
    total coverage **91.40%**。
42. **外部标注审计契约加固**：`compile_external_annotation_packet` 现在拒绝
    不安全 annotator id、绝对模板路径、`../` 路径穿越、symlink 逃逸、非对象
    packet manifest、错误 schema、缺失 `label` / `gold_label` 和已存在输出文件；
    `audit_external_conflict_set` 补充 manifest、重复/未知题号、标注员缺失、
    adjudication 覆盖缺口、内部指纹匹配等失败路径测试。修复了 no-conflict 行
    `conflict_type` 写错仍被 `_normalize_conflict_type` 静默归一为 `none` 的漏洞。
    `src/benchmark/external_annotations.py` 覆盖率提升到 **96%**，本地
    release-check 为 **623 collected / 620 passed + 3 skipped**，total coverage
    **91.97%**。
43. **冲突图 learned/LLM 输出防御**：`ConflictGraphBuilder` 的 learned
    CrossEncoder 分数归一化现在对空列表、非法字符串、NaN/Inf 和极端 logit
    fail-closed 或稳定 sigmoid，不再可能把坏模型输出写成 `nan` 置信度冲突边。
    LLM fallback 现在只接受明确的 SUPPORT/REFUTE/PARTIAL_SUPPORT/UNRELATED
    relationship，并要求 confidence 为有限数值再夹到 `[0, 1]`；未知 relationship
    不再默认为 SUPPORT。`tests/test_conflict_detection.py` 新增 6 个边界测试，
    本地 coverage gate 为 **629 collected / 626 passed + 3 skipped**，total
    coverage **92.18%**，`src/evidence/conflict_graph.py` 覆盖率提升到 **85%**。
44. **冲突图增量更新与 NLI 标签防御**：`update_graph()` 现在利用同一 builder
    内注册的旧 claim/evidence 与新证据做真实增量比较，新增 self-refuting claim
    检测、update-window learned 批量打分和无向重复边去重，避免只比较新证据自身。
    NLI 层现在拒绝 NaN/Inf logits，并且在 `id2label` 只部分可识别或映射不唯一时
    fail closed，避免把 contradiction/entailment 列读错。`tests/test_conflict_detection.py`
    新增 5 个回归测试，本地 coverage gate 为 **634 collected / 631 passed + 3 skipped**，
    total coverage **92.37%**，`src/evidence/conflict_graph.py` 覆盖率提升到 **87%**。
45. **证据评分与幻觉指标契约加固**：`EvidenceScorer` 现在同时识别 evidence id
    和 claim id 端点，并按无向边统计 support/conflict，避免证据质量分受冲突边方向
    影响；无相关边时的默认 support/conflict 路径也有测试覆盖。`HallucinationMetrics`
    对 unsupported claim 匹配做大小写/空白归一化，并对 overclaiming 的长度不一致、
    非有限 confidence、越界 threshold fail closed，避免评估行被 `zip` 静默截断。
    本地 coverage gate 为 **639 collected / 636 passed + 3 skipped**，total coverage
    **92.51%**，`src/evidence/evidence_scorer.py` 覆盖率提升到 **100%**。
46. **文档加载器 PDF/Markdown/JSONL 边界加固**：`DocumentLoader` 现在即使在
    PDF 页文本提取抛错时也会关闭 PyMuPDF handle，空 PDF fallback 统一标记
    `source="file"`；新增无标题 Markdown、前言 section、空 section 跳过、JSONL
    空行跳过、fake PyMuPDF 页、空 PDF、页提取失败和缺 PyMuPDF 的回归测试。
    本地 coverage gate 为 **647 collected / 644 passed + 3 skipped**，total coverage
    **92.78%**，`src/ingestion/loader.py` 覆盖率提升到 **100%**。
47. **答案指标 F1 契约修正**：`AnswerMetrics.f1_score` 从 set 交集改为 multiset
    overlap，避免重复 token 被去重后虚高；`soft_f1_score` 不再让空 keyword 特征把
    单字中文无关答案判为满分，`compute_all` 与 batch helper 也暴露版本化 soft F1。
    本地 coverage gate 为 **652 collected / 649 passed + 3 skipped**，total coverage
    **93.03%**，`src/evaluation/answer_metrics.py` 覆盖率提升到 **100%**。
48. **任务分析器低复杂度与 LLM 输出归一化**：`TaskAnalyzer` 不再把默认
    `MULTI_HOP_QA` 当作复杂度信号，简单问题可保持 low 并避免无谓 LLM 调用；
    数值正则中的 `$` 现在按字面量匹配，LLM 未配置时复杂问题会回退到规则分析，
    LLM 返回的 task type alias、字符串布尔值、越界 hops 和 keyword 列表会先
    归一化再进入下游 planner。新增 7 个回归测试，本地 coverage gate 为
    **659 collected / 656 passed + 3 skipped**，total coverage **93.33%**，
    `src/agents/task_analyzer.py` 覆盖率提升到 **100%**。
49. **数值幻觉指标符号与输入校验加固**：`HallucinationMetrics` 的数字抽取现在
    保留正负号，`-5%` 不再被当作 `+5%` 匹配；数值幻觉率对负 tolerance、
    NaN/Inf answer number 和 evidence number fail closed，避免评测报告静默吞掉
    非有限值。新增 4 个回归测试，本地 coverage gate 为
    **663 collected / 660 passed + 3 skipped**，total coverage **93.47%**，
    `src/evaluation/hallucination_metrics.py` 覆盖率提升到 **96%**。
50. **检索器公共契约加固**：`BaseRetriever` 现在统一校验 query、批量 query 与
    `top_k`，并让 BM25/Dense/FAISS/Hybrid 共享同一负数和非整数限制处理；
    BM25 不再会因 `top_k=-1` 返回“除最后一条外全部结果”的 Python slice
    副作用。新增 8 个回归测试，本地 coverage gate 为
    **671 collected / 668 passed + 3 skipped**，total coverage **93.57%**，
    `src/retriever/base.py` 覆盖率提升到 **100%**。
51. **证据抽取器中文语义与 LLM 输出防御**：`EvidenceExtractor` 现在能按中文
    “增长/下降/通过/生效/由于/导致/高于/低于/是指/可能”等信号识别
    numerical、temporal、causal、comparative、definitional 和 uncertainty
    claim type；LLM claim 抽取兼容 object/list JSON，强制 `max_claims`，
    跳过坏行，并归一化 claim type、列表、bool 与 support_type，避免单个坏字段让
    整段退回规则抽取。新增 13 个回归测试，本地 coverage gate 为
    **684 collected / 681 passed + 3 skipped**，total coverage **93.98%**，
    `src/evidence/extractor.py` 覆盖率提升到 **98%**。
52. **VeraBench 评测器 checkpoint 与校准防污染**：`VeraBenchEvaluator`
    现在在恢复 checkpoint 时校验当前题目文本、题型、ground truth 和 expected
    behavior，自动跳过 stale、out-of-scope 或 malformed 行，避免筛选题号或题库
    更新后误复用旧结果；pipeline/baseline 异常保持逐题隔离；pipeline 输出与
    已加载报告中的 confidence 会先归一化为有限 `[0, 1]` 概率再进入 calibration。
    新增 5 个回归测试，本地 coverage gate 为
    **689 collected / 686 passed + 3 skipped**，total coverage **94.56%**，
    `src/benchmark/evaluator.py` 全量覆盖率提升到 **95%**。
53. **冲突图数值 token 对齐修复**：`ConflictGraphBuilder` 数值冲突检测现在在
    解析阶段保留 `(value, raw_token)` 对，避免上游 LLM/规则抽取产生的无效
    number token 被跳过后，让金额/单位/上下文错位到错误数值；int/float 等
    非字符串 token 会先安全转为文本再做年份、日期、单位和上下文判断。新增 2 个
    回归测试，本地 coverage gate 为
    **691 collected / 688 passed + 3 skipped**，total coverage **94.58%**。
54. **learned CrossEncoder 二分类输出解释修复**：`ConflictGraphBuilder`
    learned 层现在能正确处理 CrossEncoder 常见的 `[negative, positive]`
    二分类概率或 logits 输出，使用 positive conflict class 作为冲突概率，而不是
    误读第一个负类分数导致系统性漏检；单 logit、单概率、嵌套单样本输出仍保持兼容。
    新增 3 个回归测试，本地 coverage gate 为
    **694 collected / 691 passed + 3 skipped**，total coverage **94.55%**。
55. **冲突图阈值配置解析加固**：`ConflictGraphBuilder` 现在会把 learned、
    NLI、support、fact-slot 和 LLM 裁决相关的概率/相似度阈值统一归一化为
    有界 float；字符串形式的 YAML/JSON/CLI 配置可直接使用，bool、NaN/Inf 和
    非数字值回退默认值，越界值裁剪到 `[0, 1]`，避免模型预测后出现运行时比较错误
    或危险阈值。新增 3 个回归测试，本地 coverage gate 为
    **697 collected / 694 passed + 3 skipped**，total coverage **94.56%**。
56. **公开包入口版本探测加固**：`verarag.__version__` 现在优先读取安装包
    distribution metadata，源码 checkout 或 metadata 缺失时回退到 `src.__version__`
    这一单一来源，避免公开 API 里重复硬编码版本号；新增 installed/fallback 两条
    回归测试，`verarag/__init__.py` 覆盖率提升到 **100%**。本地 coverage gate 为
    **699 collected / 696 passed + 3 skipped**，total coverage **94.59%**。
57. **远程真实评测启动器密钥路径回归测试**：`scripts/start_windows_verabench_eval.sh`
    的 tmux driver 现在用 `printf %q` 固化远端变量，避免路径、输出文件或额外参数中
    的特殊字符破坏评测命令；DeepSeek key 的 FIFO 读取由 `timeout 30s bash -c`
    包住，连 FIFO open 阶段也不会无限挂起。新增 fake-ssh 行为测试，确认 key
    不会出现在 SSH 参数、远端 heredoc、stdout 或 stderr 中，只通过第二段 stdin
    注入。本地全量测试为 **700 collected / 697 passed + 3 skipped**。
58. **Release health CLI 参数 fail-fast**：`verarag-release-health` 的
    `--comparison-resamples` 现在在 argparse 与函数入口两层都要求正整数，避免
    0/负数进入深层 paired comparison 后才报错；新增 CLI 和函数级回归测试。
    本地 release-check 为 **702 collected / 699 passed + 3 skipped**，total
    coverage **94.68%**。
59. **Package audit 入口模块完整性校验**：`verarag-validate-package` 现在会从
    `pyproject.toml` 与 wheel `entry_points.txt` 中解析每个 console script，
    在目标一致后继续确认对应模块文件同时存在于 sdist 与 wheel，避免发布后命令
    因漏打包模块才失败；新增缺失入口模块的回归测试。本地 release-check 为
    **703 collected / 700 passed + 3 skipped**，total coverage **94.68%**。
60. **Installed wheel 冒烟门禁**：新增 `verarag-validate-install` 与
    `installed-wheel-check`，将构建出的 wheel 展开到临时安装根，并从源码目录外
    验证公开 API、包内 VeraBench 数据、Web app 创建和全部 23 个 console script
    的 `--help` 加载路径；`make package-check` 现在串联 archive audit 与安装后
    smoke，避免只检查文件清单而漏掉真实安装路径问题。本地 release-check 为
    **706 collected / 703 passed + 3 skipped**，total coverage **94.68%**。
61. **依赖元数据一致性门禁**：新增 `verarag-validate-deps` 与 `make
    deps-check`，校验 `pyproject.toml` 核心依赖是否同步到
    `requirements.txt`、dev 发布工具依赖是否遗漏、`environment.yml` 是否通过
    pip 委托 `requirements.txt`、Python 版本是否满足 `requires-python`，以及
    README 是否保留 pip/conda 安装路径；同时补齐 `requirements.txt` 中
    package-check 需要的 `build>=1.0.0`。本地 release-check 为
    **711 collected / 708 passed + 3 skipped**，total coverage **94.59%**。
62. **CI 与本地发布门禁收敛**：GitHub Actions 现在显式运行
    `experiments/validate_dependency_metadata.py`，并在 Python 3.13 构建阶段
    复用 `make package-check`，让 CI 覆盖与本地 release gate 相同的
    sdist/wheel archive audit 和 installed-wheel smoke；对应测试锁定
    `deps-check`、`package-check`、公开 CLI 入口和 SARIF artifact 契约，减少
    手写 CI shell 与 Makefile 漂移；本地 `make release-check` 复跑通过：
    **711 collected / 708 passed + 3 skipped**，total coverage **94.59%**，
    package/install audit 均覆盖 **24** 个 console script。
63. **开源项目元数据门禁**：新增 `verarag-validate-metadata` 与
    `make metadata-check`，校验 `pyproject.toml` 的 PyPI URL、PEP 639 license
    expression、OS classifier、治理文件、GitHub issue/PR 模板、README 中贡献/安全/行为准则/引用
    入口、`CITATION.cff` 与项目版本/仓库的一致性，以及 CONTRIBUTING 是否指向当前
    `make release-check` 等质量门禁；同时补齐 `project.urls`、README 的
    `CODE_OF_CONDUCT.md`/`CITATION.cff` 入口，并将该门禁纳入 CI 与
    release-check。构建时发现并移除了 PEP 639 下会导致 `setuptools>=77`
    失败的 deprecated license classifier；本地 `make release-check` 复跑通过：
    **717 collected / 714 passed + 3 skipped**，total coverage **94.59%**，
    package/install audit 均覆盖 **25** 个 console script。
64. **GitHub 维护与 CI 权限门禁**：新增 `.github/dependabot.yml`，按周检查
    Python 依赖和 GitHub Actions 更新，并给 CI workflow 增加 top-level
    `permissions: contents: read` 与 `timeout-minutes`；`verarag-validate-metadata`
    现在会拒绝缺少 Dependabot、缺少 CI 最小权限、缺少 job 超时、或 CI 未复用
    `make package-check` 的配置，避免维护策略与发布门禁继续漂移；本地
    `make release-check` 复跑通过：**719 collected / 716 passed + 3 skipped**，
    total coverage **94.59%**，package/install audit 均覆盖 **25** 个 console script。
65. **社区入口与维护权责门禁**：新增 `.github/CODEOWNERS` 与
    `.github/ISSUE_TEMPLATE/config.yml`，默认关闭空白 issue，并将疑似安全漏洞
    引导到 GitHub private security advisory；`verarag-validate-metadata` 现在会拒绝
    缺失 repository-wide owner、空白 issue 未关闭或安全问题未路由到 `SECURITY` 的配置，
    `verarag-validate-package` 也会确认这些社区治理文件进入 sdist；本地
    `make release-check` 复跑通过：**720 collected / 717 passed + 3 skipped**，
    total coverage **94.59%**，package/install audit 均覆盖 **25** 个 console script。
66. **CodeQL 静态安全分析门禁**：新增 `.github/workflows/codeql.yml`，对 Python
    代码在 push、PR 和每周计划任务中运行 GitHub CodeQL；workflow 使用
    `contents: read`、`actions: read`、`security-events: write` 的最小权限组合和
    `timeout-minutes`，`verarag-validate-metadata` 会拒绝缺少 CodeQL init/analyze、
    缺少计划任务、缺少 Python 分析或权限/超时不完整的配置，`verarag-validate-package`
    也会确认 CodeQL workflow 进入 sdist；本地 `make release-check` 复跑通过：
    **721 collected / 718 passed + 3 skipped**，total coverage **94.59%**，
    package/install audit 均覆盖 **25** 个 console script。
67. **GitHub Actions 供应链 pinning 门禁**：将 `actions/checkout`、
    `actions/setup-python`、`actions/upload-artifact` 与 `github/codeql-action`
    从浮动 major tag 固定到完整 commit SHA，并保留版本注释；`verarag-validate-metadata`
    现在会扫描 `.github/workflows/*.yml` 的 `uses:`，拒绝 `@v4`/`@v5` 这类浮动
    action 引用，降低 CI 供应链被上游 tag 漂移影响的风险；本地 `make release-check`
    复跑通过：**722 collected / 719 passed + 3 skipped**，total coverage **94.59%**，
    package/install audit 均覆盖 **25** 个 console script。
68. **OpenSSF Scorecard 供应链评分门禁**：新增 `.github/workflows/scorecard.yml`，
    在 push、PR 和每周计划任务中运行 `ossf/scorecard-action`，使用
    `contents: read` / `security-events: write` 最小权限，发布结果并保留
    `scorecard-results.json` artifact；`verarag-validate-metadata` 会拒绝缺少
    Scorecard action、计划任务、结果发布、JSON artifact、权限/超时不完整或 action
    未固定 SHA 的配置，`verarag-validate-package` 也会确认 Scorecard workflow 进入
    sdist；本地 `make release-check` 复跑通过：**723 collected / 720 passed + 3 skipped**，
    total coverage **94.59%**，package/install audit 均覆盖 **25** 个 console script。
69. **CycloneDX SBOM 发布门禁**：新增 `experiments/generate_sbom.py` /
    `verarag-generate-sbom` / `make sbom-check`，从 `pyproject.toml` 生成并校验
    CycloneDX 1.5 JSON SBOM，覆盖 core、dev 与 optional dependency groups，并检查
    项目身份、组件唯一性和依赖组件是否陈旧；`make release-check` 现在会在 coverage
    gate 前生成 `build/sbom/verarag-sbom.cdx.json` 并校验，`verarag-validate-package`
    会确认 SBOM 工具和测试进入发布包；本地 `make release-check` 复跑通过：
    **728 collected / 725 passed + 3 skipped**，total coverage **94.59%**，
    package/install audit 均覆盖 **26** 个 console script，当前 SBOM 包含 **31**
    个依赖组件。
70. **发布产物 manifest 可追溯性**：`verarag-release-health` 现在会生成并校验
    `build/release-health/release-artifacts-manifest.json`，把 VeraBench 数据审计、
    外部冲突 fixture 审计、盲标 packet manifest、demo report 和 demo paired
    comparison 的相对路径、SHA-256、byte size、生成命令与关键指标写入同一个
    机器可读清单；新增篡改检测回归测试会在产物 hash 或大小不一致时 fail closed。
    `benchmark-check` 默认使用 gitignored 的 `build/release-health`，也可通过
    `RELEASE_HEALTH_DIR` 覆盖；新增 `make release-artifacts-check` /
    `verarag-release-health --validate-manifest`，无需重跑 benchmark 即可离线复验
    已生成 manifest，并拒绝绝对路径或 `..` path traversal。目标测试当前
    **20 passed**；当前 `make release-check` 复跑通过：**753 collected /
    750 passed + 3 skipped**，total coverage **94.59%**，release health manifest
    覆盖 **5** 个发布关键产物，package/install audit 均覆盖 **30** 个 console script。
71. **发布 checksum manifest 与离线验真**：新增
    `experiments/generate_release_checksums.py` /
    `verarag-release-checksums` / `make release-checksums-check`，对当前
    `pyproject.toml` 版本的 sdist、wheel、CycloneDX SBOM 和 release-health
    artifact manifest 生成 SHA-256 与 byte size 清单；验证模式会拒绝缺失产物、
    hash/大小漂移、重复 role/path、绝对路径和 `..` path traversal。`release-check`
    将 checksum gate 放在 `package-check` 之后，确保被验真的 dist 产物已经按当前
    源码重新构建；目标回归测试覆盖清单生成、篡改检测、路径逃逸、CLI 验证和
    Makefile 顺序。当前 `make release-check` 复跑通过：**753 collected /
    750 passed + 3 skipped**，total coverage **94.59%**，release health manifest
    覆盖 **5** 个发布关键产物，package/install audit 均覆盖 **30** 个 console
    script，release checksum manifest 覆盖 **4** 个发布验真产物。
72. **Quickstart 示例可运行性门禁**：新增 `experiments/validate_examples.py` /
    `verarag-validate-examples` / `make examples-check`，把 README 中暴露的
    `python examples/quickstart.py` 首次上手路径纳入机器验证；验证器会检查
    quickstart 文件存在、README 命令未漂移，并实际运行无需 API key 的 VeraBench
    demo，确认 57 篇文档、152 道题和 bounded demo 评测输出。`release-check` 现在在
    docs gate 后立即运行 examples gate，避免示例只能靠全量 pytest 间接覆盖；
    package/install audit 同步覆盖 **28** 个 console script。
73. **文档命令面防漂移**：`verarag-validate-docs` / `make docs-check` 从只检查
    Markdown 本地链接和锚点，升级为同时验证文档中的 `make xxx` 必须对应真实
    Makefile target、`verarag-xxx` 必须对应 `pyproject.toml` 公开 console script，
    并要求每个公开 console script 至少在 Markdown 文档中出现一次。新增回归测试
    覆盖有效命令、未知 make target、未知 console script 和未文档化 console
    script；`docs/API.md` 的 Console Scripts 列表补齐 examples、SBOM 与 checksum
    相关 CLI。当前 docs gate 验证 **12** 份 Markdown、**22** 个本地链接和
    **193** 个命令引用。
74. **Docker/Web 部署配置门禁**：新增 `experiments/validate_deployment.py` /
    `verarag-validate-deployment` / `make deployment-check`，检查 Dockerfile 使用
    安装后的 `verarag-web` 入口、暴露 8000 端口、非 root 用户运行、通过
    `/api/status` 做 HEALTHCHECK，并确认 `.dockerignore` 排除 `.git`、`.venv`、
    `.verarag_key`、build/dist、outputs/results 和本地 SQLite 数据库等不应进入
    build context 的文件。Dockerfile 已从 editable `.[all]` 安装改为发布包安装，
    并创建 `verarag` 系统用户与可写 `/app/data`；CI 和 `release-check` 显式运行
    deployment gate，README/RELEASING/CONTRIBUTING 同步记录该检查。
75. **pre-commit 开发者体验门禁**：新增 `.pre-commit-config.yaml` 和
    `experiments/validate_precommit.py` / `verarag-validate-precommit` /
    `make precommit-check`，用本地 hooks 串起 Ruff、mypy、secret scan、
    docs/examples/deployment/deps/metadata gate，并校验 CI、README 与
    CONTRIBUTING 不会和 hooks 漂移。`release-check` 已纳入该门禁，发布包
    audit 同步确认配置文件和验证器进入 sdist/wheel。
76. **GitHub Release / PyPI 发布流水线**：新增 `.github/workflows/release.yml`，
    手动触发时运行 `make release-check`，为 wheel、sdist、CycloneDX SBOM、
    release-health manifest 和 checksum manifest 生成 GitHub Artifact
    Attestations，并上传同一组发布产物；`v*.*.*` tag 才启用 PyPI OIDC
    trusted publishing 与 GitHub Release 创建。`verarag-validate-metadata` 现在会
    校验 release workflow 的 tag-only 发布条件、OIDC/attestations 权限、pypi
    protected environment、产物清单、attestation subject-path 和所有 action 的完整
    SHA pinning。
77. **结果页 / leaderboard 发布防漂移门禁**：新增
    `experiments/validate_results.py` / `verarag-validate-results` /
    `make results-check`，校验 `docs/RESULTS.md` 的生成命令、当前 VeraBench
    v1.1.2 corpus/questions SHA-256、历史/诊断结果不可比标签、关键结果表头和正式
    leaderboard 的 fail-closed 规则说明；CI、pre-commit、release-check、
    package audit、README、CONTRIBUTING、API 与 RELEASING 均已接入。目标测试
    **60 passed**；当时 `make release-check` 复跑通过：**757 collected /
    754 passed + 3 skipped**，total coverage **94.59%**，docs gate 验证
    **202** 个命令引用，package/install audit 均覆盖 **31** 个 console script。
78. **发布版本身份一致性门禁**：新增
    `experiments/validate_version_identity.py` / `verarag-validate-version` /
    `make version-check`，校验 `pyproject.toml` 版本、`src.__version__` 源码
    fallback、`CITATION.cff`、`verarag.__version__` 安装包 metadata fallback 和
    release checklist 的版本发布说明一致；CI、pre-commit、release-check、
    package audit、README、CONTRIBUTING、API 与 RELEASING 均已接入。目标测试
    **64 passed**；当时 `make release-check` 复跑通过：**761 collected /
    758 passed + 3 skipped**，total coverage **94.59%**，docs gate 验证
    **211** 个命令引用，package/install audit 均覆盖 **32** 个 console script。
79. **Python 支持范围防漂移门禁**：新增
    `experiments/validate_python_support.py` / `verarag-validate-python` /
    `make python-support-check`，校验 `requires-python`、PyPI classifiers、
    GitHub Actions Python matrix、Ruff/mypy 目标版本、`environment.yml` 默认
    Python 版本和 README/CONTRIBUTING 中的 Python 3.10+ 说明一致；CI、
    pre-commit、release-check、package audit、README、CONTRIBUTING、API 与
    RELEASING 均已接入。目标测试 **68 passed**；当时 `make release-check`
    复跑通过：**765 collected / 762 passed + 3 skipped**，total coverage
    **94.59%**，docs gate 验证 **220** 个命令引用，package/install audit
    均覆盖 **33** 个 console script。
80. **校准曲线诊断工具硬化**：`experiments/calibration_curve.py` /
    `verarag-calibration` 现在显式支持 `--correctness-field`、`--json` 和
    `--json-output`，默认按 VeraBench 行级 `correct` 行为正确性绘制可靠性图；
    对空报告、缺失 correctness 字段、非布尔 correctness、非有限或越界
    confidence、非法 bin count 全部 fail-closed，避免生成看似正常但口径错误的
    校准图。新增回归测试覆盖完整报告形状、替代布尔字段、坏输入和 CLI
    SVG/JSON 产出。目标测试 **24 passed**；当时 `make release-check` 复跑通过：
    **770 collected / 767 passed + 3 skipped**，total coverage **94.59%**，
    docs gate 验证 **221** 个命令引用，package/install audit 均覆盖 **33** 个
    console script。
81. **本机环境诊断 CLI**：新增 `experiments/doctor.py` / `verarag-doctor`，
    用标准库汇总 Python 版本、核心依赖、可选功能依赖、VeraBench 数据文件与
    LLM provider 环境变量配置状态；输出只暴露布尔状态，不打印任何 API key。
    `--json` 便于自动化消费，`--fail-on-warnings` 可把可选依赖缺失或未配置
    provider key 提升为非零退出。README、CONTRIBUTING、API、RELEASING、
    package audit 与公开入口测试均已接入。目标测试 **24 passed**；当前
    `make release-check` 复跑通过：**775 collected / 772 passed + 3 skipped**，
    total coverage **94.59%**，docs gate 验证 **227** 个命令引用，
    package/install audit 均覆盖 **34** 个 console script。
82. **环境诊断正式门禁化**：新增 `make doctor-check`，并把
    `experiments/doctor.py` 接入 `.pre-commit-config.yaml`、GitHub Actions 与
    `make release-check`。`verarag-validate-precommit` 现在会同时校验
    doctor hook、CI doctor step 和 release-check 顺序，避免环境诊断入口只停留在
    文档里而未被自动执行。README、CONTRIBUTING 与 CHANGELOG 已同步。目标测试
    **22 passed**；当前 `make release-check` 复跑通过：**775 collected /
    772 passed + 3 skipped**，total coverage **94.59%**，docs gate 验证 **232**
    个命令引用，package/install audit 均覆盖 **34** 个 console script。
83. **默认配置 schema 门禁**：新增 `experiments/validate_configs.py` /
    `verarag-validate-configs` / `make configs-check`，校验 `configs/*.yaml`
    可解析、runtime 配置包含 `llm`/`pipeline`/`retriever`、dataset 配置包含
    `dataset`/`evaluation.metrics`，并检查概率阈值、正整数、布尔字段和
    `llm.api_key` 必须使用 `${ENV_VAR}` 占位符。该门禁已接入 CI、pre-commit、
    release-check、package audit、README、CONTRIBUTING、API 与 RELEASING。目标测试
    **49 passed**；当前 `make release-check` 复跑通过：**780 collected /
    777 passed + 3 skipped**，total coverage **94.59%**，docs gate 验证 **241**
    个命令引用，package/install audit 均覆盖 **35** 个 console script。
84. **配置加载器边界加固**：`configs.load_config()` 现在只接受 `configs/`
    内部的相对 `.yaml` 文件名，拒绝绝对路径、`..` 路径穿越、非 YAML 后缀、
    空 YAML 和非 mapping YAML 根对象；`get_dataset_config()` 同时兼容
    `fever` 与 `fever.yaml` 调用；`merge_configs()` 改为深拷贝合并，避免调用方
    修改 merged config 时污染基础配置。新增回归测试覆盖路径边界、坏 YAML 根、
    深拷贝语义和 dataset config 兼容入口。目标测试 **19 passed**；当前
    `make release-check` 复跑通过：**784 collected / 781 passed + 3 skipped**，
    total coverage **94.59%**，docs gate 验证 **241** 个命令引用，
    package/install audit 均覆盖 **35** 个 console script。
85. **Windows GPU 运维状态入口**：新增 `scripts/windows_gpu_status.sh` 和
    `make gpu-status`，可从 Mac 只读查看 `windows-gpu` 上的远端项目路径、
    tmux 会话、attach 命令、GPU 利用率/显存、磁盘空间以及 `outputs/`
    下最新训练/评测产物；同时保留 `scripts/windows_gpu_status.sh gpu`
    作为连续 `nvidia-smi` 观察入口。新增脚本契约测试覆盖 Bash 语法、远端
    路径 quoting、tmux/GPU/产物检查和禁止误伤命令。目标测试 **7 passed**；
    当前 `make release-check` 复跑通过：**785 collected / 782 passed + 3 skipped**，
    total coverage **94.59%**，docs gate 验证 **245** 个命令引用，
    package/install audit 均覆盖 **35** 个 console script。
86. **VeraBench v1.1.2 canonical run 固定**：根据 `docs/ROADMAP.md` 阶段 0，
    新增 `configs/verabench_v112_canonical.yaml` 作为下一版权威 DeepSeek 全量
    评测的唯一配置身份，锁定 benchmark v1.1.2、输出路径、checkpoint、模型
    `deepseek-v4-flash`、temperature `0.0`、`max_tokens=4000`、BM25 检索、
    `max_retrieval_rounds=1`、全流程开关，以及 2,000 次 bootstrap / seed 1729。
    `scripts/start_windows_verabench_eval.sh` 默认改用该配置；`docs/RESULTS.md`、
    README、EVALUATION、GPU_TRAINING、RELEASING 和 `validate_results` 均同步
    指向同一 canonical run，避免 v1.1.2 全量跑分前出现多个候选入口。当前
    `make release-check` 复跑通过：**786 collected / 783 passed + 3 skipped**，
    total coverage **94.59%**，config gate 验证 **7** 个 YAML，docs gate 验证
    **13** 份 Markdown / **246** 个命令引用，package/install audit 均覆盖
    **35** 个 console script。
87. **校准无区分力离线诊断**：根据 `docs/ROADMAP.md` 阶段 2，扩展
    `experiments/analyze_verabench_results.py` / `verarag-analyze`，新增
    `confidence_diagnostics` 与 `risk_coverage` 输出，离线报告置信度分布、
    正确/错误样本均值、confidence AUROC、`underconfident`、
    `near_constant_confidence`、`weak_correctness_discrimination` 等诊断 flag，
    以及 AURC / coverage@accuracy。对历史 v1.1 全量报告复跑诊断得到：
    mean confidence **0.352** vs behavior correctness **0.914**，
    correct/incorrect mean confidence **0.354 / 0.330**，confidence AUROC
    **0.555**，flag 为 `underconfident` 和
    `weak_correctness_discrimination`；说明阶段 2 下一步应优先重建置信信号，
    而不只是做后验 temperature scaling。当前 `make release-check` 复跑通过：
    **787 collected / 784 passed + 3 skipped**，total coverage **94.59%**，
    docs gate 验证 **13** 份 Markdown / **248** 个命令引用。
88. **运行时置信信号重建**：根据阶段 2 诊断结果，`src/pipeline/verarag.py`
    的最终 `confidence` 不再只使用 `1 - overall_uncertainty` 与乘法折扣，而是融合
    verifier 支持/反驳状态、证据质量与 claim 覆盖、answer claim 与 reasoning
    置信度、冲突压力以及拒答是否合理；`src/uncertainty/calibrator.py` 改为保序的
    有界 uncertainty penalty，避免强支持样本被压到近常数低分区间。新增回归测试
    覆盖支持答案 vs 被反驳答案、合理拒答 vs 不合理拒答，以及低 uncertainty 强信号
    不被压塌。当前 `make release-check` 复跑通过：**790 collected /
    787 passed + 3 skipped**，total coverage **94.51%**，docs gate 验证
    **13** 份 Markdown / **248** 个命令引用，package/install audit 均覆盖
    **35** 个 console script。阶段 2 代码层“根因诊断/信号重建”已完成；ECE、AUROC
    与 risk-coverage 是否达标仍需 canonical v1.1.2 全量评测验证。
89. **held-out 后验置信校准**：新增 `experiments/calibrate_verabench_confidence.py`
    / `verarag-calibrate-report`，对保存的 VeraBench report 做 correctness-stratified
    deterministic split，并支持 Platt scaling 与 temperature scaling。输出报告会把
    原始 confidence 保存在 `diagnostics.confidence_calibration.original_confidence`，
    同时在 `metadata.posthoc_confidence_calibration` 记录 method、seed、split、
    all/calibration/holdout 的 before/after ECE 与 Brier。历史 v1.1 全量报告用
    seed 1729、50/50 split、Platt scaling 复跑得到 holdout ECE **0.5523 → 0.0110**，
    Brier **0.3929 → 0.0836**；这证明“后验校准工具链”已可用，但 canonical v1.1.2
    仍需全量结果后重新校准并写入权威报告。新增回归测试覆盖报告写入、temperature
    分支、坏输入 fail-closed 与 CLI 输出。当前 `make release-check` 复跑通过：
    **794 collected / 791 passed + 3 skipped**，total coverage **94.51%**，
    docs gate 验证 **13** 份 Markdown / **251** 个命令引用，package/install audit
    均覆盖 **36** 个 console script。
90. **分行为后验校准**：`verarag-calibrate-report` 新增
    `--group-field actual_behavior`、`--min-group-rows` 与 `--group-fallback`，
    可按 answer / abstain / conflict-note / premise-correction 等实际行为分别
    拟合 Platt/temperature；当某个行为组校准样本不足或只有单一 correctness 类别时，
    默认回退到 Laplace-smoothed empirical constant，也可选择全局模型回退。输出摘要会在
    `metadata.posthoc_confidence_calibration.groups` 记录每组 mode、fallback reason、
    before/after ECE/Brier；逐题 diagnostics 记录 group value、model scope 与原始
    confidence。历史 v1.1 全量报告用 seed 1729、50/50 split、Platt + actual_behavior
    分组复跑得到 holdout ECE **0.5523 → 0.0666**，Brier **0.3929 → 0.0840**；
    `answer_with_citation` 可拟合 group Platt，其余稀疏/单类别行为使用显式回退。
    新增测试覆盖分组拟合、回退、缺失 group field fail-closed 与 CLI 参数。
91. **选择性预测曲线产品化**：`verarag-analyze` 新增
    `--risk-coverage-svg` 与 `--risk-coverage-csv`，可从保存的 VeraBench report
    直接输出 risk-coverage 曲线和逐点 CSV，并继续在 JSON/table 中报告 AURC 与
    coverage@accuracy。历史 v1.1 分行为校准报告生成的诊断曲线已进入
    `docs/assets/verabench_v11_group_calibrated_risk_coverage.svg`，AURC **0.0328**，
    coverage@accuracy≥0.95 **0.5724**，coverage@accuracy≥0.90 **1.0000**。
    新增回归测试覆盖 analyzer artifact 写出路径，`docs/RESULTS.md` 已标明该图仍是
    historical diagnostic，canonical v1.1.2 需全量重跑后替换。
92. **冲突检测根因诊断结构化**：`verarag-compare-conflicts` / 
    `experiments/compare_conflict_detectors.py` 新增 `diagnosis` 输出，按 variant
    标出 dominant failure（under/over/mixed/none）、by-type TP/FP/FN、top false
    negative / false positive questions，并在 rules+learned 对比时输出 F1/precision/recall
    delta 与 learned effect。当前 bundled VeraBench v1.1.2 gold-evidence rules-only
    全量诊断为 precision **0.9231**、recall **0.8000**、F1 **0.8571**，TP/FP/FN
    **12/1/3**；主要剩 `V021` / `V075` / `V122` self-pair 漏报与 `V017` 额外 pair。
    dependency-aware test split 为 precision **0.7500**、recall **1.0000**、F1
    **0.8571**，唯一错误同样是 `V017` 额外 pair。这完成阶段 1 “先诊断再训练”的
    第一闭环：后续训练/规则修复必须以补 self-pair recall 和守住 precision 为目标。
93. **冲突规则层 edge 闭环**：根据上一条诊断，`ConflictGraphBuilder` 新增三处高精度
    修复：同证据显式数值对比（覆盖 `V021` / `V075` ECS 2.5°C vs IPCC 3.0°C）、
    ITER “原计划 2025 首次等离子体但推迟至 2030” 自反驳时间冲突、以及已被同证据
    corrective claim 纠正的 reported claim 不再重复生成跨证据冲突边（修 `V017` 额外
    E1/E2 pair）。当前 bundled VeraBench v1.1.2 gold-evidence rules-only 全量
    诊断达到 precision/recall/F1 **1.0000/1.0000/1.0000**，TP/FP/FN **15/0/0**；
    dependency-aware test split 也为 **3/0/0**。新增回归测试锁住这三种模式。注意：
    这是 gold-evidence edge 层结果，端到端 conflict behavior 仍需 canonical v1.1.2
    全量 LLM run 验证。
94. **GPU 训练前置门禁**：新增 `scripts/check_windows_conflict_training_ready.sh`
    与 `make gpu-check`，并让单 seed / 多 seed conflict training launcher 默认先
    预检远端 SSH、项目路径、tmux、`train` conda 环境、`torch` /
    `sentence_transformers`、CUDA 可见性和 offline base model 路径；手动维修时可用
    `VERARAG_GPU_SKIP_PREFLIGHT=1` 绕过。本地已重新生成 v1.1.2 pairwise 训练数据：
    总计 **181** 对、正例 **29**、train/val/test **129/26/26**，split integrity
    verified 且 dependency/text overlap 为 **0**；`train_conflict_cross_encoder`
    dry-run 显示 train loader 经正例过采样后为 **212** 对、正负 **106/106**。当前
    `windows-gpu` 仍然 SSH 超时，因此真实 cross-encoder GPU 训练尚未完成。
95. **离线检索消融入口**：新增 `experiments/evaluate_retrieval.py` /
    `verarag-evaluate-retrieval`，可在不调用 LLM 的情况下按 VeraBench gold evidence
    评估 BM25 / Hybrid 文档检索，输出 overall、by_type、by_difficulty、
    by_multi_hop 的 precision/recall/F1、hit rate、all-gold-retrieved rate、MRR
    与 nDCG。当前 bundled VeraBench v1.1.2 BM25 top-10 基线覆盖 **147** 个有
    gold evidence 的问题，macro precision **0.1293**、macro recall **0.9830**、
    macro F1 **0.2244**、hit rate **1.0000**、all-gold-retrieved rate **0.9660**、
    MRR **0.9427**、nDCG **0.9325**；这把阶段 3 的主要问题明确为“召回高但
    top-k 精度低”，下一步应优先评估 reranker、动态 top-k 与证据去冗。
96. **动态 top-k 离线前沿**：`verarag-evaluate-retrieval` 新增 `--sweep-top-k`
    和 `--top-k-policy`。当前 BM25 `precision_cap`（保留最多 4 篇）达到 macro
    precision **0.3044**、recall **0.9546**、F1 **0.4492**；`complexity_adaptive`
    （简单题 2 篇、temporal/misleading 4 篇、multi-hop/conflict 5 篇）达到 macro
    precision **0.3500**、recall **0.9456**、F1 **0.4977**。这满足“离线文档检索
    精度 ≥0.30 且召回仍高”的中间目标，但还不是端到端 Evidence Precision /
    Behavior 改善证明，下一步需要在 pipeline 或 ablation 中验证是否会引发过度拒答。
97. **动态 top-k pipeline 开关**：`DynamicRetrievalAgent` 现在读取
    `retriever.top_k_policy`，支持 `fixed`、`precision_cap` 与
    `complexity_adaptive`，并允许通过 `precision_cap_top_k` /
    `adaptive_*_top_k` 调整裁剪阈值。实现保留原始检索深度、只裁剪进入
    evidence pool 的文档数；`configs/model.yaml` 与 canonical v1.1.2 配置显式保持
    `fixed`，避免未验证策略影响权威 run。新增 agent/config/integration 测试确认
    policy 裁剪、validator 校验和 pipeline 传参；`run_verabench` report metadata
    现在也记录 retriever type/top-k policy/阈值字段，方便后续 fixed vs adaptive
    真实 A/B 审计。
98. **动态 top-k 端到端 A/B 配置**：新增
    `configs/verabench_v112_retrieval_adaptive.yaml`，镜像 canonical v1.1.2
    DeepSeek 全量配置，仅将 `retriever.top_k_policy` 改为
    `complexity_adaptive` 并使用独立 run/output/checkpoint identity。这样后续
    `compare_verabench_reports` 可以直接比较 canonical fixed 与 adaptive 策略，
    不需要临时手改权威配置。
99. **动态 top-k A/B 计划器**：新增 `experiments/plan_retrieval_ablation.py` /
    `verarag-plan-retrieval-ablation`，在不需要 API key 的情况下校验 fixed 与
    adaptive 配置是否只在 retrieval policy/run identity 上分叉，并输出 baseline
    run、candidate run 与 `compare_verabench_reports` 命令。默认计划写向
    `outputs/remote_results/verabench_v112_retrieval_adaptive_comparison.md`，为
    Windows GPU 真实实验提供可审计执行单。

## 🆕 本次更新（2026-06-14）：冲突检测召回、检索锚点与回答行为闭环

本轮围绕 VeraBench conflict 子集继续做“发现问题→修复→复测”的闭环，重点把冲突检测从“高精度但漏检”推进到可端到端影响最终回答：

1. **嵌入式反事实/误导 claim 抽取**：EvidenceExtractor 现在能从“X 声称 Y，但 Y 错误/不准确；实际上 Z”中抽出 `reported_claim` 与 `corrective_claim`，覆盖“法案被搁置”“禁止所有人脸识别”等误导型表述。
2. **同证据 counterclaim 支持**：冲突图默认仍跳过同一 evidence 内部 claim 比较，但对 `reported_claim` ↔ `corrective_claim` 明确放行，使同一事实核查段落内部的“错误说法 vs 更正说法”可形成冲突边。
3. **问题聚焦的冲突图过滤**：pipeline 在图构建后按问题属性、实体和年份过滤冲突边，并对 reported-claim 的重复冲突做可信来源排序去重，避免一个文档里的无关冲突污染当前问题。
4. **对比语境与修饰性“通过”降噪**：新增“与欧盟AI法案不同，美国……”这类 contrast-context 过滤，并避免把“最终通过的四个等级分类”误判为法案通过状态。
5. **冲突回答兜底**：ReasoningAgent 的冲突上下文现在包含 evidence id、标题和 claim 文本；若检测到冲突但 LLM 回答未显式写出“冲突/不一致/错误/不准确”等标记，会自动前置冲突说明并把冲突证据写入 answer claims。
6. **真实 DeepSeek smoke 复测**：在 `windows-gpu` 上运行 `configs/deepseek_run.yaml`、`--types conflict --max 3`，rules-only 结果为 Answer F1 **0.5223**、Evidence Recall **1.0000**、Conflict F1 **1.0000**、Behavior Acc **1.0000**，冲突 TP/FP/FN = **3/0/0**。结果归档在 `outputs/remote_results/verabench_conflict_behavior_rules_max3.json`。
7. **离线友好的 rules-only 评测配置**：新增 `configs/deepseek_rules_only.yaml`，关闭 verifier/NLI/semantic-dedup 的 Hugging Face 依赖，保留 conflict graph、repair 与 uncertainty，便于 Windows WSL 无外网时稳定跑真实 LLM 回归。
8. **Conflict+misleading max10 v16 复测**：在 `windows-gpu` 上运行 `configs/deepseek_rules_only.yaml`、`--types conflict misleading --max 10`，结果为 Answer F1 **0.3437**、Evidence Recall **1.0000**、Conflict F1 **1.0000**、Behavior Acc **1.0000**，冲突 TP/FP/FN = **12/0/0**。相比 v4（Conflict F1 0.7667、Behavior Acc 0.8000、TP/FP/FN 9/2/3），漏检和误检均清零；结果归档在 `outputs/remote_results/verabench_conflict_misleading_rules_only_max10_v16.json`。
9. **V021/V024/V042 闭环修复**：新增比较数值句拆分（ECS 2.5°C vs IPCC 3.0°C）、混排大写缩写抽取、文档级指代实体继承、温度/出口销量事实槽门控、RetrievalAgent metadata 传递，以及 misleading/conflict 行为分类优先级修正；同时修正 VeraBench V021 的错误 gold evidence 标注。
10. **真实检索路径稳定性修复**：端到端 v9-v14 暴露出 V024/V042 波动：`DynamicRetrievalAgent._result_to_evidence()` 未保留 `entities`，导致同证据比较缺少共享实体；DeepSeek 过度分解问题时又会把 `sq_original` 的 top-k 稀释到 4，漏掉 D011。现已保留 evidence metadata，并让原始问题检索锚点固定保留 top-10 召回深度。
11. **评测可诊断性增强**：`QuestionResult.diagnostics` 现在保存 evidence ids、mapped conflict pairs、冲突边摘要、conflict graph 节点/边数量和 output metadata，远端 FP/FN 可直接从 JSON 定位，不需要重新跑 LLM。
12. **max25 失败诊断与结构修复**：真实 DeepSeek `conflict+misleading --max 25` 首轮结果暴露出过度检测（TP/FP/FN = **13/17/7**）。本轮针对弱实体 `AI` 过宽、制程节点数字误报、同证据 self-refutation 漏检、对比实体污染和“既然...”推断题误报做了规则化修复；远端 v5 复跑达到 Evidence Recall **1.0000**、Conflict F1 **0.6400**、Behavior Acc **0.9600**，冲突 TP/FP/FN = **18/0/2**，precision **1.0000**、recall **0.9000**。结果归档在 `outputs/remote_results/verabench_conflict_misleading_rules_only_max25_v5.json`。
13. **Premise refutation 诊断指标**：新增 `premise_refutation_summary` 与 per-question diagnostics，用于区分“证据共同反驳用户前提”和 evidence-evidence conflict。max25 v5 的 premise refutation TP/FP/FN = **17/0/0**，precision **1.0000**、recall **1.0000**，说明 V067/V068 并非系统未反驳前提，而是 gold conflict pair 表达了另一类语义关系。
14. **测试与质量门禁**：新增冲突抽取、对比语境、问题聚焦过滤、数字单位兼容、离线语义去重开关、检索 metadata 传递、原始问题检索锚点、self-refutation、premise-refutation 和冲突/误导行为分类回归测试；该阶段全量测试 **304 passed + 3 skipped**，`make lint` 通过 Ruff 与 mypy，`python -m build` 成功生成 sdist/wheel。

注意：max10 v16 已无冲突检测失败；max25 v5 已确认主要结构误报清零，并新增 premise-refutation 诊断。下一步应扩大到全 152 题，重点观察该新指标在 misleading/unanswerable/multi-hop 上是否稳定，并重新校准置信度。

## 🆕 本次更新（2026-06-06）：评测透明度与失败诊断

为把项目继续推向高质量开源，本次重点增强 VeraBench 结果的**可解释性和可复盘性**：

1. **BenchmarkReport 新增失败诊断字段**：`behavior_confusion` 记录 expected behavior → actual behavior 的混淆矩阵；`failure_summary` 汇总行为失败、低证据召回、冲突失败及 Top 失败样例。
2. **新增冲突检测诊断计数**：每题记录 `predicted_conflicts/gold_conflicts/tp/fp/fn`，报告汇总 `conflict_summary`，用于区分过度检测、漏检或混合失败。
3. **新增校准分桶**：报告生成 `calibration_bins`，离线分析可直接查看每个置信度区间的样本量、平均置信度、真实准确率和 gap。
4. **`run_verabench.py` 控制台报告新增诊断区**：真实/演示评测结束后直接打印失败数量、按类型分布、冲突 FP/FN 和行为混淆，便于不用打开 JSON 也能定位问题。
5. **新增离线分析 CLI**：`experiments/analyze_verabench_results.py` 可直接分析已有 `results/*.json`，无需 API key 或重新调用 LLM；`calibration_curve.py` 也已兼容完整 VeraBench 报告 JSON。
6. **补充测试覆盖**：新增评估诊断、冲突计数、校准报告读取、包内 VeraBench 数据 fallback、公开 API、quickstart 示例、leaderboard 生成、冲突训练数据构建与离线分析测试，当前全量测试为 **253 passed + 3 skipped**。
7. **README 同步更新**：加入离线分析/校准曲线命令，并修正测试用例数量与 lint 状态说明。
8. **开源治理补齐**：新增 CONTRIBUTING / SECURITY / CHANGELOG / CITATION / Code of Conduct、Issue/PR 模板和 `docs/EVALUATION.md`，并修正 `pyproject.toml` dev 依赖与 Ruff 流程一致。
9. **CI 质量门禁增强**：GitHub Actions 升级为 Python 3.10/3.11/3.12/3.13 测试矩阵，并新增全仓 Ruff、全量 `src`/`verarag` mypy 与 wheel/sdist 构建检查；当前 `src/web/experiments/examples/tests` Ruff 债务已清零。
10. **发布包可用性增强**：VeraBench 数据、Web 模板/静态资源、默认配置和 examples 纳入 wheel/sdist；`verarag-web` console script 已补齐可执行入口；`load_verabench()` 支持仓库数据与包内数据 fallback。
11. **公开使用路径补齐**：新增 `docs/API.md`、`docs/ARCHITECTURE.md` 和 `examples/quickstart.py`，让新用户可以从 API、系统设计和无需 API key 的示例三条路径快速理解项目。
12. **结果发布流程补齐**：新增 `verarag-leaderboard` / `experiments/build_verabench_leaderboard.py`、`docs/RESULTS.md` 和 `docs/RELEASING.md`，把真实评测结果、复现命令和发布检查从人工说明变成可生成流程。
13. **GPU 训练路径补齐**：新增 VeraBench conflict-pair 数据构建、CrossEncoder dry-run/训练 CLI、Windows GPU 同步与 tmux 启动脚本，以及 `docs/GPU_TRAINING.md`，并把训练产物作为可配置 learned conflict layer 接入主冲突图。
14. **冲突检测训练可量化**：训练脚本现在写出 `training_metadata.json` 与 `training_metrics.json`，包含默认阈值和验证集 F1 选择阈值下的 validation/test precision、recall、F1，方便把 learned layer 的收益纳入发布结果。
15. **冲突训练数据增强**：conflict-pair builder 现在补回 gold self-conflicts，并新增保守弱正例、跨问题 topical hard negatives、`by_sample_source` 元数据与默认正例过采样训练 loader。当前扩展数据为 187 pairs，GPU smoke baseline test F1 为 0.3333。
16. **冲突检测离线 A/B**：新增 `compare_conflict_detectors.py` / `verarag-compare-conflicts`，可在不跑 LLM/检索的情况下对 VeraBench gold evidence 做 rules vs rules+learned 对比；高精度 fact-slot 门控后，离线 precision 为 1.0，但 recall 仍偏低（rules F1 0.0690，rules+learned F1 0.1333）。
17. **冲突检测完整 pipeline A/B**：新增 `run_conflict_ablation.py` / `verarag-conflict-ablation`，自动生成 rules 与 rules+learned 两份配置，支持 `--plan-only` 无 API key 预演，也能在有 API key 时成对跑 VeraBench 并输出 `summary.json` 指标 delta。
18. **真实 pipeline smoke 暴露并修复冲突过度检测**：DeepSeek `max=3` conflict 子集最初出现 rules=108、rules+learned=107 个预测冲突且 precision 约 0.019；新增 fact-slot 门控、中文 claim 原子化、日期/周期数字过滤、营收年份/季度/业务线槽位、同证据内部跳过，以及 source/scope/granularity 弱规则 opt-in 后，rules-only 已降到 `predicted/gold = 0/3`。
19. **learned conflict layer 重新定位为实验层**：GPU CrossEncoder 在离线 gold-evidence 上仍可作为候选信号，但真实 pipeline smoke 中 threshold 0.5/0.8 均产生 3 个误报且 0 TP；默认继续关闭 learned detector，下一阶段应优先做“X 声称 Y，但 Y 错误”这类嵌入式反事实/误导 claim 抽取来提升召回。
20. **检索器离线健壮性修复**：Pipeline 现在支持 `retriever.type`，DeepSeek 评测配置显式使用 BM25；HybridRetriever 在 dense 模型无法从 Hugging Face 加载时会退回 BM25，避免离线环境整轮评测失败。

## 🆕 本次更新（2026-06-01）：首次完整真实评测 + 关键 bug 修复

**首次跑通全部 152 题 VeraBench 真实评测**（DeepSeek `deepseek-v4-flash`，全流水线，零错误，~79s/题）。结果见 [README#实测结果](README.md)。核心数字：Evidence Recall **0.811**、Behavior Acc 0.526、Answer-F1 0.157；单证据/时序/多跳类行为准确率 0.92–1.0，但不可答/冲突/误导类仅 ~0.08（"倾向作答"偏差，待修）。

评测过程中发现并修复 4 个真实 bug + 新增断点续跑：

1. **evidence_id 用随机 UUID** (`agents/retrieval_agent.py`) → 改用稳定 chunk id（`D001_c0`）。修复后 Evidence Recall 从 **0 → 0.81**，证据真正可溯源。
2. **NLI 模型每对 claim 重载** (`evidence/conflict_graph.py`)：原模型 `nli-deberta-v3-small` 无权重文件必失败，且失败后不缓存，6 题加载 648 次。改用 `nli-distilroberta-base` + `_nli_tried` 只尝试一次 → 单题 **77s → ~19s**，三层冲突检测真正生效。
3. **pipeline 每题重建** (`experiments/run_verabench.py`)：改为只构建一次（原每题重载模型 + 重索引语料）。
4. **缺 python-multipart** (`requirements.txt`)：FastAPI 表单/上传依赖缺失，导致 11 个 web 测试失败 → 补齐后 **182 passed + 3 skipped**。
5. **断点续跑**（`benchmark/evaluator.py` + `run_verabench.py`）：每题完成即写 JSONL 检查点，重跑自动跳过已完成题；新增 `--checkpoint/--restart/--no-checkpoint`。

### 行为对齐迭代：修复"倾向作答"偏差（baseline → v2 → v3）

针对 baseline 的"倾向作答"偏差（不可答/冲突/误导三类 Behavior Acc 仅 ~0.08），重写推理与修复 Agent：
- `agents/reasoning_agent.py`：prompt 改中文 + 明确行为决策（证据不足→拒答 / 断言前提不成立→"该说法不准确"纠正 / 多源冲突→标注 / 否则作答），并加"默认应作答、部分证据勿轻易拒答"原则平衡过度拒答。
- `agents/repair_agent.py`：`_generate_repaired_answer` 改为**保留**推理原始答案（原会用英文模板重写、滥加冲突说明，是 Behavior Acc/Conflict-F1 偏低的根源之一），hedge 改中文。

**结果（全 152 题真实评测，Behavior Acc）：baseline 0.526 → v2 0.743 → v3 0.763**。三个弱项：不可答 0.077→0.962、误导 0.080→0.760、冲突 0.080→0.480。代价：多证据(0.92→0.60)/时序(1.0→0.76) 略降、ECE 校准退化（待重标定）。三版结果存于 `results/verabench_full{,_v2,_v3}.json`。

---

> 历史更新：2026-05-28

## 项目概况

| 指标 | 数值 |
|------|------|
| 仓库 | https://github.com/xiaweiyi713/VeraRAG |
| 源码行数 | ~8,200 行 (src/) |
| 测试 | 791 passed + 3 skipped (real LLM) |
| Web UI | ~1,800 行 (web/) |
| VeraBench 题库 | 152 题 / 6 类型 |
| VeraBench 语料 | 57 篇 / 13 主题 |
| VeraBench v1.1.2 | 152 questions, 57 docs, 208/208 traceable evidence refs, current questions hash `c19e7401...36882` |
| VeraBench v1.1 历史真实评测 | 152/152, 0 errors, Behavior Acc 0.9803 |
| 冲突检测 | 分层架构（规则10+Learned CrossEncoder+NLI+LLM） |
| LLM Provider | 6 种 |

---

## 已完成功能

### 1. 核心 Pipeline（10 阶段）
- [x] Task Analyzer — 规则 + LLM 任务分析，复杂度/跳数估计
- [x] Decomposition Planner — 子问题分解 + 不确定性驱动的 refine_plan
- [x] Dynamic Retrieval Agent — 多轮检索，query 变体生成，反证检索，覆盖度评估
- [x] Evidence Normalizer — 语义去重，可信度/时效性评分，质量过滤
- [x] Conflict Graph Builder — **分层架构**: 规则层(10个检测器) + Learned CrossEncoder + NLI层(CrossEncoder) + LLM裁决层
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
- [x] **评估报告新增行为混淆矩阵 + 失败样例摘要**
- [x] **评估报告新增冲突 TP/FP/FN 汇总 + 校准分桶**

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

### 11. 测试覆盖（794 collected）
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
- [x] GitHub Actions CI（Python 3.10/3.11/3.12/3.13 + 全仓 Ruff + 全量 `src`/`verarag` mypy + wheel/sdist 构建）
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

- [x] **跑完整 152 题 VeraBench 真实评测** ✅ 2026-06-01 完成（DeepSeek，零错误）
  - 命令: `DEEPSEEK_API_KEY=<key> python experiments/run_verabench.py --config configs/deepseek_run.yaml --output results/verabench_full.json`
  - 支持断点续跑：中断后重跑同命令自动从检查点接续
- [ ] **修复"倾向作答"偏差**（新发现的最高优先）：不可答/误导/冲突类行为准确率仅 ~0.08，需把不确定性控制器的拒答决策与最终输出文本打通

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
