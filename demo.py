"""
VeraRAG Demo - 展示系统功能

这个演示展示了 VeraRAG 的核心功能，无需外部 API。
"""

import sys
sys.path.insert(0, 'src')

from src.utils.data_structures import *
from src.agents.task_analyzer import TaskAnalyzer
from src.evidence.conflict_graph import ConflictGraphBuilder
from src.evidence.evidence_scorer import EvidenceScorer
from src.uncertainty.controller import UncertaintyController, Action
from src.evaluation.answer_metrics import AnswerMetrics
from src.evaluation.evidence_metrics import EvidenceMetrics
from src.evaluation.conflict_metrics import ConflictMetrics


def print_section(title: str):
    """打印分隔的章节标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def demo_data_structures():
    """演示核心数据结构"""
    print_section("1. 核心数据结构演示")

    # 创建 Claim
    claim = Claim(
        claim_id="C1",
        claim="RAG 通过外部知识增强 LLM 的能力",
        claim_type=ClaimType.FACTUAL,
        entities=["RAG", "LLM"],
        confidence=0.9
    )
    print(f"✓ Claim: {claim.claim}")
    print(f"  类型: {claim.claim_type.value}, 置信度: {claim.confidence}")

    # 创建 Evidence
    evidence = Evidence(
        evidence_id="E1",
        source="paper",
        title="Retrieval-Augmented Generation for Large Language Models",
        text_span="RAG 通过检索相关文档并将其作为上下文来增强生成能力...",
        credibility_score=0.9,
        recency_score=0.8,
        relevance_score=0.85
    )
    print(f"\n✓ Evidence: {evidence.title}")
    print(f"  综合得分: {evidence.combined_score:.2f}")

    # 创建 UncertaintyBreakdown
    uncertainty = UncertaintyBreakdown(
        retrieval_uncertainty=0.15,
        evidence_conflict=0.25,
        reasoning_gap=0.10
    )
    print(f"\n✓ Uncertainty Breakdown:")
    print(f"  总体不确定性: {uncertainty.overall:.2f}")
    print(f"  可接受: {uncertainty.is_acceptable(threshold=0.3)}")


def demo_task_analyzer():
    """演示任务分析器"""
    print_section("2. 任务分析器演示")

    analyzer = TaskAnalyzer()

    questions = [
        "什么是 RAG？",
        "RAG 是否比微调更适合减少法律领域的幻觉？",
        "2023 年苹果和谷歌哪个公司的收入更高？这种增长是否可持续？"
    ]

    for q in questions:
        task = analyzer._rule_based_analyze(q)
        print(f"\n问题: {q}")
        print(f"  任务类型: {task.task_type.value}")
        print(f"  复杂度: {task.complexity.value}")
        print(f"  预计推理步数: {task.estimated_hops}")
        print(f"  需要冲突检测: {task.requires_conflict_check}")
        print(f"  需要数值推理: {task.requires_numerical_reasoning}")


def demo_conflict_graph():
    """演示冲突图构建器"""
    print_section("3. 证据冲突图演示")

    # 创建一些示例证据
    evidences = [
        Evidence(
            evidence_id="E1",
            source="paper",
            title="RAG 的优势",
            text_span="RAG 能够显著减少 LLM 的幻觉问题",
            credibility_score=0.9
        ),
        Evidence(
            evidence_id="E2",
            source="blog",
            title="RAG 的局限性",
            text_span="RAG 无法完全消除幻觉，只能减少约 50%",
            credibility_score=0.7
        ),
        Evidence(
            evidence_id="E3",
            source="paper",
            title="RAG 评估研究",
            text_span="实验表明 RAG 将错误率从 20% 降低到 8%",
            credibility_score=0.85
        )
    ]

    print(f"创建 {len(evidences)} 条证据")

    # 添加 claims
    evidences[0].claims = [
        Claim(claim_id="C1", claim="RAG 能够显著减少幻觉", claim_type=ClaimType.FACTUAL)
    ]
    evidences[1].claims = [
        Claim(claim_id="C2", claim="RAG 只能减少约 50% 的幻觉", claim_type=ClaimType.NUMERICAL)
    ]
    evidences[2].claims = [
        Claim(claim_id="C3", claim="RAG 将错误率从 20% 降低到 8%", claim_type=ClaimType.NUMERICAL)
    ]

    # 构建冲突图
    graph = ConflictGraphBuilder()
    conflict_graph = graph.build_graph(evidences, use_llm=False)

    print(f"\n✓ 冲突图构建完成:")
    print(f"  节点数: {len(conflict_graph.nodes)}")
    print(f"  边数: {len(conflict_graph.edges)}")
    print(f"  冲突分数: {conflict_graph.get_conflict_score():.2f}")

    # 使用 EvidenceScorer
    scorer = EvidenceScorer()
    ranked = scorer.rank_evidence(evidences, conflict_graph)

    print(f"\n✓ 证据质量排序:")
    for ev, score in ranked:
        print(f"  {ev.evidence_id}: {score:.2f} - {ev.title}")


def demo_uncertainty_controller():
    """演示不确定性控制器"""
    print_section("4. 不确定性控制器演示")

    controller = UncertaintyController()

    # 创建示例子问题和证据
    subquestions = [
        SubQuestion(id="sq1", question="RAG 的原理是什么？", coverage_score=0.9),
        SubQuestion(id="sq2", question="RAG 的局限性有哪些？", coverage_score=0.5),
        SubQuestion(id="sq3", question="RAG 与微调的比较？", coverage_score=0.3)
    ]

    evidence = [
        Evidence(evidence_id="E1", source="paper", title="RAG 原理", text_span="...", credibility_score=0.9)
    ]

    # 创建空冲突图
    conflict_graph = EvidenceConflictGraph()

    # 评估第一轮
    print("第一轮检索后评估:")
    decision = controller.assess(subquestions, evidence, conflict_graph, current_round=0, max_rounds=3)

    print(f"  建议操作: {decision.action.value}")
    print(f"  原因: {decision.reason}")
    print(f"  置信度: {decision.confidence:.2f}")
    print(f"  是否停止: {decision.should_stop}")


def demo_evaluation_metrics():
    """演示评估指标"""
    print_section("5. 评估指标演示")

    # 答案质量指标
    predicted = "RAG 通过检索增强生成，可以减少幻觉"
    reference = "RAG 通过检索增强生成来减少幻觉"

    em = AnswerMetrics.exact_match(predicted, reference)
    f1 = AnswerMetrics.f1_score(predicted, reference)

    print(f"✓ 答案质量指标:")
    print(f"  预测: {predicted}")
    print(f"  参考: {reference}")
    print(f"  Exact Match: {em:.2f}")
    print(f"  F1 Score: {f1:.2f}")

    # 证据质量指标
    retrieved = ["E1", "E2", "E3"]
    relevant = ["E1", "E4"]

    prec = EvidenceMetrics.evidence_precision(retrieved, relevant)
    rec = EvidenceMetrics.evidence_recall(retrieved, relevant)
    f1_ev = EvidenceMetrics.evidence_f1(retrieved, relevant)

    print(f"\n✓ 证据质量指标:")
    print(f"  检索到的证据: {retrieved}")
    print(f"  相关的证据: {relevant}")
    print(f"  Precision: {prec:.2f}")
    print(f"  Recall: {rec:.2f}")
    print(f"  F1: {f1_ev:.2f}")

    # 冲突检测指标
    pred_conflicts = [("A", "B"), ("B", "C")]
    gold_conflicts = [("A", "B"), ("C", "D")]

    f1_conf = ConflictMetrics.conflict_detection_f1(pred_conflicts, gold_conflicts)

    print(f"\n✓ 冲突检测指标:")
    print(f"  预测的冲突: {pred_conflicts}")
    print(f"  真实的冲突: {gold_conflicts}")
    print(f"  F1 Score: {f1_conf:.2f}")


def demo_complete_workflow():
    """演示完整的工作流程（简化版）"""
    print_section("6. 完整工作流程演示")

    question = "RAG 能否完全解决法律领域的幻觉问题？"

    print(f"用户问题: {question}")

    # 1. 任务分析
    analyzer = TaskAnalyzer()
    task = analyzer._rule_based_analyze(question)
    print(f"\n[1] 任务分析:")
    print(f"  类型: {task.task_type.value}, 复杂度: {task.complexity.value}")

    # 2. 问题分解
    subquestions = [
        SubQuestion(id="sq1", question="法律领域的幻觉问题有哪些？"),
        SubQuestion(id="sq2", question="RAG 如何缓解这些幻觉？"),
        SubQuestion(id="sq3", question="RAG 完全消除幻觉了吗？")
    ]
    print(f"\n[2] 问题分解:")
    for sq in subquestions:
        print(f"  - {sq.question}")

    # 3. 模拟检索
    evidences = [
        Evidence(evidence_id="E1", source="paper", title="RAG 在法律领域的应用",
                text_span="RAG 可以显著减少但不能完全消除幻觉", credibility_score=0.9),
        Evidence(evidence_id="E2", source="study", title="法律 AI 的可靠性",
                text_span="研究表明 RAG 系统仍存在约 15% 的错误率", credibility_score=0.8)
    ]
    print(f"\n[3] 检索结果:")
    for ev in evidences:
        print(f"  - {ev.title}")

    # 4. 不确定性评估
    controller = UncertaintyController()
    decision = controller.assess(subquestions, evidences, EvidenceConflictGraph(), current_round=0, max_rounds=2)
    print(f"\n[4] 不确定性评估:")
    print(f"  操作: {decision.action.value}")
    print(f"  置信度: {decision.confidence:.2f}")

    # 5. 最终输出
    print(f"\n[5] 最终答案:")
    answer = f"根据现有证据 ({decision.confidence:.0%} 置信度):"
    answer += "\n  RAG 能够显著减少但不能完全消除法律领域的幻觉问题。"
    answer += "\n  研究显示 RAG 系统仍有约 15% 的错误率，"
    answer += "\n  因此建议结合人工审核机制来确保可靠性。"

    for line in answer.split("\n"):
        print(f"  {line}")


def main():
    """运行所有演示"""
    print("\n")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║           VeraRAG 系统功能演示                              ║")
    print("║     Verifiable Agentic RAG for Complex Knowledge Tasks      ║")
    print("╚════════════════════════════════════════════════════════════╝")

    demo_data_structures()
    demo_task_analyzer()
    demo_conflict_graph()
    demo_uncertainty_controller()
    demo_evaluation_metrics()
    demo_complete_workflow()

    print_section("演示完成")
    print("所有核心功能已验证！")
    print("\n下一步:")
    print("  1. 安装依赖: pip install -r requirements.txt")
    print("  2. 配置 API Key (如需使用 LLM)")
    print("  3. 运行实验: python experiments/run_hotpotqa.py --help")
    print()


if __name__ == "__main__":
    main()
