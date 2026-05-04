"""
VeraRAG 本地模式演示 - 无需 API Key

这个演示展示了 VeraRAG 在本地模式下的功能，
使用规则基础的方法，无需任何外部 API。
"""

import sys
sys.path.insert(0, 'src')

from src.utils.data_structures import *
from src.evidence.evidence_scorer import EvidenceScorer
from src.uncertainty.controller import UncertaintyController, Action


class LocalVeraRAG:
    """本地模式的 VeraRAG，无需 LLM API"""

    def __init__(self):
        self.evidence_scorer = EvidenceScorer()
        self.uncertainty_controller = UncertaintyController()

    def query(self, question: str, documents: list) -> dict:
        """
        本地模式查询

        Args:
            question: 用户问题
            documents: 文档列表 [{"id": "1", "title": "...", "text": "..."}]
        """
        print(f"\n{'='*60}")
        print(f"问题: {question}")
        print(f"{'='*60}")

        # 1. 分析问题关键词
        keywords = self._extract_keywords(question)
        print(f"\n[1] 提取关键词: {keywords}")

        # 2. 检索相关文档
        relevant_docs = self._retrieve_documents(question, documents)
        print(f"\n[2] 检索到 {len(relevant_docs)} 条相关文档")
        for doc in relevant_docs[:3]:
            print(f"   - {doc['title']}")

        # 3. 创建证据对象
        evidences = []
        for doc in relevant_docs:
            ev = Evidence(
                evidence_id=doc['id'],
                source="local",
                title=doc['title'],
                text_span=doc['text'][:200] + "...",
                credibility_score=0.8,
                relevance_score=self._calculate_relevance(question, doc['text'])
            )
            evidences.append(ev)

        # 4. 评估证据质量
        print(f"\n[3] 证据质量评估:")
        for ev, score in self.evidence_scorer.rank_evidence(evidences):
            print(f"   {ev.evidence_id}: {score:.2f} - {ev.title}")

        # 5. 估计不确定性
        subquestions = [
            SubQuestion(id="sq1", question=question, coverage_score=0.7)
        ]
        conflict_graph = EvidenceConflictGraph()

        decision = self.uncertainty_controller.assess(
            subquestions, evidences, conflict_graph
        )

        print(f"\n[4] 不确定性评估:")
        print(f"   建议操作: {decision.action.value}")
        print(f"   置信度: {decision.confidence:.2f}")

        # 6. 生成答案
        answer = self._generate_answer(question, evidences)

        return {
            "question": question,
            "answer": answer,
            "evidence_count": len(evidences),
            "confidence": decision.confidence,
            "uncertainty_action": decision.action.value
        }

    def _extract_keywords(self, question: str) -> list:
        """提取关键词"""
        import re
        # 简单的关键词提取
        words = re.findall(r'\b[a-zA-Z一-龥]{2,}\b', question)
        return list(set(words))

    def _retrieve_documents(self, question: str, documents: list, top_k: int = 5) -> list:
        """检索相关文档"""
        keywords = self._extract_keywords(question).lower()

        scored = []
        for doc in documents:
            score = 0
            text_lower = doc['text'].lower()
            title_lower = doc['title'].lower()

            for kw in keywords:
                if kw in text_lower:
                    score += 1
                if kw in title_lower:
                    score += 2  # 标题匹配权重更高

            if score > 0:
                scored.append({**doc, 'score': score})

        scored.sort(key=lambda x: x['score'], reverse=True)
        return scored[:top_k]

    def _calculate_relevance(self, question: str, text: str) -> float:
        """计算相关性分数"""
        keywords = set(self._extract_keywords(question).lower())
        text_lower = text.lower()

        matches = sum(1 for kw in keywords if kw in text_lower)
        return min(1.0, matches / max(1, len(keywords)))

    def _generate_answer(self, question: str, evidences: list) -> str:
        """生成答案"""
        if not evidences:
            return f"抱歉，没有找到足够的信息来回答问题：{question}"

        top_ev = max(evidences, key=lambda e: e.combined_score)

        answer = f"根据检索到的证据（置信度: {top_ev.combined_score:.0%}）:\n\n"
        answer += f"基于《{top_ev.title}》的内容，"
        answer += f"{top_ev.text_span[:100]}...\n\n"

        if len(evidences) > 1:
            answer += f"此外，还有 {len(evidences)-1} 条相关证据可供参考。"

        return answer


def demo_local_mode():
    """本地模式演示"""
    print("\n" + "╔" + "═"*58 + "╗")
    print("║" + " "*10 + "VeraRAG 本地模式演示" + " "*29 + "║")
    print("║" + "           无需 API Key，即可运行" + " "*31 + "║")
    print("╚" + "═"*58 + "╝")

    # 创建示例文档
    documents = [
        {
            "id": "doc1",
            "title": "RAG 简介",
            "text": "RAG (Retrieval-Augmented Generation) 是一种结合检索和生成的技术。"
                       "它通过从外部知识库检索相关文档，然后将这些文档作为上下文提供给大语言模型，"
                       "从而生成更准确、更有事实依据的回答。"
                       "RAG 能够显著减少 LLM 的幻觉问题，因为它基于检索到的事实信息进行生成。"
        },
        {
            "id": "doc2",
            "title": "RAG 的局限性",
            "text": "虽然 RAG 能够减少幻觉，但不能完全消除。"
                       "研究表明 RAG 系统仍有约 10-20% 的错误率，"
                       "主要原因是检索结果可能不相关、不完整或包含过时信息。"
                       "此外，RAG 在处理需要推理或综合多个信息源的任务时仍面临挑战。"
        },
        {
            "id": "doc3",
            "title": "RAG vs 微调",
            "text": "与微调相比，RAG 具有更新知识更灵活、不需要重新训练模型的优势。"
                       "但微调在特定任务上可能表现更好，因为它可以将知识内化到模型参数中。"
                       "最佳方案是结合两者：使用 RAG 提供最新知识，使用微调优化任务性能。"
        },
        {
            "id": "doc4",
            "title": "RAG 在法律领域的应用",
            "text": "在法律领域，RAG 被用于从大量法律文档中检索相关案例和法规。"
                       "这可以帮助律师更高效地研究案件，但需要注意准确性要求非常高。"
                       "研究表明，结合人工审核的 RAG 系统在法律任务中表现最佳。"
        },
        {
            "id": "doc5",
            "title": "幻觉问题的研究",
            "text": "幻觉是指大语言模型生成看似合理但实际上不正确的内容。"
                       "幻觉的主要原因包括：训练数据的偏差、模型对不确定性的过度自信、"
                       "以及缺乏外部知识的验证。"
        }
    ]

    # 创建本地模式实例
    verarag = LocalVeraRAG()

    # 演示查询
    questions = [
        "什么是 RAG？",
        "RAG 能否完全消除幻觉？",
        "RAG 与微调有什么区别？",
        "RAG 在法律领域有哪些应用？"
    ]

    for q in questions:
        result = verarag.query(q, documents)
        print(f"\n答案:\n{result['answer']}")
        print()

    print("\n" + "="*60)
    print("本地模式演示完成！")
    print("\n提示：如需完整功能，请选择以下方案之一：")
    print("  1. 安装 Ollama: brew install ollama && ollama pull qwen2.5:7b")
    print("  2. 注册通义千问: https://dashscope.aliyun.com/")
    print("  3. 注册智谱 AI: https://open.bigmodel.cn/")
    print("="*60)


if __name__ == "__main__":
    demo_local_mode()
