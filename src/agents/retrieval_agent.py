"""Dynamic Retrieval Agent for VeraRAG."""

import uuid
from typing import Dict, Any, Optional, List, Tuple

from .base import BaseAgent
from ..utils.data_structures import SubQuestion, Evidence
from ..retriever.base import BaseRetriever, RetrievalResult


class DynamicRetrievalAgent(BaseAgent):
    """
    Dynamic multi-round retrieval agent.

    Unlike static RAG that retrieves once, this agent:
    1. Retrieves for each sub-question
    2. Adapts retrieval strategy based on progress
    3. Seeks counter-evidence when needed
    4. Enforces source diversity
    """

    # Common English stopwords
    STOPWORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further", "then",
        "once", "what", "which", "who", "whom", "this", "that", "these", "those",
        "it", "its", "and", "but", "or", "nor", "not", "so", "yet", "both",
        "either", "neither", "each", "every", "all", "any", "few", "more",
        "most", "other", "some", "such", "no", "only", "own", "same", "than",
        "too", "very", "just", "because", "if", "when", "where", "how", "why",
        # Chinese stopwords
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
        "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有",
        "看", "好", "自己", "这", "他", "她", "它", "吗", "吧", "呢", "啊",
        "什么", "怎么", "哪些", "哪个", "为什么", "如何",
    }

    # Common synonym mappings for query expansion
    SYNONYM_MAP = {
        "impact": ["effect", "influence", "consequence"],
        "cause": ["reason", "factor", "driver"],
        "improve": ["enhance", "boost", "increase"],
        "reduce": ["decrease", "lower", "mitigate", "minimize"],
        "relationship": ["connection", "link", "association", "correlation"],
        "mechanism": ["process", "method", "approach", "way"],
        "result": ["outcome", "finding", "conclusion"],
        "important": ["significant", "critical", "key", "major"],
        "problem": ["issue", "challenge", "difficulty"],
        "compare": ["contrast", "differ", "distinction"],
        # Chinese synonyms
        "影响": ["效果", "后果", "作用"],
        "原因": ["因素", "起因", "驱动"],
        "提升": ["增强", "改善", "提高"],
        "减少": ["降低", "下降", "缩减"],
        "关系": ["联系", "关联", "连接"],
        "机制": ["原理", "过程", "方法"],
        "结果": ["结论", "发现", "成果"],
        "重要": ["关键", "核心", "显著"],
        "问题": ["挑战", "难点", "困境"],
        "比较": ["对比", "差异", "区别"],
        "发展": ["进展", "演变", "趋势"],
        "政策": ["规定", "法规", "制度"],
        "技术": ["科技", "工艺", "方法"],
    }

    def __init__(
        self,
        retriever: BaseRetriever,
        config: Optional[Dict[str, Any]] = None,
        llm_client: Optional[Any] = None
    ):
        super().__init__(config, llm_client)
        self.retriever = retriever
        self.system_prompt = """You are an expert at generating effective search queries.
Your goal is to create queries that will find relevant evidence.
Output ONLY valid JSON, no other text."""

    def retrieve_for_subquestion(
        self,
        subquestion: SubQuestion,
        top_k: int = 10,
        seek_counter_evidence: bool = False
    ) -> List[RetrievalResult]:
        """
        Retrieve evidence for a specific sub-question.

        Args:
            subquestion: The sub-question to retrieve for
            top_k: Number of results to retrieve
            seek_counter_evidence: Whether to also retrieve counter-evidence

        Returns:
            List of retrieval results
        """
        # Generate query variants
        queries = self._generate_query_variants(subquestion.question)

        # Retrieve for each query variant
        all_results = []
        for query in queries:
            results = self.retriever.retrieve(query, top_k=top_k)
            all_results.extend(results)

        # Deduplicate by doc_id
        seen_ids = set()
        unique_results = []
        for r in all_results:
            if r.doc_id not in seen_ids:
                seen_ids.add(r.doc_id)
                unique_results.append(r)

        # Re-rank and limit
        unique_results = sorted(unique_results, key=lambda x: x.score, reverse=True)[:top_k]

        # If counter-evidence is requested, generate negation queries
        if seek_counter_evidence or subquestion.requires_counter_evidence:
            counter_queries = self._generate_counter_evidence_queries(subquestion.question)
            for cq in counter_queries:
                c_results = self.retriever.retrieve(cq, top_k=top_k // 2)
                unique_results.extend(c_results)

        return unique_results

    def dynamic_retrieve(
        self,
        subquestions: List[SubQuestion],
        evidence_pool: List[Evidence],
        max_rounds: int = 5,
        budget_per_round: int = 50
    ) -> List[Evidence]:
        """
        Dynamically retrieve evidence until coverage is satisfactory.

        Args:
            subquestions: List of sub-questions
            evidence_pool: Current evidence pool
            max_rounds: Maximum retrieval rounds
            budget_per_round: Total retrieval budget per round

        Returns:
            Updated evidence pool
        """
        # Find unresolved sub-questions
        unresolved = [sq for sq in subquestions if sq.status != "resolved"]

        if not unresolved:
            return evidence_pool

        for round_id in range(max_rounds):
            if not unresolved:
                break

            # Select sub-question with highest uncertainty
            target = self._select_highest_uncertainty_subquestion(unresolved, evidence_pool)

            # Retrieve for this sub-question
            results = self.retrieve_for_subquestion(
                target,
                top_k=min(10, budget_per_round // len(unresolved))
            )

            # Convert to Evidence objects
            new_evidence = [
                self._result_to_evidence(r, f"E{uuid.uuid4().hex[:8]}")
                for r in results
            ]

            evidence_pool.extend(new_evidence)

            # Assess coverage
            coverage = self._assess_subquestion_coverage(target, evidence_pool)
            target.coverage_score = coverage

            if coverage >= 0.8:  # Coverage threshold
                target.status = "resolved"
                unresolved.remove(target)
            else:
                # Refine the sub-question
                target = self._refine_subquestion(target, evidence_pool)

        return evidence_pool

    def _generate_query_variants(self, question: str) -> List[str]:
        """Generate multiple query variants for better retrieval."""
        import re
        variants = [question]

        words = question.replace("?", "").split()

        # Variant 1: 去掉停用词的精简版
        content_words = [w for w in words if w.lower() not in self.STOPWORDS]
        if content_words and " ".join(content_words) != question:
            variants.append(" ".join(content_words))

        # Variant 2: 同义词替换版
        synonym_words = []
        replaced = False
        for w in words:
            w_lower = w.lower()
            if w_lower in self.SYNONYM_MAP and not replaced:
                synonyms = self.SYNONYM_MAP[w_lower]
                synonym_words.append(synonyms[0])
                replaced = True
            else:
                synonym_words.append(w)
        synonym_query = " ".join(synonym_words)
        if synonym_query != question and synonym_query not in variants:
            variants.append(synonym_query)

        # Variant 3: 实体聚焦版（提取核心实体）
        entities = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', question)
        if entities:
            entity_query = " ".join(entities)
            if entity_query not in variants:
                variants.append(entity_query)

        # Variant 4: 关键词组合
        from ..agents.task_analyzer import TaskAnalyzer
        analyzer = TaskAnalyzer()
        keywords = analyzer._extract_keywords(question)
        if keywords and " ".join(keywords) not in variants:
            variants.append(" ".join(keywords))

        return variants[:5]

    def _generate_counter_evidence_queries(self, question: str) -> List[str]:
        """Generate queries to find counter-evidence via three retrieval paths.

        Paths:
          1. challenge_query: look for explicit contradictions
          2. temporal_query: look for newer/different timeframe versions
          3. alternative_query: look for alternative viewpoints
        """
        queries = []

        # Path 1: Challenge – seek explicit contradictions
        challenge_terms_en = ["not", "false", "incorrect", "debunked", "myth", "contradiction", "disputed"]
        challenge_terms_cn = ["不实", "错误", "辟谣", "争议", "反驳", "质疑", "不同"]
        for term in challenge_terms_cn[:3]:
            queries.append(f"{question} {term}")
        for term in challenge_terms_en[:2]:
            queries.append(f"{question} {term}")

        # Path 2: Temporal – seek latest or alternative timeframe versions
        temporal_prefixes_cn = ["最新", "更新", "当前", "截至目前"]
        temporal_prefixes_en = ["latest", "updated", "current", "as of 2024"]
        for prefix in temporal_prefixes_cn[:2]:
            queries.append(f"{prefix} {question}")
        queries.append(f"{question} {temporal_prefixes_en[0]}")

        # Path 3: Alternative viewpoint
        alt_terms_cn = ["不同观点", "反对意见", "替代解释"]
        queries.append(f"{question} {alt_terms_cn[0]}")

        return queries[:8]

    def _select_highest_uncertainty_subquestion(
        self,
        subquestions: List[SubQuestion],
        evidence_pool: List[Evidence]
    ) -> SubQuestion:
        """Select the sub-question with least evidence coverage."""
        return min(subquestions, key=lambda sq: sq.coverage_score)

    def _assess_subquestion_coverage(
        self,
        subquestion: SubQuestion,
        evidence_pool: List[Evidence]
    ) -> float:
        """Assess how well a sub-question is covered by evidence."""
        # Simple assessment based on relevant evidence count
        # Can be enhanced with relevance scoring

        keywords = subquestion.question.lower().split()

        relevant_count = 0
        for ev in evidence_pool:
            ev_text = f"{ev.title} {ev.text_span}".lower()
            if any(kw in ev_text for kw in keywords):
                relevant_count += 1

        # Normalize (heuristic: 3+ relevant evidence = good coverage)
        return min(1.0, relevant_count / 3.0)

    def _refine_subquestion(
        self,
        subquestion: SubQuestion,
        evidence_pool: List[Evidence]
    ) -> SubQuestion:
        """Refine a sub-question based on current evidence."""
        q_words = set(subquestion.question.lower().replace("?", "").split())
        content_words = {w for w in q_words if w not in self.STOPWORDS and len(w) > 2}

        if not content_words:
            return subquestion

        covered_words = set()
        for ev in evidence_pool:
            ev_text = f"{ev.title} {ev.text_span}".lower()
            for w in content_words:
                if w in ev_text:
                    covered_words.add(w)

        uncovered = content_words - covered_words

        if not uncovered:
            return subquestion

        refined_question = f"Find specific information about {' '.join(uncovered)} in context of: {subquestion.question}"

        return SubQuestion(
            id=subquestion.id,
            question=refined_question,
            required_evidence_type=subquestion.required_evidence_type,
            dependency_ids=subquestion.dependency_ids,
            requires_counter_evidence=subquestion.requires_counter_evidence,
            status=subquestion.status,
            coverage_score=subquestion.coverage_score
        )

    def _result_to_evidence(self, result: RetrievalResult, evidence_id: str) -> Evidence:
        """Convert a RetrievalResult to an Evidence object."""
        return Evidence(
            evidence_id=evidence_id,
            source=result.metadata.get("source", "unknown"),
            title=result.title,
            text_span=result.content,
            url=result.metadata.get("url"),
            relevance_score=min(1.0, result.score)  # Normalize score
        )

    def run(self, *args, **kwargs) -> Any:
        """Run the dynamic retrieval agent."""
        return self.dynamic_retrieve(*args, **kwargs)
