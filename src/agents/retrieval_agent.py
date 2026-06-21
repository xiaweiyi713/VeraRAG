"""Dynamic Retrieval Agent for VeraRAG."""

import uuid
from typing import Any, ClassVar

from ..retriever.base import BaseRetriever, RetrievalResult
from ..utils.data_structures import Evidence, SubQuestion
from .base import BaseAgent


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
    STOPWORDS: ClassVar[set[str]] = {
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
    SYNONYM_MAP: ClassVar[dict[str, list[str]]] = {
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
        config: dict[str, Any] | None = None,
        llm_client: Any | None = None
    ):
        super().__init__(config, llm_client)
        self.retriever = retriever
        self.system_prompt = """You are an expert at generating effective search queries.
Your goal is to create queries that will find relevant evidence.
Output ONLY valid JSON, no other text."""
        retriever_config = self.config.get("retriever", {})
        self.top_k_policy = str(retriever_config.get("top_k_policy", "fixed"))
        self.retrieval_top_k = self._positive_config_int(
            retriever_config.get("retrieval_top_k", 10),
            "retriever.retrieval_top_k",
        )
        self.precision_cap_top_k = int(retriever_config.get("precision_cap_top_k", 4))
        self.adaptive_simple_top_k = int(retriever_config.get("adaptive_simple_top_k", 2))
        self.adaptive_medium_top_k = int(retriever_config.get("adaptive_medium_top_k", 4))
        self.adaptive_complex_top_k = int(retriever_config.get("adaptive_complex_top_k", 5))
        self.targeted_second_pass_enabled = bool(
            retriever_config.get("targeted_second_pass_enabled", False)
        )
        self.targeted_second_pass_top_k = self._positive_config_int(
            retriever_config.get(
                "targeted_second_pass_top_k",
                max(self.retrieval_top_k, self.adaptive_complex_top_k),
            ),
            "retriever.targeted_second_pass_top_k",
        )
        self.targeted_second_pass_max_new_evidence = self._positive_config_int(
            retriever_config.get("targeted_second_pass_max_new_evidence", 2),
            "retriever.targeted_second_pass_max_new_evidence",
        )
        self.targeted_second_pass_coverage_threshold = float(
            retriever_config.get("targeted_second_pass_coverage_threshold", 0.67)
        )

    @staticmethod
    def _positive_config_int(value: Any, field: str) -> int:
        parsed = int(value)
        if parsed <= 0:
            raise ValueError(f"{field} must be a positive integer")
        return parsed

    def retrieve_for_subquestion(
        self,
        subquestion: SubQuestion,
        top_k: int = 10,
        seek_counter_evidence: bool = False
    ) -> list[RetrievalResult]:
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

        # Re-rank and limit. The retrieval depth can stay high for recall, while
        # the retained evidence set is controlled by a configurable policy.
        selected_top_k = self._select_output_top_k(subquestion, top_k)
        unique_results = sorted(unique_results, key=lambda x: x.score, reverse=True)[
            :selected_top_k
        ]

        # If counter-evidence is requested, generate negation queries
        if seek_counter_evidence or subquestion.requires_counter_evidence:
            counter_queries = self._generate_counter_evidence_queries(subquestion.question)
            counter_top_k = max(1, selected_top_k // 2)
            for cq in counter_queries:
                c_results = self.retriever.retrieve(cq, top_k=counter_top_k)
                unique_results.extend(c_results)

        return unique_results

    def _select_output_top_k(self, subquestion: SubQuestion, retrieval_depth: int) -> int:
        """Select how many retrieved documents should enter the evidence pool."""
        if retrieval_depth < 0:
            raise ValueError("retrieval_depth must be non-negative")
        if self.top_k_policy == "fixed":
            return retrieval_depth
        if self.top_k_policy == "precision_cap":
            return min(retrieval_depth, max(1, self.precision_cap_top_k))
        if self.top_k_policy == "complexity_adaptive":
            if self._is_complex_retrieval_need(subquestion):
                return min(retrieval_depth, max(1, self.adaptive_complex_top_k))
            if self._is_medium_retrieval_need(subquestion):
                return min(retrieval_depth, max(1, self.adaptive_medium_top_k))
            return min(retrieval_depth, max(1, self.adaptive_simple_top_k))
        raise ValueError(
            "retriever.top_k_policy must be one of fixed, precision_cap, "
            "complexity_adaptive"
        )

    @staticmethod
    def _is_complex_retrieval_need(subquestion: SubQuestion) -> bool:
        evidence_type = subquestion.required_evidence_type.lower()
        if subquestion.requires_counter_evidence or subquestion.dependency_ids:
            return True
        return any(
            marker in evidence_type
            for marker in ("multi", "hop", "conflict", "comparative", "comparison")
        )

    @staticmethod
    def _is_medium_retrieval_need(subquestion: SubQuestion) -> bool:
        evidence_type = subquestion.required_evidence_type.lower()
        question = subquestion.question.lower()
        return any(
            marker in evidence_type or marker in question
            for marker in (
                "temporal",
                "time",
                "timeline",
                "latest",
                "最新",
                "时间",
                "进展",
                "版本",
            )
        )

    def dynamic_retrieve(
        self,
        subquestions: list[SubQuestion],
        evidence_pool: list[Evidence],
        max_rounds: int = 5,
        budget_per_round: int = 50
    ) -> list[Evidence]:
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

        for _round_id in range(max_rounds):
            if not unresolved:
                break

            # Select sub-question with highest uncertainty
            target = self._select_highest_uncertainty_subquestion(unresolved, evidence_pool)

            # Retrieve for this sub-question
            top_k = (
                self.retrieval_top_k
                if target.id == "sq_original"
                else max(1, min(self.retrieval_top_k, budget_per_round // len(unresolved)))
            )
            results = self.retrieve_for_subquestion(
                target,
                top_k=top_k
            )

            # Convert to Evidence objects.
            # Use the retriever's stable chunk id (e.g. D001_c0) as the evidence_id
            # so it can be traced back to the source document (and aligned with
            # VeraBench gold evidence). Fall back to a UUID only if missing.
            new_evidence = [
                self._result_to_evidence(
                    r, r.doc_id if getattr(r, "doc_id", "") else f"E{uuid.uuid4().hex[:8]}"
                )
                for r in results
            ]

            evidence_pool.extend(new_evidence)

            # Assess coverage
            coverage = self._assess_subquestion_coverage(target, evidence_pool)
            target.coverage_score = coverage
            if self._should_run_targeted_second_pass(target, coverage):
                second_pass_target = self._refine_subquestion(target, evidence_pool)
                second_pass_results = self.retrieve_for_subquestion(
                    second_pass_target,
                    top_k=self.targeted_second_pass_top_k,
                    seek_counter_evidence=target.requires_counter_evidence,
                )
                second_pass_evidence = [
                    self._result_to_evidence(
                        r,
                        r.doc_id if getattr(r, "doc_id", "") else f"E{uuid.uuid4().hex[:8]}",
                    )
                    for r in second_pass_results
                ]
                self._append_new_evidence(
                    evidence_pool,
                    second_pass_evidence,
                    limit=self.targeted_second_pass_max_new_evidence,
                )
                coverage = self._assess_subquestion_coverage(target, evidence_pool)
                target.coverage_score = coverage

            if coverage >= 0.8:  # Coverage threshold
                target.status = "resolved"
                unresolved.remove(target)
            else:
                # Refine the sub-question
                refined_target = self._refine_subquestion(target, evidence_pool)
                if refined_target is not target:
                    self._replace_subquestion(unresolved, target, refined_target)
                    self._replace_subquestion(subquestions, target, refined_target)

        return evidence_pool

    def _should_run_targeted_second_pass(
        self,
        subquestion: SubQuestion,
        coverage: float,
    ) -> bool:
        """Run a bounded second retrieval only for under-covered non-simple needs."""
        if not self.targeted_second_pass_enabled:
            return False
        if self._needs_current_attribute_refresh(subquestion):
            return True
        if coverage >= self.targeted_second_pass_coverage_threshold:
            return False
        return (
            self._is_complex_retrieval_need(subquestion)
            or self._is_medium_retrieval_need(subquestion)
        )

    @staticmethod
    def _append_new_evidence(
        evidence_pool: list[Evidence],
        candidates: list[Evidence],
        *,
        limit: int,
    ) -> None:
        """Append at most ``limit`` evidence items not already in the pool."""
        seen = {item.evidence_id for item in evidence_pool}
        added = 0
        for item in candidates:
            if item.evidence_id in seen:
                continue
            evidence_pool.append(item)
            seen.add(item.evidence_id)
            added += 1
            if added >= limit:
                return

    def _generate_query_variants(self, question: str) -> list[str]:
        """Generate multiple query variants for better retrieval."""
        import re
        variants = [question]

        compact_entity_groups = {
            "中美欧": ("中国", "美国", "欧盟"),
        }
        for compact, group_entities in compact_entity_groups.items():
            if compact not in question:
                continue
            variants.extend(
                question.replace(compact, entity)
                for entity in group_entities
            )

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

    def _generate_counter_evidence_queries(self, question: str) -> list[str]:
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

        # Path 3: Alternative viewpoint
        alt_terms_cn = ["不同观点", "反对意见", "替代解释"]
        queries.append(f"{question} {alt_terms_cn[0]}")

        # Path 2: Temporal – seek latest or alternative timeframe versions
        temporal_prefixes_cn = ["最新", "更新", "当前", "截至目前"]
        temporal_prefixes_en = ["latest", "updated", "current", "as of 2024"]
        for prefix in temporal_prefixes_cn[:2]:
            queries.append(f"{prefix} {question}")
        queries.append(f"{question} {temporal_prefixes_en[0]}")

        return queries[:8]

    def _select_highest_uncertainty_subquestion(
        self,
        subquestions: list[SubQuestion],
        evidence_pool: list[Evidence]
    ) -> SubQuestion:
        """Select the sub-question with least evidence coverage."""
        return min(subquestions, key=lambda sq: sq.coverage_score)

    def _replace_subquestion(
        self,
        subquestions: list[SubQuestion],
        old: SubQuestion,
        new: SubQuestion,
    ) -> None:
        """Replace a sub-question in-place by object identity or stable id."""
        for index, candidate in enumerate(subquestions):
            if candidate is old or candidate.id == old.id:
                subquestions[index] = new
                return

    def _assess_subquestion_coverage(
        self,
        subquestion: SubQuestion,
        evidence_pool: list[Evidence]
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
        evidence_pool: list[Evidence]
    ) -> SubQuestion:
        """Refine a sub-question based on current evidence."""
        if self._needs_current_attribute_refresh(subquestion):
            refreshed_query = self._current_attribute_refresh_query(subquestion.question)
            if refreshed_query != subquestion.question:
                return SubQuestion(
                    id=subquestion.id,
                    question=refreshed_query,
                    required_evidence_type=subquestion.required_evidence_type,
                    dependency_ids=subquestion.dependency_ids,
                    requires_counter_evidence=subquestion.requires_counter_evidence,
                    status=subquestion.status,
                    coverage_score=subquestion.coverage_score,
                )

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

    @classmethod
    def _needs_current_attribute_refresh(cls, subquestion: SubQuestion) -> bool:
        question = subquestion.question
        current_markers = ("目前", "当前", "现在", "截至", "最新")
        attribute_markers = (
            "员工",
            "营收",
            "收入",
            "成立",
            "创立",
            "人数",
            "规模",
            "CTO",
            "CEO",
            "CFO",
            "负责人",
            "主管",
            "高管",
            "新任",
            "现任",
            "多少",
            "哪一年",
        )
        if not any(marker in question for marker in current_markers):
            return False
        return any(marker in question for marker in attribute_markers)

    @classmethod
    def _current_attribute_refresh_query(cls, question: str) -> str:
        entity = cls._current_attribute_entity(question)
        markers: list[str] = []
        if any(marker in question for marker in ("员工", "人数", "规模")):
            markers.extend(["最新", "年度财务报告", "员工总数", "截至2023年末"])
        if any(marker in question for marker in ("成立", "创立", "哪一年")):
            markers.extend(["成立", "创立", "创始人"])
        if any(marker in question for marker in ("营收", "收入")):
            markers.extend(["最新", "财报", "营收", "收入"])
        if any(marker in question for marker in ("CTO", "CEO", "CFO", "负责人", "主管", "高管", "现任")):
            for role in ("CTO", "CEO", "CFO", "负责人", "主管", "高管"):
                if role in question:
                    markers.append(role)
            markers.extend(["最新", "现任", "新任", "加入", "财报", "公告"])
        if not markers:
            markers.extend(["最新", "官方", "报告"])
        return " ".join(dict.fromkeys([entity, *markers])).strip() or question

    @staticmethod
    def _current_attribute_entity(question: str) -> str:
        import re

        match = re.match(r"([\u4e00-\u9fffA-Za-z0-9·._-]{2,}?)(?:是|目前|当前|现在|截至|有|的)", question)
        if match:
            return match.group(1)
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9._-]+|[\u4e00-\u9fff]{2,}", question)
        return tokens[0] if tokens else ""

    def _result_to_evidence(self, result: RetrievalResult, evidence_id: str) -> Evidence:
        """Convert a RetrievalResult to an Evidence object."""
        return Evidence(
            evidence_id=evidence_id,
            source=result.metadata.get("source", "unknown"),
            title=result.title,
            text_span=result.content,
            date=result.metadata.get("date"),
            author=result.metadata.get("author"),
            url=result.metadata.get("url"),
            entities=result.metadata.get("entities", []),
            relevance_score=min(1.0, result.score)  # Normalize score
        )

    def run(self, *args, **kwargs) -> Any:
        """Run the dynamic retrieval agent."""
        return self.dynamic_retrieve(*args, **kwargs)
