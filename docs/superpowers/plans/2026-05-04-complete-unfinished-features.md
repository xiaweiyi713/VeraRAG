# VeraRAG 未完成功能补全 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 补全 VeraRAG 项目中 4 个空壳/简陋实现的功能：实体冲突检测、计划精炼、子问题精炼+查询变体、语义去重。

**Architecture:** 在现有代码结构上增量补全，不改变架构。每个功能独立实现和测试，保持与现有接口一致。所有功能有 fallback 机制。

**Tech Stack:** Python 3.8+, 正则表达式, sentence-transformers (复用), pytest

---

### Task 1: 实体冲突检测 `_check_entity_conflict()`

**Files:**
- Create: `tests/test_entity_conflict.py`
- Modify: `src/evidence/conflict_graph.py:214-223`

- [ ] **Step 1: 编写实体冲突检测的失败测试**

```python
"""Tests for entity conflict detection."""

import sys
sys.path.insert(0, 'src')

import unittest
from src.utils.data_structures import Claim, ClaimType, ConflictType
from src.evidence.conflict_graph import ConflictGraphBuilder


class TestEntityConflict(unittest.TestCase):
    """Test entity conflict detection."""

    def _make_claim(self, claim_id, text, entities=None, numbers=None, time_expressions=None):
        return Claim(
            claim_id=claim_id,
            claim=text,
            claim_type=ClaimType.FACTUAL,
            entities=entities or [],
            numbers=numbers or [],
            time_expressions=time_expressions or []
        )

    def test_negation_conflict(self):
        """检测否定冲突: 'X is Y' vs 'X is not Y'"""
        builder = ConflictGraphBuilder()
        claim_i = self._make_claim("C1", "Python is the best language", entities=["Python"])
        claim_j = self._make_claim("C2", "Python is not the best language", entities=["Python"])

        edge = builder._check_entity_conflict(claim_i, claim_j)
        self.assertIsNotNone(edge)
        self.assertEqual(edge.conflict_type, ConflictType.ENTITY_MISMATCH)

    def test_different_entity_values(self):
        """检测同一属性不同实体值: 'capital is X' vs 'capital is Y'"""
        builder = ConflictGraphBuilder()
        claim_i = self._make_claim("C1", "The capital of France is Paris", entities=["France", "Paris"])
        claim_j = self._make_claim("C2", "The capital of France is Lyon", entities=["France", "Lyon"])

        edge = builder._check_entity_conflict(claim_i, claim_j)
        self.assertIsNotNone(edge)

    def test_no_conflict_different_entities(self):
        """不同实体不冲突"""
        builder = ConflictGraphBuilder()
        claim_i = self._make_claim("C1", "Paris is the capital of France", entities=["Paris", "France"])
        claim_j = self._make_claim("C2", "Berlin is the capital of Germany", entities=["Berlin", "Germany"])

        edge = builder._check_entity_conflict(claim_i, claim_j)
        self.assertIsNone(edge)

    def test_no_conflict_supporting(self):
        """相同实体不冲突"""
        builder = ConflictGraphBuilder()
        claim_i = self._make_claim("C1", "Einstein was born in Germany", entities=["Einstein", "Germany"])
        claim_j = self._make_claim("C2", "Einstein grew up in Germany", entities=["Einstein", "Germany"])

        edge = builder._check_entity_conflict(claim_i, claim_j)
        self.assertIsNone(edge)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /Users/xuwenyao/VeraRAG && python -m pytest tests/test_entity_conflict.py -v`
Expected: FAIL (所有 4 个测试失败，因为 `_check_entity_conflict` 返回 `None`)

- [ ] **Step 3: 实现 `_check_entity_conflict()`**

在 `src/evidence/conflict_graph.py` 中替换 `_check_entity_conflict` 方法（第 214-223 行）：

```python
    def _check_entity_conflict(
        self,
        claim_i: Claim,
        claim_j: Claim
    ) -> Optional[ConflictEdge]:
        """Check for entity conflicts between claims."""
        entities_i = set(claim_i.entities)
        entities_j = set(claim_j.entities)

        shared = entities_i & entities_j
        if not shared:
            return None

        diff_i = entities_i - entities_j
        diff_j = entities_j - entities_i

        # Case 1: 共享实体 + 不同实体 -> 可能属性冲突
        if diff_i and diff_j:
            # 检查是否是 "X is A" vs "X is B" 的模式
            text_i_lower = claim_i.claim.lower()
            text_j_lower = claim_j.claim.lower()

            for entity in shared:
                entity_lower = entity.lower()
                if entity_lower in text_i_lower and entity_lower in text_j_lower:
                    return ConflictEdge(
                        source_id=claim_i.claim_id,
                        target_id=claim_j.claim_id,
                        conflict_type=ConflictType.ENTITY_MISMATCH,
                        confidence=0.6,
                        rationale=f"Different values for shared entity '{entity}': {diff_i} vs {diff_j}"
                    )

        # Case 2: 否定冲突检测
        text_i_lower = claim_i.claim.lower()
        text_j_lower = claim_j.claim.lower()

        negation_patterns = [
            (" is ", " is not "),
            (" are ", " are not "),
            (" was ", " was not "),
            (" has ", " has no "),
            (" can ", " cannot "),
        ]
        for pos, neg in negation_patterns:
            if (pos in text_i_lower and neg in text_j_lower) or \
               (neg in text_i_lower and pos in text_j_lower):
                return ConflictEdge(
                    source_id=claim_i.claim_id,
                    target_id=claim_j.claim_id,
                    conflict_type=ConflictType.ENTITY_MISMATCH,
                    confidence=0.8,
                    rationale=f"Negation conflict: '{claim_i.claim}' vs '{claim_j.claim}'"
                )

        return None
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /Users/xuwenyao/VeraRAG && python -m pytest tests/test_entity_conflict.py -v`
Expected: 4 passed

- [ ] **Step 5: 运行全部测试确认无回归**

Run: `cd /Users/xuwenyao/VeraRAG && python -m pytest tests/ -v`
Expected: 28 passed (原 24 + 新 4)

- [ ] **Step 6: Commit**

```bash
cd /Users/xuwenyao/VeraRAG && git add tests/test_entity_conflict.py src/evidence/conflict_graph.py && git commit -m "feat: 实现实体冲突检测 _check_entity_conflict()"
```

---

### Task 2: 计划精炼 `refine_plan()`

**Files:**
- Create: `tests/test_refine_plan.py`
- Modify: `src/agents/planner.py:207-223`

- [ ] **Step 1: 编写计划精炼的失败测试**

```python
"""Tests for plan refinement."""

import sys
sys.path.insert(0, 'src')

import unittest
from src.agents.planner import DecompositionPlanner
from src.utils.data_structures import SubQuestion, UncertaintyBreakdown


class TestRefinePlan(unittest.TestCase):
    """Test plan refinement based on uncertainty feedback."""

    def setUp(self):
        self.planner = DecompositionPlanner()

    def test_high_retrieval_uncertainty_adds_queries(self):
        """检索不确定性高时，为低覆盖度子问题提升优先级"""
        subquestions = [
            SubQuestion(id="sq0", question="What is RAG?", coverage_score=0.2),
            SubQuestion(id="sq1", question="How does RAG work?", coverage_score=0.9),
        ]
        uncertainty = UncertaintyBreakdown(retrieval_uncertainty=0.8)

        refined = self.planner.refine_plan(subquestions, {"uncertainty": uncertainty})

        # sq0 应被标记为 in_progress（低覆盖度 + 高检索不确定性）
        sq0 = next(sq for sq in refined if sq.id == "sq0")
        self.assertEqual(sq0.status, "in_progress")
        # sq1 覆盖度高，保持 resolved
        sq1 = next(sq for sq in refined if sq.id == "sq1")
        self.assertEqual(sq1.status, "resolved")

    def test_high_conflict_uncertainty_adds_resolution_question(self):
        """冲突不确定性高时，添加冲突解决子问题"""
        subquestions = [
            SubQuestion(id="sq0", question="What caused the crash?", coverage_score=0.7),
        ]
        uncertainty = UncertaintyBreakdown(evidence_conflict=0.8)

        refined = self.planner.refine_plan(subquestions, {
            "uncertainty": uncertainty,
            "conflicts": [{"type": "temporal"}]
        })

        # 应新增冲突解决子问题
        new_sqs = [sq for sq in refined if sq.id.startswith("sq_resolve")]
        self.assertGreater(len(new_sqs), 0)
        self.assertIn("conflict", new_sqs[0].question.lower())

    def test_low_uncertainty_returns_unchanged(self):
        """低不确定性时保持原样"""
        subquestions = [
            SubQuestion(id="sq0", question="What is X?", coverage_score=0.7),
        ]
        uncertainty = UncertaintyBreakdown()

        refined = self.planner.refine_plan(subquestions, {"uncertainty": uncertainty})

        self.assertEqual(len(refined), len(subquestions))

    def test_respects_max_subquestions_limit(self):
        """不超过最大子问题数量限制"""
        subquestions = [
            SubQuestion(id=f"sq{i}", question=f"Question {i}", coverage_score=0.3)
            for i in range(8)
        ]
        uncertainty = UncertaintyBreakdown(
            retrieval_uncertainty=0.8,
            evidence_conflict=0.8
        )

        refined = self.planner.refine_plan(subquestions, {
            "uncertainty": uncertainty,
            "conflicts": [{"type": "temporal"}],
            "max_subquestions": 10
        })

        self.assertLessEqual(len(refined), 10)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /Users/xuwenyao/VeraRAG && python -m pytest tests/test_refine_plan.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 `refine_plan()`**

在 `src/agents/planner.py` 中替换 `refine_plan` 方法（第 207-223 行）：

```python
    def refine_plan(
        self,
        subquestions: List[SubQuestion],
        uncertainty_report: Dict[str, Any]
    ) -> List[SubQuestion]:
        """
        Refine the plan based on uncertainty feedback.

        Args:
            subquestions: Current sub-questions
            uncertainty_report: Report with keys: "uncertainty" (UncertaintyBreakdown),
                                optional "conflicts" (list), optional "max_subquestions" (int)

        Returns:
            Refined list of sub-questions
        """
        uncertainty: UncertaintyBreakdown = uncertainty_report.get(
            "uncertainty", UncertaintyBreakdown()
        )
        conflicts = uncertainty_report.get("conflicts", [])
        max_sq = uncertainty_report.get("max_subquestions", 10)

        refined = list(subquestions)

        # Mark high-coverage questions as resolved
        for sq in refined:
            if sq.coverage_score >= 0.8:
                sq.status = "resolved"

        # Handle high retrieval uncertainty
        if uncertainty.retrieval_uncertainty > 0.5:
            for sq in refined:
                if sq.coverage_score < 0.5 and sq.status != "resolved":
                    sq.status = "in_progress"

        # Handle high conflict uncertainty -> add resolution sub-questions
        if uncertainty.evidence_conflict > 0.5 and conflicts and len(refined) < max_sq:
            conflict_types = set()
            for c in conflicts:
                ctype = c.get("type", "general") if isinstance(c, dict) else "general"
                conflict_types.add(ctype)

            for ctype in conflict_types:
                if len(refined) >= max_sq:
                    break
                resolve_sq = SubQuestion(
                    id=f"sq_resolve_{ctype}",
                    question=f"Resolve conflicting evidence about {ctype} relationships",
                    required_evidence_type="general",
                    dependency_ids=[sq.id for sq in refined if sq.status != "resolved"],
                    requires_counter_evidence=False,
                    status="pending",
                    coverage_score=0.0
                )
                refined.append(resolve_sq)

        # Handle high source reliability uncertainty
        if uncertainty.source_reliability > 0.5 and len(refined) < max_sq:
            source_sq = SubQuestion(
                id="sq_source_verify",
                question="Find high-credibility sources to verify key claims",
                required_evidence_type="empirical_study",
                dependency_ids=[sq.id for sq in refined[:3]],
                requires_counter_evidence=False,
                status="pending",
                coverage_score=0.0
            )
            refined.append(source_sq)

        return refined[:max_sq]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /Users/xuwenyao/VeraRAG && python -m pytest tests/test_refine_plan.py -v`
Expected: 4 passed

- [ ] **Step 5: 运行全部测试确认无回归**

Run: `cd /Users/xuwenyao/VeraRAG && python -m pytest tests/ -v`
Expected: 32 passed

- [ ] **Step 6: Commit**

```bash
cd /Users/xuwenyao/VeraRAG && git add tests/test_refine_plan.py src/agents/planner.py && git commit -m "feat: 实现计划精炼 refine_plan()"
```

---

### Task 3: 查询变体生成增强 `_generate_query_variants()`

**Files:**
- Create: `tests/test_query_variants.py`
- Modify: `src/agents/retrieval_agent.py:139-152`

- [ ] **Step 1: 编写查询变体的失败测试**

```python
"""Tests for query variant generation and subquestion refinement."""

import sys
sys.path.insert(0, 'src')

import unittest
from unittest.mock import MagicMock
from src.agents.retrieval_agent import DynamicRetrievalAgent
from src.utils.data_structures import SubQuestion, Evidence


class TestQueryVariants(unittest.TestCase):
    """Test query variant generation."""

    def setUp(self):
        mock_retriever = MagicMock()
        self.agent = DynamicRetrievalAgent(retriever=mock_retriever)

    def test_generates_multiple_variants(self):
        """应生成 3-5 个查询变体"""
        variants = self.agent._generate_query_variants("What is the relationship between RAG and hallucination?")
        self.assertGreaterEqual(len(variants), 3)
        self.assertLessEqual(len(variants), 5)

    def test_original_question_included(self):
        """原始问题应包含在变体中"""
        question = "What causes climate change?"
        variants = self.agent._generate_query_variants(question)
        self.assertIn(question, variants)

    def test_stopwords_removed_variant(self):
        """应有去掉停用词的精简版变体"""
        variants = self.agent._generate_query_variants("What is the impact of RAG on hallucination?")
        # 至少有一个变体不含 "what", "is", "the", "of", "on"
        non_stopword_variants = [
            v for v in variants
            if not any(w in v.lower().split() for w in ["what", "is", "the", "of"])
        ]
        self.assertGreater(len(non_stopword_variants), 0)

    def test_entity_focused_variant(self):
        """应有实体聚焦版变体"""
        variants = self.agent._generate_query_variants(
            "How does Einstein's theory relate to Newton's laws?"
        )
        # 至少有一个变体聚焦于核心实体
        entity_variants = [v for v in variants if "Einstein" in v or "Newton" in v]
        self.assertGreater(len(entity_variants), 0)


class TestSubquestionRefinement(unittest.TestCase):
    """Test subquestion refinement."""

    def setUp(self):
        mock_retriever = MagicMock()
        self.agent = DynamicRetrievalAgent(retriever=mock_retriever)

    def test_refine_updates_question_text(self):
        """精炼后的子问题应有不同的文本"""
        sq = SubQuestion(
            id="sq0",
            question="What is the mechanism of action?",
            coverage_score=0.2
        )
        evidence = [
            Evidence(
                evidence_id="E1", source="test", title="test",
                text_span="The mechanism involves protein synthesis"
            )
        ]

        refined = self.agent._refine_subquestion(sq, evidence)
        self.assertIsNotNone(refined)
        # 精炼后应保留原问题上下文
        self.assertEqual(refined.id, "sq0")

    def test_refine_preserves_context(self):
        """精炼后应保留原子问题的上下文"""
        sq = SubQuestion(
            id="sq1",
            question="How does RAG reduce hallucination?",
            dependency_ids=["sq0"],
            coverage_score=0.3
        )
        evidence = []

        refined = self.agent._refine_subquestion(sq, evidence)
        self.assertEqual(refined.dependency_ids, ["sq0"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /Users/xuwenyao/VeraRAG && python -m pytest tests/test_query_variants.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 `_generate_query_variants()` 增强和 `_refine_subquestion()`**

在 `src/agents/retrieval_agent.py` 中替换 `_generate_query_variants` 方法（第 139-152 行）：

```python
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
        "too", "very", "just", "because", "if", "when", "where", "how", "why"
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
    }

    def _generate_query_variants(self, question: str) -> List[str]:
        """Generate multiple query variants for better retrieval."""
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
        import re
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
```

替换 `_refine_subquestion` 方法（第 193-201 行）：

```python
    def _refine_subquestion(
        self,
        subquestion: SubQuestion,
        evidence_pool: List[Evidence]
    ) -> SubQuestion:
        """Refine a sub-question based on current evidence."""
        # Analyze which keywords from the question are not covered
        q_words = set(subquestion.question.lower().replace("?", "").split())
        content_words = {w for w in q_words if w not in self.STOPWORDS and len(w) > 2}

        if not content_words:
            return subquestion

        # Check which content words appear in evidence
        covered_words = set()
        for ev in evidence_pool:
            ev_text = f"{ev.title} {ev.text_span}".lower()
            for w in content_words:
                if w in ev_text:
                    covered_words.add(w)

        uncovered = content_words - covered_words

        if not uncovered:
            return subquestion

        # Generate refined question focusing on uncovered terms
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /Users/xuwenyao/VeraRAG && python -m pytest tests/test_query_variants.py -v`
Expected: 6 passed

- [ ] **Step 5: 运行全部测试确认无回归**

Run: `cd /Users/xuwenyao/VeraRAG && python -m pytest tests/ -v`
Expected: 38 passed

- [ ] **Step 6: Commit**

```bash
cd /Users/xuwenyao/VeraRAG && git add tests/test_query_variants.py src/agents/retrieval_agent.py && git commit -m "feat: 增强查询变体生成和子问题精炼"
```

---

### Task 4: 语义去重 `deduplicate()`

**Files:**
- Create: `tests/test_semantic_dedup.py`
- Modify: `src/evidence/normalizer.py:151-178`

- [ ] **Step 1: 编写语义去重的失败测试**

```python
"""Tests for semantic deduplication."""

import sys
sys.path.insert(0, 'src')

import unittest
from unittest.mock import patch, MagicMock
from src.evidence.normalizer import EvidenceNormalizer
from src.utils.data_structures import Evidence


class TestSemanticDedup(unittest.TestCase):
    """Test semantic deduplication."""

    def _make_evidence(self, eid, text, score=0.8):
        return Evidence(
            evidence_id=eid,
            source="test",
            title=f"Test {eid}",
            text_span=text,
            credibility_score=score,
            recency_score=0.8,
            relevance_score=0.8
        )

    def test_exact_dedup_still_works(self):
        """精确文本匹配去重仍应正常工作"""
        normalizer = EvidenceNormalizer()
        ev1 = self._make_evidence("E1", "RAG improves factuality in LLMs")
        ev2 = self._make_evidence("E2", "RAG improves factuality in LLMs")

        result = normalizer.deduplicate([ev1, ev2])
        self.assertEqual(len(result), 1)

    def test_different_text_not_deduped(self):
        """完全不同的文本不应被去重"""
        normalizer = EvidenceNormalizer()
        ev1 = self._make_evidence("E1", "RAG improves factuality in language models")
        ev2 = self._make_evidence("E2", "Climate change affects global temperature")

        result = normalizer.deduplicate([ev1, ev2])
        self.assertEqual(len(result), 2)

    def test_keeps_higher_score_on_dedup(self):
        """去重时保留综合分更高的证据"""
        normalizer = EvidenceNormalizer()
        ev1 = self._make_evidence("E1", "RAG improves factuality", score=0.9)
        ev2 = self._make_evidence("E2", "RAG improves factuality", score=0.5)

        result = normalizer.deduplicate([ev2, ev1])  # 低分在前
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].evidence_id, "E1")

    def test_hardcoded_year_fixed(self):
        """验证 current_year 不再硬编码为 2025"""
        import inspect
        from src.evidence.normalizer import EvidenceNormalizer
        source = inspect.getsource(EvidenceNormalizer._estimate_recency)
        self.assertNotIn("current_year = 2025", source)
        self.assertIn("datetime", source)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /Users/xuwenyao/VeraRAG && python -m pytest tests/test_semantic_dedup.py -v`
Expected: `test_hardcoded_year_fixed` FAIL, 其他可能 PASS (精确匹配已工作)

- [ ] **Step 3: 实现语义去重 + 修复硬编码年份**

在 `src/evidence/normalizer.py` 中：

1. 修复 `_estimate_recency` 中的硬编码年份（第 121 行）：
   将 `current_year = 2025` 改为 `current_year = datetime.now().year`

2. 替换 `deduplicate` 方法（第 151-178 行）：

```python
    def deduplicate(
        self,
        evidence_list: List[Evidence],
        similarity_threshold: float = 0.92
    ) -> List[Evidence]:
        """
        Remove duplicate evidence based on semantic similarity.

        Uses sentence-transformers for semantic similarity when available,
        falls back to exact text matching otherwise.

        Args:
            evidence_list: List of evidence objects
            similarity_threshold: Threshold for semantic dedup (0-1)

        Returns:
            Deduplicated list
        """
        if len(evidence_list) <= 1:
            return evidence_list

        # Try semantic dedup with sentence-transformers
        try:
            return self._semantic_dedup(evidence_list, similarity_threshold)
        except (ImportError, Exception):
            # Fallback: exact text match
            return self._exact_dedup(evidence_list)

    def _semantic_dedup(
        self,
        evidence_list: List[Evidence],
        threshold: float
    ) -> List[Evidence]:
        """Semantic deduplication using sentence-transformers."""
        from sentence_transformers import SentenceTransformer
        import numpy as np

        # Sort by combined score descending (keep higher-scored ones)
        sorted_ev = sorted(evidence_list, key=lambda ev: ev.combined_score, reverse=True)

        texts = [ev.text_span for ev in sorted_ev]
        model = SentenceTransformer('BAAI/bge-base-en-v1.5')
        embeddings = model.encode(texts, normalize_embeddings=True)

        # Greedy dedup: keep item if not similar to any already-kept item
        kept = []
        kept_indices = []

        for i, ev in enumerate(sorted_ev):
            is_duplicate = False
            for j in kept_indices:
                sim = float(np.dot(embeddings[i], embeddings[j]))
                if sim >= threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(ev)
                kept_indices.append(i)

        return kept

    def _exact_dedup(self, evidence_list: List[Evidence]) -> List[Evidence]:
        """Exact text matching deduplication (fallback)."""
        seen = set()
        deduplicated = []

        for ev in evidence_list:
            text_key = re.sub(r'\s+', ' ', ev.text_span.lower().strip())
            if text_key not in seen:
                seen.add(text_key)
                deduplicated.append(ev)

        return deduplicated
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /Users/xuwenyao/VeraRAG && python -m pytest tests/test_semantic_dedup.py -v`
Expected: 4 passed

- [ ] **Step 5: 运行全部测试确认无回归**

Run: `cd /Users/xuwenyao/VeraRAG && python -m pytest tests/ -v`
Expected: 42 passed

- [ ] **Step 6: Commit**

```bash
cd /Users/xuwenyao/VeraRAG && git add tests/test_semantic_dedup.py src/evidence/normalizer.py && git commit -m "feat: 实现语义去重 deduplicate() 并修复硬编码年份"
```

---

## 自审清单

- **Spec 覆盖**: 4 个功能全部有对应 Task (1: 实体冲突, 2: 计划精炼, 3: 查询变体+子问题精炼, 4: 语义去重+年份修复)
- **占位符扫描**: 无 TBD/TODO，所有步骤含完整代码
- **类型一致性**: 所有方法签名和返回类型与现有接口匹配 (ConflictEdge, Optional[ConflictEdge], List[SubQuestion], List[str], List[Evidence])
