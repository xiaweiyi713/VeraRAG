"""Task Analyzer Agent for VeraRAG."""

import json
import re
from typing import Any, ClassVar

from ..utils.data_structures import Complexity, TaskAnalysis, TaskType
from .base import BaseAgent


class TaskAnalyzer(BaseAgent):
    """
    Analyzes user questions to determine task type and complexity.

    Determines:
    - Task type (multi-hop QA, fact verification, etc.)
    - Complexity level
    - Required capabilities (retrieval, conflict check, numerical reasoning, etc.)
    - Estimated number of reasoning hops
    - Key keywords for retrieval
    """

    _TASK_TYPE_ALIASES: ClassVar[dict[str, TaskType]] = {
        "multi_hop_qa": TaskType.MULTI_HOP_QA,
        "multihop_qa": TaskType.MULTI_HOP_QA,
        "multi-hop qa": TaskType.MULTI_HOP_QA,
        "fact verification": TaskType.FACT_VERIFICATION,
        "comparative analysis": TaskType.COMPARATIVE_ANALYSIS,
        "temporal reasoning": TaskType.TEMPORAL_REASONING,
        "financial reasoning": TaskType.FINANCIAL_REASONING,
        "scientific review": TaskType.SCIENTIFIC_REVIEW,
    }

    def __init__(self, config: dict[str, Any] | None = None, llm_client: Any | None = None):
        super().__init__(config, llm_client)
        self.system_prompt = """You are a task analysis expert for complex knowledge QA systems.
Your job is to analyze questions and determine their characteristics.
Output ONLY valid JSON, no other text."""

    def analyze(self, question: str) -> TaskAnalysis:
        """
        Analyze a question to determine its characteristics.

        Args:
            question: The user's question

        Returns:
            TaskAnalysis with task characteristics
        """
        # First, try rule-based analysis for efficiency
        rule_based = self._rule_based_analyze(question)

        # For complex questions, use LLM for more accurate analysis
        if rule_based.complexity != Complexity.LOW:
            llm_based = self._llm_based_analyze(question)
            return llm_based

        return rule_based

    def _rule_based_analyze(self, question: str) -> TaskAnalysis:
        """Rule-based task analysis for efficiency."""
        q_lower = question.lower()

        # Determine task type
        task_type = TaskType.MULTI_HOP_QA
        requires_retrieval = True
        requires_conflict_check = False
        requires_numerical_reasoning = False
        requires_temporal_reasoning = False
        has_multi_hop_indicator = False

        # Check for multi-hop indicators
        multi_hop_patterns = [
            r'(how|why|what|which).*(and|then|after|before|because|due to|lead to|result in)',
            r'(compare|contrast|difference|between|versus|vs)',
            r'(relationship|connection|link|associated|correlation)',
            r'(first|second|then|next|finally|step)',
            r'(depends on|influence|affect|impact)'
        ]

        for pattern in multi_hop_patterns:
            if re.search(pattern, q_lower):
                task_type = TaskType.MULTI_HOP_QA
                has_multi_hop_indicator = True
                break

        # Check for verification indicators
        verify_patterns = [
            r'(true|false|correct|accurate|verify|confirm|valid)',
            r'(is it true|is it the case|did|was|were)',
            r'(claim|statement|assertion|allegation)'
        ]

        for pattern in verify_patterns:
            if re.search(pattern, q_lower):
                task_type = TaskType.FACT_VERIFICATION
                break

        # Check for comparative analysis
        compare_patterns = [
            r'(compare|contrast|difference|better|worse|versus|vs)',
            r'(which is|which has|what are the differences)'
        ]

        for pattern in compare_patterns:
            if re.search(pattern, q_lower):
                task_type = TaskType.COMPARATIVE_ANALYSIS
                requires_conflict_check = True
                break

        # Check for temporal reasoning
        temporal_patterns = [
            r'(when|before|after|during|until|since|earlier|later)',
            r'(timeline|chronology|evolution|history)',
            r'(change over time|trend|increase|decrease)',
            r'\d{4}'  # Years
        ]

        for pattern in temporal_patterns:
            if re.search(pattern, q_lower):
                requires_temporal_reasoning = True
                break

        # Check for numerical reasoning
        numerical_patterns = [
            r'(how many|how much|percentage|rate|ratio|proportion)',
            r'(calculate|compute|sum|total|average|median)',
            r'(more than|less than|greater|fewer|increase|decrease)',
            r'(percent|%,|\$|€|£|\d+(?:,\d{3})*(?:\.\d+)?)'
        ]

        for pattern in numerical_patterns:
            if re.search(pattern, q_lower):
                requires_numerical_reasoning = True
                break

        # Determine complexity
        complexity_indicators = sum([
            has_multi_hop_indicator,
            task_type == TaskType.COMPARATIVE_ANALYSIS,
            requires_temporal_reasoning,
            requires_numerical_reasoning,
            len(re.findall(r'\?', question)) > 1,
            len(re.findall(r'\band\b|\bor\b|\bthen\b', q_lower)) >= 2
        ])

        if complexity_indicators >= 3:
            complexity = Complexity.HIGH
        elif complexity_indicators >= 1:
            complexity = Complexity.MEDIUM
        else:
            complexity = Complexity.LOW

        # Estimate hops
        if complexity == Complexity.HIGH:
            estimated_hops = 4
        elif complexity == Complexity.MEDIUM:
            estimated_hops = 2
        else:
            estimated_hops = 1

        # Extract keywords
        keywords = self._extract_keywords(question)

        return TaskAnalysis(
            task_type=task_type,
            complexity=complexity,
            requires_retrieval=requires_retrieval,
            requires_conflict_check=requires_conflict_check,
            requires_numerical_reasoning=requires_numerical_reasoning,
            requires_temporal_reasoning=requires_temporal_reasoning,
            estimated_hops=estimated_hops,
            keywords=keywords
        )

    def _llm_based_analyze(self, question: str) -> TaskAnalysis:
        """LLM-based task analysis for complex questions."""
        prompt = f"""Analyze the following question and output JSON:

Question: "{question}"

Output JSON with this exact structure:
{{
    "task_type": "multi-hop_qa|fact_verification|comparative_analysis|temporal_reasoning|financial_reasoning|scientific_review",
    "complexity": "low|medium|high",
    "requires_retrieval": true|false,
    "requires_conflict_check": true|false,
    "requires_numerical_reasoning": true|false,
    "requires_temporal_reasoning": true|false,
    "estimated_hops": <integer 1-5>,
    "keywords": ["keyword1", "keyword2", "keyword3"]
}}

Consider:
- Does this require multiple pieces of information?
- Are there conflicting claims possible?
- Does it involve numbers, dates, or comparisons?
- How many reasoning steps are needed?
- What are the key entities/concepts?
"""

        try:
            response = self._call_llm(prompt, system_prompt=self.system_prompt, response_format="json")
            data = json.loads(response)
            if not isinstance(data, dict):
                raise ValueError("Task analysis response must be a JSON object")

            return TaskAnalysis(
                task_type=self._coerce_task_type(data.get("task_type"), TaskType.MULTI_HOP_QA),
                complexity=self._coerce_complexity(data.get("complexity"), Complexity.MEDIUM),
                requires_retrieval=self._coerce_bool(data.get("requires_retrieval"), True),
                requires_conflict_check=self._coerce_bool(data.get("requires_conflict_check"), False),
                requires_numerical_reasoning=self._coerce_bool(
                    data.get("requires_numerical_reasoning"),
                    False,
                ),
                requires_temporal_reasoning=self._coerce_bool(
                    data.get("requires_temporal_reasoning"),
                    False,
                ),
                estimated_hops=self._coerce_estimated_hops(data.get("estimated_hops"), 2),
                keywords=self._coerce_keywords(data.get("keywords")),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # Fallback to rule-based
            return self._rule_based_analyze(question)

    @classmethod
    def _coerce_task_type(cls, value: Any, default: TaskType) -> TaskType:
        if isinstance(value, TaskType):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in cls._TASK_TYPE_ALIASES:
                return cls._TASK_TYPE_ALIASES[normalized]
            try:
                return TaskType(normalized)
            except ValueError:
                return default
        return default

    @staticmethod
    def _coerce_complexity(value: Any, default: Complexity) -> Complexity:
        if isinstance(value, Complexity):
            return value
        if isinstance(value, str):
            try:
                return Complexity(value.strip().lower())
            except ValueError:
                return default
        return default

    @staticmethod
    def _coerce_bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "1"}:
                return True
            if normalized in {"false", "no", "0"}:
                return False
        return default

    @staticmethod
    def _coerce_estimated_hops(value: Any, default: int) -> int:
        if isinstance(value, bool):
            return default
        try:
            hops = int(value)
        except (TypeError, ValueError):
            return default
        return min(5, max(1, hops))

    @staticmethod
    def _coerce_keywords(value: Any) -> list[str]:
        if isinstance(value, str):
            raw_keywords: list[Any] = re.split(r"[,，;；\n]+", value)
        elif isinstance(value, list):
            raw_keywords = value
        else:
            return []

        keywords: list[str] = []
        seen: set[str] = set()
        for item in raw_keywords:
            if not isinstance(item, str):
                continue
            keyword = item.strip()
            if not keyword:
                continue
            normalized = keyword.casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            keywords.append(keyword)
            if len(keywords) >= 10:
                break
        return keywords

    def _extract_keywords(self, question: str) -> list[str]:
        """Extract key entities and concepts from the question."""
        import re

        # Remove common stop words
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
            'could', 'might', 'must', 'can', 'what', 'which', 'who', 'when',
            'where', 'why', 'how', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'as', 'about'
        }

        # Extract capitalized words (likely entities)
        entities = re.findall(r'\b[A-Z][a-z]+\b', question)

        # Extract nouns (simple heuristic)
        words = re.findall(r'\b[a-z]+\b', question.lower())
        nouns = [w for w in words if w not in stop_words and len(w) > 3]

        # Combine and deduplicate
        keywords = list(set(entities + nouns))

        return keywords[:10]  # Limit to 10 keywords

    def run(self, question: str) -> TaskAnalysis:
        """Run the task analyzer."""
        return self.analyze(question)
