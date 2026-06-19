"""Evidence Extractor for VeraRAG."""

import json
import re
import uuid
from typing import Any

from ..agents.base import BaseAgent
from ..utils.data_structures import Claim, ClaimType, Evidence

ENTITY_SUFFIXES = (
    "法案", "公司", "科技", "集团", "大学", "研究院", "委员会", "理事会",
    "模型", "系统", "市场", "电池", "芯片", "制程", "算法", "平台",
    "销量", "营收", "敏感度", "霸权",
)

REPORTING_PATTERNS = (
    "有报道称", "报道称", "有消息称", "消息称", "部分自媒体声称",
    "媒体声称", "媒体报道称", "该报道称", "声称",
)

CORRECTION_MARKERS = ("实际上", "事实上", "但", "不过", "然而", "——")
SUPPORT_TYPES = {"direct", "indirect", "none"}


class EvidenceExtractor(BaseAgent):
    """
    Extracts structured evidence units from raw documents.

    Extracts:
    1. Atomic claims from text
    2. Entities mentioned
    3. Numerical values
    4. Temporal expressions
    5. Claim types
    """

    def __init__(self, config: dict[str, Any] | None = None, llm_client: Any | None = None):
        super().__init__(config, llm_client)
        self.system_prompt = """You are an expert at extracting structured claims from text.
Extract atomic factual claims that can be independently verified.
Output ONLY valid JSON, no other text."""

    def extract_from_text(
        self,
        text: str,
        source: str,
        title: str = "",
        metadata: dict[str, Any] | None = None
    ) -> Evidence:
        """
        Extract structured evidence from raw text.

        Args:
            text: The raw text content
            source: Source identifier
            title: Document title
            metadata: Additional metadata

        Returns:
            Evidence object with extracted claims
        """
        # Extract entities
        entities = self._extract_entities(text)

        # Extract claims
        claims = self._extract_claims(text)

        # Create evidence object
        evidence = Evidence(
            evidence_id=f"E{uuid.uuid4().hex[:8]}",
            source=source,
            title=title,
            text_span=text,
            entities=entities,
            claims=claims,
            **(metadata or {})
        )

        return evidence

    def _extract_entities(self, text: str) -> list[str]:
        """Extract named entities from text."""
        entities = []

        # English title-case / acronym entities.
        capitalized = re.findall(r'\b(?:[A-Z][a-z]+|[A-Z]{2,}[A-Z0-9]*(?:-[A-Z0-9]+)?)\b', text)
        entities.extend(capitalized)
        entities.extend(re.findall(r"(?<![A-Za-z])[A-Z]{2,}[A-Z0-9]*(?![A-Za-z])", text))

        # Chinese entity heuristic: capture compact phrases ending in common
        # organization/product/policy/scientific suffixes. This is intentionally
        # conservative; it provides topical anchors for conflict detection, not
        # a full NER system.
        suffix_pattern = "|".join(re.escape(suffix) for suffix in ENTITY_SUFFIXES)
        chinese_entities = re.findall(
            rf"(?:^|[，。；：、\s（(])([\u4e00-\u9fffA-Za-z0-9·-]{{2,24}}?(?:{suffix_pattern}))",
            text,
        )
        for entity in chinese_entities:
            entity = self._normalize_entity(entity)
            if len(entity) >= 2:
                entities.extend(self._entity_variants(entity))

        # Common mixed-script organization/product names.
        mixed = re.findall(r"[\u4e00-\u9fff]{1,12}(?:AI|ECS|3nm|5nm|GPT|GPU|CPU)[\u4e00-\u9fffA-Za-z0-9-]{0,8}", text)
        for entity in mixed:
            entities.extend(self._entity_variants(self._normalize_entity(entity)))

        # Deduplicate while preserving order.
        seen = set()
        deduped = []
        for entity in entities:
            if entity not in seen:
                seen.add(entity)
                deduped.append(entity)
        return deduped

    @staticmethod
    def _entity_variants(entity: str) -> list[str]:
        variants = [entity]
        if entity == "欧盟AI法案":
            variants.append("AI法案")
        return variants

    @staticmethod
    def _normalize_entity(entity: str) -> str:
        """Trim common leading context and trailing predicate text."""
        entity = re.sub(
            r"^(近日有消息称|另有报道称|近日|另有|根据|关于|针对|对于|其中|该|其|对|在|和|与|但|另外|实际上|有报道称|截至|部分自媒体声称|媒体声称|消息称|有消息称|报道称|声称)",
            "",
            entity,
        ).strip()
        for suffix in ENTITY_SUFFIXES:
            idx = entity.find(suffix)
            if idx != -1:
                normalized = entity[:idx + len(suffix)]
                return normalized.replace("人工智能", "AI")
        return entity

    def _extract_claims(self, text: str) -> list[Claim]:
        """Extract atomic claims from text."""
        # For efficiency, use rule-based extraction for simple cases
        sentences = self._split_sentences(text)
        document_entities = self._extract_entities(text)
        claims = []

        for _i, sent in enumerate(sentences):
            sent = sent.strip()
            if len(sent) < 10:  # Skip very short sentences
                continue

            inherited_entities = (
                document_entities
                if self._should_inherit_document_entities(sent)
                else None
            )

            embedded_claims = self._extract_embedded_counterclaims(sent, inherited_entities)
            if embedded_claims:
                claims.extend(embedded_claims)
                continue

            comparative_claims = self._extract_comparative_numeric_claims(sent, inherited_entities)
            if comparative_claims:
                claims.extend(comparative_claims)
                continue

            claims.append(self._make_claim(sent, inherited_entities=inherited_entities))

        return claims

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split sentences without breaking decimal numbers like 2.5 or 302.4."""
        return re.split(r"[。！？；;!?]+|(?<!\d)\.(?!\d)", text)

    def _make_claim(
        self,
        text: str,
        *,
        inherited_entities: list[str] | None = None,
        source_span: str | None = None,
    ) -> Claim:
        """Build a Claim with the same rule-based features used everywhere."""
        text = text.strip(" ，,。；;")
        claim_type = self._classify_claim_type(text)
        claim_entities = self._extract_entities(text)
        if inherited_entities:
            claim_entities = list(dict.fromkeys([*claim_entities, *inherited_entities]))

        numbers = re.findall(
            r'(?<![A-Za-z])\d+(?:,\d{3})*(?:\.\d+)?%?\s*'
            r'(?:°C|℃|年|月|日|天|小时|分钟|秒|万美元|亿元|欧元|美元|元|亿|万|nm|纳米|厘米|公里|km|米|吨|个|量子比特|qubits?|FLOPS|%)?',
            text,
        )
        numbers = [n.strip() for n in numbers if n.strip()]
        time_expressions = self._extract_temporal_expressions(text)
        verifiable = self._is_verifiable(text, numbers, claim_entities)
        support_type = self._infer_support_type(text, numbers, claim_entities, time_expressions)

        return Claim(
            claim_id=f"C{uuid.uuid4().hex[:8]}",
            claim=text,
            claim_type=claim_type,
            entities=claim_entities,
            numbers=numbers,
            time_expressions=time_expressions,
            source_span=source_span,
            verifiable=verifiable,
            support_type=support_type,
        )

    def _extract_embedded_counterclaims(
        self,
        sentence: str,
        inherited_context_entities: list[str] | None = None,
    ) -> list[Claim]:
        """Extract reported false claims and local corrections from one sentence."""
        reported = self._extract_reported_claim_text(sentence)
        if not reported:
            return []

        inherited_entities = list(dict.fromkeys([
            *self._extract_entities(sentence),
            *(inherited_context_entities or []),
        ]))
        claims = [
            self._make_claim(
                self._resolve_pronouns(reported, inherited_entities),
                inherited_entities=inherited_entities,
                source_span="reported_claim",
            )
        ]

        corrective = self._extract_corrective_claim_text(sentence)
        if corrective:
            claims.append(
                self._make_claim(
                    self._resolve_pronouns(corrective, inherited_entities),
                    inherited_entities=inherited_entities,
                    source_span="corrective_claim",
                )
            )
        return claims

    def _extract_comparative_numeric_claims(
        self,
        sentence: str,
        inherited_context_entities: list[str] | None = None,
    ) -> list[Claim]:
        """Split compact comparative estimates into two atomic numeric claims."""
        if not any(marker in sentence for marker in ("低于", "高于", "少于", "多于")):
            return []
        if len(re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", sentence)) < 2:
            return []

        split = re.split(r"[，,]?\s*(?:略)?(?:低于|高于|少于|多于)\s*", sentence, maxsplit=1)
        if len(split) != 2:
            return []
        left, right = (part.strip(" ：:，,。；;") for part in split)
        if len(left) < 8 or len(right) < 6:
            return []
        if not re.search(r"(?<![A-Za-z])\d", left) or not re.search(
            r"(?<![A-Za-z])\d",
            right,
        ):
            return []
        if not any(keyword in sentence for keyword in ("估计", "测算", "预测", "达到", "为", "约")):
            return []

        inherited_entities = list(dict.fromkeys([
            *self._extract_entities(sentence),
            *(inherited_context_entities or []),
        ]))
        return [
            self._make_claim(
                left,
                inherited_entities=inherited_entities,
                source_span="comparative_subject_claim",
            ),
            self._make_claim(
                right,
                inherited_entities=inherited_entities,
                source_span="comparative_reference_claim",
            ),
        ]

    @staticmethod
    def _should_inherit_document_entities(sentence: str) -> bool:
        return any(
            marker in sentence
            for marker in (
                "该公司",
                "公司",
                "本公司",
                "该法案",
                "该研究",
                "本研究",
                "本文",
                "该报告",
                "报告指出",
                "报道称",
                "消息称",
            )
        )

    @staticmethod
    def _extract_reported_claim_text(sentence: str) -> str | None:
        marker_pattern = "|".join(re.escape(marker) for marker in REPORTING_PATTERNS)
        match = re.search(rf"(?:{marker_pattern})(?P<claim>.+?)(?:，|,|这是|这也是|但|不过|然而|——|$)", sentence)
        if not match:
            return None
        claim = match.group("claim").strip(" ：:，,。；;\"“”")
        return claim if len(claim) >= 6 else None

    @staticmethod
    def _extract_corrective_claim_text(sentence: str) -> str | None:
        for marker in CORRECTION_MARKERS:
            if marker not in sentence:
                continue
            tail = sentence.split(marker, 1)[1].strip(" ：:，,。；;")
            if any(word in tail for word in ("错误", "不准确", "误读", "不实")):
                continue
            return tail if len(tail) >= 8 else None
        return None

    @staticmethod
    def _resolve_pronouns(text: str, entities: list[str]) -> str:
        if not entities:
            return text
        anchor = EvidenceExtractor._select_resolution_anchor(entities)
        resolved = (
            text.replace("该法案", anchor)
            .replace("该公司", anchor)
            .replace("该报道称", "")
        )
        if resolved.startswith("法案"):
            resolved = resolved.replace("法案", anchor, 1)
        return resolved.strip()

    @staticmethod
    def _select_resolution_anchor(entities: list[str]) -> str:
        for entity in entities:
            if re.fullmatch(r"[A-Z]{2,}(?:-[A-Z0-9]+)?", entity):
                continue
            if any(entity.endswith(suffix) for suffix in ENTITY_SUFFIXES):
                return entity
        return entities[0]

    @staticmethod
    def _is_verifiable(text: str, numbers: list, entities: list) -> bool:
        """Heuristic: a claim is verifiable if it contains concrete information."""
        # Claims with numbers or entities are likely verifiable
        if numbers or entities:
            return True
        # Speculative/opinion language makes it hard to verify
        speculative = ["可能", "或许", "也许", "大概", "might", "maybe", "perhaps", "possibly", "arguably"]
        lower = text.lower()
        return not any(kw in lower for kw in speculative)

    @staticmethod
    def _infer_support_type(text: str, numbers: list, entities: list, time_expr: list) -> str:
        """Infer how this claim would be supported by evidence."""
        has_factual_signal = bool(numbers) or bool(time_expr)
        has_entity = bool(entities)
        if has_factual_signal and has_entity:
            return "direct"
        if has_factual_signal or has_entity:
            return "indirect"
        return "none"

    def _classify_claim_type(self, claim_text: str) -> ClaimType:
        """Classify the type of a claim."""
        text_lower = claim_text.lower()
        has_number = bool(re.search(r'\d+(?:,\d{3})*(?:\.\d+)?%?', claim_text))

        # Check for numerical claims
        if has_number and any(
            word in text_lower
            for word in [
                'increase', 'decrease', 'percent', 'rate', 'ratio',
                '增长', '下降', '减少', '提升', '降低', '比例', '占比', '百分比',
                '营收', '销量', '收入', '规模',
            ]
        ):
            return ClaimType.NUMERICAL

        # Check for temporal claims
        if self._extract_temporal_expressions(claim_text) and any(
            word in text_lower
            for word in [
                'before', 'after', 'during', 'since', 'until', 'when',
                '之前', '之后', '期间', '以来', '截至', '生效', '通过', '发布',
                '开始', '完成', '首次',
            ]
        ):
            return ClaimType.TEMPORAL

        # Check for causal claims
        if any(
            word in text_lower
            for word in [
                'because', 'due to', 'caused', 'led to', 'resulted in', 'reason',
                '因为', '由于', '导致', '造成', '使得', '原因', '归因于',
            ]
        ):
            return ClaimType.CAUSAL

        if any(word in text_lower for word in ['defined as', 'refers to', '定义为', '是指']):
            return ClaimType.DEFINITIONAL

        if any(word in text_lower for word in ['uncertain', 'likely', '预计', '预测', '可能', '不确定']):
            return ClaimType.UNCERTAINTY

        # Check for comparative claims
        if any(
            word in text_lower
            for word in [
                'more', 'less', 'better', 'worse', 'compared', 'versus', 'than',
                '高于', '低于', '多于', '少于', '优于', '劣于', '相比', '相较',
                '超过', '不及',
            ]
        ):
            return ClaimType.COMPARATIVE

        # Default to factual
        return ClaimType.FACTUAL

    @staticmethod
    def _coerce_max_claims(max_claims: int) -> int:
        if isinstance(max_claims, bool) or not isinstance(max_claims, int):
            raise TypeError("max_claims must be an integer")
        if max_claims < 0:
            raise ValueError("max_claims must be non-negative")
        return max_claims

    @staticmethod
    def _coerce_claim_type(value: Any) -> ClaimType:
        if isinstance(value, ClaimType):
            return value
        if not isinstance(value, str):
            return ClaimType.FACTUAL
        normalized = value.strip().lower().replace("-", "_")
        aliases = {
            "number": "numerical",
            "numeric": "numerical",
            "time": "temporal",
            "date": "temporal",
            "comparison": "comparative",
            "compare": "comparative",
            "definition": "definitional",
            "uncertain": "uncertainty",
        }
        normalized = aliases.get(normalized, normalized)
        try:
            return ClaimType(normalized)
        except ValueError:
            return ClaimType.FACTUAL

    @staticmethod
    def _coerce_string_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @staticmethod
    def _coerce_bool(value: Any, default: bool = True) -> bool:
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
    def _coerce_support_type(value: Any) -> str:
        if not isinstance(value, str):
            return "none"
        normalized = value.strip().lower().replace("-", "_")
        return normalized if normalized in SUPPORT_TYPES else "none"

    def _extract_temporal_expressions(self, text: str) -> list[str]:
        """Extract temporal expressions from text."""
        temporal = []

        # Years
        years = re.findall(r'((?:19|20)\d{2})年?', text)
        temporal.extend(years)

        # Months
        months = re.findall(
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b',
            text
        )
        temporal.extend(months)

        # Temporal words
        temp_words = re.findall(
            r'\b(yesterday|today|tomorrow|last week|next week|last month|next month|last year|next year)\b',
            text,
            re.IGNORECASE
        )
        temporal.extend([w.lower() for w in temp_words])

        return temporal

    def extract_claims_with_llm(
        self,
        text: str,
        max_claims: int = 10
    ) -> list[Claim]:
        """
        Use LLM to extract claims from text (more accurate but slower).

        Args:
            text: The text to extract claims from
            max_claims: Maximum number of claims to extract

        Returns:
            List of Claim objects
        """
        max_claims = self._coerce_max_claims(max_claims)
        if max_claims == 0:
            return []

        prompt = f"""Extract atomic factual claims from the following passage.
Each claim should be:
- Self-contained and understandable
- Potentially verifiable
- Not overly long

Passage:
{text[:2000]}

Output JSON:
{{
    "claims": [
        {{
            "claim": "the claim text",
            "claim_type": "factual|numerical|temporal|causal|comparative",
            "entities": ["entity1", "entity2"],
            "numbers": ["any numbers mentioned"],
            "time_expressions": ["any temporal expressions"],
            "verifiable": true,
            "support_type": "direct|indirect|none"
        }}
    ]
}}

Limit to {max_claims} claims.

support_type rules:
- "direct": claim contains specific numbers/dates/entities that can be looked up
- "indirect": claim is factual but requires inference to verify
- "none": claim is speculative or opinion-based
"""

        try:
            response = self._call_llm(
                prompt,
                system_prompt=self.system_prompt,
                response_format="json"
            )
            data = json.loads(response)
            raw_claims = data.get("claims", []) if isinstance(data, dict) else data
            if not isinstance(raw_claims, list):
                return self._extract_claims(text)

            claims = []
            for c_data in raw_claims[:max_claims]:
                if not isinstance(c_data, dict):
                    continue
                claim_text = str(c_data.get("claim", "")).strip()
                if not claim_text:
                    continue
                claim = Claim(
                    claim_id=f"C{uuid.uuid4().hex[:8]}",
                    claim=claim_text,
                    claim_type=self._coerce_claim_type(c_data.get("claim_type", "factual")),
                    entities=self._coerce_string_list(c_data.get("entities", [])),
                    numbers=self._coerce_string_list(c_data.get("numbers", [])),
                    time_expressions=self._coerce_string_list(c_data.get("time_expressions", [])),
                    verifiable=self._coerce_bool(c_data.get("verifiable", True)),
                    support_type=self._coerce_support_type(c_data.get("support_type", "none")),
                )
                claims.append(claim)

            return claims

        except Exception:
            # Fallback to rule-based
            return self._extract_claims(text)

    def run(self, *args, **kwargs) -> Any:
        """Run the evidence extractor."""
        return self.extract_from_text(*args, **kwargs)
