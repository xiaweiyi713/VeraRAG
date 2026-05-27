"""Evidence Extractor for VeraRAG."""

import json
import re
import uuid
from typing import Dict, Any, Optional, List

from ..utils.data_structures import Evidence, Claim, ClaimType
from ..agents.base import BaseAgent


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

    def __init__(self, config: Optional[Dict[str, Any]] = None, llm_client: Optional[Any] = None):
        super().__init__(config, llm_client)
        self.system_prompt = """You are an expert at extracting structured claims from text.
Extract atomic factual claims that can be independently verified.
Output ONLY valid JSON, no other text."""

    def extract_from_text(
        self,
        text: str,
        source: str,
        title: str = "",
        metadata: Optional[Dict[str, Any]] = None
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

    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text."""
        entities = []

        # Capitalized words (simple heuristic for proper nouns)
        capitalized = re.findall(r'\b[A-Z][a-z]+\b', text)
        entities.extend(capitalized)

        # Deduplicate
        return list(set(entities))

    def _extract_claims(self, text: str) -> List[Claim]:
        """Extract atomic claims from text."""
        # For efficiency, use rule-based extraction for simple cases
        sentences = re.split(r'[.!?]+', text)
        claims = []

        for i, sent in enumerate(sentences):
            sent = sent.strip()
            if len(sent) < 10:  # Skip very short sentences
                continue

            # Determine claim type
            claim_type = self._classify_claim_type(sent)

            # Extract entities in claim
            claim_entities = self._extract_entities(sent)

            # Extract numbers
            numbers = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?%?', sent)

            # Extract temporal expressions
            time_expressions = self._extract_temporal_expressions(sent)

            # Determine verifiability and support type
            verifiable = self._is_verifiable(sent, numbers, claim_entities)
            support_type = self._infer_support_type(sent, numbers, claim_entities, time_expressions)

            claim = Claim(
                claim_id=f"C{uuid.uuid4().hex[:8]}",
                claim=sent,
                claim_type=claim_type,
                entities=claim_entities,
                numbers=numbers,
                time_expressions=time_expressions,
                verifiable=verifiable,
                support_type=support_type,
            )
            claims.append(claim)

        return claims

    @staticmethod
    def _is_verifiable(text: str, numbers: list, entities: list) -> bool:
        """Heuristic: a claim is verifiable if it contains concrete information."""
        # Claims with numbers or entities are likely verifiable
        if numbers or entities:
            return True
        # Speculative/opinion language makes it hard to verify
        speculative = ["可能", "或许", "也许", "大概", "might", "maybe", "perhaps", "possibly", "arguably"]
        lower = text.lower()
        if any(kw in lower for kw in speculative):
            return False
        return True

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

        # Check for numerical claims
        if re.search(r'\d+(?:,\d{3})*(?:\.\d+)?%?', claim_text):
            if any(word in text_lower for word in ['increase', 'decrease', 'percent', 'rate', 'ratio']):
                return ClaimType.NUMERICAL

        # Check for temporal claims
        if self._extract_temporal_expressions(claim_text):
            if any(word in text_lower for word in ['before', 'after', 'during', 'since', 'until', 'when']):
                return ClaimType.TEMPORAL

        # Check for causal claims
        if any(word in text_lower for word in ['because', 'due to', 'caused', 'led to', 'resulted in', 'reason']):
            return ClaimType.CAUSAL

        # Check for comparative claims
        if any(word in text_lower for word in ['more', 'less', 'better', 'worse', 'compared', 'versus', 'than']):
            return ClaimType.COMPARATIVE

        # Default to factual
        return ClaimType.FACTUAL

    def _extract_temporal_expressions(self, text: str) -> List[str]:
        """Extract temporal expressions from text."""
        temporal = []

        # Years
        years = re.findall(r'\b(19|20)\d{2}\b', text)
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
    ) -> List[Claim]:
        """
        Use LLM to extract claims from text (more accurate but slower).

        Args:
            text: The text to extract claims from
            max_claims: Maximum number of claims to extract

        Returns:
            List of Claim objects
        """
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

            claims = []
            for i, c_data in enumerate(data.get("claims", [])):
                claim = Claim(
                    claim_id=f"C{uuid.uuid4().hex[:8]}",
                    claim=c_data.get("claim", ""),
                    claim_type=ClaimType(c_data.get("claim_type", "factual")),
                    entities=c_data.get("entities", []),
                    numbers=c_data.get("numbers", []),
                    time_expressions=c_data.get("time_expressions", []),
                    verifiable=c_data.get("verifiable", True),
                    support_type=c_data.get("support_type", "none"),
                )
                claims.append(claim)

            return claims

        except:
            # Fallback to rule-based
            return self._extract_claims(text)

    def run(self, *args, **kwargs) -> Any:
        """Run the evidence extractor."""
        return self.extract_from_text(*args, **kwargs)
