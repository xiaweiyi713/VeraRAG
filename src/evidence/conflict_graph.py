"""Conflict Graph Builder for VeraRAG.

Three-layer detection architecture:
  Layer 1 – Rule-based detectors (8 types, fast, deterministic)
  Layer 2 – NLI model (natural language inference, optional)
  Layer 3 – LLM adjudication (fallback for ambiguous cases)

Detects 11 types of relationships between evidence claims:
  SUPPORT, REFUTE, PARTIAL_SUPPORT,
  NUMERIC_CONFLICT, TEMPORAL_CONFLICT, ENTITY_MISMATCH,
  SOURCE_DISAGREEMENT, DEFINITIONAL_CONFLICT,
  SCOPE_CONFLICT, CAUSAL_CONFLICT, GRANULARITY_CONFLICT,
  UNRELATED.
"""

import logging
import re
from difflib import SequenceMatcher
from typing import Dict, Any, Optional, List, Tuple

from ..utils.data_structures import (
    Evidence,
    Claim,
    EvidenceConflictGraph,
    ConflictGraphNode,
    ConflictEdge,
    ConflictType,
)
from ..agents.base import BaseAgent

logger = logging.getLogger("verarag")

# --- Conflict severity levels ---
SEVERITY_LOW = "low"
SEVERITY_MEDIUM = "medium"
SEVERITY_HIGH = "high"
SEVERITY_CRITICAL = "critical"

# --- Resolver strategies ---
RESOLVE_TEMPORAL = "prefer_newer"
RESOLVE_SOURCE = "prefer_official"
RESOLVE_NUMERIC = "flag_for_verification"
RESOLVE_SCOPE = "prefer_narrower"
RESOLVE_CAUSAL = "flag_for_expert"
RESOLVE_GRANULARITY = "prefer_finer"
RESOLVE_DEFINITION = "prefer_consensus"

# --- Scope keywords (Chinese + English) ---
SCOPE_GLOBAL = {"全球", "世界", "国际", "global", "worldwide", "international", "all countries"}
SCOPE_REGIONAL = {"中国", "美国", "欧盟", "印度", "日本", "亚太", "欧洲", "北美",
                  "China", "US", "EU", "India", "Japan", "Asia", "Europe"}
SCOPE_NARROWER = {"部分", "某些", "特定", "本地", "单一", "partial", "specific", "local", "domestic"}

# --- Causal keywords ---
CAUSAL_POSITIVE = {"导致", "引起", "造成", "促使", "推动", "使", "因为", "由于", "所以",
                   "causes", "leads to", "results in", "drives", "due to", "because"}
CAUSAL_NEGATIVE = {"无关", "无关的", "不影响", "没有关系", "不导致", "并非", "不意味着",
                   "no link", "unrelated", "no effect", "does not cause", "not because"}

# --- Granularity temporal markers ---
GRANULARITY_FINE = {"季度", "月", "周", "日", "q1", "q2", "q3", "q4", "quarterly", "monthly", "weekly"}
GRANULARITY_COARSE = {"年", "年度", "全年", "annual", "yearly", "per year"}

# --- Definition patterns ---
DEFINITION_PATTERNS = [
    r"(?:是指|定义为|定义为|指的是|所谓)\s*[\"']?(.+?)[\"']?\s*(?:，|,|$)",
    r"(?:is defined as|refers to|means|is known as)\s+(.+?)(?:\.|,|$)",
]


class ConflictGraphBuilder(BaseAgent):
    """Builds and updates evidence conflict graphs with a three-layer architecture.

    Layer 1: 8 rule-based detectors (numeric, entity, temporal, scope, causal,
             granularity, definitional, source reliability) + support detection
    Layer 2: NLI model (optional, falls back gracefully)
    Layer 3: LLM adjudication (for cases where Layer 1 & 2 are inconclusive)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, llm_client: Optional[Any] = None):
        super().__init__(config, llm_client)
        self.system_prompt = """You are an expert at detecting conflicts and relationships between claims.
Identify whether claims support, refute, or partially support each other.
Output ONLY valid JSON, no other text."""

        # NLI model (Layer 2) – lazy-loaded
        self._nli_available = False
        self._nli_tried = False  # only attempt to load once, even on failure
        self._nli_model = None
        self._nli_tokenizer = None

        # Config-driven switches
        cg_config = config.get("conflict_graph", {}) if config else {}
        self.enable_nli = cg_config.get("enable_nli", True)
        self.enable_support_detection = cg_config.get("enable_support_detection", True)
        self.nli_threshold = cg_config.get("nli_threshold", 0.7)
        self.text_similarity_threshold = cg_config.get("text_similarity_threshold", 0.6)

    def build_graph(
        self,
        evidence_list: List[Evidence],
        use_llm: bool = True,
    ) -> EvidenceConflictGraph:
        graph = EvidenceConflictGraph()

        all_claims: List[tuple] = []
        for ev in evidence_list:
            for claim in ev.claims:
                node = ConflictGraphNode(
                    node_id=claim.claim_id,
                    content=claim.claim,
                    node_type="claim",
                    evidence_ids=[ev.evidence_id],
                )
                graph.add_node(node)
                all_claims.append((claim, ev))

        for i, (claim_i, ev_i) in enumerate(all_claims):
            for j, (claim_j, ev_j) in enumerate(all_claims):
                if i >= j:
                    continue
                edge = self._detect_relationship(claim_i, ev_i, claim_j, ev_j, use_llm)
                if edge:
                    graph.add_edge(edge)

        return graph

    # ------------------------------------------------------------------
    # NLI Layer 2 – lazy init
    # ------------------------------------------------------------------

    def _init_nli(self):
        """Try to load a cross-encoder NLI model for Layer 2 (once)."""
        if self._nli_available or self._nli_tried or not self.enable_nli:
            return
        # Attempt the (potentially expensive / network-bound) load at most once;
        # on failure the NLI layer is permanently disabled and we fall back to
        # rule + LLM layers instead of re-trying for every claim pair.
        self._nli_tried = True
        model_name = "cross-encoder/nli-distilroberta-base"
        try:
            from sentence_transformers import CrossEncoder
            self._nli_model = CrossEncoder(model_name)
            self._nli_available = True
            logger.info(f"NLI model loaded ({model_name})")
        except ImportError:
            logger.debug("sentence-transformers not installed, NLI layer disabled")
        except Exception as e:
            logger.debug(f"NLI model unavailable, disabling NLI layer: {e}")

    def _nli_detect(
        self, claim_i: Claim, claim_j: Claim,
    ) -> Optional[ConflictEdge]:
        """Use NLI cross-encoder to determine entailment / contradiction / neutral."""
        if not self._nli_available:
            return None

        try:
            scores = self._nli_model.predict(
                [(claim_i.claim, claim_j.claim)],
                show_progress_bar=False,
            )
            # Deberta NLI outputs: [contradiction, entailment, neutral]
            # Depending on model version, label order may vary; use softmax
            import numpy as np
            probs = np.exp(scores[0]) / np.exp(scores[0]).sum() if scores.ndim > 1 else None

            if probs is not None:
                contradiction_prob = float(probs[0])
                entailment_prob = float(probs[1])
            else:
                return None

            if entailment_prob > self.nli_threshold:
                return ConflictEdge(
                    source_id=claim_i.claim_id,
                    target_id=claim_j.claim_id,
                    conflict_type=ConflictType.SUPPORT,
                    confidence=round(entailment_prob, 3),
                    severity=SEVERITY_LOW,
                    rationale=f"NLI entailment: {entailment_prob:.2f}",
                    resolver_strategy="",
                )
            if contradiction_prob > self.nli_threshold:
                return ConflictEdge(
                    source_id=claim_i.claim_id,
                    target_id=claim_j.claim_id,
                    conflict_type=ConflictType.REFUTE,
                    confidence=round(contradiction_prob, 3),
                    severity=SEVERITY_HIGH,
                    rationale=f"NLI contradiction: {contradiction_prob:.2f}",
                    resolver_strategy=RESOLVE_SOURCE,
                )
        except Exception as e:
            logger.debug(f"NLI detection failed: {e}")

        return None

    # ------------------------------------------------------------------
    # Semantic text similarity (used by rule-based detectors)
    # ------------------------------------------------------------------

    @staticmethod
    def _text_similarity(text_a: str, text_b: str) -> float:
        """Jaccard similarity on character-level tokens for CJK + SequenceMatcher."""
        # Quick SequenceMatcher ratio
        ratio = SequenceMatcher(None, text_a, text_b).ratio()
        if ratio > 0.9:
            return ratio

        # Jaccard on character bigrams (good for Chinese)
        def bigrams(s: str) -> set:
            return {s[i:i+2] for i in range(len(s) - 1)} if len(s) > 1 else {s}

        bg_a = bigrams(text_a)
        bg_b = bigrams(text_b)
        if not bg_a or not bg_b:
            return ratio
        jaccard = len(bg_a & bg_b) / len(bg_a | bg_b)
        return max(ratio, jaccard)

    # ------------------------------------------------------------------
    # Support detection (rule-based)
    # ------------------------------------------------------------------

    def _check_support(
        self, claim_i: Claim, claim_j: Claim,
    ) -> Optional[ConflictEdge]:
        """Detect SUPPORT relationship: highly similar claims with shared entities."""
        if not self.enable_support_detection:
            return None

        sim = self._text_similarity(claim_i.claim, claim_j.claim)
        if sim < self.text_similarity_threshold:
            return None

        # Must share at least one entity or have very high similarity
        shared_entities = set(claim_i.entities) & set(claim_j.entities)
        if not shared_entities and sim < 0.85:
            return None

        return ConflictEdge(
            source_id=claim_i.claim_id,
            target_id=claim_j.claim_id,
            conflict_type=ConflictType.SUPPORT,
            confidence=round(min(0.95, sim), 3),
            severity=SEVERITY_LOW,
            rationale=f"Similar claims (sim={sim:.2f}, shared={len(shared_entities)})",
            resolver_strategy="",
        )

    # ------------------------------------------------------------------
    # Main dispatcher
    # ------------------------------------------------------------------

    def _detect_relationship(
        self,
        claim_i: Claim,
        ev_i: Evidence,
        claim_j: Claim,
        ev_j: Evidence,
        use_llm: bool,
    ) -> Optional[ConflictEdge]:
        # Layer 1: Rule-based detection
        edge = self._rule_based_conflict_detection(claim_i, ev_i, claim_j, ev_j)
        if edge:
            return edge

        # Layer 2: NLI model (if available)
        if self.enable_nli:
            self._init_nli()
            if self._nli_available:
                edge = self._nli_detect(claim_i, claim_j)
                if edge:
                    return edge

        # Layer 3: LLM adjudication
        if use_llm and self.llm_client:
            return self._llm_conflict_detection(claim_i, claim_j)
        return None

    def _rule_based_conflict_detection(
        self,
        claim_i: Claim,
        ev_i: Evidence,
        claim_j: Claim,
        ev_j: Evidence,
    ) -> Optional[ConflictEdge]:
        """Run all rule-based detectors in priority order."""

        # 1. Numeric conflict (highest priority – clear signal)
        if claim_i.numbers and claim_j.numbers:
            edge = self._check_numerical_conflict(claim_i, claim_j)
            if edge:
                return edge

        # 2. Entity mismatch
        if claim_i.entities and claim_j.entities:
            edge = self._check_entity_conflict(claim_i, claim_j)
            if edge:
                return edge

        # 3. Temporal conflict
        edge = self._check_temporal_conflict(claim_i, claim_j, ev_i, ev_j)
        if edge:
            return edge

        # 4. Scope conflict
        edge = self._check_scope_conflict(claim_i, claim_j)
        if edge:
            return edge

        # 5. Causal conflict
        edge = self._check_causal_conflict(claim_i, claim_j)
        if edge:
            return edge

        # 6. Granularity conflict
        edge = self._check_granularity_conflict(claim_i, claim_j)
        if edge:
            return edge

        # 7. Definitional conflict
        edge = self._check_definitional_conflict(claim_i, claim_j)
        if edge:
            return edge

        # 8. Source reliability conflict
        edge = self._check_source_reliability_conflict(claim_i, ev_i, claim_j, ev_j)
        if edge:
            return edge

        # 9. Semantic support detection (shared entities + high text similarity)
        edge = self._check_support(claim_i, claim_j)
        if edge:
            return edge

        # 10. Semantic contradiction via text similarity + negation
        edge = self._check_semantic_contradiction(claim_i, claim_j)
        if edge:
            return edge

        return None

    # ------------------------------------------------------------------
    # 10. Semantic contradiction (rule-based)
    # ------------------------------------------------------------------

    _NEGATION_PAIRS = [
        ("是", "不是"), ("有", "没有"), ("会", "不会"), ("能", "不能"),
        ("应该", "不应该"), ("可以", "不可以"), ("需要", "不需要"),
        ("是", "非"), ("对", "错"), ("正确", "错误"),
        ("上升", "下降"), ("增加", "减少"), ("增长", "下降"),
        (" is ", " is not "), (" are ", " are not "),
        (" was ", " was not "), (" has ", " has no "),
        (" can ", " cannot "), (" will ", " will not "),
        ("increased", "decreased"), ("rose", "fell"),
    ]

    def _check_semantic_contradiction(
        self, claim_i: Claim, claim_j: Claim,
    ) -> Optional[ConflictEdge]:
        """Detect contradiction via shared entities + negation pairs, even without
        exact keyword match — uses text similarity to confirm topical relevance."""
        ti = claim_i.claim.lower()
        tj = claim_j.claim.lower()

        shared_entities = set(claim_i.entities) & set(claim_j.entities)
        if not shared_entities:
            return None

        for pos, neg in self._NEGATION_PAIRS:
            if (pos in ti and neg in tj) or (neg in ti and pos in tj):
                return ConflictEdge(
                    source_id=claim_i.claim_id,
                    target_id=claim_j.claim_id,
                    conflict_type=ConflictType.REFUTE,
                    confidence=0.75,
                    severity=SEVERITY_HIGH,
                    rationale=f"Negation contradiction on shared entities: '{pos.strip()}' vs '{neg.strip()}'",
                    resolver_strategy=RESOLVE_SOURCE,
                )
        return None

    # ------------------------------------------------------------------
    # 1. Numeric conflict
    # ------------------------------------------------------------------

    def _check_numerical_conflict(
        self, claim_i: Claim, claim_j: Claim,
    ) -> Optional[ConflictEdge]:
        nums_i = self._parse_numbers(claim_i.numbers)
        nums_j = self._parse_numbers(claim_j.numbers)
        if not nums_i or not nums_j:
            return None

        # Only compare numbers of comparable magnitude (skip year-vs-quantity)
        for n_i, raw_i in zip(nums_i, claim_i.numbers):
            if self._is_likely_year(raw_i):
                continue
            for n_j, raw_j in zip(nums_j, claim_j.numbers):
                if self._is_likely_year(raw_j):
                    continue
                if n_i > 0 and n_j > 0:
                    ratio = max(n_i, n_j) / min(n_i, n_j)
                    # Dynamic threshold: require shared context (entities or similarity)
                    threshold = 1.1
                    if not (set(claim_i.entities) & set(claim_j.entities)):
                        threshold = 1.3  # Higher bar when no shared entities
                    if ratio > threshold:
                        severity = SEVERITY_HIGH if ratio > 1.5 else SEVERITY_MEDIUM
                        return ConflictEdge(
                            source_id=claim_i.claim_id,
                            target_id=claim_j.claim_id,
                            conflict_type=ConflictType.NUMERIC_CONFLICT,
                            confidence=min(0.9, 0.5 + (ratio - threshold) / 2),
                            severity=severity,
                            rationale=f"Numerical values {n_i} and {n_j} differ (ratio={ratio:.2f})",
                            resolver_strategy=RESOLVE_NUMERIC,
                        )
        return None

    @staticmethod
    def _is_likely_year(num_str: str) -> bool:
        """Heuristic: 4-digit numbers in 1800-2099 range are likely years."""
        clean = num_str.strip().replace(",", "")
        try:
            val = int(float(clean))
            return 1800 <= val <= 2099
        except (ValueError, TypeError):
            return False

    def _parse_numbers(self, num_strings: List[str]) -> List[float]:
        numbers = []
        for ns in num_strings:
            try:
                clean = ns.replace("%", "").replace(",", "").replace("亿", "00000000").replace("万", "0000")
                numbers.append(float(clean))
            except (ValueError, TypeError):
                pass
        return numbers

    # ------------------------------------------------------------------
    # 2. Entity conflict
    # ------------------------------------------------------------------

    def _check_entity_conflict(
        self, claim_i: Claim, claim_j: Claim,
    ) -> Optional[ConflictEdge]:
        entities_i = set(claim_i.entities)
        entities_j = set(claim_j.entities)
        shared = entities_i & entities_j
        if not shared:
            return None

        diff_i = entities_i - entities_j
        diff_j = entities_j - entities_i

        if diff_i and diff_j:
            for entity in shared:
                el = entity.lower()
                if el in claim_i.claim.lower() and el in claim_j.claim.lower():
                    return ConflictEdge(
                        source_id=claim_i.claim_id,
                        target_id=claim_j.claim_id,
                        conflict_type=ConflictType.ENTITY_MISMATCH,
                        confidence=0.6,
                        severity=SEVERITY_MEDIUM,
                        rationale=f"Different values for shared entity '{entity}': {diff_i} vs {diff_j}",
                        resolver_strategy=RESOLVE_SOURCE,
                    )

        # Negation detection
        ti = claim_i.claim.lower()
        tj = claim_j.claim.lower()
        negation_pairs = [
            (" is ", " is not "), (" are ", " are not "),
            (" was ", " was not "), (" has ", " has no "),
            (" can ", " cannot "), ("有", "没有"),
            ("是", "不是"), ("会", "不会"),
        ]
        for pos, neg in negation_pairs:
            if (pos in ti and neg in tj) or (neg in ti and pos in tj):
                return ConflictEdge(
                    source_id=claim_i.claim_id,
                    target_id=claim_j.claim_id,
                    conflict_type=ConflictType.ENTITY_MISMATCH,
                    confidence=0.8,
                    severity=SEVERITY_HIGH,
                    rationale=f"Negation conflict: '{claim_i.claim}' vs '{claim_j.claim}'",
                    resolver_strategy=RESOLVE_SOURCE,
                )
        return None

    # ------------------------------------------------------------------
    # 3. Temporal conflict
    # ------------------------------------------------------------------

    def _check_temporal_conflict(
        self,
        claim_i: Claim,
        claim_j: Claim,
        ev_i: Evidence,
        ev_j: Evidence,
    ) -> Optional[ConflictEdge]:
        entities_i = set(claim_i.entities)
        entities_j = set(claim_j.entities)

        # If claims share entities but evidence has different dates
        if entities_i & entities_j:
            if ev_i.date and ev_j.date and ev_i.date != ev_j.date:
                dates_different = ev_i.date[:4] != ev_j.date[:4] if len(ev_i.date) >= 4 and len(ev_j.date) >= 4 else False
                if dates_different:
                    newer = ev_j.date if ev_j.date > ev_i.date else ev_i.date
                    return ConflictEdge(
                        source_id=claim_i.claim_id,
                        target_id=claim_j.claim_id,
                        conflict_type=ConflictType.TEMPORAL_CONFLICT,
                        confidence=0.7,
                        severity=SEVERITY_MEDIUM,
                        rationale=f"Evidence dates differ: {ev_i.date} vs {ev_j.date} for shared entities",
                        resolver_strategy=RESOLVE_TEMPORAL,
                    )

        # Contradictory temporal markers in text
        if claim_i.time_expressions and claim_j.time_expressions:
            contradict = [("之前", "之后"), ("前", "后"), ("before", "after"), ("earlier", "later")]
            for ti in claim_i.time_expressions:
                for tj in claim_j.time_expressions:
                    for a, b in contradict:
                        if a in ti.lower() and b in tj.lower():
                            return ConflictEdge(
                                source_id=claim_i.claim_id,
                                target_id=claim_j.claim_id,
                                conflict_type=ConflictType.TEMPORAL_CONFLICT,
                                confidence=0.6,
                                severity=SEVERITY_MEDIUM,
                                rationale=f"Contradictory temporal markers: {ti} vs {tj}",
                                resolver_strategy=RESOLVE_TEMPORAL,
                            )
        return None

    # ------------------------------------------------------------------
    # 4. Scope conflict
    # ------------------------------------------------------------------

    def _check_scope_conflict(
        self, claim_i: Claim, claim_j: Claim,
    ) -> Optional[ConflictEdge]:
        ti = claim_i.claim.lower()
        tj = claim_j.claim.lower()

        scope_i = self._detect_scope(ti)
        scope_j = self._detect_scope(tj)

        if scope_i and scope_j and scope_i != scope_j:
            # Same entities but different scopes → conflict
            entities_i = set(claim_i.entities)
            entities_j = set(claim_j.entities)
            if entities_i & entities_j:
                return ConflictEdge(
                    source_id=claim_i.claim_id,
                    target_id=claim_j.claim_id,
                    conflict_type=ConflictType.SCOPE_CONFLICT,
                    confidence=0.65,
                    severity=SEVERITY_MEDIUM,
                    rationale=f"Scope mismatch: claim applies to '{scope_i}' vs '{scope_j}'",
                    resolver_strategy=RESOLVE_SCOPE,
                )
        return None

    def _detect_scope(self, text: str) -> Optional[str]:
        for kw in SCOPE_GLOBAL:
            if kw in text:
                return "global"
        for kw in SCOPE_REGIONAL:
            if kw in text:
                return "regional"
        for kw in SCOPE_NARROWER:
            if kw in text:
                return "narrow"
        return None

    # ------------------------------------------------------------------
    # 5. Causal conflict
    # ------------------------------------------------------------------

    def _check_causal_conflict(
        self, claim_i: Claim, claim_j: Claim,
    ) -> Optional[ConflictEdge]:
        ti = claim_i.claim.lower()
        tj = claim_j.claim.lower()

        has_causal_i = any(kw in ti for kw in CAUSAL_POSITIVE)
        has_negate_i = any(kw in ti for kw in CAUSAL_NEGATIVE)
        has_causal_j = any(kw in tj for kw in CAUSAL_POSITIVE)
        has_negate_j = any(kw in tj for kw in CAUSAL_NEGATIVE)

        if (has_causal_i and has_negate_j) or (has_negate_i and has_causal_j):
            entities_i = set(claim_i.entities)
            entities_j = set(claim_j.entities)
            if entities_i & entities_j:
                return ConflictEdge(
                    source_id=claim_i.claim_id,
                    target_id=claim_j.claim_id,
                    conflict_type=ConflictType.CAUSAL_CONFLICT,
                    confidence=0.6,
                    severity=SEVERITY_HIGH,
                    rationale="Causal relationship disputed between claims",
                    resolver_strategy=RESOLVE_CAUSAL,
                )
        return None

    # ------------------------------------------------------------------
    # 6. Granularity conflict
    # ------------------------------------------------------------------

    def _check_granularity_conflict(
        self, claim_i: Claim, claim_j: Claim,
    ) -> Optional[ConflictEdge]:
        ti = claim_i.claim.lower()
        tj = claim_j.claim.lower()

        fine_i = any(kw in ti for kw in GRANULARITY_FINE)
        coarse_i = any(kw in ti for kw in GRANULARITY_COARSE)
        fine_j = any(kw in tj for kw in GRANULARITY_FINE)
        coarse_j = any(kw in tj for kw in GRANULARITY_COARSE)

        # One uses fine-grained, other uses coarse for the same subject
        if (fine_i and coarse_j) or (coarse_i and fine_j):
            entities_i = set(claim_i.entities)
            entities_j = set(claim_j.entities)
            if entities_i & entities_j:
                return ConflictEdge(
                    source_id=claim_i.claim_id,
                    target_id=claim_j.claim_id,
                    conflict_type=ConflictType.GRANULARITY_CONFLICT,
                    confidence=0.5,
                    severity=SEVERITY_LOW,
                    rationale="Different granularity levels for same subject",
                    resolver_strategy=RESOLVE_GRANULARITY,
                )
        return None

    # ------------------------------------------------------------------
    # 7. Definitional conflict
    # ------------------------------------------------------------------

    def _check_definitional_conflict(
        self, claim_i: Claim, claim_j: Claim,
    ) -> Optional[ConflictEdge]:
        defn_i = self._extract_definition(claim_i.claim)
        defn_j = self._extract_definition(claim_j.claim)

        if defn_i and defn_j:
            entities_i = set(e.lower() for e in claim_i.entities)
            entities_j = set(e.lower() for e in claim_j.entities)
            shared = entities_i & entities_j
            if shared and defn_i != defn_j:
                return ConflictEdge(
                    source_id=claim_i.claim_id,
                    target_id=claim_j.claim_id,
                    conflict_type=ConflictType.DEFINITIONAL_CONFLICT,
                    confidence=0.6,
                    severity=SEVERITY_MEDIUM,
                    rationale=f"Different definitions for shared entity: '{defn_i}' vs '{defn_j}'",
                    resolver_strategy=RESOLVE_DEFINITION,
                )
        return None

    def _extract_definition(self, text: str) -> Optional[str]:
        for pattern in DEFINITION_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return None

    # ------------------------------------------------------------------
    # 8. Source reliability conflict
    # ------------------------------------------------------------------

    RELIABILITY_RANK = {
        "official": 5,
        "paper": 4,
        "report": 3,
        "wiki": 2,
        "news": 2,
        "blog": 1,
    }

    def _check_source_reliability_conflict(
        self,
        claim_i: Claim,
        ev_i: Evidence,
        claim_j: Claim,
        ev_j: Evidence,
    ) -> Optional[ConflictEdge]:
        entities_i = set(claim_i.entities)
        entities_j = set(claim_j.entities)

        if not (entities_i & entities_j):
            return None

        rank_i = self.RELIABILITY_RANK.get(ev_i.source, 2)
        rank_j = self.RELIABILITY_RANK.get(ev_j.source, 2)

        # Large reliability gap + same entities + contradictory claims
        if abs(rank_i - rank_j) >= 2:
            # Check if claims seem to contradict (simplified heuristic)
            ti = claim_i.claim.lower()
            tj = claim_j.claim.lower()
            contradiction_signals = ["不", "错误", "无", "并非", "否定", "not", "false", "no", "wrong"]
            has_contradiction = any(s in ti for s in contradiction_signals) or any(s in tj for s in contradiction_signals)
            if has_contradiction:
                reliable = "source" if rank_i > rank_j else "target"
                return ConflictEdge(
                    source_id=claim_i.claim_id,
                    target_id=claim_j.claim_id,
                    conflict_type=ConflictType.SOURCE_DISAGREEMENT,
                    confidence=0.55,
                    severity=SEVERITY_MEDIUM,
                    rationale=f"Source reliability gap: {ev_i.source}(rank={rank_i}) vs {ev_j.source}(rank={rank_j})",
                    resolver_strategy=f"{RESOLVE_SOURCE}:{reliable}",
                )
        return None

    # ------------------------------------------------------------------
    # LLM fallback
    # ------------------------------------------------------------------

    def _llm_conflict_detection(
        self, claim_i: Claim, claim_j: Claim,
    ) -> Optional[ConflictEdge]:
        import json

        prompt = f"""Determine the relationship between these two claims.

Claim A: {claim_i.claim}

Claim B: {claim_j.claim}

Output JSON:
{{
    "relationship": "SUPPORT|REFUTE|PARTIAL_SUPPORT|UNRELATED",
    "confidence": <0.0-1.0>,
    "rationale": "brief explanation in the same language as the claims",
    "conflict_type": "numeric_conflict|temporal_conflict|entity_mismatch|scope_conflict|causal_conflict|granularity_conflict|definitional_conflict|source_disagreement|none"
}}

Relationship types:
- SUPPORT: The claims agree with or entail each other
- REFUTE: The claims directly contradict each other
- PARTIAL_SUPPORT: The claims partially agree but have minor differences
- UNRELATED: The claims don't address the same topic

Guidelines:
- Two claims about the same entity with different numeric values → REFUTE
- Same fact stated differently → SUPPORT
- Overlapping scope with one more specific → PARTIAL_SUPPORT
- No topical overlap → UNRELATED
"""
        try:
            response = self._call_llm(prompt, system_prompt=self.system_prompt, response_format="json")
            data = json.loads(response)
            relationship = data.get("relationship", "UNRELATED")
            if relationship == "UNRELATED":
                return None

            conflict_type_map = {
                "SUPPORT": ConflictType.SUPPORT,
                "REFUTE": ConflictType.REFUTE,
                "PARTIAL_SUPPORT": ConflictType.PARTIAL_SUPPORT,
            }
            return ConflictEdge(
                source_id=claim_i.claim_id,
                target_id=claim_j.claim_id,
                conflict_type=conflict_type_map.get(relationship, ConflictType.SUPPORT),
                confidence=data.get("confidence", 0.5),
                severity=SEVERITY_HIGH if relationship == "REFUTE" else SEVERITY_LOW,
                rationale=data.get("rationale", ""),
                resolver_strategy=RESOLVE_SOURCE,
            )
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Graph update
    # ------------------------------------------------------------------

    def update_graph(
        self,
        graph: EvidenceConflictGraph,
        new_evidence: List[Evidence],
        use_llm: bool = True,
    ) -> EvidenceConflictGraph:
        for ev in new_evidence:
            for claim in ev.claims:
                if claim.claim_id not in graph.nodes:
                    node = ConflictGraphNode(
                        node_id=claim.claim_id,
                        content=claim.claim,
                        node_type="claim",
                        evidence_ids=[ev.evidence_id],
                    )
                    graph.add_node(node)

        existing_claims = []
        for node in graph.nodes.values():
            if node.node_type == "claim":
                for ev in new_evidence:
                    for claim in ev.claims:
                        if claim.claim_id == node.node_id:
                            existing_claims.append((claim, ev))

        new_claims = [(claim, ev) for ev in new_evidence for claim in ev.claims]

        for claim_i, ev_i in existing_claims:
            for claim_j, ev_j in new_claims:
                edge = self._detect_relationship(claim_i, ev_i, claim_j, ev_j, use_llm)
                if edge and edge.conflict_type != ConflictType.UNRELATED:
                    existing = any(
                        e.source_id == edge.source_id and e.target_id == edge.target_id
                        for e in graph.edges
                    )
                    if not existing:
                        graph.add_edge(edge)

        return graph

    def run(self, *args, **kwargs) -> Any:
        return self.build_graph(*args, **kwargs)
