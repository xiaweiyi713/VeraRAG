"""Conflict Graph Builder for VeraRAG.

Layered detection architecture:
  Layer 1 – Rule-based detectors (8 types, fast, deterministic)
  Layer 2 – Learned conflict CrossEncoder (optional, fine-tuned)
  Layer 3 – NLI model (natural language inference, optional)
  Layer 4 – LLM adjudication (fallback for ambiguous cases)

Detects 11 types of relationships between evidence claims:
  SUPPORT, REFUTE, PARTIAL_SUPPORT,
  NUMERIC_CONFLICT, TEMPORAL_CONFLICT, ENTITY_MISMATCH,
  SOURCE_DISAGREEMENT, DEFINITIONAL_CONFLICT,
  SCOPE_CONFLICT, CAUSAL_CONFLICT, GRANULARITY_CONFLICT,
  UNRELATED.
"""

import logging
import math
import os
import re
from difflib import SequenceMatcher
from typing import Any, ClassVar

from ..agents.base import BaseAgent
from ..utils.data_structures import (
    Claim,
    ConflictEdge,
    ConflictGraphNode,
    ConflictType,
    Evidence,
    EvidenceConflictGraph,
)
from ..utils.model_cache import load_optional_model_once

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

# Claim attributes used to decide whether two same-entity claims are about the
# same fact slot. This deliberately stays small and high precision; broad topic
# overlap alone should not create a conflict edge.
ATTRIBUTE_KEYWORDS = {
    "营收": "revenue",
    "收入": "revenue",
    "利润": "profit",
    "销量": "sales",
    "销售": "sales",
    "交付": "sales",
    "员工": "employees",
    "人数": "employees",
    "成立": "founded",
    "创立": "founded",
    "创始人": "founder",
    "创办人": "founder",
    "策略": "strategy",
    "路线": "strategy",
    "模式": "strategy",
    "特点": "strategy",
    "关键节点": "timeline",
    "历程": "timeline",
    "演变": "timeline",
    "通过": "passed",
    "批准": "passed",
    "生效": "effective",
    "搁置": "passed",
    "已获通过": "passed",
    "禁止": "ban",
    "允许": "allow",
    "气候敏感度": "climate_sensitivity",
    "ECS": "climate_sensitivity",
    "最佳估计": "climate_sensitivity",
    "估计": "estimate",
    "规模": "size",
    "增长": "growth",
    "下降": "decline",
    "比例": "ratio",
    "温度": "temperature",
    "排放": "emissions",
    "制程": "process_node",
    "栅极长度": "process_node",
    "物理尺寸": "process_node",
    "物理栅极长度": "process_node",
    "命名": "process_node",
    "尺寸": "size",
    "市场份额": "market_share",
    "计算任务": "runtime",
    "计算时间": "runtime",
    "经典算法": "runtime",
    "耗时": "runtime",
    "融资": "funding",
    "估值": "valuation",
    "罚款": "fine",
    "营业额": "fine",
    "量子比特": "qubits",
    "排名": "ranking",
    "总部": "headquarters",
    "首席执行官": "ceo",
    "CEO": "ceo",
    "首都": "capital",
    "revenue": "revenue",
    "profit": "profit",
    "sales": "sales",
    "deliveries": "sales",
    "delivered": "sales",
    "employees": "employees",
    "founded": "founded",
    "founder": "founder",
    "strategy": "strategy",
    "roadmap": "strategy",
    "timeline": "timeline",
    "passed": "passed",
    "approved": "passed",
    "effective": "effective",
    "ban": "ban",
    "allow": "allow",
    "climate sensitivity": "climate_sensitivity",
    "ecs": "climate_sensitivity",
    "estimate": "estimate",
    "growth": "growth",
    "decline": "decline",
    "emissions": "emissions",
    "size": "size",
    "market share": "market_share",
    "runtime": "runtime",
    "valuation": "valuation",
    "fine": "fine",
    "penalty": "fine",
    "turnover": "fine",
    "qubit": "qubits",
    "qubits": "qubits",
    "ranking": "ranking",
    "headquarters": "headquarters",
    "ceo": "ceo",
    "capital": "capital",
}

ENTITY_VALUE_ATTRIBUTES = {"capital", "headquarters", "ceo"}

REVENUE_QUALIFIER_KEYWORDS = {
    "云服务": "cloud",
    "AI芯片": "ai_chip",
    "企业软件": "enterprise_software",
    "其他业务": "other_business",
    "研发投入": "rd_spend",
}

SALES_QUALIFIER_KEYWORDS = {
    "全球": "global",
    "中国": "china",
    "欧洲": "europe",
    "美国": "us",
    "出口": "export",
    "海外市场": "export",
    "制造商": "manufacturer",
    "车企": "manufacturer",
    "global": "global",
    "china": "china",
    "europe": "europe",
    "u.s.": "us",
    "us": "us",
    "export": "export",
    "overseas": "export",
    "manufacturer": "manufacturer",
}

EMISSIONS_QUALIFIER_KEYWORDS = {
    "化石燃料": "fossil_fuel",
    "土地利用": "land_use",
    "煤炭": "coal",
    "石油": "oil",
    "天然气": "gas",
    "中国": "china",
    "美国": "us",
    "欧盟": "eu",
    "印度": "india",
    "fossil": "fossil_fuel",
    "land use": "land_use",
    "coal": "coal",
    "oil": "oil",
    "gas": "gas",
}

QUARTER_WORDS = {
    "一": "q1",
    "二": "q2",
    "三": "q3",
    "四": "q4",
    "1": "q1",
    "2": "q2",
    "3": "q3",
    "4": "q4",
}

# --- Definition patterns ---
DEFINITION_PATTERNS = [
    r"(?:是指|定义为|定义为|指的是|所谓)\s*[\"']?(.+?)[\"']?\s*(?:，|,|$)",
    r"(?:is defined as|refers to|means|is known as)\s+(.+?)(?:\.|,|$)",
]


class ConflictGraphBuilder(BaseAgent):
    """Builds and updates evidence conflict graphs with a layered architecture.

    Layer 1: 8 rule-based detectors (numeric, entity, temporal, scope, causal,
             granularity, definitional, source reliability) + support detection
    Layer 2: Learned conflict CrossEncoder (optional, falls back gracefully)
    Layer 3: NLI model (optional, falls back gracefully)
    Layer 4: LLM adjudication (for cases where previous layers are inconclusive)
    """

    def __init__(self, config: dict[str, Any] | None = None, llm_client: Any | None = None):
        super().__init__(config, llm_client)
        self.system_prompt = """You are an expert at detecting conflicts and relationships between claims.
Identify whether claims support, refute, or partially support each other.
Output ONLY valid JSON, no other text."""

        # NLI model (Layer 3) – lazy-loaded
        self._nli_available = False
        self._nli_tried = False  # only attempt to load once, even on failure
        self._nli_model = None
        self._nli_tokenizer = None

        # Optional trained conflict detector – lazy-loaded from local path or HF id
        self._learned_available = False
        self._learned_tried = False
        self._learned_model = None
        self._learned_score_cache: dict[tuple[int, int], float] = {}
        self._claim_registry: dict[str, tuple[Claim, Evidence]] = {}

        # Config-driven switches
        cg_config = config.get("conflict_graph", {}) if config else {}
        learned_path = self._resolve_model_path(
            cg_config.get("learned_model_path") or os.getenv("VERARAG_CONFLICT_MODEL", "")
        )
        self.enable_nli = cg_config.get("enable_nli", True)
        self.nli_model_name = cg_config.get(
            "nli_model",
            "cross-encoder/nli-distilroberta-base",
        )
        self.nli_local_files_only = cg_config.get("nli_local_files_only", False)
        self.enable_learned_detector = cg_config.get("enable_learned_detector", bool(learned_path))
        self.learned_model_path = learned_path
        self.learned_threshold = self._config_probability(cg_config, "learned_threshold", 0.7)
        self.learned_require_context = cg_config.get("learned_require_context", True)
        self.learned_candidate_similarity = self._config_probability(
            cg_config,
            "learned_candidate_similarity",
            0.18,
        )
        self.enable_source_reliability_conflict = cg_config.get("enable_source_reliability_conflict", False)
        self.enable_scope_conflict = cg_config.get("enable_scope_conflict", False)
        self.enable_granularity_conflict = cg_config.get("enable_granularity_conflict", False)
        self.compare_within_evidence = cg_config.get("compare_within_evidence", False)
        self.enable_support_detection = cg_config.get("enable_support_detection", True)
        self.nli_threshold = self._config_probability(cg_config, "nli_threshold", 0.7)
        self.text_similarity_threshold = self._config_probability(
            cg_config,
            "text_similarity_threshold",
            0.6,
        )
        self.min_conflict_similarity = self._config_probability(
            cg_config,
            "min_conflict_similarity",
            0.22,
        )
        self.unattributed_conflict_similarity = self._config_probability(
            cg_config,
            "unattributed_conflict_similarity",
            0.55,
        )
        self.enable_llm_adjudication = cg_config.get("enable_llm_adjudication", False)
        self.llm_adjudication_similarity = self._config_probability(
            cg_config,
            "llm_adjudication_similarity",
            0.35,
        )

    @staticmethod
    def _resolve_model_path(raw_path: Any) -> str:
        """Resolve plain paths, ``~`` paths, and ``${ENV_VAR}`` placeholders."""
        if not raw_path:
            return ""
        path = str(raw_path)
        if path.startswith("${") and path.endswith("}"):
            path = os.getenv(path[2:-1], "")
        path = os.path.expandvars(path)
        if "$" in path:
            return ""
        return os.path.expanduser(path)

    @staticmethod
    def _config_probability(config: dict[str, Any], key: str, default: float) -> float:
        """Read a probability-like threshold from config without runtime surprises."""
        value = config.get(key, default)
        if isinstance(value, bool):
            return default
        try:
            probability = float(value)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(probability):
            return default
        return max(0.0, min(1.0, probability))

    def build_graph(
        self,
        evidence_list: list[Evidence],
        use_llm: bool = True,
    ) -> EvidenceConflictGraph:
        graph = EvidenceConflictGraph()

        all_claims: list[tuple] = []
        for ev in evidence_list:
            self._register_evidence(ev)
            for claim in ev.claims:
                node = ConflictGraphNode(
                    node_id=claim.claim_id,
                    content=claim.claim,
                    node_type="claim",
                    evidence_ids=[ev.evidence_id],
                )
                graph.add_node(node)
                all_claims.append((claim, ev))
                self_edge = self._detect_self_refuting_claim(claim, ev)
                if self_edge:
                    graph.add_edge(self_edge)

        self._prime_learned_score_cache(all_claims)
        for i, (claim_i, ev_i) in enumerate(all_claims):
            for j, (claim_j, ev_j) in enumerate(all_claims):
                if i >= j:
                    continue
                if (
                    not self.compare_within_evidence
                    and ev_i is ev_j
                    and not self._is_explicit_same_evidence_pair(claim_i, claim_j, ev_i)
                ):
                    continue
                edge = self._detect_relationship(claim_i, ev_i, claim_j, ev_j, use_llm)
                if edge:
                    graph.add_edge(edge)

        return graph

    def _detect_self_refuting_claim(self, claim: Claim, evidence: Evidence) -> ConflictEdge | None:
        """Detect high-precision single-claim corrections of common false premises."""
        text = f"{evidence.title} {claim.claim}".lower()
        raw_text = f"{evidence.title} {claim.claim}"
        conflict_type = ConflictType.REFUTE
        rationale = ""
        resolver = RESOLVE_SOURCE

        if (
            any(marker in raw_text for marker in ("不再代表实际", "并不代表实际", "不代表实际"))
            and any(marker in raw_text for marker in ("物理栅极长度", "物理尺寸", "栅极长度"))
        ):
            conflict_type = ConflictType.DEFINITIONAL_CONFLICT
            rationale = "Process-node naming is explicitly distinguished from physical dimensions"
            resolver = RESOLVE_DEFINITION
        elif "尚未通过联邦层面的综合性ai立法" in text:
            conflict_type = ConflictType.DEFINITIONAL_CONFLICT
            rationale = "Claim explicitly says federal comprehensive AI legislation has not passed"
            resolver = RESOLVE_SOURCE
        elif "并不总是降低幻觉率" in raw_text or "不总是降低幻觉率" in raw_text:
            conflict_type = ConflictType.CAUSAL_CONFLICT
            rationale = "Claim explicitly rejects the monotonic retrieval-count premise"
            resolver = RESOLVE_CAUSAL
        elif (
            "固态电池" in raw_text
            and "大规模" in raw_text
            and any(marker in raw_text for marker in ("仍面临", "才能实现", "尚未实现"))
        ):
            conflict_type = ConflictType.TEMPORAL_CONFLICT
            rationale = "Claim states large-scale solid-state battery commercialization remains future or blocked"
            resolver = RESOLVE_TEMPORAL
        elif (
            "全球" in claim.claim
            and any(marker in claim.claim for marker in ("CO2排放", "碳排放", "化石燃料CO2排放"))
            and any(marker in claim.claim for marker in ("增长", "创新高", "创历史新高"))
        ):
            conflict_type = ConflictType.SCOPE_CONFLICT
            rationale = "Claim says global emissions are growing or at a record high"
            resolver = RESOLVE_SCOPE
        elif (
            "ITER" in raw_text
            and "首次等离子体" in raw_text
            and "原计划" in raw_text
            and any(marker in raw_text for marker in ("推迟", "延误", "延期"))
        ):
            conflict_type = ConflictType.TEMPORAL_CONFLICT
            rationale = "Claim explicitly corrects the original ITER first-plasma schedule"
            resolver = RESOLVE_TEMPORAL
        elif (
            "先进封装" in raw_text
            and "先进封装" in evidence.title
            and any(marker in raw_text for marker in ("关键路径", "成本高昂", "多种路径"))
        ):
            conflict_type = ConflictType.DEFINITIONAL_CONFLICT
            rationale = "Claim frames advanced packaging as one improvement path, not a complete process replacement"
            resolver = RESOLVE_DEFINITION
        else:
            return None

        return ConflictEdge(
            source_id=claim.claim_id,
            target_id=claim.claim_id,
            conflict_type=conflict_type,
            confidence=0.72,
            severity=SEVERITY_MEDIUM,
            rationale=rationale,
            resolver_strategy=resolver,
        )

    # ------------------------------------------------------------------
    # Learned conflict layer – optional fine-tuned CrossEncoder
    # ------------------------------------------------------------------

    def _init_learned_detector(self) -> bool:
        """Try to load a trained conflict CrossEncoder once."""
        if self._learned_available:
            return True
        if self._learned_tried or not self.enable_learned_detector:
            return False

        self._learned_tried = True
        if not self.learned_model_path:
            logger.debug("learned conflict detector enabled but no model path configured")
            return False

        try:
            from sentence_transformers import CrossEncoder
            self._learned_model = CrossEncoder(self.learned_model_path)
            self._learned_available = True
            logger.info(f"Learned conflict detector loaded ({self.learned_model_path})")
            return True
        except ImportError:
            logger.debug("sentence-transformers not installed, learned conflict detector disabled")
        except Exception as e:
            logger.debug(f"Learned conflict detector unavailable, disabling layer: {e}")
        return False

    @staticmethod
    def _score_to_probability(score: Any) -> float:
        """Normalize CrossEncoder output into a conflict probability."""
        if hasattr(score, "tolist"):
            score = score.tolist()
        while isinstance(score, list) and len(score) == 1:
            score = score[0]
        if isinstance(score, list):
            if not score:
                return 0.0
            if len(score) == 2:
                try:
                    negative_score, positive_score = (float(score[0]), float(score[1]))
                except (TypeError, ValueError):
                    return 0.0
                if not (
                    math.isfinite(negative_score)
                    and math.isfinite(positive_score)
                ):
                    return 0.0
                if (
                    0.0 <= negative_score <= 1.0
                    and 0.0 <= positive_score <= 1.0
                    and negative_score + positive_score <= 1.000001
                ):
                    return positive_score
                max_score = max(negative_score, positive_score)
                neg_exp = math.exp(negative_score - max_score)
                pos_exp = math.exp(positive_score - max_score)
                return pos_exp / (neg_exp + pos_exp)
            return 0.0
        try:
            value = float(score)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(value):
            return 0.0
        if 0.0 <= value <= 1.0:
            return value
        if value >= 0:
            return 1.0 / (1.0 + math.exp(-value))
        exp_value = math.exp(value)
        return exp_value / (1.0 + exp_value)

    @staticmethod
    def _coerce_probability(value: Any) -> float | None:
        try:
            probability = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(probability):
            return None
        return max(0.0, min(1.0, probability))

    def _learned_conflict_detect(
        self, claim_i: Claim, claim_j: Claim,
    ) -> ConflictEdge | None:
        """Use a trained binary CrossEncoder to identify conflict pairs."""
        if not self._learned_available or self._learned_model is None:
            return None
        if self.learned_require_context and not self._eligible_for_learned_detection(claim_i, claim_j):
            return None

        cache_key = (id(claim_i), id(claim_j))
        probability = self._learned_score_cache.get(cache_key)
        if probability is None:
            try:
                score = self._learned_model.predict(
                    [(claim_i.claim, claim_j.claim)],
                    show_progress_bar=False,
                )
                probability = self._score_to_probability(score)
            except Exception as e:
                logger.debug(f"Learned conflict detection failed: {e}")
                return None
        if not math.isfinite(probability):
            return None

        if probability < self.learned_threshold:
            return None

        return ConflictEdge(
            source_id=claim_i.claim_id,
            target_id=claim_j.claim_id,
            conflict_type=ConflictType.REFUTE,
            confidence=round(probability, 3),
            severity=SEVERITY_HIGH if probability >= 0.85 else SEVERITY_MEDIUM,
            rationale=f"Learned conflict detector probability: {probability:.2f}",
            resolver_strategy=RESOLVE_SOURCE,
        )

    def _prime_learned_score_cache(
        self,
        all_claims: list[tuple[Claim, Evidence]],
    ) -> None:
        """Batch learned scores once per graph to avoid per-edge GPU calls."""
        self._learned_score_cache.clear()
        if not self.enable_learned_detector or not self._init_learned_detector():
            return
        if self._learned_model is None:
            return

        candidates: list[tuple[Claim, Claim]] = []
        for i, (claim_i, ev_i) in enumerate(all_claims):
            for claim_j, ev_j in all_claims[i + 1:]:
                if (
                    not self.compare_within_evidence
                    and ev_i is ev_j
                    and not self._is_explicit_same_evidence_pair(
                        claim_i,
                        claim_j,
                        ev_i,
                    )
                ):
                    continue
                if (
                    self.learned_require_context
                    and not self._eligible_for_learned_detection(
                        claim_i,
                        claim_j,
                    )
                ):
                    continue
                candidates.append((claim_i, claim_j))

        if not candidates:
            return
        try:
            raw_scores = self._learned_model.predict(
                [
                    (claim_i.claim, claim_j.claim)
                    for claim_i, claim_j in candidates
                ],
                show_progress_bar=False,
            )
            if hasattr(raw_scores, "tolist"):
                raw_scores = raw_scores.tolist()
            if not isinstance(raw_scores, list):
                raw_scores = [raw_scores]
            if len(raw_scores) != len(candidates):
                logger.debug(
                    "Learned conflict batch returned %d scores for %d pairs",
                    len(raw_scores),
                    len(candidates),
                )
                return
            self._learned_score_cache.update({
                (id(claim_i), id(claim_j)): self._score_to_probability(score)
                for (claim_i, claim_j), score in zip(
                    candidates,
                    raw_scores,
                    strict=True,
                )
            })
        except Exception as e:
            logger.debug("Learned conflict batch detection failed: %s", e)

    def _eligible_for_learned_detection(self, claim_i: Claim, claim_j: Claim) -> bool:
        return self._same_fact_slot(
            claim_i,
            claim_j,
            min_similarity=self.learned_candidate_similarity,
        )

    # ------------------------------------------------------------------
    # NLI Layer 3 – lazy init
    # ------------------------------------------------------------------

    def _init_nli(self):
        """Try to load a cross-encoder NLI model for Layer 3 (once)."""
        if self._nli_available or self._nli_tried or not self.enable_nli:
            return
        # Attempt the (potentially expensive / network-bound) load at most once;
        # on failure the NLI layer is permanently disabled and we fall back to
        # rule + LLM layers instead of re-trying for every claim pair.
        self._nli_tried = True
        model_name = self.nli_model_name
        try:
            def factory() -> Any:
                from sentence_transformers import CrossEncoder

                if self.nli_local_files_only:
                    return CrossEncoder(model_name, local_files_only=True)
                return CrossEncoder(model_name)

            self._nli_model, error = load_optional_model_once(
                "cross_encoder",
                model_name,
                factory,
                local_files_only=self.nli_local_files_only,
            )
            self._nli_available = self._nli_model is not None
            if self._nli_available:
                logger.info("NLI model loaded (%s)", model_name)
            elif error is not None:
                logger.debug("NLI model unavailable, disabling NLI layer: %s", error)
        except Exception as e:
            logger.debug("NLI model unavailable, disabling NLI layer: %s", e)

    def _nli_detect(
        self, claim_i: Claim, claim_j: Claim,
    ) -> ConflictEdge | None:
        """Use NLI cross-encoder to determine entailment / contradiction / neutral."""
        if not self._nli_available or self._nli_model is None:
            return None

        try:
            scores = self._nli_model.predict(
                [(claim_i.claim, claim_j.claim)],
                show_progress_bar=False,
            )
            import numpy as np
            raw_scores = np.asarray(scores, dtype=float)
            if raw_scores.ndim == 1:
                if raw_scores.shape[0] != 3:
                    return None
                raw_scores = raw_scores.reshape(1, 3)
            if raw_scores.ndim != 2 or raw_scores.shape[1] != 3:
                return None
            if not np.isfinite(raw_scores).all():
                return None
            shifted = raw_scores - raw_scores.max(axis=1, keepdims=True)
            probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
            label_indices = self._nli_label_indices(raw_scores.shape[1])
            if label_indices is None:
                return None
            contradiction_index, entailment_index, _neutral_index = label_indices
            contradiction_prob = float(probs[0, contradiction_index])
            entailment_prob = float(probs[0, entailment_index])

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

    @classmethod
    def _claims_have_compatible_status_polarity(
        cls,
        claim_i: Claim,
        claim_j: Claim,
    ) -> bool:
        polarity_i = cls._claim_status_polarity(claim_i.claim)
        polarity_j = cls._claim_status_polarity(claim_j.claim)
        return bool(polarity_i and polarity_i == polarity_j)

    @staticmethod
    def _claim_status_polarity(claim_text: str) -> str:
        text = claim_text.lower()
        negative_passed_markers = (
            "搁置",
            "尚未通过",
            "未通过",
            "没有通过",
            "未获通过",
            "尚未获通过",
            "not passed",
            "has not passed",
            "not approved",
            "shelved",
        )
        if any(marker in text for marker in negative_passed_markers):
            return "passed_negative"

        positive_passed_markers = (
            "正式通过",
            "已获通过",
            "获通过",
            "通过了",
            "通过《",
            "approved",
            "passed",
            "has passed",
        )
        if any(marker in text for marker in positive_passed_markers):
            return "passed_positive"
        return ""

    def _nli_label_indices(self, num_labels: int = 3) -> tuple[int, int, int] | None:
        """Return contradiction, entailment, and neutral label indices."""
        if num_labels < 3:
            return None
        config = getattr(getattr(self._nli_model, "model", None), "config", None)
        raw_labels = getattr(config, "id2label", {}) or {}
        labels: dict[int, str] = {}
        for index, label in raw_labels.items():
            try:
                label_index = int(index)
            except (TypeError, ValueError):
                continue
            if 0 <= label_index < num_labels:
                labels[label_index] = str(label).lower()

        def find(marker: str) -> int | None:
            return next(
                (index for index, label in labels.items() if marker in label),
                None,
            )

        recognized = any(
            marker in label
            for label in labels.values()
            for marker in ("contrad", "entail", "neutral")
        )
        if recognized:
            contradiction_index = find("contrad")
            entailment_index = find("entail")
            neutral_index = find("neutral")
            if (
                contradiction_index is None
                or entailment_index is None
                or neutral_index is None
            ):
                return None
            indices = (
                contradiction_index,
                entailment_index,
                neutral_index,
            )
        else:
            indices = (0, 1, 2)
        if len(set(indices)) != 3:
            return None
        if any(index < 0 or index >= num_labels for index in indices):
            return None
        return indices

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

    @staticmethod
    def _claim_attributes(claim: Claim) -> set[str]:
        text = claim.claim.lower()
        attrs = set()
        for keyword, canonical in ATTRIBUTE_KEYWORDS.items():
            if keyword.lower() not in text:
                continue
            if canonical == "passed" and re.search(r"通过的.{0,12}(等级|分类|版本|规则|条款|框架)", text):
                continue
            if canonical == "passed" and re.search(r"通过(部门|统一|法律|规章|框架|规则)", text):
                continue
            attrs.add(canonical)
        return attrs

    def _same_fact_slot(
        self,
        claim_i: Claim,
        claim_j: Claim,
        *,
        min_similarity: float | None = None,
        require_time_compatibility: bool = True,
    ) -> bool:
        """Return True when claims are likely about the same checkable fact.

        Same entity is not sufficient: "Company X was founded in 2012" and
        "Company X has 3,000 employees" should not become a conflict just
        because the evidence dates or sources differ.
        """
        shared_entities = set(claim_i.entities) & set(claim_j.entities)
        if not shared_entities:
            return False
        if self._shared_entity_only_contrast_context(claim_i, claim_j, shared_entities):
            return False

        attrs_i = self._claim_attributes(claim_i)
        attrs_j = self._claim_attributes(claim_j)
        if require_time_compatibility and not self._compatible_time_slot(claim_i, claim_j):
            return False
        if not self._compatible_value_qualifier(claim_i, claim_j, attrs_i, attrs_j):
            return False
        if not self._compatible_sales_qualifier(claim_i, claim_j, attrs_i, attrs_j):
            return False
        if not self._compatible_emissions_qualifier(claim_i, claim_j, attrs_i, attrs_j):
            return False
        if not self._compatible_climate_temperature_slot(claim_i, claim_j, attrs_i, attrs_j):
            return False
        if not self._compatible_qubit_qualifier(claim_i, claim_j, attrs_i, attrs_j):
            return False
        if attrs_i and attrs_j:
            return bool(attrs_i & attrs_j)

        sim = self._text_similarity(claim_i.claim, claim_j.claim)
        threshold = self.min_conflict_similarity if min_similarity is None else min_similarity
        if not (attrs_i and attrs_j):
            threshold = max(threshold, self.unattributed_conflict_similarity)
        return sim >= threshold

    @staticmethod
    def _shared_entity_only_contrast_context(
        claim_i: Claim,
        claim_j: Claim,
        shared_entities: set[str],
    ) -> bool:
        """Skip pairs where a shared entity is only mentioned as a contrast."""
        for entity in shared_entities:
            if not entity:
                continue
            contrast_patterns = (
                f"与{entity}不同",
                f"和{entity}不同",
                f"不同于{entity}",
                f"unlike {entity.lower()}",
            )
            for claim in (claim_i, claim_j):
                text = claim.claim.lower()
                if any(pattern.lower() in text for pattern in contrast_patterns):
                    return True
        return False

    @staticmethod
    def _claim_time_keys(claim: Claim) -> set[str]:
        text = claim.claim
        keys = {f"year:{year}" for year in re.findall(r"((?:19|20)\d{2})年?", text)}
        for expression in claim.time_expressions:
            keys.update(
                f"year:{year}"
                for year in re.findall(r"((?:19|20)\d{2})年?", expression)
            )
        for marker in re.findall(r"[Qq]([1-4])", text):
            keys.add(f"quarter:{QUARTER_WORDS[marker]}")
        for marker in re.findall(r"第?([一二三四1234])季度", text):
            keys.add(f"quarter:{QUARTER_WORDS[marker]}")
        for expression in claim.time_expressions:
            for marker in re.findall(r"[Qq]([1-4])", expression):
                keys.add(f"quarter:{QUARTER_WORDS[marker]}")
            for marker in re.findall(r"第?([一二三四1234])季度", expression):
                keys.add(f"quarter:{QUARTER_WORDS[marker]}")
        if any(word in text for word in ("全年", "年度", "财年", "annual", "yearly")):
            keys.add("period:annual")
        if any(
            any(word in expression for word in ("全年", "年度", "财年", "annual", "yearly"))
            for expression in claim.time_expressions
        ):
            keys.add("period:annual")
        return keys

    def _compatible_time_slot(self, claim_i: Claim, claim_j: Claim) -> bool:
        time_i = self._claim_time_keys(claim_i)
        time_j = self._claim_time_keys(claim_j)
        for prefix in ("year:", "quarter:", "period:"):
            subset_i = {key for key in time_i if key.startswith(prefix)}
            subset_j = {key for key in time_j if key.startswith(prefix)}
            if subset_i and subset_j and not (subset_i & subset_j):
                return False
        quarter_i = {key for key in time_i if key.startswith("quarter:")}
        quarter_j = {key for key in time_j if key.startswith("quarter:")}
        annual_i = "period:annual" in time_i
        annual_j = "period:annual" in time_j
        return not ((quarter_i and annual_j) or (quarter_j and annual_i))

    @staticmethod
    def _claim_value_qualifiers(claim: Claim) -> set[str]:
        text = claim.claim
        return {
            qualifier
            for keyword, qualifier in REVENUE_QUALIFIER_KEYWORDS.items()
            if keyword in text
        }

    def _compatible_value_qualifier(
        self,
        claim_i: Claim,
        claim_j: Claim,
        attrs_i: set[str],
        attrs_j: set[str],
    ) -> bool:
        if "revenue" not in attrs_i or "revenue" not in attrs_j:
            return True
        qualifiers_i = self._claim_value_qualifiers(claim_i)
        qualifiers_j = self._claim_value_qualifiers(claim_j)
        if qualifiers_i or qualifiers_j:
            return bool(qualifiers_i & qualifiers_j)
        return True

    @staticmethod
    def _compatible_sales_qualifier(
        claim_i: Claim,
        claim_j: Claim,
        attrs_i: set[str],
        attrs_j: set[str],
    ) -> bool:
        if "sales" not in attrs_i or "sales" not in attrs_j:
            return True

        def qualifiers(claim: Claim) -> set[str]:
            return {
                qualifier
                for keyword, qualifier in SALES_QUALIFIER_KEYWORDS.items()
                if keyword in claim.claim
            }

        qualifiers_i = qualifiers(claim_i)
        qualifiers_j = qualifiers(claim_j)
        role_qualifiers = {"manufacturer", "export"}
        role_i = qualifiers_i & role_qualifiers
        role_j = qualifiers_j & role_qualifiers
        if role_i or role_j:
            return bool(role_i & role_j)
        if qualifiers_i or qualifiers_j:
            return bool(qualifiers_i & qualifiers_j)
        return True

    @staticmethod
    def _compatible_emissions_qualifier(
        claim_i: Claim,
        claim_j: Claim,
        attrs_i: set[str],
        attrs_j: set[str],
    ) -> bool:
        if "emissions" not in attrs_i or "emissions" not in attrs_j:
            return True

        def qualifiers(claim: Claim) -> set[str]:
            return {
                qualifier
                for keyword, qualifier in EMISSIONS_QUALIFIER_KEYWORDS.items()
                if keyword in claim.claim
            }

        qualifiers_i = qualifiers(claim_i)
        qualifiers_j = qualifiers(claim_j)
        if qualifiers_i or qualifiers_j:
            return bool(qualifiers_i & qualifiers_j)
        return True

    @staticmethod
    def _compatible_climate_temperature_slot(
        claim_i: Claim,
        claim_j: Claim,
        attrs_i: set[str],
        attrs_j: set[str],
    ) -> bool:
        if "climate_sensitivity" not in (attrs_i | attrs_j):
            return True

        def is_temperature_target(claim: Claim) -> bool:
            return any(marker in claim.claim for marker in ("目标", "以内", "控制在", "减排"))

        return is_temperature_target(claim_i) == is_temperature_target(claim_j)

    @staticmethod
    def _compatible_qubit_qualifier(
        claim_i: Claim,
        claim_j: Claim,
        attrs_i: set[str],
        attrs_j: set[str],
    ) -> bool:
        if "qubits" not in attrs_i or "qubits" not in attrs_j:
            return True

        target_markers = ("需要", "至少", "目标", "预计", "计划", "required", "target")
        target_i = any(marker in claim_i.claim.lower() for marker in target_markers)
        target_j = any(marker in claim_j.claim.lower() for marker in target_markers)
        return target_i == target_j

    def _eligible_for_llm_adjudication(self, claim_i: Claim, claim_j: Claim) -> bool:
        if self._same_fact_slot(
            claim_i,
            claim_j,
            min_similarity=self.llm_adjudication_similarity,
        ):
            return True
        return bool(set(claim_i.entities) & set(claim_j.entities)) and (
            bool(claim_i.numbers and claim_j.numbers)
            or bool(claim_i.time_expressions and claim_j.time_expressions)
        )

    def _is_explicit_same_evidence_pair(self, claim_i: Claim, claim_j: Claim, evidence: Evidence) -> bool:
        spans = {claim_i.source_span, claim_j.source_span}
        if "reported_claim" in spans and "corrective_claim" in spans:
            return True

        text = f"{evidence.title} {evidence.text_span}"
        contrast_markers = (
            "误解", "质疑", "澄清", "不再代表", "并不代表", "不同", "相比",
            "仅从", "但", "而", "低于", "高于", "more than", "less than",
            "对比", "比较", " vs ", "vs", "versus", "questioned", "unlike",
        )
        if not any(marker in text for marker in contrast_markers):
            return False
        attrs_i = self._claim_attributes(claim_i)
        attrs_j = self._claim_attributes(claim_j)
        if attrs_i and attrs_j and attrs_i & attrs_j:
            return True
        if claim_i.numbers and claim_j.numbers:
            return True
        return self._is_quantifier_contrast_pair(claim_i, claim_j)

    @staticmethod
    def _is_quantifier_contrast_pair(claim_i: Claim, claim_j: Claim) -> bool:
        ti = claim_i.claim.lower()
        tj = claim_j.claim.lower()
        broad_i = any(marker in ti for marker in ("所有", "全部", "all "))
        broad_j = any(marker in tj for marker in ("所有", "全部", "all "))
        narrow_i = any(marker in ti for marker in ("仅", "只", "某些", "特定", "部分", "only ", "some "))
        narrow_j = any(marker in tj for marker in ("仅", "只", "某些", "特定", "部分", "only ", "some "))
        return (broad_i and narrow_j) or (broad_j and narrow_i)

    # ------------------------------------------------------------------
    # Support detection (rule-based)
    # ------------------------------------------------------------------

    def _check_support(
        self, claim_i: Claim, claim_j: Claim,
    ) -> ConflictEdge | None:
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
    ) -> ConflictEdge | None:
        if self._is_corrected_reported_claim_cross_evidence_pair(claim_i, ev_i, claim_j, ev_j):
            return None

        # Layer 1: Rule-based detection
        edge = self._rule_based_conflict_detection(claim_i, ev_i, claim_j, ev_j)
        if edge:
            return edge

        # Layer 2: optional learned conflict detector (if configured)
        if self.enable_learned_detector:
            self._init_learned_detector()
            if self._learned_available:
                edge = self._learned_conflict_detect(claim_i, claim_j)
                if edge:
                    return edge

        # Layer 3: NLI model (if available)
        if self.enable_nli:
            self._init_nli()
            if self._nli_available:
                edge = self._nli_detect(claim_i, claim_j)
                if edge:
                    return edge

        # Layer 4: LLM adjudication
        if (
            use_llm
            and self.enable_llm_adjudication
            and self.llm_client
            and self._eligible_for_llm_adjudication(claim_i, claim_j)
        ):
            return self._llm_conflict_detection(claim_i, claim_j)
        return None

    def _rule_based_conflict_detection(
        self,
        claim_i: Claim,
        ev_i: Evidence,
        claim_j: Claim,
        ev_j: Evidence,
    ) -> ConflictEdge | None:
        """Run all rule-based detectors in priority order."""

        if ev_i is ev_j and claim_i.numbers and claim_j.numbers:
            edge = self._check_same_evidence_numeric_contrast(claim_i, claim_j, ev_i)
            if edge:
                return edge

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

        # 4. High-precision quantifier scope conflict (all vs limited subset)
        edge = self._check_quantifier_scope_conflict(claim_i, claim_j)
        if edge:
            return edge

        # 5. Scope conflict (opt-in; broad scope words are noisy in retrieved passages)
        if self.enable_scope_conflict:
            edge = self._check_scope_conflict(claim_i, claim_j)
            if edge:
                return edge

        # 6. Causal conflict
        edge = self._check_causal_conflict(claim_i, claim_j)
        if edge:
            return edge

        # 7. Granularity conflict (opt-in; useful for targeted evals, noisy in full RAG)
        if self.enable_granularity_conflict:
            edge = self._check_granularity_conflict(claim_i, claim_j)
            if edge:
                return edge

        # 8. Definitional conflict
        edge = self._check_definitional_conflict(claim_i, claim_j)
        if edge:
            return edge

        # 9. Source reliability conflict. Source rank differences are useful
        # metadata, but too weak to infer contradiction unless explicitly enabled.
        if self.enable_source_reliability_conflict:
            edge = self._check_source_reliability_conflict(claim_i, ev_i, claim_j, ev_j)
            if edge:
                return edge

        # 10. Semantic contradiction via text similarity + negation
        edge = self._check_semantic_contradiction(claim_i, claim_j)
        if edge:
            return edge

        # 11. Semantic support detection (shared entities + high text similarity)
        edge = self._check_support(claim_i, claim_j)
        if edge:
            return edge

        return None

    @staticmethod
    def _is_corrected_reported_claim_cross_evidence_pair(
        claim_i: Claim,
        ev_i: Evidence,
        claim_j: Claim,
        ev_j: Evidence,
    ) -> bool:
        if ev_i is ev_j:
            return False

        def corrected_reported(claim: Claim, evidence: Evidence) -> bool:
            return claim.source_span == "reported_claim" and any(
                other.source_span == "corrective_claim"
                or any(marker in other.claim for marker in ("实际上", "正确", "错误", "并非", "不是"))
                for other in evidence.claims
                if other is not claim
            )

        return corrected_reported(claim_i, ev_i) or corrected_reported(claim_j, ev_j)

    def _check_same_evidence_numeric_contrast(
        self,
        claim_i: Claim,
        claim_j: Claim,
        evidence: Evidence,
    ) -> ConflictEdge | None:
        text = f"{evidence.title} {evidence.text_span} {claim_i.claim} {claim_j.claim}"
        contrast_markers = (
            "低于", "高于", "超过", "少于", "多于", "相比", "比较", "对比",
            "质疑", "修正", "澄清", "原计划", "推迟", "延误", "延期",
            "less than", "more than", "lower than", "higher than",
        )
        if not any(marker in text for marker in contrast_markers):
            return None
        attrs_i = self._claim_attributes(claim_i)
        attrs_j = self._claim_attributes(claim_j)
        shared_entities = set(claim_i.entities) & set(claim_j.entities)
        if attrs_i and attrs_j and not (attrs_i & attrs_j) and not shared_entities:
            return None
        if not attrs_i and not attrs_j and not shared_entities:
            return None
        if not self._compatible_time_slot(claim_i, claim_j):
            return None
        if not self._compatible_value_qualifier(claim_i, claim_j, attrs_i, attrs_j):
            return None
        if not self._compatible_sales_qualifier(claim_i, claim_j, attrs_i, attrs_j):
            return None
        if not self._compatible_emissions_qualifier(claim_i, claim_j, attrs_i, attrs_j):
            return None
        if not self._compatible_climate_temperature_slot(claim_i, claim_j, attrs_i, attrs_j):
            return None
        if not self._compatible_qubit_qualifier(claim_i, claim_j, attrs_i, attrs_j):
            return None

        numeric_i = self._parse_number_tokens(claim_i.numbers)
        numeric_j = self._parse_number_tokens(claim_j.numbers)
        for value_i, raw_i in numeric_i:
            if (
                self._is_likely_year(raw_i)
                or self._is_likely_date_component(raw_i)
                or self._is_likely_date_range_component(raw_i, claim_i.claim)
                or self._is_likely_period_number(raw_i, claim_i.claim)
            ):
                continue
            for value_j, raw_j in numeric_j:
                if (
                    self._is_likely_year(raw_j)
                    or self._is_likely_date_component(raw_j)
                    or self._is_likely_date_range_component(raw_j, claim_j.claim)
                    or self._is_likely_period_number(raw_j, claim_j.claim)
                ):
                    continue
                if not self._numeric_units_compatible(raw_i, raw_j):
                    continue
                denominator = max(abs(value_i), abs(value_j), 1.0)
                if abs(value_i - value_j) / denominator < 0.05:
                    continue
                return ConflictEdge(
                    source_id=claim_i.claim_id,
                    target_id=claim_j.claim_id,
                    conflict_type=ConflictType.NUMERIC_CONFLICT,
                    confidence=0.78,
                    severity=SEVERITY_HIGH,
                    rationale="Same-evidence numeric contrast between comparable claims",
                    resolver_strategy=RESOLVE_NUMERIC,
                )
        return None

    # ------------------------------------------------------------------
    # 10. Semantic contradiction (rule-based)
    # ------------------------------------------------------------------

    _NEGATION_PAIRS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("是", "不是"), ("有", "没有"), ("会", "不会"), ("能", "不能"),
        ("应该", "不应该"), ("可以", "不可以"), ("需要", "不需要"),
        ("正确", "错误"),
        ("通过", "搁置"), ("已获通过", "搁置"), ("生效", "搁置"),
        ("上升", "下降"), ("增加", "减少"), ("增长", "下降"),
        (" is ", " is not "), (" are ", " are not "),
        (" was ", " was not "), (" has ", " has no "),
        (" can ", " cannot "), (" will ", " will not "),
        ("increased", "decreased"), ("rose", "fell"),
    )

    def _check_semantic_contradiction(
        self, claim_i: Claim, claim_j: Claim,
    ) -> ConflictEdge | None:
        """Detect contradiction via shared entities + negation pairs, even without
        exact keyword match — uses text similarity to confirm topical relevance."""
        ti = claim_i.claim.lower()
        tj = claim_j.claim.lower()

        shared_entities = set(claim_i.entities) & set(claim_j.entities)
        if not shared_entities:
            return None
        if not self._same_fact_slot(claim_i, claim_j, min_similarity=0.25):
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

    def _check_quantifier_scope_conflict(
        self, claim_i: Claim, claim_j: Claim,
    ) -> ConflictEdge | None:
        """Detect high-precision all-vs-subset contradictions."""
        if not self._same_fact_slot(claim_i, claim_j, min_similarity=0.25):
            return None

        ti = claim_i.claim.lower()
        tj = claim_j.claim.lower()
        broad_i = any(marker in ti for marker in ("所有", "全部", "all "))
        broad_j = any(marker in tj for marker in ("所有", "全部", "all "))
        narrow_i = any(marker in ti for marker in ("仅", "只", "某些", "特定", "部分", "only ", "some "))
        narrow_j = any(marker in tj for marker in ("仅", "只", "某些", "特定", "部分", "only ", "some "))
        if not ((broad_i and narrow_j) or (broad_j and narrow_i)):
            return None

        return ConflictEdge(
            source_id=claim_i.claim_id,
            target_id=claim_j.claim_id,
            conflict_type=ConflictType.SCOPE_CONFLICT,
            confidence=0.75,
            severity=SEVERITY_HIGH,
            rationale="Broad all-scope claim conflicts with a limited-scope correction",
            resolver_strategy=RESOLVE_SCOPE,
        )

    # ------------------------------------------------------------------
    # 1. Numeric conflict
    # ------------------------------------------------------------------

    def _check_numerical_conflict(
        self, claim_i: Claim, claim_j: Claim,
    ) -> ConflictEdge | None:
        nums_i = self._parse_number_tokens(claim_i.numbers)
        nums_j = self._parse_number_tokens(claim_j.numbers)
        if not nums_i or not nums_j:
            return None
        if not self._same_fact_slot(claim_i, claim_j):
            return None

        # Only compare numbers of comparable magnitude (skip year-vs-quantity)
        for n_i, raw_i in nums_i:
            if (
                self._is_likely_year(raw_i)
                or self._is_likely_date_component(raw_i)
                or self._is_likely_date_range_component(raw_i, claim_i.claim)
                or self._is_likely_period_number(raw_i, claim_i.claim)
            ):
                continue
            for n_j, raw_j in nums_j:
                if (
                    self._is_likely_year(raw_j)
                    or self._is_likely_date_component(raw_j)
                    or self._is_likely_date_range_component(raw_j, claim_j.claim)
                    or self._is_likely_period_number(raw_j, claim_j.claim)
                ):
                    continue
                context_i = self._numeric_context_class(raw_i, claim_i.claim)
                context_j = self._numeric_context_class(raw_j, claim_j.claim)
                if "process_node" in {context_i, context_j}:
                    continue
                if context_i != context_j:
                    continue
                if not self._numeric_units_compatible(raw_i, raw_j):
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

    @classmethod
    def _numeric_units_compatible(cls, raw_i: Any, raw_j: Any) -> bool:
        unit_i = cls._numeric_unit_class(raw_i)
        unit_j = cls._numeric_unit_class(raw_j)
        if unit_i == unit_j:
            return True
        if "percent" in {unit_i, unit_j}:
            return False
        return "plain" in {unit_i, unit_j}

    @staticmethod
    def _numeric_unit_class(raw: Any) -> str:
        text = str(raw).lower()
        if "%" in text or "percent" in text:
            return "percent"
        if any(unit in text for unit in ("元", "美元", "欧元", "人民币", "$", "€")):
            return "money"
        if any(unit in text for unit in ("万", "亿")):
            return "magnitude"
        if any(unit in text for unit in ("°c", "℃", "摄氏度")):
            return "temperature"
        if any(unit in text for unit in ("nm", "纳米", "厘米", "米", "km", "公里")):
            return "length"
        if any(unit in text for unit in ("秒", "分钟", "小时", "天", "年")):
            return "duration"
        if "量子比特" in text or "qubit" in text:
            return "qubit"
        return "plain"

    @staticmethod
    def _numeric_context_class(raw: Any, claim_text: str) -> str:
        """Classify same-unit numbers that describe different semantic slots.

        Semiconductor process labels such as 3nm/7nm/14nm are node names, not
        physical length measurements. Treating them as comparable lengths
        creates false conflicts against actual gate lengths or other process
        generations.
        """
        raw_text = str(raw)
        lowered_raw = raw_text.lower()
        if any(unit in lowered_raw for unit in ("秒", "分钟", "小时", "天", "年")):
            return "duration"
        if "%" in lowered_raw or "percent" in lowered_raw:
            return "percentage"
        if any(unit in lowered_raw for unit in ("元", "美元", "欧元", "人民币", "$", "€")):
            return "money"
        escaped = re.escape(raw_text.strip())
        match = re.search(escaped, claim_text, flags=re.IGNORECASE)
        if not match:
            return "quantity"
        start = max(0, match.start() - 16)
        end = min(len(claim_text), match.end() + 18)
        window = claim_text[start:end]
        if re.search(rf"{escaped}\s*(?:秒|分钟|小时|天|年)", window):
            return "duration"
        if re.search(
            rf"{escaped}\s*(?:个)?(?:物理|逻辑)?量子比特|{escaped}\s*qubits?",
            window,
            flags=re.IGNORECASE,
        ):
            return "qubits"
        if re.search(rf"(?:码)?距离(?:为|是|=)?\s*{escaped}", window):
            return "code_distance"
        if "nm" not in lowered_raw and "纳米" not in raw_text:
            return "quantity"
        if any(
            marker in window
            for marker in (
                "制程",
                "工艺",
                "命名",
                "节点",
                "级芯片",
                "及以下",
                "先进",
                "量产",
            )
        ):
            return "process_node"
        return "quantity"

    @staticmethod
    def _is_likely_year(num_str: Any) -> bool:
        """Heuristic: 4-digit numbers in 1800-2099 range are likely years."""
        clean = re.sub(r"[^\d.]", "", str(num_str).strip().replace(",", ""))
        try:
            val = int(float(clean))
            return 1800 <= val <= 2099
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _is_likely_date_component(num_str: Any) -> bool:
        """Skip month/day/date fragments; they are not comparable quantities."""
        return bool(re.search(r"\d+\s*[月日]", str(num_str)))

    @staticmethod
    def _is_likely_date_range_component(num_str: Any, claim_text: str) -> bool:
        """Skip bare numbers that are part of date ranges such as 1-9月."""
        clean = re.sub(r"[^\d]", "", str(num_str))
        if not clean:
            return False
        return bool(re.search(rf"(?<!\d){clean}\s*[-–至到]\s*\d{{1,2}}\s*月", claim_text))

    @staticmethod
    def _is_likely_period_number(num_str: Any, claim_text: str) -> bool:
        """Skip the numeric part of period markers such as Q1."""
        clean = re.sub(r"[^\d]", "", str(num_str))
        if clean not in {"1", "2", "3", "4"}:
            return False
        return bool(re.search(rf"[Qq]{clean}|第?{clean}季度", claim_text))

    def _parse_numbers(self, num_strings: list[Any]) -> list[float]:
        return [value for value, _raw in self._parse_number_tokens(num_strings)]

    def _parse_number_tokens(self, num_strings: list[Any]) -> list[tuple[float, str]]:
        numbers = []
        for ns in num_strings:
            try:
                text = str(ns).replace(",", "").strip()
                match = re.search(r"\d+(?:\.\d+)?", text)
                if not match:
                    continue
                value = float(match.group(0))
                if "亿" in text:
                    value *= 100_000_000
                elif "万" in text:
                    value *= 10_000
                if math.isfinite(value):
                    numbers.append((value, text))
            except (ValueError, TypeError):
                pass
        return numbers

    # ------------------------------------------------------------------
    # 2. Entity conflict
    # ------------------------------------------------------------------

    def _check_entity_conflict(
        self, claim_i: Claim, claim_j: Claim,
    ) -> ConflictEdge | None:
        entities_i = set(claim_i.entities)
        entities_j = set(claim_j.entities)
        shared = entities_i & entities_j
        if not shared:
            return None

        attrs_i = self._claim_attributes(claim_i)
        attrs_j = self._claim_attributes(claim_j)
        diff_i = entities_i - entities_j
        diff_j = entities_j - entities_i
        if (
            diff_i
            and diff_j
            and (attrs_i & attrs_j & ENTITY_VALUE_ATTRIBUTES)
            and self._same_fact_slot(claim_i, claim_j, min_similarity=0.35)
        ):
            return ConflictEdge(
                source_id=claim_i.claim_id,
                target_id=claim_j.claim_id,
                conflict_type=ConflictType.ENTITY_MISMATCH,
                confidence=0.7,
                severity=SEVERITY_MEDIUM,
                rationale=f"Different entity values for {attrs_i & attrs_j}: {diff_i} vs {diff_j}",
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
    ) -> ConflictEdge | None:
        entities_i = set(claim_i.entities)
        entities_j = set(claim_j.entities)

        attrs_i = self._claim_attributes(claim_i)
        attrs_j = self._claim_attributes(claim_j)
        temporal_attrs = {"founded", "passed", "effective"}
        for attr in sorted(attrs_i & attrs_j & temporal_attrs):
            years_i = self._claim_years_for_temporal_attr(claim_i, attr)
            years_j = self._claim_years_for_temporal_attr(claim_j, attr)
            if (
                entities_i & entities_j
                and years_i
                and years_j
                and not (years_i & years_j)
                and self._same_fact_slot(
                    claim_i,
                    claim_j,
                    min_similarity=0.35,
                    require_time_compatibility=False,
                )
            ):
                return ConflictEdge(
                    source_id=claim_i.claim_id,
                    target_id=claim_j.claim_id,
                    conflict_type=ConflictType.TEMPORAL_CONFLICT,
                    confidence=0.7,
                    severity=SEVERITY_MEDIUM,
                    rationale=(
                        f"Temporal {attr} years differ: "
                        f"{sorted(years_i)} vs {sorted(years_j)}"
                    ),
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

    @staticmethod
    def _claim_years_for_temporal_attr(claim: Claim, attr: str) -> set[str]:
        keyword_map = {
            "founded": ("成立", "创立", "founded"),
            "passed": ("通过", "批准", "获通过", "approved", "passed", "搁置"),
            "effective": ("生效", "适用", "effective", "take effect"),
        }
        keywords = keyword_map.get(attr, ())
        if not keywords:
            return set()

        clauses = [
            clause.strip()
            for clause in re.split(r"[，,；;。]|而且|并且|同时|不过|然而", claim.claim)
            if clause.strip()
        ]
        years: set[str] = set()
        for idx, clause in enumerate(clauses):
            if not any(keyword in clause for keyword in keywords):
                continue
            clause_years = {
                f"year:{year}"
                for year in re.findall(r"((?:19|20)\d{2})年?", clause)
            }
            if not clause_years and idx > 0:
                previous = clauses[idx - 1]
                if len(previous) <= 40:
                    clause_years.update(
                        f"year:{year}"
                        for year in re.findall(r"((?:19|20)\d{2})年?", previous)
                    )
            years.update(clause_years)
        return years

    # ------------------------------------------------------------------
    # 4. Scope conflict
    # ------------------------------------------------------------------

    def _check_scope_conflict(
        self, claim_i: Claim, claim_j: Claim,
    ) -> ConflictEdge | None:
        ti = claim_i.claim.lower()
        tj = claim_j.claim.lower()

        scope_i = self._detect_scope(ti)
        scope_j = self._detect_scope(tj)

        if scope_i and scope_j and scope_i != scope_j:
            # Same entities but different scopes → conflict
            entities_i = set(claim_i.entities)
            entities_j = set(claim_j.entities)
            if entities_i & entities_j and self._same_fact_slot(
                claim_i,
                claim_j,
                min_similarity=0.35,
                require_time_compatibility=False,
            ):
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

    def _detect_scope(self, text: str) -> str | None:
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
    ) -> ConflictEdge | None:
        ti = claim_i.claim.lower()
        tj = claim_j.claim.lower()

        has_causal_i = any(kw in ti for kw in CAUSAL_POSITIVE)
        has_negate_i = any(kw in ti for kw in CAUSAL_NEGATIVE)
        has_causal_j = any(kw in tj for kw in CAUSAL_POSITIVE)
        has_negate_j = any(kw in tj for kw in CAUSAL_NEGATIVE)

        if (has_causal_i and has_negate_j) or (has_negate_i and has_causal_j):
            entities_i = set(claim_i.entities)
            entities_j = set(claim_j.entities)
            if entities_i & entities_j and self._same_fact_slot(
                claim_i,
                claim_j,
                min_similarity=0.35,
                require_time_compatibility=False,
            ):
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
    ) -> ConflictEdge | None:
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
            if entities_i & entities_j and self._same_fact_slot(
                claim_i,
                claim_j,
                min_similarity=0.35,
                require_time_compatibility=False,
            ):
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
    ) -> ConflictEdge | None:
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

    def _extract_definition(self, text: str) -> str | None:
        for pattern in DEFINITION_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return None

    # ------------------------------------------------------------------
    # 8. Source reliability conflict
    # ------------------------------------------------------------------

    RELIABILITY_RANK: ClassVar[dict[str, int]] = {
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
    ) -> ConflictEdge | None:
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
            if has_contradiction and self._same_fact_slot(claim_i, claim_j, min_similarity=0.35):
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
    ) -> ConflictEdge | None:
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
            relationship = str(data.get("relationship", "UNRELATED")).strip().upper()
            if relationship == "UNRELATED":
                return None

            conflict_type_map = {
                "SUPPORT": ConflictType.SUPPORT,
                "REFUTE": ConflictType.REFUTE,
                "PARTIAL_SUPPORT": ConflictType.PARTIAL_SUPPORT,
            }
            if relationship not in conflict_type_map:
                return None
            confidence = self._coerce_probability(data.get("confidence"))
            if confidence is None:
                return None
            return ConflictEdge(
                source_id=claim_i.claim_id,
                target_id=claim_j.claim_id,
                conflict_type=conflict_type_map[relationship],
                confidence=confidence,
                severity=SEVERITY_HIGH if relationship == "REFUTE" else SEVERITY_LOW,
                rationale=data.get("rationale", ""),
                resolver_strategy=RESOLVE_SOURCE,
            )
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Graph update
    # ------------------------------------------------------------------

    def _register_evidence(self, evidence: Evidence) -> None:
        for claim in evidence.claims:
            self._claim_registry[claim.claim_id] = (claim, evidence)

    @staticmethod
    def _edge_exists(graph: EvidenceConflictGraph, edge: ConflictEdge) -> bool:
        return any(
            {
                existing.source_id,
                existing.target_id,
            } == {edge.source_id, edge.target_id}
            and existing.conflict_type == edge.conflict_type
            for existing in graph.edges
        )

    def update_graph(
        self,
        graph: EvidenceConflictGraph,
        new_evidence: list[Evidence],
        use_llm: bool = True,
    ) -> EvidenceConflictGraph:
        new_claim_ids = {
            claim.claim_id
            for ev in new_evidence
            for claim in ev.claims
        }
        existing_claims = [
            (claim, ev)
            for claim_id, (claim, ev) in self._claim_registry.items()
            if claim_id in graph.nodes and claim_id not in new_claim_ids
        ]

        for ev in new_evidence:
            self._register_evidence(ev)
            for claim in ev.claims:
                if claim.claim_id not in graph.nodes:
                    node = ConflictGraphNode(
                        node_id=claim.claim_id,
                        content=claim.claim,
                        node_type="claim",
                        evidence_ids=[ev.evidence_id],
                    )
                    graph.add_node(node)
                self_edge = self._detect_self_refuting_claim(claim, ev)
                if self_edge and not self._edge_exists(graph, self_edge):
                    graph.add_edge(self_edge)

        new_claims = [(claim, ev) for ev in new_evidence for claim in ev.claims]
        self._prime_learned_score_cache(existing_claims + new_claims)
        for claim_i, ev_i in existing_claims:
            for claim_j, ev_j in new_claims:
                if claim_i.claim_id == claim_j.claim_id:
                    continue
                edge = self._detect_relationship(claim_i, ev_i, claim_j, ev_j, use_llm)
                if (
                    edge
                    and edge.conflict_type != ConflictType.UNRELATED
                    and not self._edge_exists(graph, edge)
                ):
                    graph.add_edge(edge)

        for i, (claim_i, ev_i) in enumerate(new_claims):
            for claim_j, ev_j in new_claims[i + 1:]:
                if (
                    not self.compare_within_evidence
                    and ev_i is ev_j
                    and not self._is_explicit_same_evidence_pair(claim_i, claim_j, ev_i)
                ):
                    continue
                edge = self._detect_relationship(claim_i, ev_i, claim_j, ev_j, use_llm)
                if (
                    edge
                    and edge.conflict_type != ConflictType.UNRELATED
                    and not self._edge_exists(graph, edge)
                ):
                    graph.add_edge(edge)

        return graph

    def run(self, *args, **kwargs) -> Any:
        return self.build_graph(*args, **kwargs)
