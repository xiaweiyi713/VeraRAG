"""VeraRAG Main Pipeline."""

import os
import time
from typing import Any

from ..agents.base import LLMClient
from ..agents.planner import DecompositionPlanner
from ..agents.reasoning_agent import ReasoningAgent
from ..agents.repair_agent import RepairAgent
from ..agents.retrieval_agent import DynamicRetrievalAgent
from ..agents.task_analyzer import TaskAnalyzer
from ..agents.verifier_agent import VerifierAgent
from ..evidence.conflict_graph import ConflictGraphBuilder
from ..evidence.extractor import EvidenceExtractor
from ..evidence.normalizer import EvidenceNormalizer
from ..retriever.hybrid import HybridRetriever
from ..uncertainty.controller import Action, UncertaintyController
from ..utils.data_structures import (
    Evidence,
    EvidenceConflictGraph,
    UncertaintyBreakdown,
    VeraRAGOutput,
)


class VeraRAG:
    """
    Main VeraRAG pipeline for verifiable agentic RAG.

    Pipeline stages:
    1. Task Analysis: Analyze the question
    2. Decomposition: Break into sub-questions
    3. Dynamic Retrieval: Multi-round evidence collection
    4. Evidence Normalization: Structure the evidence
    5. Conflict Graph Building: Model evidence relationships
    6. Uncertainty Assessment: Estimate uncertainty
    7. Reasoning: Generate answer with claims
    8. Verification: Check claims against evidence
    9. Repair: Fix any issues identified
    10. Final Output: Assemble result
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize VeraRAG pipeline.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self._setup_components()

    def _setup_components(self):
        """Setup all pipeline components."""
        # LLM Client
        llm_config = self.config.get("llm", {})
        raw_api_key = llm_config.get("api_key", "")
        if raw_api_key and raw_api_key.startswith("${") and raw_api_key.endswith("}"):
            env_var = raw_api_key[2:-1]
            raw_api_key = os.getenv(env_var, "")
        api_key = raw_api_key or os.getenv(llm_config.get("api_key_env", "OPENAI_API_KEY"))
        self.llm_client = LLMClient(
            provider=llm_config.get("provider", "openai"),
            model=llm_config.get("model", "gpt-4o"),
            api_key=api_key,
            base_url=llm_config.get("base_url"),
            temperature=llm_config.get("temperature", 0.7),
            max_tokens=llm_config.get("max_tokens", 2000)
        )

        # Retriever — fall back to BM25 if sentence-transformers not available
        retriever_config = self.config.get("retriever", {})
        try:
            self.retriever = HybridRetriever(
                config=retriever_config,
                sparse_weight=retriever_config.get("sparse_weight", 0.3),
                dense_weight=retriever_config.get("dense_weight", 0.7)
            )
        except ImportError:
            from ..retriever.bm25 import BM25Retriever
            self.retriever = BM25Retriever(config=retriever_config)
            import logging
            logging.getLogger("verarag").warning("sentence-transformers not installed, using BM25 only")

        # Agents
        self.task_analyzer = TaskAnalyzer(self.config, self.llm_client)
        self.planner = DecompositionPlanner(self.config, self.llm_client)
        self.retrieval_agent = DynamicRetrievalAgent(
            retriever=self.retriever,
            config=self.config,
            llm_client=self.llm_client
        )
        self.reasoning_agent = ReasoningAgent(self.config, self.llm_client)
        self.verifier_agent = VerifierAgent(self.config, self.llm_client)
        self.repair_agent = RepairAgent(self.config, self.llm_client)

        # Evidence processing
        self.evidence_extractor = EvidenceExtractor(self.config, self.llm_client)
        self.evidence_normalizer = EvidenceNormalizer(self.config)
        self.conflict_graph_builder = ConflictGraphBuilder(self.config, self.llm_client)

        # Uncertainty control
        uncertainty_config = self.config.get("uncertainty", {})
        self.uncertainty_controller = UncertaintyController(uncertainty_config)

        # Pipeline settings
        pipeline_config = self.config.get("pipeline", {})
        self.max_retrieval_rounds = pipeline_config.get("max_retrieval_rounds", 5)
        self.max_subquestions = pipeline_config.get("max_subquestions", 10)
        self.enable_conflict_graph = pipeline_config.get("enable_conflict_graph", True)
        self.enable_uncertainty = pipeline_config.get("enable_uncertainty", True)
        self.enable_verification = pipeline_config.get("enable_verification", True)
        self.enable_repair = pipeline_config.get("enable_repair", True)

    def index_documents(self, documents: list[dict[str, Any]]) -> None:
        """
        Build retrieval index from documents.

        Args:
            documents: List of documents with 'id' and 'text' fields
        """
        self.retriever.index_documents(documents)

    def query(
        self,
        question: str,
        max_rounds: int | None = None
    ) -> VeraRAGOutput:
        """
        Process a question through the VeraRAG pipeline.

        Args:
            question: The user's question
            max_rounds: Maximum retrieval rounds (overrides config)

        Returns:
            VeraRAGOutput with answer, evidence, and metadata
        """
        return self.query_stream(question, max_rounds=max_rounds, callback=None)

    def _normalize_evidence(self, evidence_list: list[Evidence]) -> list[Evidence]:
        """Normalize a list of evidence."""
        # Filter and deduplicate
        evidence_list = self.evidence_normalizer.filter_low_quality(evidence_list)
        evidence_list = self.evidence_normalizer.deduplicate(evidence_list)
        return evidence_list

    def _retrieval_result_to_evidence(self, result: Any) -> Evidence:
        """Convert retrieval result to evidence."""
        # Handle both RetrievalResult and Evidence types
        if hasattr(result, 'evidence_id'):
            return result  # type: ignore[no-any-return]

        from ..utils.data_structures import Evidence

        # Use chunked doc_id (e.g. D006_c0) as evidence_id for uniqueness
        # The evaluator maps D006_c0 → D006 → gold evidence_id via metadata
        chunk_id = result.doc_id if hasattr(result, 'doc_id') and result.doc_id else ""
        stable_id = chunk_id or result.metadata.get("doc_id", "")

        return Evidence(
            evidence_id=stable_id,
            source=result.metadata.get("source", "unknown"),
            title=result.title,
            text_span=result.content,
            url=result.metadata.get("url"),
            relevance_score=min(1.0, result.score)
        )

    def query_stream(
        self,
        question: str,
        max_rounds: int | None = None,
        callback=None
    ) -> VeraRAGOutput:
        """
        Process a question with streaming callbacks for each pipeline stage.

        Args:
            question: The user's question
            max_rounds: Maximum retrieval rounds (overrides config)
            callback: Callable(event_type: str, data: dict) called at each stage

        Returns:
            VeraRAGOutput with answer, evidence, and metadata
        """
        def emit(event_type: str, data: dict):
            if callback:
                callback(event_type, data)

        start_time = time.time()
        max_rounds = max_rounds or self.max_retrieval_rounds

        # Stage 1: Task Analysis
        emit("stage", {"stage": "task_analysis", "status": "started"})
        task_analysis = self.task_analyzer.analyze(question)
        emit("task_analysis", task_analysis.to_dict())

        # Stage 2: Decomposition
        emit("stage", {"stage": "decomposition", "status": "started"})
        subquestions = self.planner.decompose(
            question,
            task_analysis,
            self.max_subquestions
        )
        reasoning_plan = self.planner.get_reasoning_plan(question, subquestions)
        emit("decomposition", {
            "subquestions": [sq.to_dict() for sq in subquestions]
        })

        # Stage 3 & 4 & 5: Dynamic Retrieval, Normalization, Conflict Graph
        evidence_pool: list[Evidence] = []
        conflict_graph = EvidenceConflictGraph()
        prev_decision = None

        for round_id in range(max_rounds):
            emit("stage", {
                "stage": "retrieval",
                "round": round_id + 1,
                "total_rounds": max_rounds,
                "status": "started"
            })

            # Determine retrieval strategy based on previous uncertainty decision
            retrieve_budget = 50
            if round_id > 0 and prev_decision is not None:
                if prev_decision.action == Action.RESOLVE_CONFLICTS:
                    # Prioritize counter-evidence for conflicting claims
                    retrieve_budget = 30
                elif prev_decision.action == Action.CONTINUE_RETRIEVAL:
                    # Broader retrieval with higher budget
                    retrieve_budget = 80
            prev_decision = None

            evidence_pool = self.retrieval_agent.dynamic_retrieve(
                subquestions,
                evidence_pool,
                max_rounds=1,
                budget_per_round=retrieve_budget
            )

            if round_id == 0:
                evidence_pool = self._normalize_evidence(
                    [self._retrieval_result_to_evidence(r) for r in evidence_pool]
                )
                evidence_pool = self.evidence_normalizer.filter_low_quality(evidence_pool)

            emit("evidence", {
                "round": round_id + 1,
                "new_count": len(evidence_pool),
                "total": len(evidence_pool),
                "evidence": [e.to_dict() for e in evidence_pool[-5:]]
            })

            if self.enable_conflict_graph:
                # Extract claims from evidence if not already populated
                for ev in evidence_pool:
                    if not ev.claims:
                        ev.claims = self.evidence_extractor._extract_claims(ev.text_span)
                conflict_graph = self.conflict_graph_builder.build_graph(
                    evidence_pool,
                    use_llm=True
                )
                emit("conflict", {
                    "conflicts": len(conflict_graph.get_conflicts()),
                    "conflict_score": conflict_graph.get_conflict_score(),
                    "edges": [e.to_dict() for e in conflict_graph.edges[:10]]
                })

            if self.enable_uncertainty:
                decision = self.uncertainty_controller.assess(
                    subquestions,
                    evidence_pool,
                    conflict_graph,
                    current_round=round_id,
                    max_rounds=max_rounds
                )
                prev_decision = decision
                emit("uncertainty", {
                    "action": decision.action.value,
                    "confidence": decision.confidence,
                    "reason": decision.reason
                })

                if decision.should_stop:
                    break

        # Stage 6: Reasoning
        emit("stage", {"stage": "reasoning", "status": "started"})
        answer, answer_claims, reasoning_chain = self.reasoning_agent.reason(
            question,
            subquestions,
            evidence_pool,
            conflict_graph,
            reasoning_plan
        )
        emit("reasoning", {
            "answer": answer,
            "claims": [c.to_dict() for c in answer_claims],
            "steps": [r.to_dict() for r in reasoning_chain]
        })

        # Stage 7: Verification
        verification_report = None
        if self.enable_verification:
            emit("stage", {"stage": "verification", "status": "started"})
            verification_report = self.verifier_agent.verify_answer(
                answer,
                answer_claims,
                evidence_pool,
                conflict_graph
            )
            emit("verification", verification_report.to_dict())

        # Stage 8: Repair
        if self.enable_repair and verification_report and verification_report.has_critical_issues():
            emit("stage", {"stage": "repair", "status": "started"})
            answer, answer_claims = self.repair_agent.repair_answer(
                answer,
                answer_claims,
                verification_report,
                evidence_pool
            )

        # Stage 9: Final uncertainty
        uncertainty = UncertaintyBreakdown()
        final_confidence = 0.5

        if self.enable_uncertainty:
            uncertainty = self.uncertainty_controller.get_uncertainty_breakdown(
                subquestions,
                evidence_pool,
                conflict_graph
            )

            if verification_report:
                verification_conf = (
                    1.0 if verification_report.overall_status.value == "supported"
                    else 0.5
                )
                uncertainty = self.uncertainty_controller.estimator.estimate_for_answer(
                    final_confidence,
                    verification_conf,
                    uncertainty
                )

            final_confidence = self.uncertainty_controller.calibrator.calibrate_confidence(
                1.0 - uncertainty.overall,
                uncertainty
            )

        # Stage 10: Output
        elapsed_time = time.time() - start_time
        output = VeraRAGOutput(
            question=question,
            answer=answer,
            answer_claims=answer_claims,
            evidence=evidence_pool[:20],
            reasoning_chain=reasoning_chain,
            conflict_report=conflict_graph.to_dict(),
            verification_report=verification_report,
            confidence=final_confidence,
            uncertainty=uncertainty,
            metadata={
                "task_analysis": task_analysis.to_dict(),
                "num_subquestions": len(subquestions),
                "num_evidence": len(evidence_pool),
                "num_conflicts": len(conflict_graph.get_conflicts()),
                "elapsed_time": elapsed_time,
                "retrieval_rounds": round_id + 1
            }
        )

        emit("complete", {
            "elapsed_time": elapsed_time,
            "confidence": final_confidence,
            "num_evidence": len(evidence_pool),
            "num_conflicts": len(conflict_graph.get_conflicts())
        })

        return output

    def batch_query(
        self,
        questions: list[str],
        max_rounds: int | None = None
    ) -> list[VeraRAGOutput]:
        """
        Process multiple questions.

        Args:
            questions: List of questions
            max_rounds: Maximum retrieval rounds

        Returns:
            List of VeraRAGOutput objects
        """
        return [self.query(q, max_rounds) for q in questions]


def create_verarag(config_path: str | None = None) -> VeraRAG:
    """
    Factory function to create VeraRAG instance.

    Args:
        config_path: Optional path to config file

    Returns:
        VeraRAG instance
    """
    config = None

    if config_path:
        import yaml  # type: ignore[import-untyped]
        with open(config_path) as f:
            config = yaml.safe_load(f)

    return VeraRAG(config)
