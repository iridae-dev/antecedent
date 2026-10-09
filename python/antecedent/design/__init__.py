"""Design ranking: which study to run next, ranked on an explicit basis.

One entry point, :func:`rank_designs`, and one result, :class:`DesignRankingResult`, whose
``basis`` says what the order means:

* ``"identification"`` -- no decision declared: plans (:class:`Measurement`,
  :class:`Experiment`, :class:`Environment`, :class:`Sampling`) ranked by the probability that
  a query becomes identified under a :class:`StructurePrior`.
* ``"net_value"`` / ``"evsi"`` -- a :class:`DesignDecision` declared: studies
  (:class:`Candidate`) ranked by expected value of sample information, net of cost under a
  :class:`CostMap` (``"evsi"`` when no cost map is given). Identifiability is reported as a
  gate. The result exports a durable artifact that :func:`consume` replays::

    ranked = design.rank_designs(plans, prior=design.StructurePrior.uniform([True, False]))
    ranked = design.rank_designs(studies, decision=decision, signal=spec, cost_map=cost_map)
    print(ranked.explain())
    design.consume(ranked.export(), expected_identity=ranked.expectation())

:func:`rank_structural` is the no-model fallback ordering and :func:`evaluate_decision` scores
one decision with a Python utility callback.
"""

from __future__ import annotations

from .._native import DecisionEvaluation
from .._native import evaluate_decision_py as evaluate_decision
from ._declarations import (
    ARTIFACT_KIND,
    CALIBRATION,
    RESULT_LINK_ID,
    ActionUtility,
    BinomialSignal,
    Candidate,
    CandidateValue,
    ConsumedEntry,
    ConsumedRanking,
    CostMap,
    CostUnitsRefusal,
    DesignDecision,
    DesignRankingRefusal,
    DesignSearchReceipt,
    Expectation,
    ExternalLaw,
    ExternalSignal,
    GaussianMeanSignal,
    IntegrationReport,
    MonteCarlo,
    ProviderIdentity,
    RolloutResult,
    SignalProvider,
    SignalProviderRefusal,
    SignalSpec,
    SourceOverlapDiagnostics,
    SourceOverlapRefusal,
    StatePrior,
    consume,
    consume_rollout,
)
from .plans import (
    DesignPlan,
    Environment,
    Experiment,
    Measurement,
    Sampling,
    StructurePrior,
)
from .ranking import (
    Basis,
    ConstraintViolation,
    DesignRankingResult,
    GateEntry,
    IdentificationCandidate,
    IdentificationGate,
    evsi,
    rank_designs,
)
from .structural import (
    StructuralCandidate,
    StructuralEntry,
    StructuralRanking,
    rank_structural,
)

__all__ = [
    "ARTIFACT_KIND",
    "CALIBRATION",
    "RESULT_LINK_ID",
    "ActionUtility",
    "Basis",
    "BinomialSignal",
    "Candidate",
    "CandidateValue",
    "ConstraintViolation",
    "ConsumedEntry",
    "ConsumedRanking",
    "CostMap",
    "CostUnitsRefusal",
    "DecisionEvaluation",
    "DesignDecision",
    "DesignPlan",
    "DesignRankingRefusal",
    "DesignRankingResult",
    "DesignSearchReceipt",
    "Environment",
    "Expectation",
    "Experiment",
    "ExternalLaw",
    "ExternalSignal",
    "GateEntry",
    "GaussianMeanSignal",
    "IdentificationCandidate",
    "IdentificationGate",
    "IntegrationReport",
    "Measurement",
    "MonteCarlo",
    "ProviderIdentity",
    "Sampling",
    "SignalProvider",
    "SignalProviderRefusal",
    "SignalSpec",
    "SourceOverlapDiagnostics",
    "SourceOverlapRefusal",
    "StatePrior",
    "StructuralCandidate",
    "StructuralEntry",
    "StructuralRanking",
    "StructurePrior",
    "consume",
    "consume_rollout",
    "RolloutResult",
    "evaluate_decision",
    "evsi",
    "rank_designs",
    "rank_structural",
]
