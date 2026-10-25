"""Value-of-information requests compiled against, and evaluated for, a study.

:func:`~antecedent.design.rank_designs` with a ``decision=`` ranks candidate studies by EVSI
over the decision's own state belief. These functions bind the same request to a **study**: a
binary average effect of ``treatment`` on ``outcome`` over named columns and a DAG, whose
inference publishes a parameter posterior. The study is compiled first, as the state belief of
the declared evaluation:

* :func:`compile_for_study` compiles and nothing more: no model is fitted and nothing is
  simulated. The lowered program carries the decision, the demanded carrier (a parameter
  posterior), the candidates in canonical order, the prior observations, the ranking schedule,
  the seed and tolerance, and the quoted Monte Carlo store. A study whose inference is a
  sampling realization is refused (a sampling law is not a state belief) and the refusal names
  the Bayesian realization; a store the admission pool cannot hold is refused before any
  allocation;
* :func:`evaluate_for_study` runs that compile first, then the evaluation, and binds the sealed
  ranking to the lowered program's identity.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import _native
from .._refusal import raise_refusal
from ..errors import CausalTypeError, CausalValueError
from ._declarations import (
    Candidate,
    CostMap,
    DesignDecision,
    MonteCarlo,
    SignalSpec,
    _candidate_from_wire,
    _cost_map,
    _refusal_type,
    _request_wire,
    _search,
)
from .ranking import DesignRankingResult, _ValueBody

__all__ = [
    "RankingStudy",
    "StudyProgram",
    "StudyRanking",
    "compile_for_study",
    "evaluate_for_study",
]


@dataclass(frozen=True)
class RankingStudy:
    """The study a value-of-information request is compiled against.

    ``data`` maps column names to numeric arrays, ``edges`` are the DAG's directed edges as
    ``(cause, effect)`` names, and the query is the binary average effect of ``treatment`` on
    ``outcome``. ``posterior=True`` declares conjugate Bayesian inference (``n_draws`` draws under
    an isotropic prior of scale ``prior_scale``); ``posterior=False`` leaves the default sampling
    realization, which publishes no parameter posterior and is refused by the compile.
    """

    data: Mapping[str, Any]
    edges: tuple[tuple[str, str], ...]
    treatment: str
    outcome: str
    posterior: bool = True
    n_draws: int = 256
    prior_scale: float = 10.0

    def _arguments(self) -> tuple[list[str], list[list[float]], list[tuple[str, str]], str, str]:
        if not isinstance(self.data, Mapping) or not self.data:
            raise CausalTypeError("data must be a non-empty mapping of column name to array")
        names = [str(name) for name in self.data]
        columns = []
        for name in names:
            values = np.asarray(self.data[name], dtype=np.float64)
            if values.ndim != 1:
                raise CausalValueError(f"column {name!r} must be one-dimensional")
            columns.append([float(v) for v in values])
        return (
            names,
            columns,
            [(str(a), str(b)) for a, b in self.edges],
            str(self.treatment),
            str(self.outcome),
        )


@dataclass(frozen=True)
class StudyProgram:
    """The lowered program a study compiled for a value-of-information request."""

    program_identity: str
    carrier: str
    contract_identity: str | None
    utility_unit: str | None
    candidates: tuple[str, ...]
    prior_observations: tuple[str, ...]
    rng_seed: int
    mc_error_tolerance: float
    schedule: Mapping[str, int]
    """``min_batches``, ``max_batches``, ``batch_size`` and ``draws_per_candidate``."""
    store: Mapping[str, int]
    """The quoted Monte Carlo store: ``candidates``, ``draws``, ``tasks`` and ``total_bytes``."""


@dataclass(frozen=True)
class StudyRanking:
    """A value ranking bound to the lowered program of the study compiled for it."""

    result: DesignRankingResult
    lowered_program_identity: str


def _request(
    decision: DesignDecision,
    candidates: Sequence[Candidate],
    signal: SignalSpec | None,
    cost_map: CostMap | None,
    require_net_value: bool,
    prior_observations: Sequence[str],
    source_digests: Sequence[str],
    mc_error_tolerance: float,
    tie_tolerance: float,
    max_candidates: int,
    monte_carlo: MonteCarlo | None,
    rng_seed: int,
) -> str:
    if not isinstance(decision, DesignDecision):
        raise CausalTypeError("decision must be a design.DesignDecision")
    studies = list(candidates)
    if not studies or not all(isinstance(c, Candidate) for c in studies):
        raise CausalTypeError("candidates must be a non-empty sequence of design.Candidate")
    wire = _request_wire(
        decision,
        studies,
        signal=signal,
        cost_map=cost_map,
        require_net_value=require_net_value,
        prior_observations=prior_observations,
        source_digests=source_digests,
        rng_seed=rng_seed,
        mc_error_tolerance=mc_error_tolerance,
        tie_tolerance=tie_tolerance,
        max_candidates=max_candidates,
        monte_carlo=monte_carlo,
    )
    try:
        return json.dumps(wire, allow_nan=False)
    except ValueError as error:
        raise CausalValueError("a declaration contains a non-finite number") from error


def _study_options(study: RankingStudy) -> dict[str, Any]:
    if not isinstance(study, RankingStudy):
        raise CausalTypeError("study must be a RankingStudy")
    return {
        "posterior": bool(study.posterior),
        "n_draws": int(study.n_draws),
        "prior_scale": float(study.prior_scale),
    }


def compile_for_study(
    study: RankingStudy,
    decision: DesignDecision,
    candidates: Sequence[Candidate],
    *,
    signal: SignalSpec | None = None,
    cost_map: CostMap | None = None,
    require_net_value: bool = False,
    prior_observations: Sequence[str] = (),
    source_digests: Sequence[str] = (),
    mc_error_tolerance: float = 1e-3,
    tie_tolerance: float = 1e-12,
    max_candidates: int = 1024,
    monte_carlo: MonteCarlo | None = None,
    rng_seed: int = 0,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> StudyProgram:
    """Compile ``study`` as the state belief of the declared EVSI evaluation.

    Nothing is fitted or simulated. Declaration order of the candidates does not change the
    compiled program; a new decision utility changes it.

    Raises:
        DesignRankingRefusal: the study publishes no parameter posterior
            (``distribution_meaning_mismatch``, naming the Bayesian realization), the request
            is incoherent, or the Monte Carlo store does not fit ``memory_bytes``.
    """
    names, columns, edges, treatment, outcome = study._arguments()
    request = _request(
        decision,
        candidates,
        signal,
        cost_map,
        require_net_value,
        prior_observations,
        source_digests,
        mc_error_tolerance,
        tie_tolerance,
        max_candidates,
        monte_carlo,
        rng_seed,
    )
    text, refusal = _native.compile_design_ranking_for_study(
        request,
        names,
        columns,
        edges,
        treatment,
        outcome,
        checked_prior=decision.prior._checked,
        memory_bytes=memory_bytes,
        cancel=cancel,
        **_study_options(study),
    )
    raise_refusal(refusal, _refusal_type)
    assert text is not None
    wire = json.loads(text)
    return StudyProgram(
        program_identity=str(wire["program_identity"]),
        carrier=str(wire["carrier"]),
        contract_identity=wire["contract_identity"],
        utility_unit=wire["utility_unit"],
        candidates=tuple(wire["candidates"]),
        prior_observations=tuple(wire["prior_observations"]),
        rng_seed=int(wire["rng_seed"]),
        mc_error_tolerance=float(wire["mc_error_tolerance"]),
        schedule=dict(wire["schedule"]),
        store=dict(wire["store"]),
    )


def evaluate_for_study(
    study: RankingStudy,
    decision: DesignDecision,
    candidates: Sequence[Candidate],
    *,
    signal: SignalSpec | None = None,
    cost_map: CostMap | None = None,
    require_net_value: bool = False,
    prior_observations: Sequence[str] = (),
    source_digests: Sequence[str] = (),
    mc_error_tolerance: float = 1e-3,
    tie_tolerance: float = 1e-12,
    max_candidates: int = 1024,
    artifact_id: str = "design-ranking",
    monte_carlo: MonteCarlo | None = None,
    rng_seed: int = 0,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> StudyRanking:
    """Evaluate the request bound to the lowered program of ``study`` compiled for it.

    The compile runs first, so a study that cannot carry the request or a store the pool cannot
    hold refuses before any candidate is scored. The returned ranking was recomputed
    independently before it is handed back, exactly as for
    :func:`~antecedent.design.rank_designs`, and ``lowered_program_identity`` is the identity
    :func:`compile_for_study` reports for the same arguments.
    """
    names, columns, edges, treatment, outcome = study._arguments()
    request = _request(
        decision,
        candidates,
        signal,
        cost_map,
        require_net_value,
        prior_observations,
        source_digests,
        mc_error_tolerance,
        tie_tolerance,
        max_candidates,
        monte_carlo,
        rng_seed,
    )
    body, artifact, program, refusal = _native.evaluate_design_ranking_for_study(
        request,
        artifact_id,
        names,
        columns,
        edges,
        treatment,
        outcome,
        checked_prior=decision.prior._checked,
        memory_bytes=memory_bytes,
        cancel=cancel,
        **_study_options(study),
    )
    raise_refusal(refusal, _refusal_type)
    assert body is not None and artifact is not None and program is not None
    wire = json.loads(body)
    data = bytes(artifact)
    consumed_text, consumed_refusal = _native.consume_design_ranking(data, None)
    raise_refusal(consumed_refusal, _refusal_type)
    assert consumed_text is not None
    replays = {e["semantic_id"]: e for e in json.loads(consumed_text)["entries"]}
    action_ids = tuple(wire["decision"]["action_ids"])
    result = DesignRankingResult(
        basis=wire["basis"],
        candidates=tuple(
            _candidate_from_wire(c, replays[c["semantic_id"]])
            for c in sorted(wire["candidates"], key=lambda c: int(c["rank"]))
        ),
        search=_search(wire["search"]),
        calibration=wire["calibration"],
        _value=_ValueBody(
            decision_contract_identity=wire["decision"]["contract_identity"],
            utility_unit=wire["decision"]["utility_unit"],
            action_ids=action_ids,
            bayes_action=action_ids[int(wire["bayes_action"])],
            prior_expected_utility=float(wire["prior_expected_utility"]),
            evpi=float(wire["evpi"]),
            cost_map=_cost_map(wire["cost_map"]),
            rng_seed=int(wire["rng_seed"]),
            source_digests=tuple(sorted(set(wire["source_digests"]))),
            ties=tuple((a, b) for a, b in wire["ties"]),
            identity=wire["digest"],
            ranking_identity=wire["ranking_identity"],
        ),
        _bytes=data,
    )
    return StudyRanking(result=result, lowered_program_identity=str(program))
