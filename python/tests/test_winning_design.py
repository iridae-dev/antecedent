"""The winning design is the rank and sample size the ranking already stored.

The same three rows are the Rust fixture in
``crates/antecedent-design/src/winning_design.rs``
(``rank_zero_sample_size_is_copied_when_neighbours_are_larger_and_smaller``):
rank 0 is ``kept`` at n 40, between a larger n and a smaller n, and it is not
listed first. A projection that re-ranked by sample size or by list position
would return a different declaration.
"""

from __future__ import annotations

import json

import pytest
from antecedent._native import project_winning_design as project_rows
from antecedent.choice_inference import StudyRank
from antecedent.design import (
    CandidateValue,
    DesignRankingResult,
    DesignSearchReceipt,
    Environment,
    IdentificationCandidate,
    IntegrationReport,
    Measurement,
    Sampling,
    SourceOverlapDiagnostics,
    StructuralEntry,
    StructuralRanking,
    WinningDesign,
    project_winning_design,
)
from antecedent.errors import CausalTypeError, CausalValueError

KEPT = "kept"
BIGGER = "bigger"
SMALLER = "smaller"
KEPT_N = 40
BIGGER_N = 500
SMALLER_N = 10

FIXTURE = {
    "rows": [
        {"candidate_id": BIGGER, "rank": 1, "n": BIGGER_N},
        {"candidate_id": KEPT, "rank": 0, "n": KEPT_N},
        {"candidate_id": SMALLER, "rank": 2, "n": SMALLER_N},
    ]
}

DECLARATION = {"candidate_id": KEPT, "n": KEPT_N}


def test_rank_zero_sample_size_is_copied_when_neighbours_are_larger_and_smaller() -> None:
    assert json.loads(project_rows(json.dumps(FIXTURE))) == DECLARATION


def test_native_refusals_name_the_stored_ranking() -> None:
    with pytest.raises(CausalValueError, match="the ranking stored no candidates"):
        project_rows(json.dumps({"rows": []}))
    with pytest.raises(CausalValueError, match="the ranking stored no rank-0 candidate"):
        project_rows(json.dumps({"rows": [{"candidate_id": BIGGER, "rank": 1, "n": BIGGER_N}]}))
    with pytest.raises(CausalValueError, match="the ranking stored more than one rank-0 candidate"):
        project_rows(
            json.dumps(
                {
                    "rows": [
                        {"candidate_id": KEPT, "rank": 0, "n": KEPT_N},
                        {"candidate_id": BIGGER, "rank": 0, "n": BIGGER_N},
                    ]
                }
            )
        )
    with pytest.raises(CausalValueError, match="winning design rows are not the stored ranking"):
        project_rows(
            json.dumps(
                {"rows": [{"candidate_id": KEPT, "rank": 0, "n": KEPT_N, "evsi": 1}]}
            )
        )


def _search() -> DesignSearchReceipt:
    return DesignSearchReceipt(3, 3, False, ())


def _structural(candidate_id: str, rank: int, sample_budget: int) -> StructuralEntry:
    return StructuralEntry(
        id=candidate_id,
        rank=rank,
        verified_sufficient=True,
        cost_units=1,
        sample_budget=sample_budget,
    )


def test_design_ranking_reads_rank_zero_sample_budget() -> None:
    ranking = DesignRankingResult(
        basis="structural_sufficiency_cost",
        candidates=(
            _structural(BIGGER, 1, BIGGER_N),
            _structural(KEPT, 0, KEPT_N),
            _structural(SMALLER, 2, SMALLER_N),
        ),
        search=_search(),
        calibration="not_measured",
    )
    assert project_winning_design(ranking) == WinningDesign(candidate_id=KEPT, n=KEPT_N)
    assert ranking.best is not None
    assert ranking.best.id == BIGGER


def _value(candidate_id: str, rank: int, sample_size: int, evsi: float) -> CandidateValue:
    return CandidateValue(
        id=candidate_id,
        rank=rank,
        evsi=evsi,
        evpi=1.0,
        sample_size=sample_size,
        study_cost=0.0,
        cost_unit="utility",
        study_cost_utility=None,
        net_value=evsi,
        integration=IntegrationReport("exact", 0, 0.0, None, True, False),
        rank_uncertain=False,
        provider_trust="native_supported",
        update_mode="native_update",
        attestor=None,
        provider=None,
        signal_family="gaussian_mean",
        request_fingerprint="request",
        signal_identity="signal",
        source_overlap=SourceOverlapDiagnostics(0, ()),
        assumptions=(),
        replay="point_only",
        natively_replayed=True,
        trust_limit="native",
    )


def test_value_ranking_reads_stored_sample_size() -> None:
    ranking = DesignRankingResult(
        basis="evsi",
        candidates=(
            _value(BIGGER, 1, BIGGER_N, 9.0),
            _value(KEPT, 0, KEPT_N, 0.1),
            _value(SMALLER, 2, SMALLER_N, 0.2),
        ),
        search=_search(),
        calibration="not_measured",
    )
    assert project_winning_design(ranking) == WinningDesign(candidate_id=KEPT, n=KEPT_N)


def _identified(
    candidate_id: str, rank: int, plan: Sampling | Environment | Measurement, score: float
) -> IdentificationCandidate:
    return IdentificationCandidate(
        id=candidate_id,
        index=rank,
        rank=rank,
        kind=plan.kind,
        tag=0,
        score=score,
        stderr=0.0,
        rank_uncertain=False,
        evaluation="exact",
        plan=plan,
        identification_probability=score,
    )


def test_sampling_plan_reads_additional_samples() -> None:
    ranking = DesignRankingResult(
        basis="identification",
        candidates=(
            _identified(BIGGER, 1, Sampling(BIGGER_N, id=BIGGER), 0.9),
            _identified(KEPT, 0, Sampling(KEPT_N, id=KEPT), 0.1),
        ),
        search=DesignSearchReceipt(2, 2, False, ()),
        calibration="not_measured",
    )
    assert project_winning_design(ranking) == WinningDesign(candidate_id=KEPT, n=KEPT_N)


def test_environment_plan_reads_additional_rows() -> None:
    ranking = DesignRankingResult(
        basis="identification",
        candidates=(
            _identified(BIGGER, 1, Sampling(BIGGER_N, id=BIGGER), 0.9),
            _identified(
                KEPT,
                0,
                Environment(1, additional_rows=KEPT_N, sample_budget=SMALLER_N, id=KEPT),
                0.1,
            ),
        ),
        search=DesignSearchReceipt(2, 2, False, ()),
        calibration="not_measured",
    )
    assert project_winning_design(ranking) == WinningDesign(candidate_id=KEPT, n=KEPT_N)


def test_measurement_plan_reads_sample_budget() -> None:
    ranking = DesignRankingResult(
        basis="identification",
        candidates=(
            _identified(KEPT, 0, Measurement((1,), sample_budget=KEPT_N, id=KEPT), 0.1),
        ),
        search=DesignSearchReceipt(1, 1, False, ()),
        calibration="not_measured",
    )
    assert project_winning_design(ranking) == WinningDesign(candidate_id=KEPT, n=KEPT_N)


def test_structural_ranking_reads_sample_budget() -> None:
    ranking = StructuralRanking(
        entries=(
            _structural(BIGGER, 1, BIGGER_N),
            _structural(KEPT, 0, KEPT_N),
        ),
        identity="stored",
    )
    assert project_winning_design(ranking) == WinningDesign(candidate_id=KEPT, n=KEPT_N)


def test_choice_study_reads_the_stored_order() -> None:
    ranking = (
        StudyRank(
            label=KEPT,
            observations=KEPT_N,
            cost=0.0,
            evsi=0.1,
            standard_error=0.0,
            net_value=0.1,
            rank_uncertain=False,
            method={"enumeration": "exact"},
        ),
        StudyRank(
            label=BIGGER,
            observations=BIGGER_N,
            cost=0.0,
            evsi=9.0,
            standard_error=0.0,
            net_value=9.0,
            rank_uncertain=False,
            method={"enumeration": "exact"},
        ),
    )
    assert project_winning_design(ranking) == WinningDesign(candidate_id=KEPT, n=KEPT_N)


def test_empty_and_ambiguous_rankings_are_refused() -> None:
    empty = DesignRankingResult(
        basis="evsi",
        candidates=(),
        search=DesignSearchReceipt(0, 0, False, ()),
        calibration="not_measured",
    )
    with pytest.raises(CausalValueError, match="the ranking stored no candidates"):
        project_winning_design(empty)
    ambiguous = DesignRankingResult(
        basis="structural_sufficiency_cost",
        candidates=(_structural(KEPT, 0, KEPT_N), _structural(BIGGER, 0, BIGGER_N)),
        search=DesignSearchReceipt(2, 2, False, ()),
        calibration="not_measured",
    )
    with pytest.raises(CausalValueError, match="the ranking stored more than one rank-0 candidate"):
        project_winning_design(ambiguous)


def test_a_non_ranking_is_refused() -> None:
    with pytest.raises(CausalTypeError, match="project_winning_design reads"):
        project_winning_design(object())  # type: ignore[arg-type]
