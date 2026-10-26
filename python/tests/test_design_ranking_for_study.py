"""Value-of-information requests compiled against, and evaluated for, a study.

The study is a binary average effect of ``t`` on ``y`` with conjugate Bayesian inference, so it
publishes a parameter posterior and can be the state belief of the decision. Truth: the program
quotes a Monte Carlo store of ``candidates x draws`` scores and ``draws`` seeds (8 bytes each);
declaration order does not change the program; a study with a sampling realization carries no
posterior and is refused; and the evaluation equals the ranking the same request gives without a
study, bound to the program's identity.
"""

from __future__ import annotations

import pytest
from antecedent import design as dr
from antecedent.design import study_ranking as sr
from antecedent.errors import CausalError
from antecedent.joint_distribution import ScientificQuantity

TREATMENT = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
OUTCOME = [1.1, 3.0, 0.9, 3.1, 1.0, 2.9, 1.0, 3.0]


def _quantity(name: str) -> ScientificQuantity:
    return ScientificQuantity(
        variable_id=f"schema:{name}",
        variable_name=name,
        role="outcome",
        units="dimensionless",
        population_id="target",
        regime_id="observational",
        horizon=0,
        functional_id="state",
    )


def _signal() -> dr.SignalSpec:
    return dr.SignalSpec(
        prior_id="prior-1",
        state=_quantity("state"),
        observation=_quantity("signal"),
        evidence_lineage=("snapshot:a",),
        rng_seed=3,
    )


def _bet() -> dr.DesignDecision:
    return dr.DesignDecision(
        contract="contract-bet",
        actions=(dr.ActionUtility("abstain", 0.0, 0.0), dr.ActionUtility("bet", -0.5, 1.0)),
        prior=dr.StatePrior.draws([0.25, 0.75]),
        utility_units="utility",
    )


def _study(candidate_id: str) -> dr.Candidate:
    return dr.Candidate(candidate_id, 2, dr.BinomialSignal(), cost=0.0, cost_unit="utility")


def _study_of(posterior: bool = True) -> sr.RankingStudy:
    return sr.RankingStudy(
        data={"t": TREATMENT, "y": OUTCOME},
        edges=(("t", "y"),),
        treatment="t",
        outcome="y",
        posterior=posterior,
    )


def test_compiling_fills_the_value_of_information_program_and_quotes_its_store():
    program = sr.compile_for_study(
        _study_of(),
        _bet(),
        [_study("study-b"), _study("study-a")],
        signal=_signal(),
        rng_seed=3,
    )
    assert program.carrier == "parameter_posterior"
    assert program.contract_identity == "contract-bet"
    assert program.utility_unit == "utility"
    assert program.candidates == ("study-a", "study-b")
    assert program.rng_seed == 3
    assert program.mc_error_tolerance > 0.0
    draws = program.schedule["draws_per_candidate"]
    assert program.store["candidates"] == 2
    assert program.store["draws"] == draws
    assert program.store["tasks"] == 2 * draws
    assert program.store["total_bytes"] == 2 * draws * 8 + draws * 8
    assert len(program.program_identity) == 64


def test_declaration_order_does_not_change_the_program_and_the_candidate_set_does():
    study = _study_of()
    first = sr.compile_for_study(
        study, _bet(), [_study("study-b"), _study("study-a")], signal=_signal()
    )
    reordered = sr.compile_for_study(
        study, _bet(), [_study("study-a"), _study("study-b")], signal=_signal()
    )
    fewer = sr.compile_for_study(study, _bet(), [_study("study-a")], signal=_signal())
    assert first.program_identity == reordered.program_identity
    assert first.program_identity != fewer.program_identity


def test_the_evaluation_is_the_ranking_of_the_same_request_bound_to_the_program():
    studies = [_study("study-a"), _study("study-b")]
    ranking = sr.evaluate_for_study(_study_of(), _bet(), studies, signal=_signal())
    program = sr.compile_for_study(_study_of(), _bet(), studies, signal=_signal())
    plain = dr.rank_designs(studies, decision=_bet(), signal=_signal())
    assert ranking.lowered_program_identity == program.program_identity
    assert [c.id for c in ranking.result.candidates] == [c.id for c in plain.candidates]
    for bound, free in zip(ranking.result.candidates, plain.candidates, strict=True):
        assert bound.evsi == pytest.approx(free.evsi, abs=1e-12)
    assert ranking.result.basis == plain.basis
    assert bytes(
        ranking.result.export()
    )  # the sealed artifact was recomputed before it was returned


def test_a_study_with_a_sampling_realization_publishes_no_posterior_and_is_refused():
    studies = [_study("study-a")]
    for call in (sr.compile_for_study, sr.evaluate_for_study):
        with pytest.raises(dr.DesignRankingRefusal) as info:
            call(_study_of(posterior=False), _bet(), studies, signal=_signal())
        assert info.value.reason_code == "distribution_meaning_mismatch"


def test_a_store_the_admission_pool_cannot_hold_is_refused_before_anything_is_scored():
    with pytest.raises(CausalError, match="Monte Carlo"):
        sr.compile_for_study(
            _study_of(), _bet(), [_study("study-a")], signal=_signal(), memory_bytes=64
        )
    with pytest.raises(CausalError, match="Monte Carlo"):
        sr.evaluate_for_study(
            _study_of(), _bet(), [_study("study-a")], signal=_signal(), memory_bytes=64
        )
