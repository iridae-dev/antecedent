"""The participant-level adaptive family: three arms, three or six binary responses per
participant, historical informative priors, Thompson allocation with a minimum exploration
probability, and expected-loss, futility and recruitment-cap decisions at completed-participant
batch boundaries.

Truth is computed here in pure Python, never by native code:

* the exact posterior of each arm mean under the beta-binomial likelihood of the completed
  participants' success counts (a dense midpoint sum over the success probability, with the
  gamma-function form of the mass function, jointly over the correlation when it is estimated),
  from the retained history;
* the prior ``Beta(1 + sum a0 s, 1 + sum a0 (n - s))`` of the borrowed sources and its effective
  sample size ``alpha + beta``;
* the running intersection of the predictable plug-in empirical Bernstein intervals (Waudby-Smith
  and Ramdas 2024), written out from the paper's formulas, for the confidence-sequence reading at
  every look, at the level ``alpha / K`` per arm.

Every result is ``derived_unmeasured``. The posterior claim and the anytime-valid frequentist claim
of a look are separate, labelled fields.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from antecedent import _native
from antecedent import adaptive_inference as ai
from antecedent.adaptive_inference import AdaptiveInferenceRefusal

from _refusal import assert_registered_refusal

RESPONSES = 3
RHO = 0.25
FLOOR = 0.15
ALPHA = 0.06
TUNING = 30.0
TRUTH = (0.35, 0.45, 0.4)


def borrowing(discount: float = 0.5) -> tuple[ai.ArmPrior, ...]:
    registry = ai.ArmPrior(
        historical=(
            ai.HistoricalPrior("control registry", 40, 100, discount),
            ai.HistoricalPrior("earlier pilot", 9, 30, 1.0),
        )
    )
    return (registry, ai.ArmPrior(), ai.ArmPrior())


def declaration(**changes) -> ai.ParticipantDeclaration:
    fields = {
        "arms": 3,
        "responses": RESPONSES,
        "correlation": RHO,
        "borrowing": borrowing(),
        "enrolment": (40, 40, 40, 40, 40),
        "lag_looks": 1,
        "min_exploration": FLOOR,
        # Arms within a tenth of one another keep an expected loss far above this at every size
        # used here, and the control is not futile against its best competitor.
        "decisions": ai.ParticipantDecisions(
            loss_threshold=1e-12, futility=ai.FutilityRule(margin=0.1, threshold=0.02)
        ),
        "anytime": ai.AnytimeSettings(alpha=ALPHA, tuning_samples=TUNING),
    }
    fields.update(changes)
    return ai.ParticipantDeclaration(**fields)


def refused(call, *, detail: str | None = None, code: str | None = None):
    with pytest.raises(AdaptiveInferenceRefusal) as caught:
        call()
    assert_registered_refusal(caught.value)
    if detail is not None:
        assert caught.value.detail == detail
    if code is not None:
        assert caught.value.reason_code == code
    return caught.value


@pytest.fixture(scope="module")
def procedure() -> ai.ParticipantProcedure:
    return ai.ParticipantProcedure.declare(declaration())


@pytest.fixture(scope="module")
def history(procedure) -> ai.ParticipantHistory:
    return procedure.simulate_history(TRUTH, seed=12)


# --------------------------------------------------------------------------- declaration


def test_every_missing_element_is_its_own_refusal():
    error = refused(
        lambda: ai.ParticipantProcedure.declare(ai.ParticipantDeclaration(arms=3)),
        code="adaptive_contract_incomplete",
    )
    assert error.details == (
        "adaptive_inference.outcome_structure_missing",
        "adaptive_inference.within_unit_correlation_missing",
        "adaptive_inference.historical_borrowing_missing",
        "adaptive_inference.interim_analyses_missing",
        "adaptive_inference.policy_history_missing",
        "adaptive_inference.stopping_rule_missing",
        "adaptive_inference.anytime_settings_missing",
    )


@pytest.mark.parametrize(
    ("changes", "detail"),
    [
        ({"responses": 1}, "adaptive_inference.measurements_invalid"),
        ({"correlation": 1.0}, "adaptive_inference.correlation_invalid"),
        ({"lag_looks": 33}, "adaptive_inference.outcome_availability_invalid"),
        ({"enrolment": (40,)}, "adaptive_inference.looks_not_ordered"),
        ({"min_exploration": 0.4}, "adaptive_inference.rule_mismatch"),
        (
            {"anytime": ai.AnytimeSettings(alpha=1.0, tuning_samples=30.0)},
            "adaptive_inference.anytime_settings_invalid",
        ),
        (
            {"borrowing": (ai.ArmPrior(), ai.ArmPrior())},
            "adaptive_inference.historical_prior_invalid",
        ),
        (
            {
                "borrowing": (
                    ai.ArmPrior(historical=(ai.HistoricalPrior("registry", 3, 10, 1.5),)),
                    ai.ArmPrior(),
                    ai.ArmPrior(),
                )
            },
            "adaptive_inference.historical_prior_invalid",
        ),
    ],
)
def test_an_inconsistent_element_is_refused_with_its_own_detail(changes, detail):
    refused(
        lambda: ai.ParticipantProcedure.declare(declaration(**changes)),
        detail=detail,
        code="invalid_argument",
    )


# --------------------------------------------------------------------------- historical priors


def test_borrowed_sources_resolve_to_a_beta_with_its_effective_sample_size(procedure):
    control = procedure.priors[0]
    # 1 + 0.5 * 40 + 9 and 1 + 0.5 * 60 + 21.
    assert control["alpha"] == pytest.approx(1 + 20 + 9)
    assert control["beta"] == pytest.approx(1 + 30 + 21)
    assert control["effective_sample_size"] == pytest.approx(82)
    assert control["borrowed_sample_size"] == pytest.approx(0.5 * 100 + 30)
    assert [s["source"] for s in control["sources"]] == ["control registry", "earlier pilot"]
    assert [s["discount"] for s in control["sources"]] == [0.5, 1.0]
    # The treatment arms start from the initial prior alone.
    assert procedure.priors[1]["effective_sample_size"] == pytest.approx(2)
    assert not procedure.priors[1]["sources"]


def test_borrowing_is_part_of_the_identity_and_frozen_after_the_first_look(procedure):
    other = ai.ParticipantProcedure.declare(declaration(borrowing=borrowing(0.6)))
    assert other.identity != procedure.identity
    amended = procedure.amend_borrowing(borrowing(0.6), looks_performed=0)
    assert amended.identity == other.identity
    error = refused(
        lambda: procedure.amend_borrowing(borrowing(0.6), looks_performed=1),
        detail="adaptive_inference.borrowing_frozen",
        code="adaptive_assumption_violated",
    )
    assert "frozen" in str(error)
    # The same plan is not a change.
    assert (
        procedure.amend_borrowing(borrowing(0.5), looks_performed=3).identity == procedure.identity
    )


# --------------------------------------------------------------------------- readings


def completed(history: ai.ParticipantHistory, look: int, lag: int):
    """Participants whose follow-up completed by ``look``, in enrolment order."""
    end = sum(history.batches[: max(look - lag + 1, 0)])
    return [(history.arms[i], history.successes[i]) for i in range(end)]


def log_beta(a, b):
    return np.vectorize(math.lgamma)(a) + np.vectorize(math.lgamma)(b) - np.vectorize(math.lgamma)(a + b)


def log_pmf(m: int, s: int, p, rho: float):
    """Log mass of ``s`` successes among ``m`` responses at arm mean ``p``, correlation ``rho``."""
    log_choose = math.lgamma(m + 1) - math.lgamma(s + 1) - math.lgamma(m - s + 1)
    if rho == 0.0:
        return log_choose + s * np.log(p) + (m - s) * np.log1p(-p)
    kappa = (1.0 - rho) / rho
    return log_choose + log_beta(s + p * kappa, m - s + (1.0 - p) * kappa) - log_beta(
        p * kappa, (1.0 - p) * kappa
    )


GRID = 20_001


def arm_log_density(prior, counts, rho: float):
    p = (np.arange(GRID) + 0.5) / GRID
    out = (prior["alpha"] - 1.0) * np.log(p) + (prior["beta"] - 1.0) * np.log1p(-p)
    for s, count in enumerate(counts):
        if count:
            out = out + count * log_pmf(RESPONSES, s, p, rho)
    return p, out


def exact_moments(prior, counts, rho: float):
    """Mean and standard deviation of the exact posterior of an arm mean at a known rho."""
    p, log_density = arm_log_density(prior, counts, rho)
    weight = np.exp(log_density - log_density.max())
    weight /= weight.sum()
    mean = float((weight * p).sum())
    return mean, math.sqrt(float((weight * (p - mean) ** 2).sum()))


def histograms(history: ai.ParticipantHistory, look: int, lag: int):
    done = completed(history, look, lag)
    return [
        [sum(1 for a, c in done if a == arm and c == s) for s in range(RESPONSES + 1)]
        for arm in range(3)
    ]


def test_the_posterior_claim_is_the_exact_beta_binomial_posterior(procedure, history):
    readings = procedure.read_history(history)
    assert readings.claim == "derived_unmeasured"
    assert len(readings.looks) == len(history.batches)
    for look, reading in enumerate(readings.looks):
        posterior = reading["posterior"]
        assert posterior["paradigm"] == "bayesian_posterior"
        counts = histograms(history, look, lag=1)
        for arm in range(3):
            mean, sd = exact_moments(procedure.priors[arm], counts[arm], RHO)
            assert posterior["mean"][arm] == pytest.approx(mean, abs=1e-6)
            assert posterior["sd"][arm] == pytest.approx(sd, abs=1e-6)
            assert reading["completed"][arm] == sum(counts[arm])
        assert posterior["correlation_mean"] == pytest.approx(RHO)
        assert posterior["correlation_sd"] == 0.0
        assert sum(posterior["probability_best"]) == pytest.approx(1.0, abs=1e-8)


def test_an_estimated_correlation_is_integrated_jointly_with_the_arm_means():
    prior = ai.CorrelationPrior(alpha=2.0, beta=6.0)
    procedure = ai.ParticipantProcedure.declare(
        declaration(correlation=None, correlation_prior=prior)
    )
    assert procedure.identity != ai.ParticipantProcedure.declare(declaration()).identity
    history = procedure.simulate_history(TRUTH, seed=12, true_correlation=0.45)
    readings = procedure.read_history(history)
    rhos = (np.arange(400) + 0.5) / 400
    for look, reading in enumerate(readings.looks):
        posterior = reading["posterior"]
        counts = histograms(history, look, lag=1)
        # Joint posterior over (arm means, rho): the arms are independent given rho, so the
        # weight of rho is its prior times the product of the arms' marginal likelihoods.
        log_weight = []
        moments = []
        for rho in rhos:
            total = (prior.alpha - 1.0) * math.log(rho) + (prior.beta - 1.0) * math.log1p(-rho)
            per_arm = []
            for arm in range(3):
                p, log_density = arm_log_density(procedure.priors[arm], counts[arm], float(rho))
                peak = log_density.max()
                weight = np.exp(log_density - peak)
                total += peak + math.log(weight.sum() / GRID)
                weight /= weight.sum()
                mean = float((weight * p).sum())
                per_arm.append((mean, float((weight * p * p).sum())))
            log_weight.append(total)
            moments.append(per_arm)
        q = np.exp(np.array(log_weight) - max(log_weight))
        q /= q.sum()
        for arm in range(3):
            mean = sum(q[i] * moments[i][arm][0] for i in range(len(rhos)))
            second = sum(q[i] * moments[i][arm][1] for i in range(len(rhos)))
            assert posterior["mean"][arm] == pytest.approx(mean, abs=2e-5)
            assert posterior["sd"][arm] == pytest.approx(math.sqrt(second - mean**2), abs=2e-5)
        rho_mean = float((q * rhos).sum())
        rho_sd = math.sqrt(float((q * (rhos - rho_mean) ** 2).sum()))
        assert posterior["correlation_mean"] == pytest.approx(rho_mean, abs=2e-5)
        assert posterior["correlation_sd"] == pytest.approx(rho_sd, abs=2e-5)
        assert sum(posterior["probability_best"]) == pytest.approx(1.0, abs=1e-8)
    # More completed participants say more about the correlation.
    sds = [reading["posterior"]["correlation_sd"] for reading in readings.looks]
    assert sds[-1] < sds[1]


def test_simulating_an_estimated_correlation_needs_the_true_correlation():
    procedure = ai.ParticipantProcedure.declare(
        declaration(
            correlation=None,
            correlation_prior=ai.CorrelationPrior(alpha=2.0, beta=6.0),
            enrolment=(15, 15, 15),
        )
    )
    with pytest.raises(AdaptiveInferenceRefusal) as caught:
        procedure.simulate(TRUTH, replications=2, seed=1)
    assert_registered_refusal(caught.value)
    simulation = procedure.simulate(TRUTH, replications=3, seed=1, true_correlation=0.3)
    assert simulation.claim == "derived_unmeasured"
    assert len(simulation.replicates) == 3


def running_intersection(values: list[float], alpha: float) -> tuple[float, float]:
    """The predictable plug-in empirical Bernstein sequence, intersected over the prefixes."""
    lower, upper = 0.0, 1.0
    total = squares = tilts = tilted = penalty = 0.0
    log_term = math.log(2.0 / alpha)
    for t, x in enumerate(values, start=1):
        previous_mean = (0.5 + total) / t
        previous_variance = (0.25 + squares) / t
        lam = min(math.sqrt(2.0 * log_term / (previous_variance * t * math.log(1.0 + t))), 0.75)
        tilts += lam
        tilted += lam * x
        penalty += (x - previous_mean) ** 2 * (-math.log(1.0 - lam) - lam)
        total += x
        squares += (x - (0.5 + total) / (t + 1.0)) ** 2
        centre, radius = tilted / tilts, (log_term + penalty) / tilts
        lower = max(lower, centre - radius)
        upper = min(upper, centre + radius)
    return lower, upper


def test_the_frequentist_claim_is_a_confidence_sequence_read_at_every_look(procedure, history):
    readings = procedure.read_history(history)
    alpha_arm = ALPHA / 3
    for look, reading in enumerate(readings.looks):
        claim = reading["frequentist"]
        assert claim["paradigm"] == "anytime_valid_frequentist"
        assert claim["calibration"] == "derived_unmeasured"
        assert claim["sequence"] == "predictable_plug_in_empirical_bernstein"
        assert claim["alpha_per_arm"] == pytest.approx(alpha_arm)
        assert claim["alpha_per_comparison"] == pytest.approx(2 * alpha_arm)
        done = completed(history, look, lag=1)
        for arm in range(3):
            proportions = [count / RESPONSES for a, count in done if a == arm]
            lower, upper = running_intersection(proportions, alpha_arm)
            assert claim["arms"][arm]["count"] == len(proportions)
            assert claim["arms"][arm]["lower"] == pytest.approx(lower, abs=1e-12)
            assert claim["arms"][arm]["upper"] == pytest.approx(upper, abs=1e-12)
        control = claim["arms"][0]
        for comparison in claim["comparisons"]:
            treatment = claim["arms"][comparison["arm"]]
            assert comparison["lower"] == pytest.approx(treatment["lower"] - control["upper"])
            assert comparison["upper"] == pytest.approx(treatment["upper"] - control["lower"])
        # The data-selected best treatment is covered by the same simultaneous family.
        best = claim["best_treatment"]
        assert claim["best_treatment_comparison"] == claim["comparisons"][best - 1]
    # Posterior and frequentist claims are separate objects.
    assert set(readings.looks[0]) >= {"posterior", "frequentist", "decision"}
    assert readings.looks[0]["posterior"] is not readings.looks[0]["frequentist"]


def test_every_batch_is_randomized_under_probabilities_at_or_above_the_floor(procedure, history):
    readings = procedure.read_history(history)
    index = 0
    for look, size in enumerate(history.batches):
        row = readings.looks[look]["assignment_probabilities"]
        assert sum(row) == pytest.approx(1.0)
        assert min(row) >= FLOOR - 1e-12
        for _ in range(size):
            assert history.assignment[index] == pytest.approx(row)
            index += 1


def test_nothing_is_decided_before_a_participant_has_completed(procedure, history):
    first = procedure.read_history(history).looks[0]
    assert first["completed"] == (0, 0, 0)
    assert first["decision"]["kind"] == "continue"
    assert first["posterior"]["mean"] == pytest.approx(
        [prior["alpha"] / (prior["alpha"] + prior["beta"]) for prior in procedure.priors],
        abs=1e-8,
    )


# --------------------------------------------------------------------------- decisions


def test_the_recruitment_cap_ends_the_procedure_at_the_batch_boundary_that_reaches_it():
    capped = ai.ParticipantProcedure.declare(
        declaration(
            lag_looks=0,
            decisions=ai.ParticipantDecisions(loss_threshold=1e-12, participant_cap=80),
        )
    )
    simulation = capped.simulate((0.4, 0.4, 0.4), replications=20, seed=4)
    assert simulation.claim == "derived_unmeasured"
    assert {r["reason"] for r in simulation.replicates} == {"recruitment_cap"}
    assert {(r["look"], r["enrolled"]) for r in simulation.replicates} == {(1, 80)}
    assert simulation.summary["stop_fractions"]["recruitment_cap"] == pytest.approx(1.0)


def test_expected_loss_decides_for_the_best_arm_and_futility_for_the_control():
    decisive = ai.ParticipantProcedure.declare(
        declaration(
            lag_looks=0,
            enrolment=(150, 150),
            decisions=ai.ParticipantDecisions(loss_threshold=0.01),
        )
    )
    simulation = decisive.simulate((0.1, 0.9, 0.1), replications=20, seed=11)
    assert {(r["reason"], r["look"], r["decision"]) for r in simulation.replicates} == {
        ("expected_loss", 0, 1)
    }
    futile = ai.ParticipantProcedure.declare(
        declaration(
            lag_looks=0,
            enrolment=(300, 300),
            decisions=ai.ParticipantDecisions(
                loss_threshold=1e-12, futility=ai.FutilityRule(margin=0.15, threshold=0.1)
            ),
        )
    )
    simulation = futile.simulate((0.3, 0.3, 0.3), replications=30, seed=21)
    stopped = [r for r in simulation.replicates if r["reason"] == "futility"]
    assert len(stopped) >= 24
    assert {r["decision"] for r in stopped} == {0}


# --------------------------------------------------------------------------- simulation and capacity


def test_the_simulation_covers_the_selected_treatment_under_selection_and_returns_a_receipt():
    equal = ai.ParticipantProcedure.declare(declaration(lag_looks=0))
    simulation = equal.simulate((0.4, 0.4, 0.4), replications=60, seed=77)
    receipt = simulation.receipt
    assert receipt["replications"] == 60
    assert receipt["work_planned"] == 60 * 200 * RESPONSES
    assert receipt["work_done"] <= receipt["work_planned"]
    assert simulation.summary["covered_every_look"] >= 0.95
    assert simulation.summary["contrast_covered_at_stop"] >= 0.95


@pytest.mark.parametrize("participants", [600, 1200, 2400])
def test_planned_procedures_run(participants):
    planned = ai.ParticipantProcedure.declare(
        declaration(
            arms=4,
            responses=6,
            borrowing=(ai.ArmPrior(),) * 4,
            enrolment=(participants // 12,) * 12,
            min_exploration=0.1,
            decisions=ai.ParticipantDecisions(loss_threshold=1e-12),
        )
    )
    simulation = planned.simulate((0.4, 0.4, 0.4, 0.4), replications=2, seed=3)
    assert all(r["enrolled"] == participants for r in simulation.replicates)
    assert simulation.receipt["work_planned"] == 2 * participants * 6
    assert simulation.receipt["peak_units"] == participants


def test_capacity_refusals_name_their_own_class_of_limit(procedure):
    refused(
        lambda: ai.ParticipantProcedure.declare(declaration(enrolment=(40_000, 40_000))),
        detail="adaptive_inference.declared_units_exceeded",
        code="invalid_argument",
    )
    refused(
        lambda: ai.ParticipantProcedure.declare(declaration(enrolment=(1,) * 65)),
        detail="adaptive_inference.declared_looks_exceeded",
    )
    large = ai.ParticipantProcedure.declare(declaration(enrolment=(32_000, 32_000)))
    refused(
        lambda: large.simulate((0.4, 0.4, 0.4), replications=30_000, seed=1),
        detail="adaptive_inference.simulation_work_exceeded",
    )
    refused(
        lambda: procedure.simulate((0.4, 0.4, 0.4), replications=10, seed=1, memory_bytes=1_000),
        detail="adaptive_inference.simulation_memory_exceeded",
    )
    refused(
        lambda: procedure.simulate((0.4, 0.4, 0.4), replications=1_000_001, seed=1),
        detail="adaptive_inference.simulation_replications_exceeded",
    )


def test_cancellation_is_a_refusal_not_a_verdict(procedure):
    token = _native.CancellationToken()
    token.cancel()
    refused(
        lambda: procedure.simulate((0.4, 0.4, 0.4), replications=10, seed=1, cancel=token),
        code="cancelled_no_claim",
    )


# --------------------------------------------------------------------------- replay


def test_a_tampered_history_does_not_replay(procedure, history):
    rows = [list(row) for row in history.assignment]
    rows[0][0] += 1e-6
    rows[0][1] -= 1e-6
    tampered = ai.ParticipantHistory(history.batches, history.arms, history.successes, rows)
    refused(
        lambda: procedure.read_history(tampered),
        detail="adaptive_inference.assignment_not_replayed",
        code="adaptive_assumption_violated",
    )
    # A procedure that continued past a look at which its declared decisions stop it.
    capped = ai.ParticipantProcedure.declare(
        declaration(decisions=ai.ParticipantDecisions(loss_threshold=1e-12, participant_cap=80))
    )
    refused(
        lambda: capped.read_history(history),
        detail="adaptive_inference.stopping_not_replayed",
        code="adaptive_assumption_violated",
    )


def test_the_retained_contract_is_sealed_with_its_procedure_and_replayed(procedure, history):
    session = procedure.prepare_session("dose finding", ("control", "low", "high"), history)
    assert session.procedure == (
        "thompson_beta_bernoulli/repeated_measures/fixed_interim_looks/decision_theoretic"
    )
    artifact = session.export_contract_with_procedure("bayesian_posterior", procedure)
    consumed = ai.consume_adaptive_monitoring_contract_artifact(artifact)
    section = consumed.participant_procedure
    assert section is not None
    assert section["procedure_identity"] == procedure.identity
    readings = procedure.read_history(history)
    assert len(section["readings"]) == len(readings.looks)
    for replayed, produced in zip(section["readings"], readings.looks, strict=True):
        assert replayed["assignment_probabilities"] == pytest.approx(
            produced["assignment_probabilities"]
        )
        assert replayed["posterior"]["mean"] == pytest.approx(produced["posterior"]["mean"])
        assert (
            replayed["frequentist"]["best_treatment"] == produced["frequentist"]["best_treatment"]
        )
    # A contract sealed without the section replays without one.
    plain = ai.consume_adaptive_monitoring_contract_artifact(
        session.export_contract("bayesian_posterior")
    )
    assert plain.participant_procedure is None
    # Fixed-sample inference of an adaptive design is refused.
    refused(
        lambda: session.export_contract_with_procedure("frequentist_fixed_sample", procedure),
        code="adaptive_design_ignored",
    )
    # Sealing a history under other borrowing: the allocation does not replay.
    other = procedure.amend_borrowing(borrowing(1.0), looks_performed=0)
    refused(
        lambda: session.export_contract_with_procedure("bayesian_posterior", other),
        detail="adaptive_inference.assignment_not_replayed",
    )


def test_the_procedure_matrix_licenses_the_family_cell_for_its_three_paradigms():
    cell = next(
        c
        for c in ai.procedure_matrix()
        if c.id
        == "thompson_beta_bernoulli/repeated_measures/fixed_interim_looks/decision_theoretic"
    )
    assert cell.status == "licensed"
    assert cell.paradigms == (
        "bayesian_posterior",
        "decision_theoretic_stopping",
        "anytime_valid_frequentist",
    )
