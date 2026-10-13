"""Inference from adaptive experiments through the Python facade.

Truth is computed here in pure Python, never by native code:

* the Beta-Bernoulli posterior of counts (conjugate closed form), the probability that one arm
  exceeds another (the finite sum for integer shapes) and the posterior expected loss (an integral
  of binomial-sum distribution functions);
* an exact enumeration of a two-step Thompson tree: the posterior mean is a martingale of the
  prior predictive however the arms were chosen;
* the augmented inverse-probability scores and their weighted mean, written out directly;
* the mixture-Hoeffding confidence-sequence radius, ``0.5 sqrt((m + n0) ln((m + n0) / (n0 a^2)))``;
* published one-sided O'Brien-Fleming-type group-sequential boundaries (3.7103, 2.5114, 1.9930 for
  three equal looks at alpha 0.025) and Welch statistics.

The route is point-only: the tests that ask for an interval assert the typed refusal.
"""

from __future__ import annotations

import itertools
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import pytest
from antecedent import _native
from antecedent import adaptive_inference as ai
from antecedent.adaptive_inference import AdaptiveInferenceRefusal

from _refusal import assert_registered_refusal
from _repo_text import load_toml

REPO = Path(__file__).resolve().parents[2]
TOL = 1e-9


# --------------------------------------------------------------------------- pure-Python truth


def beta_cdf(x: float, a: int, b: int) -> float:
    """Distribution function of Beta(a, b) for integer shapes: P(Bin(a + b - 1, x) >= a)."""
    n = a + b - 1
    return sum(math.comb(n, j) * x**j * (1.0 - x) ** (n - j) for j in range(a, n + 1))


def p_second_exceeds_first(first: tuple[int, int], second: tuple[int, int]) -> float:
    """P(theta_second > theta_first) for integer shapes: Evan Miller's finite sum."""
    (a1, b1), (a2, b2) = first, second

    def ln_b(x: float, y: float) -> float:
        return math.lgamma(x) + math.lgamma(y) - math.lgamma(x + y)

    return sum(
        math.exp(ln_b(a1 + i, b1 + b2) - ln_b(1 + i, b2) - ln_b(a1, b1)) / (b2 + i)
        for i in range(a2)
    )


def expected_loss(chosen: tuple[int, int], other: tuple[int, int]) -> float:
    """E[(theta_other - theta_chosen)^+] = integral of F_chosen (1 - F_other), by Simpson."""
    steps = 2000
    h = 1.0 / steps
    total = 0.0
    for i in range(steps + 1):
        t = i * h
        f = beta_cdf(t, *chosen) * (1.0 - beta_cdf(t, *other))
        total += f * (1 if i in (0, steps) else (4 if i % 2 else 2))
    return total * h / 3.0


def counts_of(arms: list[int], outcomes: list[list[float]], units: list[int]) -> list[list[int]]:
    counts = [[0, 0], [0, 0]]  # [successes, failures] per arm
    for u in units:
        counts[arms[u]][0 if outcomes[u][0] > 0.5 else 1] += 1
    return counts


def posterior_shapes(counts: list[list[int]]) -> list[tuple[int, int]]:
    return [(1 + s, 1 + f) for s, f in counts]


def aipw_estimate(
    arms: list[int],
    probs: list[list[float]],
    y: list[float],
    estimand: tuple[int, int | None],
    scheme: str,
    m0: float = 0.5,
    kappa: float = 1.0,
) -> tuple[float, float]:
    """Weighted mean and plug-in variance of augmented inverse-probability scores, from scratch."""
    k = len(probs[0])
    s, c = [0.0] * k, [0] * k
    values, weights = [], []
    plus, minus = estimand
    for i, arm in enumerate(arms):
        gamma = []
        for a in range(k):
            m = (kappa * m0 + s[a]) / (kappa + c[a])
            gamma.append(m + ((y[i] - m) / probs[i][a] if arm == a else 0.0))
        s[arm] += y[i]
        c[arm] += 1
        if minus is None:
            values.append(gamma[plus])
            inverse = 1.0 / probs[i][plus]
        else:
            values.append(gamma[plus] - gamma[minus])
            inverse = 1.0 / probs[i][plus] + 1.0 / probs[i][minus]
        weights.append(1.0 if scheme == "uniform" else 1.0 / math.sqrt(inverse))
    total = sum(weights)
    mean = sum(h * v for h, v in zip(weights, values)) / total
    variance = sum((h * (v - mean)) ** 2 for h, v in zip(weights, values)) / total**2
    return mean, variance


def sequence_radius(m: int, alpha: float, n0: float) -> float:
    return 0.5 * math.sqrt((m + n0) * math.log((m + n0) / (n0 * alpha * alpha))) / m


def sequence_bounds(values: list[float], alpha: float, n0: float) -> tuple[float, float]:
    lo, hi, total = 0.0, 1.0, 0.0
    for m, v in enumerate(values, 1):
        total += v
        r = sequence_radius(m, alpha, n0)
        lo = max(lo, total / m - r)
        hi = min(hi, total / m + r)
    return lo, hi


def welch(control: list[float], treated: list[float]) -> float:
    def moments(v: list[float]) -> tuple[int, float, float]:
        n = len(v)
        mean = sum(v) / n
        return n, mean, sum((x - mean) ** 2 for x in v) / (n - 1)

    n0, m0, v0 = moments(control)
    n1, m1, v1 = moments(treated)
    return (m1 - m0) / math.sqrt(v0 / n0 + v1 / n1)


# --------------------------------------------------------------------------- trial builders

COHORT = ("control", "treatment")


def look_of(sizes: tuple[int, ...], n: int) -> list[int]:
    return [next(k for k, size in enumerate(sizes) if u < size) for u in range(n)]


def interim(sizes: tuple[int, ...]) -> list[ai.InterimAnalysis]:
    return [ai.InterimAnalysis(s, s / sizes[-1]) for s in sizes]


def thompson_design(
    seed: int = 1, sizes: tuple[int, ...] = (30, 60), truth: tuple[float, float] = (0.3, 0.6)
) -> ai.AdaptiveDesign:
    rng = random.Random(seed)
    n = sizes[-1]
    counts = [[0, 0], [0, 0]]
    arms, rows, outcomes = [], [], []
    for _ in range(n):
        shapes = posterior_shapes(counts)
        p1 = p_second_exceeds_first(shapes[0], shapes[1])
        e = [1.0 - p1, p1]
        arm = int(rng.random() < p1)
        success = rng.random() < truth[arm]
        counts[arm][0 if success else 1] += 1
        arms.append(arm)
        rows.append(e)
        outcomes.append([1.0 if success else 0.0])
    return ai.AdaptiveDesign(
        label="thompson",
        assignment="thompson_beta_bernoulli",
        outcome="single_outcome",
        monitoring="fixed_interim_looks",
        stopping="efficacy",
        arm_labels=COHORT,
        assigned_arm=arms,
        outcomes=outcomes,
        policy_history=[ai.PolicyEpoch(0, ai.policy_identity("thompson/uniform"))],
        assignment_probabilities=rows,
        exposure=arms,
        availability=ai.Availability(look_of(sizes, n), "immediate"),
        interim_analyses=interim(sizes),
        stopping_rule=ai.StoppingRule.posterior_probability_best(0.95),
        probability_floor=1e-9,
    )


def epsilon_greedy_design(
    seed: int = 2,
    sizes: tuple[int, ...] = (50, 100),
    truth: tuple[float, float] = (0.4, 0.6),
    epsilon: float = 0.3,
    measurements: int = 1,
) -> ai.AdaptiveDesign:
    rng = random.Random(seed)
    n = sizes[-1]
    total, count = [0.0, 0.0], [0, 0]
    arms, rows, outcomes = [], [], []
    for _ in range(n):
        means = [total[a] / count[a] if count[a] else 0.5 for a in (0, 1)]
        e = [epsilon / 2, epsilon / 2]
        if means[0] == means[1]:
            e = [0.5, 0.5]
        else:
            e[int(means[1] > means[0])] += 1 - epsilon
        arm = int(rng.random() >= e[0])
        if measurements == 1:
            record = [1.0 if rng.random() < truth[arm] else 0.0]
        else:
            intercept = 0.3 * rng.gauss(0, 1)
            record = [truth[arm] + intercept + 0.5 * rng.gauss(0, 1) for _ in range(measurements)]
        total[arm] += sum(record) / len(record)
        count[arm] += 1
        arms.append(arm)
        rows.append(e)
        outcomes.append(record)
    repeated = measurements > 1
    return ai.AdaptiveDesign(
        label="epsilon_greedy",
        assignment="epsilon_greedy",
        outcome="repeated_measures" if repeated else "single_outcome",
        monitoring="final_analysis_only" if repeated else "fixed_interim_looks",
        stopping="no_stopping" if repeated else "efficacy",
        arm_labels=COHORT,
        assigned_arm=arms,
        outcomes=outcomes,
        policy_history=[ai.PolicyEpoch(0, ai.policy_identity("epsilon_greedy/0.3"))],
        assignment_probabilities=rows,
        exposure=arms,
        availability=ai.Availability(look_of((n,) if repeated else sizes, n), "immediate"),
        interim_analyses=interim((n,) if repeated else sizes),
        stopping_rule=ai.StoppingRule.no_stopping()
        if repeated
        else ai.StoppingRule.posterior_probability_best(0.95),
        probability_floor=0.05,
    )


def group_sequential_design(
    seed: int = 3, sizes: tuple[int, ...] = (40, 80, 120), effect: float = 0.5
) -> ai.AdaptiveDesign:
    rng = random.Random(seed)
    n = sizes[-1]
    arms = [u % 2 for u in range(n)]
    outcomes = [[effect * a + rng.gauss(0, 1)] for a in arms]
    return ai.AdaptiveDesign(
        label="group_sequential",
        assignment="fixed_randomization",
        outcome="single_outcome",
        monitoring="group_sequential",
        stopping="efficacy",
        arm_labels=COHORT,
        assigned_arm=arms,
        outcomes=outcomes,
        policy_history=[ai.PolicyEpoch(0, ai.policy_identity("fixed/1:1"))],
        assignment_probabilities=[[0.5, 0.5]] * n,
        exposure=arms,
        availability=ai.Availability(look_of(sizes, n), "immediate"),
        interim_analyses=interim(sizes),
        stopping_rule=ai.StoppingRule.alpha_spending(
            "obrien_fleming_type", 0.025, "one_sided_upper"
        ),
        probability_floor=0.5,
    )


def prepare(design: ai.AdaptiveDesign) -> ai.AdaptiveInferenceSession:
    return ai.AdaptiveInferenceSession.prepare(design)


def refused(call) -> AdaptiveInferenceRefusal:
    with pytest.raises(AdaptiveInferenceRefusal) as caught:
        call()
    assert_registered_refusal(caught.value)
    return caught.value


def replace(design: ai.AdaptiveDesign, **changes) -> ai.AdaptiveDesign:
    fields = {f: getattr(design, f) for f in design.__dataclass_fields__}
    fields.update(changes)
    return ai.AdaptiveDesign(**fields)


# --------------------------------------------------------------------------- Bayesian posterior


def test_the_posterior_is_the_conjugate_closed_form_of_the_counts():
    design = thompson_design()
    session = prepare(design)
    result = session.analyse(ai.BayesianPosterior(look=1))
    assert result.paradigm == "bayesian_posterior"
    assert result.claim == "point_only"
    n = len(design.assigned_arm)
    counts = counts_of(list(design.assigned_arm), [list(o) for o in design.outcomes], list(range(n)))
    shapes = posterior_shapes(counts)
    v = result.values
    for arm, (a, b) in enumerate(shapes):
        assert v[arm * 6] == a and v[arm * 6 + 1] == b
        assert v[arm * 6 + 2] == pytest.approx(a / (a + b), abs=1e-14)
        assert v[arm * 6 + 3] == pytest.approx(a * b / ((a + b) ** 2 * (a + b + 1)), abs=1e-14)
    p_best_1 = p_second_exceeds_first(shapes[0], shapes[1])
    assert v[10] == pytest.approx(p_best_1, abs=1e-8)
    assert v[4] + v[10] == pytest.approx(1.0, abs=1e-8)
    assert v[11] == pytest.approx(expected_loss(shapes[1], shapes[0]), abs=1e-7)
    assert v[5] == pytest.approx(expected_loss(shapes[0], shapes[1]), abs=1e-7)
    assert result.closed_guarantee == "adaptive_inference.posterior_interval_closed"
    # The first look saw half the data.
    early = session.analyse(ai.BayesianPosterior(look=0))
    first = counts_of(list(design.assigned_arm), [list(o) for o in design.outcomes], list(range(30)))
    assert early.values[0] == 1 + first[0][0]


def two_step_design(path: tuple[tuple[int, int], tuple[int, int]]) -> ai.AdaptiveDesign:
    """A two-unit Thompson trial; ``path`` is ``(arm, success)`` for each unit."""
    (arm1, y1), (arm2, y2) = path
    counts = [[0, 0], [0, 0]]
    counts[arm1][0 if y1 else 1] += 1
    shapes = posterior_shapes(counts)
    p1 = p_second_exceeds_first(shapes[0], shapes[1])
    rows = [[0.5, 0.5], [1.0 - p1, p1]]
    arms = [arm1, arm2]
    return ai.AdaptiveDesign(
        label="tree",
        assignment="thompson_beta_bernoulli",
        outcome="single_outcome",
        monitoring="fixed_interim_looks",
        stopping="efficacy",
        arm_labels=COHORT,
        assigned_arm=arms,
        outcomes=[[float(y1)], [float(y2)]],
        policy_history=[ai.PolicyEpoch(0, ai.policy_identity("tree"))],
        assignment_probabilities=rows,
        exposure=arms,
        availability=ai.Availability([0, 1], "immediate"),
        interim_analyses=[ai.InterimAnalysis(1, 0.5), ai.InterimAnalysis(2, 1.0)],
        stopping_rule=ai.StoppingRule.posterior_probability_best(0.95),
        probability_floor=1e-6,
    )


def test_an_enumerated_thompson_tree_has_a_martingale_posterior():
    """E over every history of the posterior mean equals the prior mean 1/2, for both arms."""
    total = mean_zero = mean_one = superiority = 0.0
    for arm1, y1, y2 in itertools.product((0, 1), (0, 1), (0, 1)):
        counts = [[0, 0], [0, 0]]
        counts[arm1][0 if y1 else 1] += 1
        shapes = posterior_shapes(counts)
        p1 = p_second_exceeds_first(shapes[0], shapes[1])
        arm2_prob = (1.0 - p1, p1)
        for arm2 in (0, 1):
            a, b = shapes[arm2]
            p_y2 = a / (a + b) if y2 else b / (a + b)
            probability = 0.5 * (0.5 if y1 else 0.5) * arm2_prob[arm2] * p_y2
            # P(arm1) = 1/2; P(y1) is 1/2 under the uniform prior.
            session = prepare(two_step_design(((arm1, y1), (arm2, y2))))
            v = session.analyse(ai.BayesianPosterior(look=1)).values
            total += probability
            mean_zero += probability * v[2]
            mean_one += probability * v[8]
            superiority += probability * v[10]
    assert total == pytest.approx(1.0, abs=1e-12)
    assert mean_zero == pytest.approx(0.5, abs=1e-9)
    assert mean_one == pytest.approx(0.5, abs=1e-9)
    assert superiority == pytest.approx(0.5, abs=1e-8)


# --------------------------------------------------------------------------- refusals


MISSING = [
    ("policy_history", "adaptive_inference.policy_history_missing"),
    ("assignment_probabilities", "adaptive_inference.assignment_probabilities_missing"),
    ("exposure", "adaptive_inference.exposure_history_missing"),
    ("availability", "adaptive_inference.outcome_availability_missing"),
    ("interim_analyses", "adaptive_inference.interim_analyses_missing"),
    ("stopping_rule", "adaptive_inference.stopping_rule_missing"),
]


@pytest.mark.parametrize(("element", "detail"), MISSING)
def test_each_missing_element_is_its_own_refusal(element, detail):
    error = refused(lambda: prepare(replace(thompson_design(), **{element: None})))
    assert error.reason_code == "adaptive_contract_incomplete"
    assert error.detail == detail
    assert error.details == (detail,)


def test_every_missing_element_at_once_names_each_exactly_once():
    error = refused(lambda: prepare(replace(thompson_design(), **{e: None for e, _ in MISSING})))
    assert sorted(error.details) == sorted(d for _, d in MISSING)


def test_fixed_sample_inference_of_adaptive_data_is_refused_but_the_posterior_runs():
    session = prepare(thompson_design())
    assert session.is_adaptive
    error = refused(session.fixed_sample)
    assert error.reason_code == "adaptive_design_ignored"
    assert error.detail == "adaptive_inference.fixed_sample_on_adaptive_data"
    assert session.analyse(ai.BayesianPosterior(look=1)).values


def test_no_interval_is_licensed_for_any_route():
    session = prepare(thompson_design())
    error = refused(session.interval)
    assert error.reason_code == "cell_not_licensed"
    assert error.detail == "adaptive_inference.interval_closed"


def test_a_paradigm_the_cell_is_not_licensed_for_is_refused():
    session = prepare(thompson_design())
    error = refused(lambda: session.analyse(ai.AdaptiveWeighting(look=1, plus_arm=1, minus_arm=0)))
    assert error.reason_code == "cell_not_licensed"
    assert error.detail == "adaptive_inference.paradigm_not_licensed_for_cell"
    error = refused(lambda: session.analyse(ai.GroupSequential(look=1)))
    assert error.detail == "adaptive_inference.paradigm_not_licensed_for_cell"


def test_noncompliance_and_outcome_dependent_delay_are_refused_before_any_number():
    design = thompson_design()
    exposure = list(design.exposure)
    exposure[3] = 1 - exposure[3]
    session = prepare(replace(design, exposure=exposure))
    error = refused(lambda: session.analyse(ai.BayesianPosterior(look=1)))
    assert error.reason_code == "adaptive_assumption_violated"
    assert error.detail == "adaptive_inference.exposure_differs_from_assignment"
    delayed = replace(
        design,
        availability=ai.Availability(list(design.availability.available_at_look), "outcome_dependent"),
    )
    error = refused(lambda: prepare(delayed).analyse(ai.BayesianPosterior(look=1)))
    assert error.detail == "adaptive_inference.delay_not_ignorable"


def test_cancellation_is_a_refusal_not_a_verdict():
    session = prepare(thompson_design())
    token = _native.CancellationToken()
    token.cancel()
    error = refused(lambda: session.analyse(ai.BayesianPosterior(look=1), cancel=token))
    assert error.reason_code == "cancelled_no_claim"


# --------------------------------------------------------------------------- weighting


@pytest.mark.parametrize("scheme", ["uniform", "variance_stabilizing"])
def test_adaptive_weighting_matches_an_independent_calculation(scheme):
    design = epsilon_greedy_design()
    session = prepare(design)
    look = 1
    result = session.analyse(
        ai.AdaptiveWeighting(look=look, plus_arm=1, minus_arm=0, scheme=scheme)
    )
    assert result.paradigm == "frequentist_adaptive_weighting"
    y = [o[0] for o in design.outcomes]
    estimate, variance = aipw_estimate(
        list(design.assigned_arm), [list(r) for r in design.assignment_probabilities], y, (1, 0), scheme
    )
    assert result.values[0] == pytest.approx(estimate, abs=1e-10)
    assert result.values[1] == pytest.approx(variance, abs=1e-10)
    assert result.values[2] == len(y)
    assert result.closed_guarantee == "adaptive_inference.interval_closed"


def test_a_probability_below_the_floor_is_refused_not_clipped():
    design = replace(epsilon_greedy_design(), probability_floor=0.4)
    error = refused(
        lambda: prepare(design).analyse(ai.AdaptiveWeighting(look=1, plus_arm=1, minus_arm=0))
    )
    assert error.reason_code == "adaptive_assumption_violated"
    assert error.detail == "adaptive_inference.positivity_floor"


def test_repeated_measures_use_the_subject_mean_as_the_unit():
    design = epsilon_greedy_design(measurements=3, sizes=(80,))
    session = prepare(design)
    assert session.procedure == "epsilon_greedy/repeated_measures/final_analysis_only/no_stopping"
    result = session.analyse(
        ai.AdaptiveWeighting(
            look=0, plus_arm=1, minus_arm=0, scheme="uniform", prior_mean=0.5, prior_strength=1.0
        )
    )
    means = [sum(o) / len(o) for o in design.outcomes]
    estimate, variance = aipw_estimate(
        list(design.assigned_arm),
        [list(r) for r in design.assignment_probabilities],
        means,
        (1, 0),
        "uniform",
    )
    assert result.values[0] == pytest.approx(estimate, abs=1e-10)
    assert result.values[1] == pytest.approx(variance, abs=1e-10)
    # Reordering the measurements within each subject changes the data identity, not the estimate.
    reversed_design = replace(design, outcomes=[list(reversed(o)) for o in design.outcomes])
    other = prepare(reversed_design)
    assert other.contract_identity != session.contract_identity
    again = other.analyse(
        ai.AdaptiveWeighting(look=0, plus_arm=1, minus_arm=0, scheme="uniform", prior_mean=0.5)
    )
    assert again.values[0] == pytest.approx(result.values[0], abs=1e-12)


def test_the_hierarchical_cluster_cell_is_carried_forward():
    design = replace(epsilon_greedy_design(measurements=3, sizes=(40,)), outcome="hierarchical_clustered")
    error = refused(
        lambda: prepare(design).analyse(ai.AdaptiveWeighting(look=0, plus_arm=1, minus_arm=0))
    )
    assert error.reason_code == "cell_not_licensed"
    assert error.detail == "adaptive_inference.cell_carried_forward"


# --------------------------------------------------------------------------- confidence sequence


def test_the_confidence_sequence_matches_the_closed_form_and_union_bounds_a_contrast():
    design = epsilon_greedy_design(sizes=(100, 200))
    session = prepare(design)
    result = session.analyse(
        ai.ConfidenceSequence(look=1, alpha=0.05, tuning_samples=50.0, contrast=(1, 0))
    )
    by_arm = [[o[0] for o, a in zip(design.outcomes, design.assigned_arm) if a == arm] for arm in (0, 1)]
    bounds = [sequence_bounds(v, 0.025, 50.0) for v in by_arm]
    v = result.values
    for arm in (0, 1):
        assert v[arm * 4] == len(by_arm[arm])
        assert v[arm * 4 + 1] == pytest.approx(sum(by_arm[arm]) / len(by_arm[arm]), abs=1e-12)
        assert v[arm * 4 + 2] == pytest.approx(bounds[arm][0], abs=1e-12)
        assert v[arm * 4 + 3] == pytest.approx(bounds[arm][1], abs=1e-12)
    assert v[8] == pytest.approx(bounds[1][0] - bounds[0][1], abs=1e-12)
    assert v[9] == pytest.approx(bounds[1][1] - bounds[0][0], abs=1e-12)
    assert result.paradigm == "anytime_valid_frequentist"


# --------------------------------------------------------------------------- group sequential


def test_group_sequential_boundaries_and_decisions_match_published_values():
    design = group_sequential_design()
    session = prepare(design)
    assert session.status["class"] == "licensed"
    result = session.analyse(ai.GroupSequential(look=2))
    v = result.values
    published = (3.7103, 2.5114, 1.9930)
    sizes = (40, 80, 120)
    arms, y = list(design.assigned_arm), [o[0] for o in design.outcomes]
    statistics = []
    for k, n in enumerate(sizes):
        control = [y[u] for u in range(n) if arms[u] == 0]
        treated = [y[u] for u in range(n) if arms[u] == 1]
        statistics.append(welch(control, treated))
        assert v[1 + 3 * k] == pytest.approx(published[k], abs=2e-4)
        assert v[2 + 3 * k] == pytest.approx(statistics[k], abs=1e-9)
    crossing = next((k for k, z in enumerate(statistics) if z >= published[k]), -1)
    assert v[0] == crossing
    spent = sum(v[3 + 3 * k] for k in range(3))
    assert spent == pytest.approx(0.025, abs=1e-9)
    error = refused(session.fixed_sample)
    assert error.reason_code == "adaptive_design_ignored"


def test_group_sequential_with_adaptive_assignment_is_a_closed_pairing():
    design = replace(
        thompson_design(),
        monitoring="group_sequential",
        stopping_rule=ai.StoppingRule.alpha_spending(
            "pocock_type", 0.05, "two_sided_symmetric"
        ),
    )
    error = refused(lambda: prepare(design).analyse(ai.BayesianPosterior(look=1)))
    assert error.detail == "adaptive_inference.pairing_not_licensed"


# --------------------------------------------------------------------------- evidence ledger


def test_interim_looks_count_each_observation_once_and_are_never_pooled():
    session = prepare(thompson_design(sizes=(30, 60)))
    evidence = session.interim_evidence()
    assert evidence.look_sizes == (30, 60)
    assert evidence.declared_slots == 90
    assert evidence.distinct_observations == 60
    assert evidence.repeated_slots == 30
    assert evidence.independent_pooling_refusal == (
        "evidence_double_counted",
        "adaptive_inference.looks_share_observations",
    )


# --------------------------------------------------------------------------- procedure matrix


AXES = {
    "assignment": [
        "fixed_randomization",
        "thompson_beta_bernoulli",
        "epsilon_greedy",
        "upper_confidence_bound",
        "contextual_bandit",
        "batched_adaptive",
    ],
    "outcome": ["single_outcome", "repeated_measures", "hierarchical_clustered"],
    "monitoring": [
        "final_analysis_only",
        "fixed_interim_looks",
        "group_sequential",
        "continuous_monitoring",
    ],
    "stopping": ["no_stopping", "efficacy", "futility", "resource", "decision_theoretic"],
}


def test_the_procedure_matrix_lists_every_cell_once_and_agrees_with_the_registry_table():
    expected_ids = {"/".join(c) for c in itertools.product(*AXES.values())}
    assert len(expected_ids) == 360
    matrix = ai.procedure_matrix()
    assert {c.id for c in matrix} == expected_ids, "a cell is unlisted or invented"
    assert len(matrix) == len(expected_ids)
    table = load_toml(REPO / "parity" / "adaptive_procedures.toml")["cell"]
    assert [row["id"] for row in table] == [c.id for c in matrix]
    for row, cell in zip(table, matrix):
        assert row["status"] == cell.status, row["id"]
        assert tuple(row.get("paradigms", ())) == cell.paradigms, row["id"]
        assert row.get("owner") == cell.owner, row["id"]
        assert row["status"] in {"licensed", "closed_incoherent", "closed_pairing", "carried_forward"}
    classes = [c.status for c in matrix]
    assert {s: classes.count(s) for s in set(classes)} == {
        "licensed": 6,
        "closed_incoherent": 144,
        "closed_pairing": 18,
        "carried_forward": 192,
    }
    assert all(c.owner for c in matrix if c.status == "carried_forward")


# --------------------------------------------------------------------------- artifacts


def test_the_monitoring_contract_round_trips_and_a_fresh_reader_sees_rule_and_paradigm():
    design = group_sequential_design()
    session = prepare(design)
    artifact = session.export_contract("frequentist_group_sequential")
    consumed = ai.consume_adaptive_monitoring_contract_artifact(artifact)
    assert consumed.paradigm == "frequentist_group_sequential"
    assert consumed.procedure == "fixed_randomization/single_outcome/group_sequential/efficacy"
    assert consumed.stopping_rule["spec"] == "alpha_spending"
    assert consumed.stopping_rule["spending"] == "obrien_fleming_type"
    assert consumed.contract_identity == session.contract_identity
    assert consumed.boundaries is not None
    for got, want in zip(consumed.boundaries["critical"], (3.7103, 2.5114, 1.9930)):
        assert got == pytest.approx(want, abs=2e-4)
    # Deterministic, and accepted against its own retained identity.
    assert session.export_contract("frequentist_group_sequential") == artifact
    again = ai.consume_adaptive_monitoring_contract_artifact(artifact, consumed.identity)
    assert again.identity == consumed.identity


def test_a_paradigm_the_design_may_not_use_cannot_be_exported():
    session = prepare(group_sequential_design())
    error = refused(lambda: session.export_contract("frequentist_fixed_sample"))
    assert error.reason_code == "adaptive_design_ignored"
    error = refused(lambda: session.export_contract("bayesian_posterior"))
    assert error.reason_code == "cell_not_licensed"


def test_the_result_artifact_is_recomputed_by_the_consumer_and_refuses_other_experiments():
    session = prepare(thompson_design())
    request = ai.BayesianPosterior(look=1)
    artifact = session.export_result(request)
    produced = session.analyse(request)
    consumed = ai.consume_adaptive_inference_result_artifact(artifact)
    assert consumed.paradigm == "bayesian_posterior"
    assert consumed.procedure == produced.procedure
    assert consumed.stopping_rule["spec"] == "posterior_probability_best"
    assert consumed.closed_guarantee == "adaptive_inference.posterior_interval_closed"
    for a, b in zip(consumed.values, produced.values):
        assert a == pytest.approx(b, abs=1e-9)
    # A different experiment (one outcome changed) is another identity: the consumer that
    # retained the first identity refuses the second artifact.
    design = thompson_design()
    outcomes = [list(o) for o in design.outcomes]
    outcomes[0] = [1.0 - outcomes[0][0]]
    other = prepare(replace(design, outcomes=outcomes))
    other_artifact = other.export_result(request)
    assert other_artifact != artifact
    error = refused(
        lambda: ai.consume_adaptive_inference_result_artifact(other_artifact, consumed.identity)
    )
    assert error.reason_code == "route_not_supported"
    assert error.detail == "adaptive_inference.wrong_contract"
    # A corrupted container is refused, not partially read.
    damaged = bytearray(artifact)
    damaged[len(damaged) // 2] ^= 0xFF
    refused(lambda: ai.consume_adaptive_inference_result_artifact(bytes(damaged)))
    refused(lambda: ai.consume_adaptive_inference_result_artifact(artifact[: len(artifact) // 2]))


def test_a_fresh_process_consumes_both_artifacts_against_the_retained_identity(tmp_path):
    session = prepare(group_sequential_design())
    request = ai.GroupSequential(look=2)
    contract = session.export_contract("frequentist_group_sequential")
    result = session.export_result(request)
    retained = ai.consume_adaptive_inference_result_artifact(result).identity
    (tmp_path / "contract.bin").write_bytes(contract)
    (tmp_path / "result.bin").write_bytes(result)
    script = (
        "import json, sys\n"
        "from antecedent import adaptive_inference as ai\n"
        "c = ai.consume_adaptive_monitoring_contract_artifact(open(sys.argv[1], 'rb').read())\n"
        "r = ai.consume_adaptive_inference_result_artifact(open(sys.argv[2], 'rb').read(), sys.argv[3])\n"
        "print(json.dumps({'procedure': c.procedure, 'paradigm': c.paradigm, "
        "'rule': c.stopping_rule['spec'], 'critical': list(c.boundaries['critical']), "
        "'identity': r.identity, 'values': list(r.values)}))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "contract.bin"), str(tmp_path / "result.bin"), retained],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=True,
    )
    wire = json.loads(done.stdout)
    assert wire["procedure"] == "fixed_randomization/single_outcome/group_sequential/efficacy"
    assert wire["paradigm"] == "frequentist_group_sequential"
    assert wire["rule"] == "alpha_spending"
    assert wire["identity"] == retained
    assert wire["critical"][0] == pytest.approx(3.7103, abs=2e-4)
    produced = session.analyse(request).values
    assert wire["values"] == pytest.approx(list(produced), abs=1e-9)
    refusal = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "contract.bin"), str(tmp_path / "result.bin"), "0" * 64],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert refusal.returncode != 0
    assert "wrong_contract" in refusal.stderr
