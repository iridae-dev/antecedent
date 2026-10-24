"""Uncertainty carried through a chain of transformations, through the Python surface.

Truth is closed form or an independent numpy computation on the same draws: a normal net benefit
under an exponential utility (``1 - exp(-m + v / 2)``), the 2^3 sign design whose pooled variance
is ``(v + c) / 2``, the EVSI of a normal-normal study (``s (phi(d) - d Phi(-d))``), enumerated
couplings of two {0, 2} marginals, and interval arithmetic by hand. The rule, the arithmetic, the
identity and every refusal are Rust's; Python must agree with the numpy reference bit for bit where
the arithmetic is the same and within a stated band where it is a closed form on a finite grid.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pytest
from antecedent import propagation_chain as pc
from antecedent.errors import CausalValueError

R = pc.PropagationChainRefusal
ROWS = 1597
STRIDE = 987
TABLE = "joint-table"
_NORMAL = NormalDist()


def _grid(mean: float, sd: float) -> np.ndarray:
    """Mid-quantile grid of N(mean, sd^2)."""
    return np.array([mean + sd * _NORMAL.inv_cdf((i + 0.5) / ROWS) for i in range(ROWS)])


def _partner(mean: float, sd: float) -> np.ndarray:
    """A Fibonacci-lattice partner of ``_grid``: draws of an independent normal."""
    base = _grid(mean, sd)
    return base[(np.arange(ROWS) * STRIDE) % ROWS]


def _release() -> pc.PropagationChain:
    effect = pc.draws(
        "effect",
        _grid(0.5, 0.6),
        meaning="posterior",
        alignment=TABLE,
        factors=[("row", "trial-rows")],
    )
    cost = pc.draws(
        "cost",
        _partner(0.2, 0.3),
        meaning="posterior",
        alignment=TABLE,
        factors=[("prior", "cost-prior")],
    )
    stages = [
        pc.affine("mapped", "measurement_mapping", ["effect"], scale=0.8, shift=0.1),
        pc.affine("transported", "transport", ["mapped"], scale=0.9, shift=0.0),
        pc.binary("net", "derived_quantity", ["transported", "cost"], "difference"),
        pc.unary(
            "utility",
            "decision_utility",
            ["net"],
            "exponential_utility",
            risk_aversion=1.0,
            decision_reference=0.0,
        ),
        pc.affine("cost_doubled", "derived_quantity", ["cost"], scale=2.0, shift=0.0),
    ]
    return pc.propagate([effect, cost], stages, output="utility")


def test_four_stage_chain_agrees_with_numpy_and_the_closed_form() -> None:
    chain = _release()
    assert chain.label == "uncalibrated_spread"
    effect, cost = _grid(0.5, 0.6), _partner(0.2, 0.3)
    net = 0.9 * (0.8 * effect + 0.1) - cost
    utility = 1.0 - np.exp(-net)
    assert chain.stage("net").value("mean") == pytest.approx(net.mean(), rel=1e-12)
    assert chain.stage("net").value("variance") == pytest.approx(net.var(ddof=1), rel=1e-12)
    assert chain.stage("utility").value("mean") == pytest.approx(utility.mean(), rel=1e-12)
    mean, variance = 0.25, 0.72**2 * 0.36 + 0.09
    truth = 1.0 - math.exp(-mean + variance / 2)
    assert chain.answer.value("mean") == pytest.approx(truth, abs=5e-3)
    # g(E[X]) != E[g(X)]: the utility of the mean is a different number.
    assert abs((1.0 - math.exp(-mean)) - truth) > 0.1
    assert chain.answer.verdict == "act"
    assert chain.stage("utility").values["n_rows"] == ROWS


def test_export_is_consumed_independently_and_edits_are_refused() -> None:
    chain = _release()
    consumed = pc.consume_propagation_chain(chain.export(), expected_identity=chain.identity)
    assert consumed.identity == chain.identity
    assert consumed.results == chain.results
    assert consumed.answer.value("mean") == chain.answer.value("mean")
    with pytest.raises(R):
        pc.consume_propagation_chain(chain.export(), expected_identity="0" * 64)
    with pytest.raises(R):
        pc.consume_propagation_chain(chain.export()[:-7])


def _overlapping_estimates() -> tuple[np.ndarray, np.ndarray]:
    t, e1, e2 = (np.array(c, dtype=float) for c in zip(*np.ndindex(2, 2, 2), strict=True))
    t, e1, e2 = 2 * t - 1, 2 * e1 - 1, 2 * e2 - 1
    return 10 + t + e1, 10 + t + e2


def test_overlapping_evidence_is_counted_once() -> None:
    a, b = _overlapping_estimates()
    sources = [
        pc.draws("a", a, meaning="sampling", alignment=TABLE, factors=[("row", "trial-rows")]),
        pc.draws("b", b, meaning="sampling", alignment=TABLE, factors=[("row", "trial-rows")]),
    ]
    with pytest.raises(R) as refused:
        pc.propagate(sources, [pc.pool("pooled", ["a", "b"], dependent=False)], output="pooled")
    assert refused.value.code == "evidence_double_counted"
    assert refused.value.detail == "propagation_chain.pool_shares_evidence"
    assert refused.value.next_attempt == "pool_with_joint_covariance"
    chain = pc.propagate(sources, [pc.pool("pooled", ["a", "b"], dependent=True)], output="pooled")
    pooled = chain.stage("pooled")
    assert pooled.value("weight_0") == pytest.approx(0.5)
    assert pooled.value("joint_variance") == pytest.approx(12 / 7, rel=1e-12)
    assert pooled.value("independent_variance") == pytest.approx(8 / 7, rel=1e-12)
    assert pooled.value("independent_variance") < pooled.value("variance")
    # numpy agrees with the pooled column's variance.
    assert pooled.value("variance") == pytest.approx(np.var(0.5 * a + 0.5 * b, ddof=1), rel=1e-12)


def test_unaligned_draws_are_not_paired_by_index() -> None:
    sources = [
        pc.draws("x", [0.0, 1.0, 2.0], meaning="posterior", alignment="one"),
        pc.draws("y", [2.0, 1.0, 0.0], meaning="posterior", alignment="two"),
    ]
    stage = pc.binary("sum", "derived_quantity", ["x", "y"], "sum")
    with pytest.raises(R) as refused:
        pc.propagate(sources, [stage], output="sum")
    assert refused.value.code == "joint_law_required"
    assert refused.value.next_attempt == "build_independent_product_table"


def test_the_same_marginals_under_different_couplings_decide_oppositely() -> None:
    def run(x: list[float], y: list[float]) -> float:
        stages = [
            pc.binary("product", "derived_quantity", ["x", "y"], "product"),
            pc.affine("shifted", "derived_quantity", ["product"], scale=1.0, shift=-1.0),
            pc.affine("utility", "decision_utility", ["shifted"]),
        ]
        sources = [
            pc.draws("x", x, meaning="posterior", alignment=TABLE),
            pc.draws("y", y, meaning="posterior", alignment=TABLE),
        ]
        return pc.propagate(sources, stages, output="utility").answer.value("mean")

    assert run([0, 0, 2, 2], [0, 0, 2, 2]) == pytest.approx(1.0)
    assert run([0, 0, 2, 2], [2, 2, 0, 0]) == pytest.approx(-1.0)


def test_independent_sources_join_by_their_product() -> None:
    table = pc.independent_product_table([[0.0, 2.0], [0.0, 2.0]])
    xs, ys = np.meshgrid([0.0, 2.0], [0.0, 2.0], indexing="ij")
    assert np.array_equal(table[0], xs.ravel())
    assert np.array_equal(table[1], ys.ravel())
    sources = [
        pc.draws("x", table[0], meaning="posterior", alignment="product"),
        pc.draws("y", table[1], meaning="posterior", alignment="product"),
    ]
    stage = pc.binary("product", "derived_quantity", ["x", "y"], "product")
    chain = pc.propagate(
        sources, [stage], output="product", construction="independent_product"
    )
    assert chain.answer.value("mean") == pytest.approx(1.0)
    with pytest.raises(R) as refused:
        pc.independent_product_table([[1.0] * 10, [1.0] * 10], max_rows=50)
    assert refused.value.next_attempt == "raise_budget_or_thin_draws"


def test_changed_assumptions_reach_exactly_the_dependent_stages() -> None:
    chain = _release()
    assert chain.dependents_of("cost") == ["net", "utility", "cost_doubled"]
    assert chain.dependents_of("effect") == ["mapped", "transported", "net", "utility"]
    withdrawn = chain.withdraw("cost")
    for name in ("mapped", "transported"):
        assert withdrawn.stage(name).status == "current"
        assert withdrawn.stage(name).values == chain.stage(name).values
    for name in ("net", "utility", "cost_doubled"):
        assert withdrawn.stage(name).status == "invalidated"
        assert withdrawn.stage(name).invalidated_by == ("cost",)
        assert not withdrawn.stage(name).values
    tilt = 1.0 * _grid(0.5, 0.6)
    reweighted = chain.reweight("effect", tilt)
    assert reweighted.stage("mapped").status == "reweighted"
    assert reweighted.stage("mapped").value("mean") == pytest.approx(0.8 * 0.86 + 0.1, abs=1.5e-2)
    assert reweighted.stage("cost_doubled").status == "current"
    assert reweighted.stage("cost_doubled").values == chain.stage("cost_doubled").values
    # The original is not modified and the changed record replays.
    assert chain.stage("net").status == "current"
    consumed = pc.consume_propagation_chain(reweighted.export())
    assert consumed.results == reweighted.results
    with pytest.raises(R):
        chain.withdraw("nowhere")
    with pytest.raises(R):
        chain.reweight("effect", [0.0, 1.0])


def _evsi_reference(theta: np.ndarray, noise: np.ndarray, sd: float, n: float) -> float:
    """The preposterior value, computed with a dense matrix in numpy."""
    scale = sd / math.sqrt(n)
    observed = theta + scale * noise
    log_likelihood = -0.5 * ((observed[:, None] - theta[None, :]) / scale) ** 2
    q = np.exp(log_likelihood - log_likelihood.max(axis=1, keepdims=True))
    updated = (q * theta[None, :]).sum(axis=1) / q.sum(axis=1)
    return float(np.maximum(updated, 0.0).mean() - max(theta.mean(), 0.0))


def test_a_future_study_updates_the_posterior_before_the_decision() -> None:
    theta, noise = _grid(0.5, 1.0), _partner(0.0, 1.0)
    sd = math.sqrt(12.0)
    sources = [pc.draws("theta", theta, meaning="posterior", alignment=TABLE)]
    stages = [
        pc.affine("utility", "decision_utility", ["theta"], decision_reference=0.0),
        pc.preposterior(
            "update",
            "theta",
            "utility",
            noise_sd=sd,
            sample_size=4.0,
            reference_utility=0.0,
            noise=noise,
        ),
    ]
    chain = pc.propagate(sources, stages, output="update")
    update = chain.answer
    reference = _evsi_reference(theta, noise, sd, 4.0)
    assert update.value("value_of_information") == pytest.approx(reference, rel=1e-9, abs=1e-12)
    d = 1.0
    closed = 0.5 * (_NORMAL.pdf(d) - d * _NORMAL.cdf(-d))
    assert update.value("value_of_information") == pytest.approx(closed, abs=2e-2)
    assert update.value("current_expected_utility") == pytest.approx(0.5, abs=1e-3)
    assert update.value("probability_decision_changes") > 0.05
    consumed = pc.consume_propagation_chain(chain.export())
    assert consumed.answer.values == update.values


def test_the_representation_must_carry_the_uncertainty() -> None:
    stage = pc.unary("u", "decision_utility", ["a"], "exp")
    sampling = pc.draws("a", [0.0, 1.0, 2.0], meaning="sampling", alignment=TABLE)
    with pytest.raises(R) as refused:
        pc.propagate([sampling], [stage], output="u", reading="posterior")
    assert refused.value.code == "uncertainty_representation_insufficient"
    assert refused.value.next_attempt == "supply_posterior_draws"
    # A sampling law is not updated by a future study.
    update = pc.preposterior(
        "update", "a", "u", noise_sd=1.0, sample_size=1.0, reference_utility=0.0, noise=[0, 0, 0]
    )
    with pytest.raises(R) as refused:
        pc.propagate([sampling], [stage, update], output="update")
    assert refused.value.next_attempt == "supply_posterior_draws"
    # A point has a value and no spread.
    fixed_point = pc.point("p", 3.0)
    doubled = pc.affine("d", "derived_quantity", ["p"], scale=2.0, shift=1.0)
    assert pc.propagate([fixed_point], [doubled], output="d", reading="point").answer.value(
        "mean"
    ) == pytest.approx(7.0)
    with pytest.raises(R) as refused:
        pc.propagate([fixed_point], [doubled], output="d", reading="spread")
    assert refused.value.next_attempt == "supply_draws_for_fixed_input"
    # An interval composed with anything has no interval reading without a coverage record.
    with pytest.raises(R) as refused:
        _release().interval()
    assert refused.value.code == "uncertainty_representation_insufficient"
    assert refused.value.next_attempt == "measure_composed_coverage"


def test_intervals_run_into_an_enclosure_under_every_coupling() -> None:
    sources = [pc.interval("effect", 0.0, 1.0), pc.interval("cost", 0.1, 0.4), pc.point("rate", 2.0)]
    stages = [
        pc.binary("scaled", "transport", ["effect", "rate"], "product"),
        pc.binary("net", "derived_quantity", ["scaled", "cost"], "difference"),
        pc.unary("utility", "decision_utility", ["net"], "exponential_utility", risk_aversion=1.0),
    ]
    chain = pc.propagate(sources, stages, output="utility", reading="enclosure")
    assert chain.label == "propagation_enclosure"
    assert chain.stage("net").value("lower") == pytest.approx(-0.4)
    assert chain.stage("net").value("upper") == pytest.approx(1.9)
    assert chain.answer.value("lower") == pytest.approx(1.0 - math.exp(0.4))
    assert chain.answer.value("upper") == pytest.approx(1.0 - math.exp(-1.9))
    for reading in ("spread", "posterior"):
        with pytest.raises(R) as refused:
            pc.propagate(sources, stages, output="utility", reading=reading)
        assert refused.value.code == "uncertainty_representation_insufficient"
    with pytest.raises(R):
        chain.reweight("effect", [0.0, 0.0])


def test_declarations_are_validated_before_the_native_layer() -> None:
    with pytest.raises(CausalValueError):
        pc.draws("a", [0.0, float("nan")], meaning="posterior", alignment=TABLE)
    with pytest.raises(CausalValueError):
        pc.draws("a", [0.0, 1.0], meaning="bootstrap", alignment=TABLE)
    with pytest.raises(CausalValueError):
        pc.interval("a", 1.0, 0.0)
    with pytest.raises(CausalValueError):
        pc.binary("s", "derived_quantity", ["a", "b"], "power")


def test_a_chain_compiles_before_it_runs_and_the_result_carries_its_program() -> None:
    effect = pc.draws("effect", _grid(0.5, 0.6), meaning="posterior", alignment=TABLE)
    stages = [
        pc.affine("mapped", "measurement_mapping", ["effect"], scale=0.8, shift=0.1),
        pc.unary("utility", "decision_utility", ["mapped"], "exponential_utility", risk_aversion=1.0),
    ]
    view = pc.plan_view([effect], stages, output="utility")
    assert view["route"] == "PropagationChain"
    assert view["family"] == "propagation_chain"
    chain = pc.propagate([effect], stages, output="utility")
    assert chain.program_identity == view["identity"]
    # A changed declaration is a different program.
    other = pc.plan_view([effect], stages, output="utility", reading="point")
    assert other["identity"] != view["identity"]
    # A change is compiled and executed as a new program.
    assert chain.reweight("effect", 0.5 * _grid(0.5, 0.6)).program_identity != chain.program_identity
    with pytest.raises(R):
        pc.plan_view([effect], stages, output="utility", reading="interval")
