"""Typed (nonlinear / non-Gaussian) mechanisms carried into a target population.

The fixture ``conformance/transport/typed_mechanism_shift`` holds a source and a target sample:
income and price mechanisms are invariant, and six outcome mechanisms (Gaussian, Poisson,
Bernoulli, gamma, log-normal, heteroscedastic Gaussian, each with a quadratic price effect) shift.
``reference.py`` fits the target sample independently with numpy and scipy and evaluates the
target interventional mean by quadrature over the invariant income law; nothing from native code
decides what the truth is here.
"""

from __future__ import annotations

import json
from pathlib import Path

import antecedent as at
import numpy as np
import pytest
from antecedent import typed_mechanism_transport as tmt
from antecedent.errors import CausalTypeError, CausalValueError

FIXTURE = (
    Path(__file__).resolve().parents[2] / "conformance" / "transport" / "typed_mechanism_shift"
)
DATA = json.loads((FIXTURE / "data.json").read_text())
EXPECTED = json.loads((FIXTURE / "expected.json").read_text())
OUTCOMES = ("demand", "orders", "buys", "spend", "basket", "noise_h")
FAMILY = {
    "demand": "gaussian",
    "orders": "poisson",
    "buys": "bernoulli",
    "spend": "gamma",
    "basket": "log_normal",
    "noise_h": "heteroscedastic_gaussian",
}
INDEPENDENT = tmt.TypedDependence.independent_by_design("disjoint independently generated samples")
DRAWS = 200_000


def spec(variable: str) -> tmt.MechanismSpecification:
    if variable == "income":
        return tmt.MechanismSpecification("income")
    if variable == "price":
        return tmt.MechanismSpecification("price", parents=("income",))
    return tmt.MechanismSpecification(
        variable,
        parents=("price", "income"),
        family=FAMILY[variable],
        mean=tmt.Polynomial(degree=2),
    )


@pytest.fixture(scope="module")
def sources() -> dict[str, tmt.FittedTypedMechanism]:
    return {
        v: tmt.fit_typed_mechanism(spec(v), DATA["source"], population="source")
        for v in ("income", "price", *OUTCOMES)
    }


@pytest.fixture(scope="module")
def targets(sources) -> dict[str, tmt.FittedTypedMechanism]:
    return {
        v: sources[v].refit_in_target(DATA["target"], population="target")
        for v in ("income", "price", *OUTCOMES)
    }


def transport(shifted=OUTCOMES, **kw) -> tmt.TypedTransport:
    return tmt.TypedTransport(
        source_population="source",
        target_population="target",
        selection_nodes=(tmt.SelectionNode("S", tuple(shifted)),),
        dependence=kw.pop("dependence", INDEPENDENT),
        **kw,
    )


def level(outcome: str, price: float) -> tmt.MeanContrast:
    return tmt.MeanContrast.mean_at(outcome, {"price": price})


@pytest.fixture(scope="module")
def transported(sources, targets) -> tmt.TypedTransportedComposition:
    return tmt.transport_typed_mechanisms(
        transport(), list(sources.values()), [targets[v] for v in ("income", "price", *OUTCOMES)]
    )


@pytest.mark.parametrize("outcome", OUTCOMES)
def test_transported_response_matches_the_independent_target_fit(transported, outcome) -> None:
    o = EXPECTED["outcomes"][outcome]
    execution = at.Execution(seed=7)
    for i, p in enumerate(EXPECTED["grid"]):
        est = transported.mean_contrast(level(outcome, p), draws=DRAWS, execution=execution)
        assert abs(est.value - o["reference"][i]) <= 6 * est.monte_carlo_se + 1e-6
        band = 4.5 * np.hypot(o["reference_se"][i], est.monte_carlo_se)
        assert abs(est.value - o["truth"][i]) <= band


@pytest.mark.parametrize("outcome", OUTCOMES)
def test_parameter_uncertainty_agrees_with_the_independent_delta_method(
    transported, outcome
) -> None:
    ref = EXPECTED["outcomes"][outcome]["reference_se"][2]
    u = transported.mean_contrast_uncertainty(
        level(outcome, 2.0), draws=300_000, execution=at.Execution(seed=13)
    )
    assert abs(u.standard_error - ref) / ref < 0.08
    assert u.scope == "fitted_parameters_only_transport_assumption_not_propagated"
    assert u.dependence == "independent_by_design"
    assert np.isfinite(u.estimate.monte_carlo_se)


def test_receipt_records_carried_and_refit_components(transported) -> None:
    receipt = transported.receipt
    assert receipt.transported_variables == ("income", "price")
    assert set(receipt.native_variables) == set(OUTCOMES)
    assert receipt.support == "extrapolative" and receipt.inference == "point_only"
    assert receipt.selection_targets == tuple(sorted(OUTCOMES))
    assert receipt.untested_invariances == ()
    for d in receipt.dispositions:
        if d.role == "transported":
            assert d.covariance_source == "source_fit" and d.fitted_population == "source"
            expected = EXPECTED["invariance"][d.variable]
            assert d.consistency is not None and not d.consistency.rejected
            assert abs(d.consistency.p_value - expected["p_value"]) <= 1e-6
        else:
            assert d.covariance_source == "target_fit" and d.fitted_population == "target"
    assert receipt.as_transport_receipt().support == "extrapolative"


def test_a_shifted_mechanism_without_target_evidence_is_refused(sources, targets) -> None:
    with pytest.raises(tmt.TypedMechanismRefusal) as caught:
        tmt.transport_typed_mechanisms(
            transport(shifted=("demand",)),
            list(sources.values()),
            [targets["income"], targets["price"]],
        )
    assert caught.value.details == ("typed_mechanism_transport.shifted_without_target_evidence",)
    assert caught.value.reason_code == "transport_missing_evidence"


def test_a_wrongly_declared_invariance_is_flagged_by_target_data(sources, targets) -> None:
    shifted_price = sources["price"].refit_in_target(
        {"income": DATA["target"]["income"], "price": DATA["target"]["price_shifted"]},
        population="target",
    )
    check = tmt.check_invariance(sources["price"], shifted_price)
    assert check.rejected and check.p_value < 1e-12
    clean = tmt.check_invariance(sources["price"], targets["price"])
    assert not clean.rejected
    with pytest.raises(tmt.TypedMechanismRefusal) as caught:
        tmt.transport_typed_mechanisms(
            transport(),
            list(sources.values()),
            [targets["income"], shifted_price, *(targets[v] for v in OUTCOMES)],
        )
    assert caught.value.details == ("typed_mechanism_transport.invariance_contradicted",)


def test_an_invariance_without_target_data_is_recorded_as_untested(sources, targets) -> None:
    done = tmt.transport_typed_mechanisms(
        transport(),
        list(sources.values()),
        [targets["income"], *(targets[v] for v in OUTCOMES)],
    )
    assert done.receipt.untested_invariances == ("price",)


def test_unknown_dependence_refuses_the_variance_but_not_the_point(sources, targets) -> None:
    done = tmt.transport_typed_mechanisms(
        transport(dependence=tmt.TypedDependence.unknown()),
        list(sources.values()),
        [targets[v] for v in ("income", "price", *OUTCOMES)],
    )
    assert done.mean_contrast(level("demand", 1.0), draws=5_000).draws == 5_000
    with pytest.raises(tmt.TypedMechanismRefusal) as caught:
        done.mean_contrast_uncertainty(level("demand", 1.0), draws=5_000)
    assert caught.value.details == ("typed_mechanism_transport.parameter_dependence_unknown",)


def test_the_interval_route_is_closed(transported) -> None:
    with pytest.raises(tmt.TypedMechanismRefusal) as caught:
        transported.interval()
    assert caught.value.reason_code == "cell_not_licensed"


def test_identity_is_verified_only_against_what_it_re_derives(
    sources, targets, transported
) -> None:
    args = (
        transport(),
        list(sources.values()),
        [targets[v] for v in ("income", "price", *OUTCOMES)],
    )
    receipt = tmt.verify_typed_transport(*args, expected_identity=transported.identity)
    assert receipt.identity == transported.identity
    with pytest.raises(tmt.TypedMechanismRefusal) as caught:
        tmt.verify_typed_transport(*args, expected_identity="0" * 64)
    assert caught.value.details == ("typed_mechanism_transport.transport_identity_mismatch",)


def test_fit_refuses_support_violations_and_separation() -> None:
    x = [i / 10 for i in range(40)]
    bad = tmt.MechanismSpecification("n", parents=("x",), family="poisson")
    with pytest.raises(tmt.TypedMechanismRefusal):
        tmt.fit_typed_mechanism(bad, {"x": x, "n": [-1.0] + [1.0] * 39}, population="p")
    separated = tmt.MechanismSpecification("n", parents=("x",), family="bernoulli")
    with pytest.raises(tmt.TypedMechanismRefusal) as caught:
        tmt.fit_typed_mechanism(separated, {"x": x, "n": [0.0] * 20 + [1.0] * 20}, population="p")
    assert caught.value.reason_code == "mechanism_fit_not_converged"


def test_an_expression_mean_is_fitted_and_reported_with_its_covariance() -> None:
    rng = np.random.default_rng(3)
    x = rng.uniform(-2, 2, 3000)
    y = 0.5 + 3.0 * np.tanh(1.5 * x) + 0.3 * rng.standard_normal(3000)
    expression = {
        "op": "add",
        "terms": [
            {"op": "parameter", "parameter": 0},
            {
                "op": "multiply",
                "factors": [
                    {"op": "parameter", "parameter": 1},
                    {
                        "op": "tanh",
                        "argument": {
                            "op": "multiply",
                            "factors": [
                                {"op": "parameter", "parameter": 2},
                                {"op": "variable", "variable": 0},
                            ],
                        },
                    },
                ],
            },
        ],
    }
    fit = tmt.fit_typed_mechanism(
        tmt.MechanismSpecification(
            "y", parents=("x",), mean=tmt.Expression(expression, start=(0.0, 1.0, 1.0))
        ),
        {"x": x, "y": y},
        population="p",
    )
    sd = np.sqrt(np.diag(fit.covariance))[:3]
    assert np.all(np.abs(fit.parameters[:3] - np.array([0.5, 3.0, 1.5])) < 4.5 * sd)
    assert fit.conditional_mean([1.0]) == pytest.approx(0.5 + 3.0 * np.tanh(1.5), abs=0.1)


def test_declarations_are_validated() -> None:
    with pytest.raises(CausalValueError):
        tmt.MechanismSpecification("y", family="weibull")
    with pytest.raises(CausalTypeError):
        tmt.MechanismSpecification("y", parents="x")
    with pytest.raises(CausalTypeError):
        tmt.transport_typed_mechanisms("not a transport", [])


def test_the_module_resolves_lazily_on_the_package_root() -> None:
    assert at.typed_mechanism_transport is tmt
