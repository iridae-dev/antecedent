"""Fitted linear-Gaussian mechanisms carried into a target population's composition under a
declared selection diagram.

Scenario. A source population fits ``X``, ``M | X`` and ``Y | X, M`` by least squares. In the
target population a selection node ``S`` points into ``M`` only: the mechanism of ``M`` differs
there (it is refitted on target data) while the mechanisms of ``X`` and ``Y`` are invariant and
are carried from the source fit.

Independent oracles in this file: ordinary least squares per mechanism computed with numpy, the
joint law of the resulting structural system by path algebra ``V = (I - B)^-1 (c + e)``, and the
delta-method variance ``g' Sigma g`` with ``g`` from the closed form of the interventional mean
``E[Y | do(X = x)] = a_Y + b_YX x + b_YM (a_M + b_MX x)``. No value is read from native code to
decide what the truth is.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
from antecedent import mechanism_composition as mc
from antecedent.errors import CausalTypeError, CausalUnsupportedError
from antecedent.mechanism_composition import MechanismCompositionRefusal
from antecedent.model import fit_gcm
from antecedent.outcomes import require_success

TOL = 1e-9
ROLES = {"X": "treatment", "M": "mediator", "Y": "outcome"}
EDGES = [("X", "M"), ("X", "Y"), ("M", "Y")]
INTERCEPT = mc.ParameterTerm.intercept()
SCOPE = "fitted_parameters_only_transport_assumption_not_propagated"


def coef(parent: str) -> mc.ParameterTerm:
    return mc.ParameterTerm.coefficient(parent)


def marginal(terms, covariance) -> mc.MechanismUncertainty:
    return mc.MechanismUncertainty(
        parameters=tuple(terms), covariance=tuple(covariance), meaning="sampling_distribution"
    )


def quantity(variable: str, population: str = "pop") -> mc.Quantity:
    return mc.Quantity(
        variable,
        units="u",
        population_id=population,
        role=ROLES.get(variable, "covariate"),
        variable_name=f"label of {variable}",
    )


def declaration(variable: str, parents=(), *, suffix: str = "") -> mc.MechanismDeclaration:
    return mc.MechanismDeclaration(
        output=quantity(variable),
        parents=tuple(quantity(p) for p in parents),
        artifact_id=f"fit:{variable}{suffix}",
        evidence_id=f"data:{variable}{suffix}",
    )


def world(n: int, seed: int, m_intercept: float, m_slope: float):
    rng = np.random.default_rng(seed)
    x = 1.0 + 2.0 * rng.standard_normal(n)
    m = m_intercept + m_slope * x + rng.standard_normal(n)
    y = -1.0 + 0.5 * x + 3.0 * m + 0.5 * rng.standard_normal(n)
    return x, m, y


def ols(target, regressors):
    n = len(target)
    design = np.column_stack([np.ones(n), *regressors])
    beta = np.linalg.lstsq(design, target, rcond=None)[0]
    resid = target - design @ beta
    # The fitted sigma is the unbiased residual scale, divisor n - (number of coefficients).
    return beta, float(np.sqrt(resid @ resid / (n - design.shape[1])))


def scm_law(spec: dict, do: dict | None = None):
    names = sorted(spec)
    index = {v: i for i, v in enumerate(names)}
    k = len(names)
    c = np.zeros(k)
    b = np.zeros((k, k))
    s = np.zeros(k)
    for v, (intercept, sigma, parents) in spec.items():
        i = index[v]
        if do and v in do:
            c[i], s[i] = do[v], 0.0
            continue
        c[i], s[i] = intercept, sigma
        for p, value in parents.items():
            b[i, index[p]] = value
    inv = np.linalg.inv(np.eye(k) - b)
    return names, inv @ c, inv @ np.diag(s**2) @ inv.T


SOURCE = world(600, 11, 0.5, 2.0)
TARGET = world(600, 29, 1.5, 0.5)
BX, SX = ols(SOURCE[0], [])
BM_SOURCE, SM_SOURCE = ols(SOURCE[1], [SOURCE[0]])
BY, SY = ols(SOURCE[2], [SOURCE[0], SOURCE[1]])
BM_TARGET, SM_TARGET = ols(TARGET[1], [TARGET[0]])

# The target composition's structural system: X and Y carried, M refitted in the target.
TARGET_SPEC = {
    "X": (BX[0], SX, {}),
    "M": (BM_TARGET[0], SM_TARGET, {"X": BM_TARGET[1]}),
    "Y": (BY[0], SY, {"X": BY[1], "M": BY[2]}),
}
SOURCE_SPEC = {
    "X": (BX[0], SX, {}),
    "M": (BM_SOURCE[0], SM_SOURCE, {"X": BM_SOURCE[1]}),
    "Y": (BY[0], SY, {"X": BY[1], "M": BY[2]}),
}


@pytest.fixture(scope="module")
def source_gcm():
    return fit_gcm(["X", "M", "Y"], list(SOURCE), EDGES)


@pytest.fixture(scope="module")
def target_gcm():
    return fit_gcm(["X", "M", "Y"], list(TARGET), EDGES)


def source_mechanisms(gcm, *, uncertain: bool = False) -> list[mc.Mechanism]:
    uncertainty = {
        "X": marginal([INTERCEPT], [0.25]) if uncertain else None,
        "M": marginal([INTERCEPT, coef("X")], [0.04, -0.01, -0.01, 0.01]) if uncertain else None,
        "Y": marginal([INTERCEPT, coef("X"), coef("M")], [0.09, 0, 0, 0, 0.01, 0, 0, 0, 0.0025])
        if uncertain
        else None,
    }
    return [
        mc.Mechanism.fitted_linear(declaration("X"), gcm, uncertainty=uncertainty["X"]),
        mc.Mechanism.fitted_linear(declaration("M", ["X"]), gcm, uncertainty=uncertainty["M"]),
        mc.Mechanism.fitted_linear(declaration("Y", ["X", "M"]), gcm, uncertainty=uncertainty["Y"]),
    ]


def source_composition(
    gcm, *, uncertain: bool = False, independent: bool | None = None
) -> mc.PreparedMechanismComposition:
    declared_independent = uncertain if independent is None else independent
    return mc.compose_mechanisms(
        source_mechanisms(gcm, uncertain=uncertain),
        target_population_id="pop",
        dependence=mc.Dependence.independent_by_design("disjoint trials, independently generated")
        if declared_independent
        else None,
    )


def refit_m(target_gcm, *, uncertain: bool = False) -> mc.MechanismRefit:
    return mc.MechanismRefit.fitted_linear(
        "M",
        target_gcm,
        artifact_id="fit:M:target",
        evidence_id="data:M:target",
        uncertainty=marginal([INTERCEPT, coef("X")], [0.04, -0.01, -0.01, 0.01])
        if uncertain
        else None,
    )


def transport(
    *,
    into=("M",),
    refits=(),
    invariances=(),
    dependence=None,
    target: str = "clinic_b",
) -> mc.MechanismTransport:
    return mc.MechanismTransport(
        target_population_id=target,
        selection_nodes=(mc.SelectionNode("S", tuple(into)),),
        target_refits=tuple(refits),
        invariances=tuple(invariances),
        dependence=dependence,
    )


def refusal_of(call) -> MechanismCompositionRefusal:
    with pytest.raises(MechanismCompositionRefusal) as caught:
        call()
    assert isinstance(caught.value, CausalUnsupportedError)
    return caught.value


# ------------------------------------------------------------------------------ values


def test_carried_and_refitted_components_give_the_path_algebra_law(source_gcm, target_gcm) -> None:
    source = source_composition(source_gcm)
    transported = source.transport_to(transport(refits=[refit_m(target_gcm)]))

    receipt = transported.receipt
    assert receipt.route == "mechanism"
    assert transported.identity == receipt.identity and len(receipt.identity) == 64
    assert receipt.source_plan_identity == source.plan_identity
    assert receipt.source_plan_identity_version == source.plan_identity_version == 2
    assert receipt.source_population == "pop" and receipt.target_population == "clinic_b"
    assert receipt.selection_targets == ("M",)
    assert receipt.transported_variables == ("X", "Y")
    assert receipt.native_variables == ("M",)
    by_variable = {d.variable: d for d in receipt.dispositions}
    assert by_variable["X"].role == "transported" and by_variable["X"].fitted_population == "pop"
    assert by_variable["M"].role == "native_in_target"
    assert by_variable["M"].fitted_population == "clinic_b"
    assert by_variable["M"].evidence_id == "data:M:target"
    assert receipt.support == "extrapolative" and receipt.inference == "point_only"
    assert receipt.uncertainty_scope == SCOPE
    assert transported.native_target_data is False
    raw = json.loads(transported._handle.receipt_json())
    assert raw.pop("source_plan_identity_version") == 2
    legacy = mc._receipt_of(raw)
    assert legacy.source_plan_identity_version is None
    with pytest.raises(ValueError, match="unknown or legacy mechanism plan identity version"):
        legacy.as_transport_receipt()
    shared = receipt.as_transport_receipt()
    assert shared.route == "mechanism" and shared.transport_identities == (receipt.identity,)
    assert shared.source_identity == f"antecedent.mechanism_plan.v2:{source.plan_identity}"
    for version in (None, 1, 3, 2.0, True, "2"):
        with pytest.raises(ValueError, match="unknown or legacy mechanism plan identity version"):
            dataclasses.replace(
                receipt, source_plan_identity_version=version
            ).as_transport_receipt()
    assert type(shared)._from_wire(shared._wire()).source_identity == shared.source_identity

    # The value is the target structural system's, not the source's.
    ate = mc.MeanContrast.ate("Y", "X", 1.0, 0.0)
    expected = BY[1] + BY[2] * BM_TARGET[1]
    assert require_success(transported.mean_contrast(ate)) == pytest.approx(expected, abs=TOL)
    _, mean_one, _ = scm_law(TARGET_SPEC, {"X": 1.0})
    _, mean_zero, _ = scm_law(TARGET_SPEC, {"X": 0.0})
    names = sorted(TARGET_SPEC)
    assert (mean_one - mean_zero)[names.index("Y")] == pytest.approx(expected, abs=1e-8)
    assert expected != pytest.approx(BY[1] + BY[2] * BM_SOURCE[1], abs=0.5)
    # The source composition is untouched by the transport.
    assert require_success(source.mean_contrast(ate)) == pytest.approx(
        BY[1] + BY[2] * BM_SOURCE[1], abs=TOL
    )


def test_a_refit_replaces_the_carried_mechanism_and_is_never_averaged(
    source_gcm, target_gcm
) -> None:
    source = source_composition(source_gcm)
    # No selection arrow at all: every mechanism is carried unless a refit is supplied; the
    # refit then wins.
    carried = source.transport_to(transport(into=()))
    assert carried.receipt.transported_variables == ("M", "X", "Y")
    ate = mc.MeanContrast.ate("Y", "X", 1.0, 0.0)
    assert require_success(carried.mean_contrast(ate)) == pytest.approx(
        BY[1] + BY[2] * BM_SOURCE[1], abs=TOL
    )
    replaced = source.transport_to(transport(into=(), refits=[refit_m(target_gcm)]))
    assert replaced.receipt.native_variables == ("M",)
    assert require_success(replaced.mean_contrast(ate)) == pytest.approx(
        BY[1] + BY[2] * BM_TARGET[1], abs=TOL
    )
    assert replaced.identity != carried.identity


def test_changed_numbers_at_an_unchanged_shape_move_the_identity(source_gcm, target_gcm) -> None:
    source = source_composition(source_gcm)
    base = source.transport_to(transport(refits=[refit_m(target_gcm)]))
    other_target = fit_gcm(["X", "M", "Y"], list(world(600, 71, 2.5, -0.5)), EDGES)
    moved = source.transport_to(transport(refits=[refit_m(other_target)]))
    assert moved.identity != base.identity
    assert moved.receipt.transported_variables == base.receipt.transported_variables
    # The same declaration is re-derived to the same identity by a fresh consumer.
    again = require_success(
        source.verify_transport(
            transport(refits=[refit_m(target_gcm)]), expected_identity=base.identity
        )
    )
    assert again == base.receipt


def test_a_transport_is_verified_only_against_the_identity_it_re_derives(
    source_gcm, target_gcm
) -> None:
    source = source_composition(source_gcm)
    declared = transport(refits=[refit_m(target_gcm)])
    base = source.transport_to(declared)
    assert (
        require_success(source.verify_transport(declared, expected_identity=base.identity))
        == base.receipt
    )
    # The identity of a different declaration is not the identity of this one.
    other = source.transport_to(transport(into=()))
    assert other.identity != base.identity
    mismatch = refusal_of(
        lambda: require_success(source.verify_transport(declared, expected_identity=other.identity))
    )
    assert mismatch.detail == "mechanism_transport.transport_identity_mismatch"
    with pytest.raises(CausalTypeError):
        source.verify_transport("not a transport", expected_identity=base.identity)  # type: ignore[arg-type]


# ------------------------------------------------------------------------------ uncertainty


def test_variance_is_parameter_only_scoped_and_has_no_interval(source_gcm, target_gcm) -> None:
    source = source_composition(source_gcm, uncertain=True)
    transported = source.transport_to(
        transport(
            refits=[refit_m(target_gcm, uncertain=True)],
            dependence=mc.Dependence.independent_by_design("disjoint trials, independently fitted"),
        )
    )
    contrast = mc.MeanContrast.mean_at("Y", {"X": 2.0})
    a_m, b_mx = BM_TARGET
    _a_x, a_y, b_yx, b_ym = BX[0], BY[0], BY[1], BY[2]
    # Parameter order: (a_M, b_MX, a_X, a_Y, b_YX, b_YM); closed-form gradient of
    # E[Y | do(X = 2)] = a_Y + 2 b_YX + b_YM (a_M + 2 b_MX).
    gradient = np.array([b_ym, 2.0 * b_ym, 0.0, 1.0, 2.0, a_m + 2.0 * b_mx])
    sigma = np.zeros((6, 6))
    sigma[0:2, 0:2] = [[0.04, -0.01], [-0.01, 0.01]]
    sigma[2, 2] = 0.25
    sigma[3:6, 3:6] = np.diag([0.09, 0.01, 0.0025])
    variance = float(gradient @ sigma @ gradient)

    value = require_success(transported.mean_contrast_uncertainty(contrast))
    assert value.point == pytest.approx(a_y + 2.0 * b_yx + b_ym * (a_m + 2.0 * b_mx), abs=TOL)
    assert value.variance == pytest.approx(variance, abs=1e-8)
    assert value.standard_error == pytest.approx(np.sqrt(variance), abs=1e-8)
    assert value.scope == SCOPE
    assert value.method == "first_order_delta_method"
    closed = refusal_of(transported.interval)
    assert closed.reason_code == "cell_not_licensed"
    assert closed.detail == "mechanism_transport.interval_route_closed"


def test_an_unknown_dependence_refuses_the_variance_but_not_the_point(
    source_gcm, target_gcm
) -> None:
    source = source_composition(source_gcm, uncertain=True, independent=False)
    transported = source.transport_to(transport(refits=[refit_m(target_gcm, uncertain=True)]))
    contrast = mc.MeanContrast.mean_at("Y", {"X": 2.0})
    # The point is the carried-and-refitted structural system's, whatever the dependence.
    a_m, b_mx = BM_TARGET
    expected = BY[0] + 2.0 * BY[1] + BY[2] * (a_m + 2.0 * b_mx)
    assert require_success(transported.mean_contrast(contrast)) == pytest.approx(expected, abs=TOL)
    error = refusal_of(lambda: require_success(transported.mean_contrast_uncertainty(contrast)))
    assert error.reason_code == "joint_law_required"


# ------------------------------------------------------------------------------ export


def test_the_transported_artifact_is_the_existing_one_and_replays(source_gcm, target_gcm) -> None:
    source = source_composition(source_gcm)
    transported = source.transport_to(transport(refits=[refit_m(target_gcm)]))
    exported = transported.export({"X": 2.0}, outcome="Y")
    produced = exported.metadata["result"]
    digest = exported.expectation.identity
    names, mean, cov = scm_law(TARGET_SPEC, {"X": 2.0})
    assert produced.mean == pytest.approx(mean[names.index("Y")], abs=1e-8)
    assert produced.variance == pytest.approx(cov[names.index("Y"), names.index("Y")], abs=1e-8)
    consumed = require_success(mc.consume_mechanism_composition_artifact(exported.data, digest))
    assert consumed.family == "linear_gaussian"
    assert consumed.result.mean == pytest.approx(produced.mean, abs=0.0)
    assert consumed.premises_digest == digest == exported.metadata["premises_digest"]


# ------------------------------------------------------------------------------ refusals


def test_a_selection_arrow_dominates_an_invariance_declaration(source_gcm) -> None:
    source = source_composition(source_gcm)
    invariance = mc.TransportInvariance("M", "population", "pop", "clinic_b", "assume:m-invariant")
    error = refusal_of(lambda: source.transport_to(transport(invariances=[invariance])))
    assert error.reason_code == "quantity_semantics_mismatch"
    assert "mechanism_composition.population_selection_arrow_into_mechanism" in error.details
    assert error.remedy


def test_declared_and_unknown_refits_are_refused_by_name(source_gcm, target_gcm) -> None:
    source = source_composition(source_gcm)
    # A target mechanism the analyst wrote down is not a natively refitted one.
    written = mc.MechanismRefit.declared_linear(
        "M",
        intercept=1.5,
        sigma=1.0,
        coefficients={"X": 0.5},
        artifact_id="fit:M:declared",
        evidence_id="data:M:declared",
    )
    error = refusal_of(lambda: source.transport_to(transport(refits=[written])))
    assert "mechanism_composition.provenance_declared_mechanism_unacknowledged" in error.details
    # A refit of a variable the composition has no mechanism for.
    stray = dataclasses.replace(refit_m(target_gcm), variable_id="Q")
    error = refusal_of(lambda: source.transport_to(transport(refits=[stray])))
    assert error.reason_code == "quantity_semantics_mismatch"
    assert error.detail == "mechanism_composition.identity_declaration_unknown_variable"


def test_declared_source_mechanisms_are_not_carried() -> None:
    declared = mc.compose_mechanisms(
        [
            mc.Mechanism.declared_linear(
                declaration("X"), intercept=1.0, sigma=2.0, coefficients={}
            ),
            mc.Mechanism.declared_linear(
                declaration("M", ["X"]), intercept=0.5, sigma=1.0, coefficients={"X": 2.0}
            ),
        ],
        target_population_id="pop",
        admit_declared_mechanisms=True,
    )
    error = refusal_of(lambda: declared.transport_to(transport(into=("M",))))
    assert "mechanism_transport.origin_not_fitted" in error.details
    origin = [r for r in error.refusals if r["detail"] == "mechanism_transport.origin_not_fitted"]
    assert origin and all(r["code"] == "mechanism_composition_incoherent" for r in origin)


def test_declaration_types_are_checked(source_gcm) -> None:
    source = source_composition(source_gcm)
    with pytest.raises(CausalTypeError):
        source.transport_to({"target_population_id": "clinic_b"})  # type: ignore[arg-type]
    with pytest.raises(CausalTypeError):
        mc.SelectionNode("S", "M")  # type: ignore[arg-type]
