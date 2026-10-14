"""Mechanism composition through the Python facade.

Hand-computed oracle (linear-Gaussian chain), with ::

    X = 1 + 2 e_X
    M = 0.5 + 2 X + 1 e_M
    Y = -1 + 0.5 X + 3 M + 0.5 e_Y          so Y = 0.5 + 6.5 X + 3 e_M + e_Y

so that ``E[X]=1, E[M]=2.5, E[Y]=7``, ``Var X=4, Var M=17, Var Y=178.25``,
``Cov(X,M)=8, Cov(X,Y)=26, Cov(M,Y)=55``; under ``do(X=2)``: ``E[M]=4.5, E[Y]=13.5, Var M=1,
Var Y=9.25, Cov(M,Y)=3``; under ``do(M=1)``: ``E[Y]=2.5, Var Y=1.25``.

The finite chain ``X -> M -> Y, X -> Y`` over binary variables has ``P(Y=1)=0.396``,
``P(Y=1 | do(X=1))=0.72``, ``P(Y=1 | do(X=0))=0.18`` and ``P(Y=1 | do(M=1))=0.66``.

Independent oracles in this file: ``scm_law`` computes the joint law of any linear-Gaussian
structural system by path algebra with numpy (``V = (I - B)^-1 (c + e)``), ``finite_joint``
enumerates a finite SCM by brute force and applies the truncated factorization, and the
delta-method variance is ``g' Sigma g`` with ``g`` obtained by central differences of the
interventional mean as a function of the parameters. No value is read from native code to decide
what the truth is.
"""

from __future__ import annotations

import itertools
import json
import subprocess
import sys

import numpy as np
import pytest
from antecedent import _native
from antecedent import mechanism_composition as mc
from antecedent.errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from antecedent.mechanism_composition import MechanismCompositionRefusal
from antecedent.model import fit_gcm

from _refusal import assert_registered_refusal

TOL = 1e-12
ROLES = {"X": "treatment", "M": "mediator", "Y": "outcome"}


# --------------------------------------------------------------------------- fixtures


def quantity(variable: str, units: str = "u", population: str = "pop") -> mc.Quantity:
    return mc.Quantity(
        variable,
        units=units,
        population_id=population,
        role=ROLES.get(variable, "covariate"),
        variable_name=f"label of {variable}",
    )


def declaration(variable: str, parents=(), units: str = "u", **kwargs) -> mc.MechanismDeclaration:
    return mc.MechanismDeclaration(
        output=quantity(variable, units),
        parents=tuple(quantity(p, units) for p in parents),
        artifact_id=f"fit:{variable}",
        evidence_id=f"data:{variable}",
        **kwargs,
    )


def linear(variable, intercept, sigma, parents=(), **kwargs) -> mc.Mechanism:
    names = [p for p, _ in parents]
    return mc.Mechanism.declared_linear(
        declaration(variable, names),
        intercept=intercept,
        sigma=sigma,
        coefficients=dict(parents),
        **kwargs,
    )


def chain() -> list[mc.Mechanism]:
    return [
        linear("X", 1.0, 2.0),
        linear("M", 0.5, 1.0, [("X", 2.0)]),
        linear("Y", -1.0, 0.5, [("X", 0.5), ("M", 3.0)]),
    ]


def compose(
    mechanisms, *, dependence=None, admit=True, **kwargs
) -> mc.PreparedMechanismComposition:
    return mc.compose_mechanisms(
        mechanisms,
        target_population_id="pop",
        dependence=dependence,
        admit_declared_mechanisms=admit,
        **kwargs,
    )


def refusal_of(call) -> MechanismCompositionRefusal:
    with pytest.raises(MechanismCompositionRefusal) as caught:
        call()
    error = caught.value
    assert isinstance(error, CausalUnsupportedError)
    assert_registered_refusal(error)
    return error


def scm_law(spec: dict, do: dict | None = None):
    """Joint law of a linear-Gaussian SCM: ``spec[v] = (intercept, sigma, {parent: coef})``."""
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
        for p, coef in parents.items():
            b[i, index[p]] = coef
    inv = np.linalg.inv(np.eye(k) - b)
    mean = inv @ c
    cov = inv @ np.diag(s**2) @ inv.T
    return names, mean, cov


CHAIN_SPEC = {
    "X": (1.0, 2.0, {}),
    "M": (0.5, 1.0, {"X": 2.0}),
    "Y": (-1.0, 0.5, {"X": 0.5, "M": 3.0}),
}


def marginal(terms, covariance, meaning="sampling_distribution") -> mc.MechanismUncertainty:
    return mc.MechanismUncertainty(
        parameters=tuple(terms), covariance=tuple(covariance), meaning=meaning
    )


INTERCEPT = mc.ParameterTerm.intercept()


def coef(parent: str) -> mc.ParameterTerm:
    return mc.ParameterTerm.coefficient(parent)


def uncertain_chain() -> list[mc.Mechanism]:
    """Marginal parameter laws: X (a_X); M (a_M, b_MX); Y listed out of canonical order."""
    return [
        linear("X", 1.0, 2.0, uncertainty=marginal([INTERCEPT], [0.25])),
        linear(
            "M",
            0.5,
            1.0,
            [("X", 2.0)],
            uncertainty=marginal([INTERCEPT, coef("X")], [0.04, -0.01, -0.01, 0.01]),
        ),
        linear(
            "Y",
            -1.0,
            0.5,
            [("X", 0.5), ("M", 3.0)],
            uncertainty=marginal(
                [INTERCEPT, coef("X"), coef("M")],
                [0.09, 0, 0, 0, 0.01, 0, 0, 0, 0.0025],
            ),
        ),
    ]


M_TERMS = (INTERCEPT, coef("X"))
X_TERMS = (INTERCEPT,)
Y_TERMS = (INTERCEPT, coef("M"), coef("X"))


def cross_blocks() -> list[mc.CrossCovarianceBlock]:
    # cov(b_MX, b_YM) = 0.002 adds 2 * 6 * 4.5 * 0.002 = 0.108; every other pair is an explicit
    # zero block (a declared independence).
    return [
        mc.CrossCovarianceBlock("M", "Y", M_TERMS, Y_TERMS, (0, 0, 0, 0, 0.002, 0)),
        mc.CrossCovarianceBlock("M", "X", M_TERMS, X_TERMS, (0, 0)),
        mc.CrossCovarianceBlock("X", "Y", X_TERMS, Y_TERMS, (0, 0, 0)),
    ]


def independent() -> mc.Dependence:
    return mc.Dependence.independent_by_design("disjoint trials, independently generated")


# The delta-method oracle: E[Y | do(X=x)] as a function of the parameters
# (a_M, b_MX, a_X, a_Y, b_YX, b_YM), differentiated numerically.
PARAMS = np.array([0.5, 2.0, 1.0, -1.0, 0.5, 3.0])


def mean_y_do_x(p: np.ndarray, x: float = 2.0) -> float:
    a_m, b_mx, _a_x, a_y, b_yx, b_ym = p
    return a_y + b_yx * x + b_ym * (a_m + b_mx * x)


def numeric_gradient(f, p: np.ndarray, h: float = 1.0) -> np.ndarray:
    grad = np.zeros_like(p)
    for i in range(len(p)):
        up, down = p.copy(), p.copy()
        up[i] += h
        down[i] -= h
        grad[i] = (f(up) - f(down)) / (2 * h)
    return grad


SIGMA_INDEPENDENT = np.zeros((6, 6))
SIGMA_INDEPENDENT[0:2, 0:2] = [[0.04, -0.01], [-0.01, 0.01]]
SIGMA_INDEPENDENT[2, 2] = 0.25
SIGMA_INDEPENDENT[3:6, 3:6] = np.diag([0.09, 0.01, 0.0025])


# --------------------------------------------------------------------------- closed form


def test_declared_mechanisms_compose_into_the_hand_computed_scm():
    composition = compose(chain())
    assert composition.family == "linear_gaussian"
    assert composition.variables == ("M", "X", "Y")
    assert composition.declared_mechanisms == ("M", "X", "Y")  # numbers the analyst wrote down

    observational = composition.intervention_law()
    assert observational.mean_of("X") == pytest.approx(1.0, abs=TOL)
    assert observational.mean_of("M") == pytest.approx(2.5, abs=TOL)
    assert observational.mean_of("Y") == pytest.approx(7.0, abs=TOL)
    assert observational.variance_of("X") == pytest.approx(4.0, abs=TOL)
    assert observational.variance_of("M") == pytest.approx(17.0, abs=TOL)
    assert observational.variance_of("Y") == pytest.approx(178.25, abs=TOL)
    assert observational.covariance_of("X", "M") == pytest.approx(8.0, abs=TOL)
    assert observational.covariance_of("X", "Y") == pytest.approx(26.0, abs=TOL)
    assert observational.covariance_of("M", "Y") == pytest.approx(55.0, abs=TOL)

    do_x = composition.intervention_law({"X": 2.0})
    assert do_x.mean_of("M") == pytest.approx(4.5, abs=TOL)
    assert do_x.mean_of("Y") == pytest.approx(13.5, abs=TOL)
    assert do_x.variance_of("M") == pytest.approx(1.0, abs=TOL)
    assert do_x.variance_of("Y") == pytest.approx(9.25, abs=TOL)
    assert do_x.covariance_of("M", "Y") == pytest.approx(3.0, abs=TOL)
    do_m = composition.intervention_law({"M": 1.0})
    assert do_m.mean_of("Y") == pytest.approx(2.5, abs=TOL)
    assert do_m.variance_of("Y") == pytest.approx(1.25, abs=TOL)
    assert composition.mean_contrast(mc.MeanContrast.ate("Y", "X", 1.0, 0.0)) == pytest.approx(
        6.5, abs=TOL
    )

    # The same numbers from the independent path-algebra oracle, for every regime.
    for do in ({}, {"X": 2.0}, {"M": 1.0}, {"X": -3.0, "M": 0.5}):
        names, mean, cov = scm_law(CHAIN_SPEC, do)
        law = composition.intervention_law(do)
        assert law.variables == tuple(names)
        assert law.mean == pytest.approx(mean, abs=1e-10)
        assert law.covariance == pytest.approx(cov, abs=1e-10)


def test_listing_order_does_not_change_the_plan_or_the_law_and_sampling_is_deterministic():
    forward = compose(chain())
    reversed_list = list(reversed(chain()))
    edges = [("X", "M"), ("X", "Y"), ("M", "Y")][::-1]
    backward = compose(reversed_list, edges=edges)
    assert forward.plan_identity == backward.plan_identity
    assert forward.intervention_law({"X": 2.0}).mean == pytest.approx(
        backward.intervention_law({"X": 2.0}).mean, abs=0
    )
    assert np.array_equal(
        forward.intervention_law({"X": 2.0}).covariance,
        backward.intervention_law({"X": 2.0}).covariance,
    )

    n = 200_000
    table = forward.sample({"X": 2.0}, n_rows=n, seed=20_260_101)
    y = table["Y"]
    # SE of the mean sqrt(9.25 / 200000) = 0.0068: six SE; relative SE of the variance sqrt(2/n).
    assert abs(y.mean() - 13.5) < 0.05
    assert abs(y.var() / 9.25 - 1.0) < 0.02
    again = forward.sample({"X": 2.0}, n_rows=n, seed=20_260_101)
    assert all(np.array_equal(table[v], again[v]) for v in table)
    other = forward.sample({"X": 2.0}, n_rows=1_000, seed=20_260_102)
    assert not np.array_equal(table["Y"][:1_000], other["Y"])

    # The noise of a variable is keyed by its name: its draws do not depend on the others.
    alone = compose([linear("X", 1.0, 2.0)]).sample({}, n_rows=500, seed=7)["X"]
    within = forward.sample({}, n_rows=500, seed=7)["X"]
    assert np.array_equal(alone, within)


def test_a_fitted_model_composes_and_matches_the_least_squares_oracle():
    rng = np.random.default_rng(12345)
    n = 20_000
    x = 1.0 + 2.0 * rng.standard_normal(n)
    m = 0.5 + 2.0 * x + rng.standard_normal(n)
    y = -1.0 + 0.5 * x + 3.0 * m + 0.5 * rng.standard_normal(n)
    gcm = fit_gcm(["X", "M", "Y"], [x, m, y], [("X", "M"), ("X", "Y"), ("M", "Y")])
    mechanisms = [
        mc.Mechanism.fitted_linear(declaration("X"), gcm),
        mc.Mechanism.fitted_linear(declaration("M", ["X"]), gcm),
        mc.Mechanism.fitted_linear(declaration("Y", ["X", "M"]), gcm),
    ]
    assert not any(mech.declared for mech in mechanisms)
    # Fitted: no acknowledgement needed, and nothing is listed as declared.
    composition = mc.compose_mechanisms(mechanisms, target_population_id="pop")
    assert composition.declared_mechanisms == ()

    # Oracle: ordinary least squares per mechanism, then the path algebra.
    def ols(target, regressors):
        design = np.column_stack([np.ones(n), *regressors])
        beta = np.linalg.lstsq(design, target, rcond=None)[0]
        resid = target - design @ beta
        return beta, float(np.sqrt(resid @ resid / n))

    bx, sx = ols(x, [])
    bm, sm = ols(m, [x])
    by, sy = ols(y, [x, m])
    spec = {
        "X": (bx[0], sx, {}),
        "M": (bm[0], sm, {"X": bm[1]}),
        "Y": (by[0], sy, {"X": by[1], "M": by[2]}),
    }
    for do in ({}, {"X": 2.0}):
        _names, mean, cov = scm_law(spec, do)
        law = composition.intervention_law(do)
        assert law.mean == pytest.approx(mean, abs=1e-6)
        # sigma is the residual standard deviation up to the (n - p) / n divisor convention.
        assert law.covariance == pytest.approx(cov, rel=2e-3, abs=1e-3)
    # Truth: the DGP's own interventional mean is 13.5; sampling error is far below 0.2.
    assert abs(composition.intervention_law({"X": 2.0}).mean_of("Y") - 13.5) < 0.2
    # The source numbers are the OLS coefficients, from the independent fit.
    source = dict(mechanisms[2].source)
    assert source["kind"] == "fitted_linear"
    assert source["intercept"] == pytest.approx(by[0], abs=1e-8)
    assert dict((p, c) for p, c in source["coefficients"]) == pytest.approx(
        {"X": by[1], "M": by[2]}, abs=1e-8
    )


def test_a_fitted_slot_that_is_not_plain_linear_gaussian_is_refused_by_name():
    rng = np.random.default_rng(3)
    n = 400
    a = rng.integers(0, 2, n).astype(float)
    b = rng.standard_normal(n)
    gcm = fit_gcm(["A", "B"], [a, b], [("A", "B")])
    # A two-level column is typed categorical, so its slot is a discrete mechanism ...
    assert dict(gcm.mechanism_kinds())["A"] == "discrete"
    # ... which is outside the bounded family and is refused by name, not coerced.
    error = refusal_of(lambda: mc.Mechanism.fitted_linear(declaration("A"), gcm))
    assert error.detail == "mechanism_composition.slot_not_linear_gaussian"
    assert error.reason_code == "mechanism_composition_incoherent"
    # A variable that is not a node of the fitted model is an input error.
    with pytest.raises(CausalValueError):
        mc.Mechanism.fitted_linear(declaration("Q"), gcm)


def test_declared_mechanisms_are_labelled_and_need_acknowledgement():
    mechanisms = chain()
    error = refusal_of(lambda: compose(mechanisms, admit=False))
    assert error.reason_code == "mechanism_composition_incoherent"
    assert set(error.details) == {
        "mechanism_composition.provenance_declared_mechanism_unacknowledged"
    }
    prepared = compose(mechanisms, admit=True)
    assert prepared.declared_mechanisms == ("M", "X", "Y")
    assert prepared.intervention_law({"X": 2.0}).mean_of("Y") == pytest.approx(13.5, abs=TOL)


# --------------------------------------------------------------------------- refusals


def test_individually_valid_mechanisms_that_cannot_compose_are_refused_with_structure():
    # Units: M is produced in mmHg but Y was fitted on M in kPa.
    m = mc.Mechanism.declared_linear(
        mc.MechanismDeclaration(
            output=quantity("M", "mmHg"),
            parents=(quantity("X"),),
            artifact_id="fit:M",
            evidence_id="data:M",
        ),
        intercept=0.5,
        sigma=1.0,
        coefficients={"X": 2.0},
    )
    y = mc.Mechanism.declared_linear(
        mc.MechanismDeclaration(
            output=quantity("Y"),
            parents=(quantity("X"), quantity("M", "kPa")),
            artifact_id="fit:Y",
            evidence_id="data:Y",
        ),
        intercept=-1.0,
        sigma=0.5,
        coefficients={"X": 0.5, "M": 3.0},
    )
    error = refusal_of(lambda: compose([linear("X", 1.0, 2.0), m, y]))
    assert error.reason_code == "quantity_semantics_mismatch"
    assert "mechanism_composition.units_mismatch" in error.details
    assert error.offending == "M->Y"
    assert error.remedy

    # Parents: Y was fitted on {X} only; the declared graph has {X, M}.
    lonely = linear("Y", -1.0, 0.5, [("X", 0.5)])
    error = refusal_of(
        lambda: compose(
            [linear("X", 1.0, 2.0), linear("M", 0.5, 1.0, [("X", 2.0)]), lonely],
            edges=[("X", "M"), ("X", "Y"), ("M", "Y")],
        )
    )
    assert error.reason_code == "graph_invalid"
    assert error.details == ("mechanism_composition.parents_declared_graph_mismatch",)
    assert error.refusals[0]["expected"] == "{M,X}"
    assert error.refusals[0]["supplied"] == "{X}"

    # A cycle is refused before anything executes.
    cyclic = [linear("A", 0.0, 1.0, [("B", 0.5)]), linear("B", 0.0, 1.0, [("A", 0.5)])]
    error = refusal_of(lambda: compose(cyclic))
    assert error.reason_code == "graph_invalid"
    assert "mechanism_composition.cycle_directed_cycle" in error.details

    # Support: Y was fitted on X in [0, 10]; X is declared on the whole line.
    bounded = [
        linear("X", 1.0, 2.0),
        linear("M", 0.5, 1.0, [("X", 2.0)]),
        linear("Y", -1.0, 0.5, [("X", 0.5), ("M", 3.0)], parent_domains={"X": (0.0, 10.0)}),
    ]
    error = refusal_of(lambda: compose(bounded))
    assert error.reason_code == "mechanism_composition_incoherent"
    assert error.details == ("mechanism_composition.support_not_contained",)

    # A parent with no mechanism of its own.
    error = refusal_of(lambda: compose([linear("Y", 0.0, 1.0, [("W", 1.0)])]))
    assert "mechanism_composition.identity_parent_without_mechanism" in error.details

    # Provenance: an unbound mechanism.
    unbound = mc.Mechanism.declared_linear(
        mc.MechanismDeclaration(
            output=quantity("M"),
            parents=(quantity("X"),),
            artifact_id=" ",
            evidence_id="data:M",
        ),
        intercept=0.5,
        sigma=1.0,
        coefficients={"X": 2.0},
    )
    error = refusal_of(
        lambda: compose(
            [linear("X", 1.0, 2.0), unbound, linear("Y", -1.0, 0.5, [("X", 0.5), ("M", 3.0)])]
        )
    )
    assert error.details == ("mechanism_composition.provenance_missing_artifact",)

    # Families are never mixed.
    table = mc.Mechanism.declared_table(
        declaration("Z", units="category"),
        levels=["0", "1"],
        rows=[({}, {"0": 0.5, "1": 0.5})],
    )
    error = refusal_of(lambda: compose([*chain(), table]))
    assert "mechanism_composition.family_mixed" in error.details


def test_a_mechanism_from_another_population_needs_a_declared_admission():
    def from_source():
        mechanisms = chain()
        m = mc.Mechanism.declared_linear(
            mc.MechanismDeclaration(
                output=quantity("M", population="source"),
                parents=(quantity("X", population="source"),),
                artifact_id="fit:M",
                evidence_id="data:M",
            ),
            intercept=0.5,
            sigma=1.0,
            coefficients={"X": 2.0},
        )
        return [mechanisms[0], m, mechanisms[2]]

    error = refusal_of(lambda: compose(from_source()))
    assert error.reason_code == "quantity_semantics_mismatch"
    assert error.details == ("mechanism_composition.population_undeclared_transport",)
    assert error.refusals[0]["expected"] == "pop"
    assert error.refusals[0]["supplied"] == "source"

    def admission(**basis):
        return mc.Admission("M", "population", "source", "pop", **basis)

    # A selection diagram with no arrow into M admits it ...
    ok = compose(from_source(), admissions=[admission(selection_targets=("X",))])
    assert ok.intervention_law({"X": 2.0}).mean_of("Y") == pytest.approx(13.5, abs=TOL)
    # ... one that points into M does not ...
    error = refusal_of(
        lambda: compose(from_source(), admissions=[admission(selection_targets=("M",))])
    )
    assert error.details == ("mechanism_composition.population_selection_arrow_into_mechanism",)
    # ... and an explicit invariance assumption admits it regardless.
    assert compose(from_source(), admissions=[admission(assumption_id="assumption:m-invariant")])
    with pytest.raises(CausalValueError):
        admission().to_wire()  # a basis is stated: exactly one of the two


def test_shared_latents_are_exact_and_a_dropped_latent_is_refused():
    def build(with_b: bool, declare_loading: bool = True):
        a = mc.Mechanism.declared_linear(
            declaration("A", latent_causes=(mc.LatentCause("U", ("B",)),)),
            intercept=0.0,
            sigma=1.0,
            coefficients={},
            latent_loadings={"U": 2.0} if declare_loading else {},
        )
        mechanisms = [a]
        if with_b:
            mechanisms.append(
                mc.Mechanism.declared_linear(
                    declaration("B", latent_causes=(mc.LatentCause("U", ("A",)),)),
                    intercept=1.0,
                    sigma=1.0,
                    coefficients={},
                    latent_loadings={"U": 3.0},
                )
            )
        return mechanisms

    # A = e + 2U, B = 1 + e' + 3U: Var A = 5, Var B = 10, Cov = 6.
    law = compose(build(True)).intervention_law()
    assert law.variance_of("A") == pytest.approx(5.0, abs=TOL)
    assert law.variance_of("B") == pytest.approx(10.0, abs=TOL)
    assert law.covariance_of("A", "B") == pytest.approx(6.0, abs=TOL)
    # Dropping B: A's latent is shared with a mechanism that is not here.
    error = refusal_of(lambda: compose(build(False)))
    assert "mechanism_composition.latent_shared_with_absent_mechanism" in error.details
    # The dependence exists but its strength is not declared.
    error = refusal_of(lambda: compose(build(True, declare_loading=False)))
    assert "mechanism_composition.latent_loading_undeclared" in error.details


# --------------------------------------------------------------------------- uncertainty


def test_unknown_dependence_stays_unknown_and_refuses_the_uncertainty_claim():
    composition = compose(uncertain_chain())
    # Point execution is unaffected ...
    assert composition.mean_contrast(mc.MeanContrast.mean_at("Y", {"X": 2.0})) == pytest.approx(
        13.5, abs=TOL
    )
    # ... but the uncertainty claim is a joint-law refusal, not independence.
    error = refusal_of(
        lambda: composition.mean_contrast_uncertainty(mc.MeanContrast.mean_at("Y", {"X": 2.0}))
    )
    assert error.reason_code == "joint_law_required"
    assert error.details == ("mechanism_composition.posterior_dependence_unknown",)
    assert error.remedy
    # The interval route is closed regardless, typed.
    closed = refusal_of(composition.interval)
    assert closed.reason_code == "cell_not_licensed"
    assert closed.details == ("mechanism_composition.interval_route_closed",)

    # Independence declared over shared evidence is contradicted, not accepted.
    shared = uncertain_chain()
    shared[1] = mc.Mechanism.declared_linear(
        mc.MechanismDeclaration(
            output=quantity("M"),
            parents=(quantity("X"),),
            artifact_id="fit:M",
            evidence_id="data:X",
        ),
        intercept=0.5,
        sigma=1.0,
        coefficients={"X": 2.0},
        uncertainty=marginal([INTERCEPT, coef("X")], [0.04, -0.01, -0.01, 0.01]),
    )
    error = refusal_of(lambda: compose(shared, dependence=independent()))
    assert error.reason_code == "joint_law_required"
    assert error.details == (
        "mechanism_composition.posterior_dependence_shared_evidence_declared_independent",
    )


def test_independence_and_a_declared_covariance_give_different_checked_variances():
    contrast = mc.MeanContrast.mean_at("Y", {"X": 2.0})

    # Oracle: gradient by central differences of the interventional mean, then g' Sigma g.
    grad = numeric_gradient(mean_y_do_x, PARAMS)
    assert grad == pytest.approx([3.0, 6.0, 0.0, 1.0, 2.0, 4.5], abs=1e-8)
    variance_independent = float(grad @ SIGMA_INDEPENDENT @ grad)
    assert variance_independent == pytest.approx(0.540625, abs=1e-12)

    independent_prepared = compose(uncertain_chain(), dependence=independent())
    u = independent_prepared.mean_contrast_uncertainty(contrast)
    assert u.point == pytest.approx(13.5, abs=TOL)
    assert u.variance == pytest.approx(variance_independent, abs=1e-10)
    assert u.standard_error == pytest.approx(np.sqrt(variance_independent), abs=1e-10)
    assert u.method == "first_order_delta_method"
    assert u.dependence == "declared_independent_by_design"
    assert u.meaning == "sampling_distribution"
    by_parameter = {(p.mechanism, p.term): g for p, g in u.gradient}
    assert by_parameter[("M", INTERCEPT)] == pytest.approx(3.0, abs=1e-9)
    assert by_parameter[("M", coef("X"))] == pytest.approx(6.0, abs=1e-9)
    assert by_parameter[("Y", INTERCEPT)] == pytest.approx(1.0, abs=1e-9)
    assert by_parameter[("Y", coef("X"))] == pytest.approx(2.0, abs=1e-9)
    assert by_parameter[("Y", coef("M"))] == pytest.approx(4.5, abs=1e-9)

    # cov(b_MX, b_YM) = 0.002 adds 2 * 6 * 4.5 * 0.002 = 0.108.
    sigma = SIGMA_INDEPENDENT.copy()
    sigma[1, 5] = sigma[5, 1] = 0.002
    variance_declared = float(grad @ sigma @ grad)
    assert variance_declared == pytest.approx(0.648625, abs=1e-12)
    declared = compose(
        uncertain_chain(), dependence=mc.Dependence.cross_covariance(cross_blocks())
    ).mean_contrast_uncertainty(contrast)
    assert declared.variance == pytest.approx(variance_declared, abs=1e-10)
    assert declared.dependence == "declared_cross_covariance"
    assert declared.variance - u.variance == pytest.approx(0.108, abs=1e-10)

    # The same dependence as one aligned joint law, in an arbitrary order.
    order = [
        ("Y", coef("X")),
        ("M", INTERCEPT),
        ("X", INTERCEPT),
        ("Y", coef("M")),
        ("M", coef("X")),
        ("Y", INTERCEPT),
    ]
    position = {
        ("M", INTERCEPT): 0,
        ("M", coef("X")): 1,
        ("X", INTERCEPT): 2,
        ("Y", INTERCEPT): 3,
        ("Y", coef("X")): 4,
        ("Y", coef("M")): 5,
    }
    full = np.array([[sigma[position[a], position[b]] for b in order] for a in order])
    joint = mc.Dependence.aligned_joint_law(
        "joint-fit-1",
        [mc.ParameterId(m, t) for m, t in order],
        full.reshape(-1).tolist(),
    )
    aligned = compose(uncertain_chain(), dependence=joint).mean_contrast_uncertainty(contrast)
    assert aligned.variance == pytest.approx(variance_declared, abs=1e-10)
    assert aligned.dependence == "aligned_joint_law"


def test_a_missing_cross_covariance_pair_is_unknown_for_that_pair():
    prepared = compose(
        uncertain_chain(), dependence=mc.Dependence.cross_covariance(cross_blocks()[:1])
    )
    error = refusal_of(
        lambda: prepared.mean_contrast_uncertainty(mc.MeanContrast.mean_at("Y", {"X": 2.0}))
    )
    assert error.reason_code == "joint_law_required"
    assert len(error.details) == 2
    assert set(error.details) == {
        "mechanism_composition.posterior_dependence_cross_covariance_missing"
    }


# --------------------------------------------------------------------------- finite tables


def finite_chain() -> list[mc.Mechanism]:
    def table(variable, parents, rows):
        return mc.Mechanism.declared_table(
            declaration(variable, parents, units="category"),
            levels=["0", "1"],
            rows=[
                (dict(zip(parents, config, strict=True)), {"0": p0, "1": p1})
                for config, (p0, p1) in rows
            ],
        )

    return [
        table("X", [], [((), (0.6, 0.4))]),
        table("M", ["X"], [(("0",), (0.8, 0.2)), (("1",), (0.3, 0.7))]),
        # parents in declared order (M, X)
        table(
            "Y",
            ["M", "X"],
            [
                (("0", "0"), (0.9, 0.1)),
                (("1", "0"), (0.5, 0.5)),
                (("0", "1"), (0.7, 0.3)),
                (("1", "1"), (0.1, 0.9)),
            ],
        ),
    ]


FINITE_TABLES = {
    "X": lambda a: {"0": 0.6, "1": 0.4},
    "M": lambda a: {"0": 0.8, "1": 0.2} if a["X"] == "0" else {"0": 0.3, "1": 0.7},
    "Y": lambda a: {
        ("0", "0"): {"0": 0.9, "1": 0.1},
        ("1", "0"): {"0": 0.5, "1": 0.5},
        ("0", "1"): {"0": 0.7, "1": 0.3},
        ("1", "1"): {"0": 0.1, "1": 0.9},
    }[(a["M"], a["X"])],
}


def finite_joint(do: dict | None = None) -> dict:
    """Brute-force joint law of the binary chain, with truncated factorization under do()."""
    do = do or {}
    law = {}
    for x, m, y in itertools.product("01", repeat=3):
        a = {"X": x, "M": m, "Y": y}
        if any(a[v] != level for v, level in do.items()):
            continue
        p = 1.0
        for v in ("X", "M", "Y"):
            if v not in do:
                p *= FINITE_TABLES[v](a)[a[v]]
        law[(x, m, y)] = p
    return law


def prob(law: dict, event: dict) -> float:
    index = {"X": 0, "M": 1, "Y": 2}
    return sum(
        p for state, p in law.items() if all(state[index[v]] == lv for v, lv in event.items())
    )


def test_finite_tables_match_the_enumerated_scm():
    prepared = compose(finite_chain())
    assert prepared.family == "tabular"
    assert prepared.declared_mechanisms == ("M", "X", "Y")
    # A Gaussian query of a finite composition is refused, not coerced.
    error = refusal_of(prepared.intervention_law)
    assert error.reason_code == "route_not_supported"

    for do in ({}, {"X": "1"}, {"X": "0"}, {"M": "1"}):
        oracle = finite_joint(do)
        law = prepared.finite_law(do)
        assert law.variables == ("M", "X", "Y")
        assert law.probability({"Y": "1"}) == pytest.approx(prob(oracle, {"Y": "1"}), abs=TOL)
        assert law.probability({"X": "1"}) == pytest.approx(prob(oracle, {"X": "1"}), abs=TOL)
        assert float(law.probabilities.sum()) == pytest.approx(1.0, abs=TOL)
    # And the literal numbers of the hand calculation.
    assert prepared.finite_law().probability({"Y": "1"}) == pytest.approx(0.396, abs=TOL)
    assert prepared.finite_law({"X": "1"}).probability({"Y": "1"}) == pytest.approx(0.72, abs=TOL)
    assert prepared.finite_law({"X": "0"}).probability({"Y": "1"}) == pytest.approx(0.18, abs=TOL)
    assert prepared.finite_law({"M": "1"}).probability({"Y": "1"}) == pytest.approx(0.66, abs=TOL)
    assert prepared.finite_law().probability({"X": "1", "M": "1", "Y": "1"}) == pytest.approx(
        0.252, abs=TOL
    )
    assert prepared.finite_law().marginal("Y") == pytest.approx({"0": 0.604, "1": 0.396}, abs=TOL)

    # A linear-Gaussian query of the other family is refused as well.
    gaussian = compose(chain())
    error = refusal_of(lambda: gaussian.finite_law({"X": "1"}))
    assert error.reason_code == "route_not_supported"

    # A table with a missing parent configuration is refused.
    broken = finite_chain()
    short = mc.Mechanism.declared_table(
        declaration("M", ["X"], units="category"),
        levels=["0", "1"],
        rows=[({"X": "0"}, {"0": 0.8, "1": 0.2})],
    )
    error = refusal_of(lambda: compose([broken[0], short, broken[2]]))
    assert "mechanism_composition.support_missing_parent_configuration" in error.details


# --------------------------------------------------------------------------- artifact


def contrast_query() -> mc.MeanContrast:
    return mc.MeanContrast.mean_at("Y", {"X": 2.0})


def test_the_artifact_replays_and_refuses_semantic_mutation():
    prepared = compose(uncertain_chain(), dependence=independent())
    export = prepared.artifact(contrast_query())
    consumed = mc.consume_mechanism_composition_artifact(export.data, export.expectation.identity)
    assert consumed.family == "linear_gaussian"
    assert consumed.premises_digest == export.expectation.identity
    result = consumed.result
    assert isinstance(result, mc.GaussianFunctionalResult)
    assert result.point == pytest.approx(13.5, abs=TOL)
    assert result.variance == pytest.approx(0.540625, abs=1e-12)
    assert result.dependence == "declared_independent_by_design"
    assert result.method == "first_order_delta_method"
    assert result == export.metadata["result"]

    # A retained identity that is not the artifact's is refused, typed.
    error = refusal_of(lambda: mc.consume_mechanism_composition_artifact(export.data, "0" * 64))
    assert error.details == ("mechanism_composition.premises_mismatch",)
    # Damaged and truncated bytes are refused.
    damaged = bytearray(export.data)
    damaged[len(damaged) // 2] ^= 0xFF
    refusal_of(lambda: mc.consume_mechanism_composition_artifact(bytes(damaged)))
    refusal_of(lambda: mc.consume_mechanism_composition_artifact(export.data[:40]))
    with pytest.raises(CausalTypeError):
        mc.consume_mechanism_composition_artifact("not bytes")  # type: ignore[arg-type]

    # Semantic mutations are different premises: each is self-consistent and consumable on its
    # own, and refused by a consumer that retained the original digest.
    mutated_numbers = uncertain_chain()
    mutated_numbers[1] = linear(
        "M",
        0.6,
        1.0,
        [("X", 2.0)],
        uncertainty=marginal([INTERCEPT, coef("X")], [0.04, -0.01, -0.01, 0.01]),
    )
    mutated = compose(mutated_numbers, dependence=independent()).artifact(contrast_query())
    own = mc.consume_mechanism_composition_artifact(mutated.data, mutated.expectation.identity)
    assert own.result.point == pytest.approx(13.8, abs=TOL)  # -1 + 1 + 3 * (0.6 + 4)
    assert mutated.expectation.identity != export.expectation.identity
    error = refusal_of(
        lambda: mc.consume_mechanism_composition_artifact(mutated.data, export.expectation.identity)
    )
    assert error.details == ("mechanism_composition.premises_mismatch",)

    other_query = prepared.artifact(mc.MeanContrast.mean_at("Y", {"X": 3.0}))
    assert other_query.expectation.identity != export.expectation.identity
    refusal_of(
        lambda: mc.consume_mechanism_composition_artifact(
            other_query.data, export.expectation.identity
        )
    )
    assert other_query.metadata["result"].point == pytest.approx(
        20.0, abs=TOL
    )  # -1 + 1.5 + 3 * (0.5 + 6)

    # An uncertainty claim without a declared joint law is not exported.
    unknown = compose(uncertain_chain())
    error = refusal_of(lambda: unknown.export(contrast_query()))
    assert error.reason_code == "joint_law_required"


def test_a_finite_artifact_matches_the_enumerated_truth_and_a_law_artifact_replays():
    prepared = compose(finite_chain())
    export = prepared.artifact({"X": "1"}, outcome="Y")
    consumed = mc.consume_mechanism_composition_artifact(export.data)
    assert consumed.family == "tabular"
    oracle = finite_joint({"X": "1"})
    expected = {level: prob(oracle, {"Y": level}) for level in "01"}
    levels = dict(consumed.result.probabilities)
    assert levels["1"] == pytest.approx(expected["1"], abs=TOL)
    assert levels["0"] == pytest.approx(expected["0"], abs=TOL)
    assert levels["1"] == pytest.approx(0.72, abs=TOL)

    gaussian = compose(chain())
    law = gaussian.artifact({"X": 2.0}, outcome="Y")
    replayed = mc.consume_mechanism_composition_artifact(law.data, law.metadata["premises_digest"])
    assert isinstance(replayed.result, mc.GaussianLawResult)
    assert replayed.result.mean == pytest.approx(13.5, abs=TOL)
    assert replayed.result.variance == pytest.approx(9.25, abs=TOL)


def test_a_fresh_process_consumes_the_artifact_against_the_retained_digest(tmp_path):
    export = compose(chain()).artifact({"X": 2.0}, outcome="Y")
    path = tmp_path / "artifact.bin"
    path.write_bytes(export.data)
    script = (
        "import json, sys\n"
        "from antecedent import mechanism_composition as mc\n"
        "c = mc.consume_mechanism_composition_artifact(open(sys.argv[1], 'rb').read(), sys.argv[2])\n"
        "print(json.dumps({'mean': c.result.mean, 'variance': c.result.variance, "
        "'family': c.family}))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script, str(path), export.expectation.identity],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=True,
    )
    wire = json.loads(done.stdout)
    assert wire["mean"] == pytest.approx(13.5, abs=TOL)
    assert wire["variance"] == pytest.approx(9.25, abs=TOL)
    assert wire["family"] == "linear_gaussian"
    refused = subprocess.run(
        [sys.executable, "-c", script, str(path), "f" * 64],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert refused.returncode != 0
    assert "premises_mismatch" in refused.stderr


# --------------------------------------------------------------------------- lifecycle


def test_refresh_keeps_the_plan_recomputes_the_numbers_and_refuses_other_mechanisms():
    prepared = compose(chain())
    before = prepared.intervention_law({"X": 2.0})
    # M refitted on new data: intercept 1, coefficient 3. E[M | do(X=2)] = 7 and
    # E[Y | do(X=2)] = -1 + 0.5 * 2 + 3 * 7 = 21.
    refit = mc.MechanismRefit.declared_linear(
        "M",
        intercept=1.0,
        sigma=1.0,
        coefficients={"X": 3.0},
        artifact_id="fit:M:2",
        evidence_id="data:M:2",
    )
    refreshed = prepared.refresh_mechanisms([refit])
    after = refreshed.intervention_law({"X": 2.0})
    assert after.mean_of("M") == pytest.approx(7.0, abs=TOL)
    assert after.mean_of("Y") == pytest.approx(21.0, abs=TOL)
    # Oracle for the refit chain.
    spec = dict(CHAIN_SPEC)
    spec["M"] = (1.0, 1.0, {"X": 3.0})
    _names, mean, _cov = scm_law(spec, {"X": 2.0})
    assert after.mean == pytest.approx(mean, abs=1e-10)
    # Kept: the declared variables and edges. Recomputed: the provenance, hence the identity.
    assert refreshed.variables == prepared.variables
    assert refreshed.plan_identity != prepared.plan_identity
    # The prepared composition it was called on is unchanged.
    assert np.array_equal(prepared.intervention_law({"X": 2.0}).mean, before.mean)
    assert before.mean_of("Y") == pytest.approx(13.5, abs=TOL)

    # Refused: another mechanism family, other parents, an unknown variable.
    table = mc.MechanismRefit(
        "M",
        {"kind": "declared_table", "levels": ["0", "1"], "rows": []},
        "fit:M:2",
        "data:M:2",
    )
    error = refusal_of(lambda: prepared.refresh_mechanisms([table]))
    assert error.details == ("mechanism_composition.family_mixed",)
    other_parents = mc.MechanismRefit.declared_linear(
        "M",
        intercept=1.0,
        sigma=1.0,
        coefficients={"X": 3.0, "Z": 1.0},
        artifact_id="fit:M:2",
        evidence_id="data:M:2",
    )
    error = refusal_of(lambda: prepared.refresh_mechanisms([other_parents]))
    assert "mechanism_composition.shape_coefficient_parent_mismatch" in error.details
    unknown = mc.MechanismRefit.declared_linear(
        "Q",
        intercept=1.0,
        sigma=1.0,
        coefficients={},
        artifact_id="fit:Q",
        evidence_id="data:Q",
    )
    error = refusal_of(lambda: prepared.refresh_mechanisms([unknown]))
    assert error.reason_code == "quantity_semantics_mismatch"
    assert error.details == ("mechanism_composition.identity_declaration_unknown_variable",)
    # Still unchanged after three refusals.
    assert np.array_equal(prepared.intervention_law({"X": 2.0}).mean, before.mean)


def test_a_dependence_declared_for_the_earlier_fit_is_unknown_again_after_a_refit():
    prepared = compose(uncertain_chain(), dependence=mc.Dependence.cross_covariance(cross_blocks()))
    refit = mc.MechanismRefit.declared_linear(
        "M",
        intercept=0.5,
        sigma=1.0,
        coefficients={"X": 2.0},
        artifact_id="fit:M:2",
        evidence_id="data:M:2",
        uncertainty=marginal([INTERCEPT, coef("X")], [0.04, -0.01, -0.01, 0.01]),
    )
    error = refusal_of(lambda: prepared.refresh_mechanisms([refit]))
    assert error.reason_code == "joint_law_required"
    assert error.details == ("mechanism_composition.posterior_dependence_unknown",)
    # Redeclared, the refit composes and the variance is recomputed.
    refreshed = prepared.refresh_mechanisms(
        [refit],
        mc.Dependence.independent_by_design("refit on a new, disjoint trial"),
    )
    value = refreshed.mean_contrast_uncertainty(contrast_query())
    assert value.variance == pytest.approx(0.540625, abs=1e-10)
    assert value.dependence == "declared_independent_by_design"


def test_cancellation_stops_before_a_composed_result_and_a_bad_declaration_is_typed():
    token = _native.CancellationToken()
    token.cancel()
    error = refusal_of(
        lambda: mc.compose_mechanisms(
            chain(), target_population_id="pop", admit_declared_mechanisms=True, cancel=token
        )
    )
    assert error.reason_code == "cancelled_no_claim"
    prepared = compose(chain())
    error = refusal_of(lambda: prepared.sample({}, n_rows=100, cancel=token))
    assert error.reason_code == "cancelled_no_claim"
    error = refusal_of(lambda: prepared.export(contrast_query(), cancel=token))
    assert error.reason_code == "cancelled_no_claim"

    with pytest.raises(CausalTypeError):
        mc.compose_mechanisms(["not a mechanism"], target_population_id="pop")  # type: ignore[list-item]
    with pytest.raises(CausalValueError):
        mc.Quantity("X", units="u", population_id="pop", role="not-a-role")
    with pytest.raises(CausalValueError):
        linear("X", float("nan"), 1.0)
