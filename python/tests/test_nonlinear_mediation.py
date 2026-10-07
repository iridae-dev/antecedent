"""B4 nonlinear continuous-mediator mediation through the Python facade.

Every expected number is the closed form of an exact structural model on a deterministic crossed
grid (the same dataset as ``crates/antecedent-estimate/tests/nonlinear_mediation.rs``).

SCM: ``M = alpha*A + gamma*X + e_M`` and
``Y = b0 + b1*A + c1*M + c2*M^2 + c3*M^3 + delta*A*M + bx*X`` (no outcome noise). ``X`` is
uniform on ``{-1, 0, 1, 2, 3}`` (``E[X] = 1``, ``E[X^2] = 3``, ``E[X^3] = 7``). The grid crosses
``A`` in {0, 1}, the five ``X`` values and four symmetric residual levels, scaled so the
degrees-of-freedom corrected residual variance equals ``sigma2``. Closed form:

    NDE = b1 + delta*gamma*E[X]
    NIE = (c1 + delta)*alpha + c2*(alpha^2 + 2*alpha*gamma*E[X]) + c3*(m3(1) - m3(0))
    TE  = NDE + NIE

Calibration is deliberately NOT measured here: bootstrap standard errors are reported with
calibration ``"unmeasured"`` and no interval.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from antecedent.errors import CausalSerializationError, CausalTypeError, CausalUnsupportedError
from antecedent.nonlinear_mediation import (
    MediationEstimand,
    MediationPremises,
    MediationRefusal,
    consume_mediation_artifact,
    mediation_effects,
)

from _refusal import assert_registered_refusal

XS = (-1.0, 0.0, 1.0, 2.0, 3.0)
EX, EX2, EX3 = 1.0, 3.0, 7.0

BASE = {
    "alpha": 0.5,
    "gamma": 0.5,
    "sigma2": 0.64,
    "b0": 1.0,
    "b1": 0.8,
    "c1": 0.6,
    "c2": 0.4,
    "c3": 0.0,
    "delta": 0.3,
    "bx": 0.7,
}


def scm(**overrides: float) -> dict[str, float]:
    return {**BASE, **overrides}


def grid(s: dict[str, float]) -> dict[str, np.ndarray]:
    rows = 2.0 * 5.0 * 4.0
    v = s["sigma2"] * (rows - 3.0) / rows
    lo = 0.4
    hi = math.sqrt(2.0 * v - lo * lo)
    a, m, y, x = [], [], [], []
    for arm in (0.0, 1.0):
        for xv in XS:
            for e in (-hi, -lo, lo, hi):
                mv = s["alpha"] * arm + s["gamma"] * xv + e
                yv = (
                    s["b0"]
                    + s["b1"] * arm
                    + s["c1"] * mv
                    + s["c2"] * mv**2
                    + s["c3"] * mv**3
                    + s["delta"] * arm * mv
                    + s["bx"] * xv
                )
                a.append(arm)
                m.append(mv)
                y.append(yv)
                x.append(xv)
    return {
        "treatment": np.array(a),
        "mediator": np.array(m),
        "outcome": np.array(y),
        "x": np.array(x),
    }


def m3(s: dict[str, float], ap: float) -> float:
    am = s["alpha"] * ap
    e_mu3 = (
        am**3
        + 3.0 * am * am * s["gamma"] * EX
        + 3.0 * am * s["gamma"] ** 2 * EX2
        + s["gamma"] ** 3 * EX3
    )
    return e_mu3 + 3.0 * s["sigma2"] * (am + s["gamma"] * EX)


def truth(s: dict[str, float]) -> tuple[float, float, float]:
    nde = s["b1"] + s["delta"] * s["gamma"] * EX
    nie = (
        (s["c1"] + s["delta"]) * s["alpha"]
        + s["c2"] * (s["alpha"] ** 2 + 2.0 * s["alpha"] * s["gamma"] * EX)
        + s["c3"] * (m3(s, 1.0) - m3(s, 0.0))
    )
    return nde, nie, nde + nie


IGNORABLE = MediationPremises.sequentially_ignorable()


def run(g: dict[str, np.ndarray], **kwargs):
    kwargs.setdefault("premises", IGNORABLE)
    kwargs.setdefault("quadrature_nodes", 8)
    kwargs.setdefault("bootstrap_replicates", 0)
    return mediation_effects(g["treatment"], g["mediator"], g["outcome"], {"x": g["x"]}, **kwargs)


def test_b4_mediation_exact_scm_recovers_closed_form_effects() -> None:
    s = scm()
    result = run(grid(s))
    nde, nie, te = truth(s)
    assert result.natural_direct == pytest.approx(nde, abs=1e-8)
    assert result.natural_indirect == pytest.approx(nie, abs=1e-8)
    assert result.total == pytest.approx(te, abs=1e-8)
    assert result.total == pytest.approx(result.natural_direct + result.natural_indirect, abs=1e-10)
    assert result.estimand is MediationEstimand.NATURAL_EFFECTS
    assert result.mediator_coefficients[1] == pytest.approx(s["alpha"], abs=1e-9)
    assert result.mediator_residual_variance == pytest.approx(s["sigma2"], abs=1e-9)
    assert result.n_rows == 40
    assert result.natural_direct_se is None


def test_b4_mediation_total_equals_direct_plus_indirect_off_base() -> None:
    s = scm(c2=-0.35, delta=-0.5, alpha=0.4, b1=-0.2)
    result = run(grid(s), quadrature_nodes=10)
    nde, nie, te = truth(s)
    assert result.identity_residual < 1e-10
    assert result.natural_direct == pytest.approx(nde, abs=1e-8)
    assert result.natural_indirect == pytest.approx(nie, abs=1e-8)
    assert result.total == pytest.approx(te, abs=1e-8)


def test_b4_mediation_reports_integration_error_and_refuses_above_tolerance() -> None:
    s = scm(c3=0.2)
    g = grid(s)
    nde, nie, te = truth(s)
    fine = run(g, outcome_degree=3, quadrature_nodes=8)
    assert fine.coarse_nodes == 8
    assert fine.fine_nodes == 16
    assert fine.integration_error < 1e-10
    assert fine.natural_direct == pytest.approx(nde, abs=1e-8)
    assert fine.natural_indirect == pytest.approx(nie, abs=1e-8)
    assert fine.total == pytest.approx(te, abs=1e-8)
    # One node is the plug-in mean; the n versus 2n disagreement is sigma2-sized and refused.
    with pytest.raises(MediationRefusal) as caught:
        run(g, outcome_degree=3, quadrature_nodes=1)
    assert caught.value.detail == "nonlinear_mediation.integration_error"
    assert caught.value.reason_code == "mechanism_fit_not_converged"
    assert_registered_refusal(caught.value)
    # A loose tolerance returns the estimate with the error reported, not hidden.
    loose = run(g, outcome_degree=3, quadrature_nodes=1, integration_tolerance=10.0)
    assert loose.integration_error > 1e-3
    assert loose.natural_indirect == pytest.approx(nie, abs=1e-8)


@pytest.mark.parametrize(
    "flag",
    [
        "unmeasured_treatment_outcome_confounding",
        "unmeasured_treatment_mediator_confounding",
        "unmeasured_mediator_outcome_confounding",
    ],
)
def test_b4_mediation_confounding_flags_refuse(flag: str) -> None:
    premises = dataclasses.replace(IGNORABLE, **{flag: True})
    with pytest.raises(MediationRefusal) as caught:
        run(grid(scm()), premises=premises)
    assert caught.value.detail == "nonlinear_mediation.confounding"
    assert caught.value.reason_code == "effect_not_identified"
    assert isinstance(caught.value, CausalUnsupportedError)
    assert_registered_refusal(caught.value)


def test_b4_mediation_treatment_induced_confounder_and_cross_world_refuse() -> None:
    induced = MediationPremises(treatment_induced_confounders=("L",), cross_world_independence=True)
    with pytest.raises(MediationRefusal) as caught:
        run(grid(scm()), premises=induced)
    assert caught.value.detail == "nonlinear_mediation.treatment_induced_confounding"
    assert caught.value.reason_code == "cross_world_not_identified"
    undeclared = MediationPremises()
    with pytest.raises(MediationRefusal) as caught:
        run(grid(scm()), premises=undeclared)
    assert caught.value.detail == "nonlinear_mediation.cross_world_independence_not_declared"
    assert_registered_refusal(caught.value)


def test_b4_mediation_interventional_estimand_is_closed() -> None:
    with pytest.raises(MediationRefusal) as caught:
        run(grid(scm()), estimand=MediationEstimand.INTERVENTIONAL_EFFECTS)
    assert caught.value.detail == "nonlinear_mediation.interventional_effects_closed"
    assert caught.value.reason_code == "route_not_supported"
    assert_registered_refusal(caught.value)
    with pytest.raises(MediationRefusal):
        run(grid(scm()), estimand="interventional_effects")


def test_b4_mediation_weak_overlap_refuses() -> None:
    # alpha = 6: the control mediator law sits far outside the treated arm's mediator range.
    with pytest.raises(MediationRefusal) as caught:
        run(grid(scm(alpha=6.0)))
    assert caught.value.detail == "nonlinear_mediation.overlap"
    # An unpopulated treated arm is the same refusal on counts.
    full = grid(scm())
    keep = np.ones(len(full["treatment"]), dtype=bool)
    keep[np.flatnonzero(full["treatment"] > 0.5)[3:]] = False
    small = {name: column[keep] for name, column in full.items()}
    with pytest.raises(MediationRefusal) as caught:
        run(small)
    assert caught.value.detail == "nonlinear_mediation.overlap"


def test_b4_mediation_bootstrap_is_seeded_unmeasured_and_has_no_interval() -> None:
    g = grid(scm())
    a = run(g, bootstrap_replicates=30, seed=7)
    b = run(g, bootstrap_replicates=30, seed=7)
    assert a.to_dict() == b.to_dict(), "the same seed reproduces the whole estimate"
    assert a.natural_direct_se is not None and a.natural_direct_se > 0.0
    assert a.natural_indirect_se is not None and a.total_se is not None
    assert a.bootstrap.seed == 7
    assert a.bootstrap.replicate_ids == tuple(range(30))
    assert a.bootstrap.replicates_succeeded + len(a.bootstrap.failed_replicate_ids) == 30
    assert a.calibration == "unmeasured"
    assert a.interval_status == "closed_calibration_unmeasured"
    assert a.bootstrap.interval_status == "closed_calibration_unmeasured"
    assert a.inference_claim == "point_with_diagnostics"
    assert any("unmeasured" in caveat for caveat in a.caveats)
    assert not hasattr(a, "interval") and not hasattr(a, "natural_direct_interval")
    other = run(g, bootstrap_replicates=30, seed=8)
    assert other.natural_direct_se != a.natural_direct_se


def test_b4_mediation_records_premises_declared_versus_checked() -> None:
    result = run(grid(scm()))
    status = {p.name: p.status for p in result.premises}
    assert status["cross_world_independence"] == "declared"
    assert status["no_unmeasured_mediator_outcome_confounding"] == "declared"
    assert status["arm_overlap"] == "checked"
    assert status["integration_error_within_tolerance"] == "checked"
    assert all(p.holds for p in result.premises)
    assert result.overlap.treated_count == 20
    assert result.overlap.control_count == 20
    assert result.overlap.mediator_support_violation <= 0.25
    assert result.model_specs["outcome_degree"] == 2
    assert "nonlinear_mediation.cross_world_independence" in result.assumptions


def test_b4_mediation_artifact_round_trips_and_checks_the_retained_identity() -> None:
    result = run(grid(scm()), bootstrap_replicates=8, seed=3)
    again = consume_mediation_artifact(result.export(), expected=result.identity)
    assert again.to_dict() == result.to_dict()
    assert again.identity == result.identity
    from_mapping = consume_mediation_artifact(result.export(), expected=result.identity._wire())
    assert from_mapping.identity == result.identity


def test_b4_mediation_artifact_refuses_a_resealed_change_against_retained_identity() -> None:
    original = run(grid(scm()))
    g = grid(scm())
    g["outcome"] = g["outcome"].copy()
    g["outcome"][0] += 1.0
    resealed = run(g)
    # The changed artifact is internally consistent ...
    assert consume_mediation_artifact(resealed.export()).identity == resealed.identity
    # ... and refused by the identity the consumer retained independently.
    with pytest.raises(MediationRefusal) as caught:
        consume_mediation_artifact(resealed.export(), expected=original.identity)
    assert caught.value.detail == "nonlinear_mediation.wrong_contract"
    assert caught.value.offending == "data"
    assert caught.value.reason_code == "route_not_supported"
    assert_registered_refusal(caught.value)


def test_b4_mediation_artifact_corruption_and_type_errors() -> None:
    result = run(grid(scm()))
    corrupt = bytearray(result.export())
    corrupt[len(corrupt) // 2] ^= 0xFF
    with pytest.raises(CausalSerializationError):
        consume_mediation_artifact(bytes(corrupt))
    with pytest.raises(CausalTypeError):
        consume_mediation_artifact("not bytes")  # type: ignore[arg-type]
    with pytest.raises(CausalTypeError):
        consume_mediation_artifact(result.export(), expected=3)  # type: ignore[arg-type]
    g = grid(scm())
    with pytest.raises(CausalTypeError):
        mediation_effects(
            g["treatment"],
            g["mediator"],
            g["outcome"],
            premises=None,  # type: ignore[arg-type]
        )
