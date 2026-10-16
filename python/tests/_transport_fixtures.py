"""Small deterministic fixtures shared by the transported-route tests.

They are the Python spelling of the fixtures of the Rust facade tests
(``crates/antecedent/tests/{inverse_forward_sources,transported_decision,transport_sensitivity,
transport_revalidation,discretized_composition,whole_path_account}_facade.rs``), so each oracle is
the hand value the Rust test states.

Linear-Gaussian chain ``X = 1 + 2 e``, ``M = 0.5 + 2 X + e``, ``Y = -1 + 0.5 X + 3 M + 0.5 e``:
``E[Y | do(X = v)] = 0.5 + 6.5 v``, so ``0.5, 7.0, 13.5`` at ``v = 0, 1, 2``. After ``M`` is refit
as ``1 + 3 X``: ``E[Y | do(X = v)] = 2 + 9.5 v``.

Binary chain ``X -> M -> Y, X -> Y``: ``P(Y = 1 | do(X = 0)) = 0.18`` and
``P(Y = 1 | do(X = 1)) = 0.72``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from antecedent import decision
from antecedent import mechanism_composition as mc
from antecedent import transported_response as tr
from antecedent.joint_distribution import (
    DistributionIdentity,
    JointDistributionArtifact,
    ScientificQuantity,
)

ROLES = {"X": "treatment", "M": "mediator", "Y": "outcome"}


# --------------------------------------------------------------------------- compositions


def quantity(variable: str, units: str = "u", population: str = "pop") -> mc.Quantity:
    return mc.Quantity(
        variable,
        units=units,
        population_id=population,
        role=ROLES.get(variable, "covariate"),
        variable_name=variable,
    )


def declaration(
    variable: str, parents=(), units: str = "u", population: str = "pop"
) -> mc.MechanismDeclaration:
    return mc.MechanismDeclaration(
        output=quantity(variable, units, population),
        parents=tuple(quantity(p, units, population) for p in parents),
        artifact_id=f"fit:{variable}",
        evidence_id=f"data:{variable}",
    )


def linear(variable, intercept, sigma, parents=(), *, population: str = "pop") -> mc.Mechanism:
    names = [p for p, _ in parents]
    return mc.Mechanism.declared_linear(
        declaration(variable, names, population=population),
        intercept=intercept,
        sigma=sigma,
        coefficients=dict(parents),
    )


def gaussian_chain(m_intercept: float = 0.5, m_slope: float = 2.0) -> list[mc.Mechanism]:
    return [
        linear("X", 1.0, 2.0),
        linear("M", m_intercept, 1.0, [("X", m_slope)]),
        linear("Y", -1.0, 0.5, [("X", 0.5), ("M", 3.0)]),
    ]


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


def compose(mechanisms, **kwargs: Any) -> mc.PreparedMechanismComposition:
    return mc.compose_mechanisms(
        mechanisms, target_population_id="pop", admit_declared_mechanisms=True, **kwargs
    )


# --------------------------------------------------------------------------- decisions


def outcome_under(
    arm: str, functional: str = "mean", population: str = "pop"
) -> ScientificQuantity:
    """``functional`` of ``Y`` under ``do(X = arm)``."""
    return ScientificQuantity(
        variable_id="Y",
        variable_name="Y",
        role="outcome",
        units="u",
        population_id=population,
        regime_id=f"do(X={arm})",
        horizon=0,
        functional_id=functional,
    )


def identity_contract(arms, functional: str = "mean") -> decision.Contract:
    return decision.Contract(
        actions=tuple(
            decision.Action(
                f"x{arm}", inputs=(outcome_under(arm, functional),), utility=decision.x(0)
            )
            for arm in arms
        ),
        utility_units="u",
        criterion=decision.Criterion.expected_utility(),
        target_population="pop",
    )


# --------------------------------------------------------------------------- transported


def effect(population: str = "clinic_b", variable: str = "effect") -> ScientificQuantity:
    return ScientificQuantity(
        variable_id=variable,
        variable_name=variable,
        role="outcome",
        units="units",
        population_id=population,
        regime_id="contrast:do(t=1)-do(t=0)",
        horizon=0,
        functional_id="mean",
    )


def transport_declaration(
    target: str = "clinic_b",
    *,
    source: str = "trial",
    identity: str = "transport-1",
    diagram: str = "diagram-1",
    snapshot: str = "snapshot-1",
    route: str = "declared",
    standard_error: float | None = 0.3,
) -> tr.TransportDeclaration:
    return tr.TransportDeclaration(
        route=route,  # type: ignore[arg-type]
        source_identity=identity,
        source_populations=(source,),
        target_population=target,
        transport_identities=(diagram,),
        assumptions=("selection acts on age only",),
        provider_id="transport-provider",
        snapshot_id=snapshot,
        causal_contract_id="contract-1",
        uncalibrated_standard_error=standard_error,
    )


def point(
    value: float,
    target: str = "clinic_b",
    *,
    variable: str = "effect",
    **declaration_options: Any,
) -> tr.TransportedResponse:
    return tr.TransportedResponse.declared_point(
        [effect(target, variable)], [value], transport_declaration(target, **declaration_options)
    )


def transported_law(target: str = "clinic_b") -> tr.TransportedResponse:
    outcome = effect(target, "y")
    identity = DistributionIdentity(
        semantic="interventional_predictive",
        quantities=(outcome,),
        alignment="joint",
        source_id="transport",
        provider_id="transport-provider",
        rng_id="deterministic_exact",
        snapshot_id="snapshot-1",
        causal_contract_id="contract-1",
    )
    law = JointDistributionArtifact(
        identity,
        np.array([[1.0], [1.0], [0.0], [0.0]], dtype=np.float64),
        calibration="point_only",
    )
    return tr.TransportedResponse.declared_law(law, transport_declaration(target))


def digest(byte: int) -> str:
    """A 64-hex identity that is the byte repeated."""
    return f"{byte:02x}" * 32


# --------------------------------------------------------------------------- prepared analysis


def prepared_analysis():
    """A prepared static ATE analysis, whose causal contract a transported decision binds to."""
    import antecedent

    from _scm import confounded_scm

    data, edges = confounded_scm(n=200, seed=31)
    query = antecedent.AverageEffect(treatment="t", outcome="y")
    return antecedent.estimation.PreparedAnalysis.prepare(
        data, query=query, graph=edges, refute=False, seed=1
    )
