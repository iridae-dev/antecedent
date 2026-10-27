"""Correct a measurement, pool disjoint evidence, compose risk, and revise a decision.

All declarations are synthetic; point decisions carry no coverage claim.
"""

import math

from antecedent import decision
from antecedent import evidence_ledger as el
from antecedent import measurement_models as mm
from antecedent import mechanism_composition as mc
from antecedent.joint_distribution import ScientificQuantity
from antecedent.outcomes import require_success

_ROLES = {"X": "treatment", "Y": "outcome"}


def _quantity(variable: str) -> mc.Quantity:
    return mc.Quantity(
        variable,
        units="category",
        population_id="pop",
        role=_ROLES[variable],
        variable_name=f"label of {variable}",
    )


def _table(
    variable: str,
    tag: str,
    parents: list[str],
    rows: list[tuple[tuple[str, ...], tuple[float, float]]],
):
    declaration = mc.MechanismDeclaration(
        output=_quantity(variable),
        parents=tuple(_quantity(p) for p in parents),
        artifact_id=f"declared:{variable}:{tag}",
        evidence_id=f"evidence:{variable}:{tag}",
    )
    return mc.Mechanism.declared_table(
        declaration,
        levels=["0", "1"],
        rows=[
            (dict(zip(parents, config, strict=True)), {"0": p0, "1": p1})
            for config, (p0, p1) in rows
        ],
    )


def compose(prevalence: float, baseline: float, effect: float) -> mc.PreparedMechanismComposition:
    """``X ~ Bernoulli(prevalence)``; ``Y | X ~ Bernoulli(baseline + effect * X)``."""
    exposed = baseline + effect
    return mc.compose_mechanisms(
        [
            _table("X", repr(prevalence), [], [((), (1.0 - prevalence, prevalence))]),
            _table(
                "Y",
                f"{baseline!r}:{effect!r}",
                ["X"],
                [
                    (("0",), (1.0 - baseline, baseline)),
                    (("1",), (1.0 - exposed, exposed)),
                ],
            ),
        ],
        target_population_id="pop",
        dependence=None,
        admit_declared_mechanisms=True,
    )


def outcome_risk(
    composition: mc.PreparedMechanismComposition, do: dict[str, str] | None = None
) -> float:
    return require_success(composition.finite_law(do or {})).probability({"Y": "1"})


def excess_risk(composition: mc.PreparedMechanismComposition) -> float:
    """``P(Y = 1) - P(Y = 1 | do(X = 0))``."""
    return outcome_risk(composition) - outcome_risk(composition, {"X": "0"})


def main():
    representation = mm.RepresentationVersion("scalar", 1)
    model = mm.MeasurementModel(
        indicator=mm.MeasurementVariable("recorded", "noisy_misclassified", "indicator"),
        construct=mm.MeasurementVariable("exposure", "latent_scientific", "indicator"),
        parameters=mm.BinaryParameters((mm.BinaryRates(0.9, 0.8, 0.0, 0.0),), min_youden=0.05),
        population_id="pop",
        regime_id="observational",
        evidence=mm.ParameterEvidence.declared("tutorial"),
        representation=representation,
        model_id="exposure-misclassification",
    )
    binding = mm.AnalysisBinding("exposure", "pop", "observational", representation=representation)
    measured = require_success(
        mm.prepare_measurement_model(model, binding, mm.BinaryPrevalenceData(410, 1000)).estimate()
    )
    assert math.isclose(measured.point, 0.3)
    quantity = ScientificQuantity(
        variable_id="excess_risk",
        variable_name="attributable excess risk",
        role="outcome",
        units="risk",
        population_id="pop",
        regime_id="observational",
        horizon=0,
        functional_id="attributable_excess_risk",
    )
    contract = decision.Contract(
        actions=(
            decision.Action("act", inputs=(quantity,), utility=1000 * decision.x(0) - 30),
            decision.Action("hold", inputs=(quantity,), utility=0 * decision.x(0)),
        ),
        utility_units="net_cases_per_1000",
        target_population="pop",
        criterion=decision.Criterion.expected_utility(),
    )
    ledger = el.EvidenceLedger()
    source_a = ledger.add_source(
        el.SourceDeclaration("A", "observation_set", observations=el.rows("study-A", range(300)))
    )
    source_b = ledger.add_source(
        el.SourceDeclaration("B", "observation_set", observations=el.rows("study-B", range(300)))
    )
    previous_pool = None
    for label, second_effect, second_variance, expected_pool, expected_utility in [
        ("initial", 0.2, 0.0004, 0.15, 15.0),
        ("revised", 0.07, 0.0002, 0.08, -6.0),
    ]:
        if previous_pool is not None:
            revision = require_success(
                ledger.revise_source(
                    el.SourceRevision.update(
                        source_b,
                        el.SourceDeclaration(
                            "B-grown",
                            "observation_set",
                            observations=el.rows("study-B", range(600)),
                        ),
                    )
                )
            )
            assert revision.created_source is not None
            source_b = revision.created_source
            assert ledger.standing(previous_pool) == "recompute"
        pooled = require_success(
            ledger.pool_summaries(
                label,
                [
                    el.StudySummary(source_a, 0.1, 0.0004),
                    el.StudySummary(source_b, second_effect, second_variance),
                ],
                replaces=previous_pool,
            )
        )
        previous_pool = pooled.result
        effect = pooled.value.estimate
        assert math.isclose(effect, expected_pool)
        composition = compose(measured.point, 0.1, effect)
        excess = excess_risk(composition)
        source = decision.MeanSource(
            coordinates=(quantity,),
            means=(excess,),
            provider_id="declared-finite-composition",
            snapshot_id=ledger.state_digest,
            causal_contract_id=composition.plan_identity,
        )
        result = require_success(contract.evaluate(source))
        utilities = {action.id: action.expected_utility for action in result.actions}
        assert math.isclose(utilities["act"], expected_utility, abs_tol=1e-10)
        assert utilities["hold"] == 0
        print(
            label,
            "prevalence",
            measured.point,
            "pooled effect",
            effect,
            "utilities",
            utilities,
            "choose",
            max(utilities, key=utilities.get),
        )


if __name__ == "__main__":
    main()
