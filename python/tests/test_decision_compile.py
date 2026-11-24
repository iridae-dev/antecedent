"""``Contract.compile`` for ordinary contracts and trade-off ranges alike.

Fixture: a table with ``t -> y`` and a mean response curve of ``y`` under ``do(t = 0.5)`` and
``do(t = 1.5)``. The actions ``low`` and ``high`` read one dose's benefit each; ``high`` gains
about 1.5 over ``low``.
"""

from __future__ import annotations

import math
from typing import Any

import antecedent as ant
import numpy as np
import pytest
from antecedent import _reason_details as reason_details
from antecedent import decision
from antecedent.joint_distribution import ScientificQuantity
from antecedent.outcomes import require_success

DOSES = [0.5, 1.5]
BENEFIT = decision.attribute("benefit", units="points")
COST = decision.attribute("cost", units="usd")


def _dose_table(name: str = "y", rows: int = 240, slope: float = 1.5, rate: float = 1.7):
    t = np.array([2.0 * i / (rows - 1) for i in range(rows)])
    noise = np.array([np.sin(i * rate) for i in range(rows)])
    return {"t": t, name: 0.5 + slope * t + 0.8 * noise}


def _prepared(data, outcome: str = "y", *, bayes: bool = True):
    inference = (
        ant.Bayesian(backend="conjugate", n_draws=64) if bayes else ant.Frequentist(bootstrap=0)
    )
    return ant.prepare(
        data,
        graph=[("t", outcome)],
        query=ant.ResponseCurve("t", outcome, grid=DOSES),
        refute="none",
        inference=inference,
        execution=ant.Execution(seed=11),
    )


def _dose(outcome: str, dose: float, units: str = "points") -> ScientificQuantity:
    return ScientificQuantity(
        variable_id=outcome,
        variable_name=outcome,
        role="outcome",
        units=units,
        population_id="target",
        regime_id=f"do(t={dose})",
        horizon=0,
        functional_id="mean",
    )


def _dose_contract(value: decision.Value | None = None, **kwargs: Any) -> decision.Contract:
    value = value or decision.Value(BENEFIT, units="points")
    return decision.Contract(
        actions=(
            decision.Action("low", inputs={"benefit": _dose("y", 0.5)}),
            decision.Action("high", inputs={"benefit": _dose("y", 1.5)}),
        ),
        value=value,
        target_population="target",
        **kwargs,
    )


def _hand_dose_contract(**kwargs: Any) -> decision.Contract:
    return decision.Contract(
        actions=(
            decision.Action("low", inputs=(_dose("y", 0.5),), utility=decision.x(0)),
            decision.Action("high", inputs=(_dose("y", 1.5),), utility=decision.x(0)),
        ),
        utility_units="points",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
        **kwargs,
    )


def _priced_value(lower: float = 0.2, upper: float = 1.0) -> decision.Value:
    price = decision.tradeoff(
        "price", units="points/usd", lower=lower, upper=upper, assumption="price"
    )
    return decision.Value(BENEFIT - price * COST, units="points")


def _priced_contract(value: decision.Value) -> decision.Contract:
    return decision.Contract(
        actions=(
            decision.Action("low", inputs={"benefit": _dose("y", 0.5), "cost": 0.5}),
            decision.Action("high", inputs={"benefit": _dose("y", 1.5), "cost": 1.5}),
        ),
        value=value,
        target_population="target",
    )


# --------------------------------------------------------------------------- ordinary contracts


def test_an_ordinary_contract_compiles_with_a_prepared_study():
    prepared = _prepared(_dose_table())
    contract = _hand_dose_contract()
    compiled = require_success(contract.compile(prepared))

    # The plan is stated before anything is fitted.
    described = compiled.inspect()
    assert described["demand"]["representation"] == "mean"
    assert [q["quantity"]["regime_id"] for q in described["demand"]["quantities"]] == [
        "do(t=0.5)",
        "do(t=1.5)",
    ]
    assert compiled.contract_identity == contract.identity
    assert compiled.representation == "mean"
    assert compiled.trade_offs == ()
    assert compiled.joint_law is None

    decided = require_success(compiled.execute())
    assert type(decided) is decision.Decision
    assert decided.choice == "high"
    assert decided.selected == ("high",)
    assert decided.standing == "best"
    assert decided.contract_identity == contract.identity

    # The compiled decision is the ordinary decision of the same estimated quantities.
    receipt = decided.receipt
    assert receipt is not None
    assert receipt.contract_identity == contract.identity
    assert receipt.bound_identity == compiled.bound_identity
    assert receipt.executed


def test_ordinary_compile_across_studies_and_design():
    key = ant.UnitKey("patient", namespace="hospital_mrn")

    def keyed(outcome: str, patients: list[str]):
        table = _dose_table(outcome, 200)
        table["patient"] = np.array(patients, dtype=object)
        return ant.prepare(
            table,
            graph=[("t", outcome)],
            query=ant.ResponseCurve("t", outcome, grid=DOSES),
            refute="none",
            inference=ant.Bayesian(backend="conjugate", n_draws=64),
            execution=ant.Execution(seed=11),
            unit_key=key,
        )

    benefit = keyed("b", [f"p{i}" for i in range(200)])
    harm = keyed("h", [f"q{i}" for i in range(200)])
    value = decision.Value(
        decision.attribute("benefit", units="points")
        - 0.2 * decision.attribute("harm", units="events"),
        units="points",
    )
    contract = decision.Contract(
        actions=(
            decision.Action(
                "high", inputs={"benefit": _dose("b", 1.5), "harm": _dose("h", 1.5, "events")}
            ),
            decision.Action(
                "low", inputs={"benefit": _dose("b", 0.5), "harm": _dose("h", 0.5, "events")}
            ),
        ),
        value=value,
        # The median reads the joint law of the two outcomes, which separate studies join.
        criterion=decision.Criterion.quantile(0.5),
        target_population="target",
    )
    joined = require_success(contract.compile([benefit, harm], draws=2000, seed=7))
    assert joined.inspect()["joint_law"]["dependence"] == "distinct_evidence"
    decided = require_success(joined.execute())
    assert type(decided) is decision.Decision
    assert decided.choice == "high"

    # Studies of one table are shared evidence, ordinary or not.
    overlapping = keyed("h", [f"p{i}" for i in range(100, 300)])
    with pytest.raises(decision.DecisionRefusal) as shared:
        require_success(contract.compile([benefit, overlapping], draws=2000, seed=7))
    assert shared.value.reason_code == "evidence_double_counted"
    assert shared.value.detail == reason_details.preference_compile.STUDIES_SHARE_EVIDENCE

    # A compiled design: clusters assigned at random, read as arm means.
    design = _cluster_design()

    def arm(regime: str) -> dict[str, Any]:
        return {
            "benefit": _design_quantity("benefit", "qaly", regime),
            "harm": _design_quantity("harm", "events", regime),
        }

    by_design = decision.Contract(
        actions=(
            decision.Action("treat", inputs=arm("do(a=1)")),
            decision.Action("wait", inputs=arm("do(a=0)")),
        ),
        value=decision.Value(
            decision.attribute("benefit", units="qaly")
            - 0.2 * decision.attribute("harm", units="events"),
            units="qaly",
        ),
        target_population="target",
    )
    compiled = require_success(by_design.compile(design))
    assert compiled.inspect()["demand"]["representation"] == "mean"
    chosen = require_success(compiled.execute())
    assert type(chosen) is decision.Decision
    assert chosen.choice == "treat"


def _design_quantity(variable: str, units: str, regime: str) -> ScientificQuantity:
    return ScientificQuantity(
        variable_id=variable,
        variable_name=variable,
        role="outcome",
        units=units,
        population_id="target",
        regime_id=regime,
        horizon=0,
        functional_id="mean",
    )


def _cluster_design(clusters: int = 64, rows: int = 10):
    from antecedent import joint_panel as jp

    assigned = [c % 2 == 0 for c in range(clusters)]
    shocks = [math.sin((c + 1.0) * 12.9898) for c in range(clusters)]
    benefit = [
        rows * (2.0 + (1.0 if a else 0.0) + 0.2 * s) for a, s in zip(assigned, shocks, strict=True)
    ]
    harm = [
        rows * (1.0 + (0.5 if a else 0.0) + 0.1 * s) for a, s in zip(assigned, shocks, strict=True)
    ]
    return jp.prepare_native_joint_design(
        {
            "kind": "decision_cluster",
            "recipe": {
                "population": "target",
                "treated_regime_id": "do(a=1)",
                "control_regime_id": "do(a=0)",
                "horizon": 0,
                "cluster_ids": [f"cluster-{c}" for c in range(clusters)],
                "cluster_sizes": [rows] * clusters,
                "assignment_law": "complete randomization of half the clusters",
                "treated_clusters": sum(assigned),
                "consistency": "consistency",
                "assignments": assigned,
                "outcomes": [
                    {
                        "component_name": "benefit",
                        "variable_id": "benefit",
                        "units": "qaly",
                        "values": benefit,
                    },
                    {
                        "component_name": "harm",
                        "variable_id": "harm",
                        "units": "events",
                        "values": harm,
                    },
                ],
                "adjustments": [],
                "exposure": {"kind": "no_interference", "assumption": "assumption.no_spillover"},
            },
        }
    )


def _edge(names: list[str], source: str, target: str) -> int:
    n, i, j = len(names), names.index(source), names.index(target)
    return 1 << (i * (n - 1) + (j if j < i else j - 1))


def test_ordinary_compile_refuses_a_quantity_the_study_does_not_estimate():
    # A study of one structure that does not identify its query refuses when it is prepared,
    # and an ordinary contract over a graph posterior refuses at compile (see
    # test_ordinary_compile_over_graph_posterior), so what the compile itself checks of an
    # ordinary contract is that every input is a coordinate the study estimates. The refusal
    # names the action and the input position.
    contract = decision.Contract(
        actions=(
            decision.Action("low", inputs=(_dose("y", 0.5),), utility=decision.x(0)),
            decision.Action("high", inputs=(_dose("y", 0.7),), utility=decision.x(0)),
        ),
        utility_units="points",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
    )
    with pytest.raises(decision.DecisionRefusal) as refused:
        require_success(contract.compile(_prepared(_dose_table())))
    assert refused.value.reason_code == "quantity_semantics_mismatch"
    assert refused.value.detail == reason_details.program_binding.DOSE_GRID_CHANGED
    assert "action high" in str(refused.value) and "input 0" in str(refused.value)


def test_fixed_value_compiles_as_its_ordinary_contract():
    prepared = _prepared(_dose_table())
    by_value = require_success(_dose_contract().compile(prepared))
    by_hand = require_success(_hand_dose_contract().compile(prepared))
    assert by_value.contract_identity == by_hand.contract_identity
    assert by_value.bound_identity == by_hand.bound_identity
    assert by_value.representation == by_hand.representation == "mean"

    decided = require_success(by_value.execute())
    expected = require_success(by_hand.execute())
    assert type(decided) is type(expected) is decision.Decision
    assert decided.to_dict() == expected.to_dict()

    # A plain number beside an estimated quantity is a fixed trade-off with no name.
    priced = _priced_contract(
        decision.Value(BENEFIT - 0.8 * COST, units="points"),
    )
    plain = require_success(require_success(priced.compile(prepared)).execute())
    assert type(plain) is decision.Decision
    assert plain.choice == "high"


# --------------------------------------------------------------------------- ranged contracts


def test_ranged_compile_through_decision():
    prepared = _prepared(_dose_table())
    contract = _priced_contract(_priced_value())
    compiled = require_success(contract.compile(prepared))
    assert compiled.contract_identity == contract.identity
    assert compiled.route == "exact"
    assert compiled.trade_offs == (
        decision.TradeOffRange("price", "points/usd", 0.2, 1.0, "price"),
    )
    assert compiled.parameter_ranges == {"price": (0.2, 1.0)}

    decided = require_success(compiled.execute())
    assert type(decided) is decision.TradeoffDecision
    assert decided.kind == "necessarily_best"
    assert decided.choice == "high"
    assert decided.standing == "best"
    assert decided.receipt is not None
    assert decided.receipt.contract_identity == compiled.contract_identity
    assert decided.receipt.bound_identity == compiled.bound_identity


def test_ordinary_compile_over_graph_posterior():
    names = ["t", "y"]
    posterior = ant.discovery.GraphPosterior.from_atoms(
        names, [1.0], [_edge(names, "t", "y")], atom_kind="Dag"
    )
    prepared = ant.prepare(
        _dose_table(),
        discovery=posterior,
        query=ant.ResponseCurve("t", "y", grid=DOSES),
        refute="none",
        inference=ant.Bayesian(backend="conjugate", n_draws=64),
        execution=ant.Execution(seed=11),
    )
    # The compiled route decides an ordinary contract on the source of one structure; over a
    # graph posterior it refuses before anything is fitted.
    with pytest.raises(decision.DecisionRefusal) as refused:
        require_success(_hand_dose_contract().compile(prepared))
    assert refused.value.reason_code == "route_not_supported"
    assert refused.value.detail == (
        reason_details.preference_compile.ORDINARY_STRUCTURES_NOT_COMPILED
    )


@pytest.mark.parametrize("form", ["ordinary", "tradeoff"])
def test_an_analysis_error_behind_a_compiled_decision_is_the_analysis_own(form):
    # A memory budget the response curve cannot fit in is the analysis's own resource error,
    # raised as the analysis raises it, not a refusal of the decision.
    from antecedent.errors import CausalResourceError

    contract = _hand_dose_contract() if form == "ordinary" else _dose_contract()
    compiled = require_success(contract.compile(_prepared(_dose_table())))
    starved = ant.Execution(seed=11, limits=ant.ResourceLimits(memory_bytes=1))
    with pytest.raises(CausalResourceError) as raised:
        require_success(compiled.execute(execution=starved))
    assert not isinstance(raised.value, decision.DecisionRefusal)
    assert "memory budget" in str(raised.value)
