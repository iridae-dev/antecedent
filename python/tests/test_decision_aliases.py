"""Every spelling in ``antecedent.preferences`` and ``antecedent.preference_learning`` names the
same object as its ``antecedent.decision`` / ``antecedent.decision_learning`` spelling.

Fixture: two actions over aligned joint draws of benefit and harm; treat leads below a harm
weight of 2 and wait above it.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import warnings
from typing import Any

import numpy as np
import pytest
from antecedent import decision
from antecedent import decision_learning as dl
from antecedent import preference_learning as pl
from antecedent import preferences as pf
from antecedent.joint_distribution import (
    DistributionIdentity,
    JointDistributionArtifact,
    ScientificQuantity,
)
from antecedent.outcomes import require_success

# Names that are the same object under both spellings: (preferences, decision).
SAME_OBJECT = [
    ("attribute", "attribute"),
    ("Attribute", "Attribute"),
    ("Parameter", "Tradeoff"),
    ("Expr", "ValueExpr"),
    ("exp", "exp"),
    ("ln", "ln"),
    ("excess", "excess"),
    ("shortfall", "shortfall"),
    ("segment", "segment"),
    ("interaction", "interaction"),
    ("both", "both"),
    ("Curve", "Curve"),
    ("Relation", "Relation"),
    ("relation", "relation"),
    ("Measure", "Measure"),
    ("Structures", "Structures"),
    ("Policy", "TradeoffPolicy"),
    ("Action", "Action"),
    ("Criterion", "Criterion"),
    ("CompiledDecision", "CompiledDecision"),
    ("CompiledQuantity", "CompiledQuantity"),
    ("TradeOffRange", "TradeOffRange"),
    ("JointLawInput", "JointLawInput"),
    ("JointLawPlan", "JointLawPlan"),
    ("DesignLawPlan", "DesignLawPlan"),
    ("PreferenceDecision", "TradeoffDecision"),
    ("ActionRecord", "ActionRecord"),
    ("Switch", "Switch"),
    ("SwitchThreshold", "SwitchThreshold"),
    ("Region", "Region"),
    ("StructureOutcome", "StructureOutcome"),
    ("ClassSampling", "ClassSampling"),
    ("SampledMass", "SampledMass"),
    ("SampledMargin", "SampledMargin"),
    ("joint_law", "joint_law"),
    ("chain_law", "chain_law"),
    ("experiment_means", "experiment_means"),
    ("experiment_quantity", "experiment_quantity"),
    ("PreferenceRefusal", "PreferenceRefusal"),
]


def _quantity(variable: str, units: str, regime: str) -> ScientificQuantity:
    return ScientificQuantity(
        variable_id=variable,
        variable_name=variable,
        role="outcome",
        units=units,
        population_id="target",
        regime_id=regime,
        horizon=0,
        functional_id="outcome",
    )


TREAT_BENEFIT = _quantity("benefit", "qaly", "do(a=1)")
TREAT_HARM = _quantity("harm", "events", "do(a=1)")
WAIT_BENEFIT = _quantity("benefit", "qaly", "do(a=0)")
WAIT_HARM = _quantity("harm", "events", "do(a=0)")
COLUMNS = (TREAT_BENEFIT, TREAT_HARM, WAIT_BENEFIT, WAIT_HARM)
ROWS = [[1.0, 0.0, 1.0, 0.0], [3.0, 1.0, 1.0, 0.0]]


def _source() -> JointDistributionArtifact:
    identity = DistributionIdentity(
        semantic="interventional_predictive",
        quantities=COLUMNS,
        alignment="joint",
        source_id="enumerated",
        provider_id="exact-law",
        rng_id="deterministic_exact",
        snapshot_id="enumeration-1",
        causal_contract_id="checked-contract",
    )
    return JointDistributionArtifact(
        identity, np.array(ROWS, dtype=np.float64), calibration="exact"
    )


def _inputs() -> tuple[dict[str, ScientificQuantity], dict[str, ScientificQuantity]]:
    return (
        {"benefit": TREAT_BENEFIT, "harm": TREAT_HARM},
        {"benefit": WAIT_BENEFIT, "harm": WAIT_HARM},
    )


def _preferences_contract(lower: float = 0.5, upper: float = 4.0, **kwargs: Any) -> pf.Contract:
    benefit = pf.attribute("benefit", units="qaly")
    harm = pf.attribute("harm", units="events")
    w = pf.strength("harm_weight", units="qaly/event", lower=lower, upper=upper, assumption="a")
    treat, wait = _inputs()
    return pf.Contract(
        model=pf.ValueModel(benefit - w * harm, utility_units="qaly"),
        actions=(pf.Action("treat", inputs=treat), pf.Action("wait", inputs=wait)),
        target_population="target",
        **kwargs,
    )


def _decision_contract(lower: float = 0.5, upper: float = 4.0, **kwargs: Any) -> decision.Contract:
    benefit = decision.attribute("benefit", units="qaly")
    harm = decision.attribute("harm", units="events")
    w = decision.tradeoff(
        "harm_weight", units="qaly/event", lower=lower, upper=upper, assumption="a"
    )
    treat, wait = _inputs()
    return decision.Contract(
        value=decision.Value(benefit - w * harm, units="qaly"),
        actions=(decision.Action("treat", inputs=treat), decision.Action("wait", inputs=wait)),
        target_population="target",
        **kwargs,
    )


# --------------------------------------------------------------------------- the objects


def test_every_preferences_name_is_the_decision_object():
    for spelled, name in SAME_OBJECT:
        assert getattr(pf, spelled) is getattr(decision, name), (spelled, name)
    assert pf.PreferenceDecision is decision.TradeoffDecision

    # The names that adapt rather than alias keep a documented relation to their decision form.
    assert issubclass(pf.ValueModel, decision.Value)
    assert issubclass(pf.Contract, decision.Contract) or callable(pf.Contract)
    model = pf.ValueModel(
        pf.attribute("benefit", units="qaly") - 1.0 * pf.attribute("harm", units="events"),
        utility_units="qaly",
    )
    assert isinstance(model, decision.Value)
    assert model.utility_units == "qaly"
    assert model.value is not None
    assert pf.strength("w", units="u", lower=0.0, upper=1.0, assumption="a") == decision.tradeoff(
        "w", units="u", lower=0.0, upper=1.0, assumption="a"
    )
    assert pf.fixed("w", units="u", value=1.0) == decision.tradeoff("w", units="u", value=1.0)
    assert isinstance(pf.fixed("w", units="u", value=1.0), decision.Tradeoff)
    harm = decision.attribute("harm", units="events")
    assert pf.constraint("cap", harm, at_most=0.5, units="events") == decision.Constraint(
        "cap", harm, bound=0.5, units="events"
    )
    # The structural policy is a string on the decision spelling; the class is accepted.
    assert pf.StructuralPolicy.maximin().kind == "maximin"


def test_every_preference_learning_name_is_decision_learning():
    assert set(pl.__all__) == set(dl.__all__)
    for name in pl.__all__:
        assert getattr(pl, name) is getattr(dl, name), name
    assert pl.load is dl.load
    assert pl._raise is dl._raise
    assert pl._mean_wire is dl._mean_wire


def test_preferences_spellings_give_equal_results():
    for kwargs in ({}, {"policy": decision.TradeoffPolicy.maximin()}):
        in_preferences, in_decision = _preferences_contract(**kwargs), _decision_contract(**kwargs)
        assert in_preferences.identity == in_decision.identity
        by_preferences = require_success(in_preferences.evaluate(_source()))
        by_decision = require_success(in_decision.evaluate(_source()))
        assert type(by_preferences) is type(by_decision) is decision.TradeoffDecision
        assert by_preferences.to_dict() == by_decision.to_dict()

    # A learned-set object built through either module is equal.
    for name in ("Stated", "Pooled", "PerDecider", "Forecast"):
        assert getattr(pl, name) is getattr(dl, name)

    # A fixed value through either spelling is a trade-off decision with the same numbers.
    by_preferences_fixed = pf.Contract(
        model=pf.ValueModel(
            pf.attribute("benefit", units="qaly")
            - pf.fixed("w", units="qaly/event", value=1.0) * pf.attribute("harm", units="events"),
            utility_units="qaly",
        ),
        actions=(
            pf.Action("treat", inputs=_inputs()[0]),
            pf.Action("wait", inputs=_inputs()[1]),
        ),
        target_population="target",
    )
    by_decision_fixed = decision.Contract(
        value=decision.Value(
            decision.attribute("benefit", units="qaly")
            - decision.tradeoff("w", units="qaly/event", value=1.0)
            * decision.attribute("harm", units="events"),
            units="qaly",
        ),
        actions=(
            decision.Action("treat", inputs=_inputs()[0]),
            decision.Action("wait", inputs=_inputs()[1]),
        ),
        target_population="target",
    )
    assert by_preferences_fixed.identity == by_decision_fixed.identity
    assert (
        require_success(by_preferences_fixed.evaluate(_source())).to_dict()
        == require_success(by_decision_fixed.evaluate(_source())).to_dict()
    )


def test_preferences_spellings_emit_no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        declared = _preferences_contract()
        result = require_success(declared.evaluate(_source()))
        assert result.kind == "preference_dependent"
        for name in pf.__all__:
            getattr(pf, name)
        for name in pl.__all__:
            getattr(pl, name)
        _ = (pf.PreferenceDecision, pf.Policy, pf.Parameter, pf.Expr)

    probe = textwrap.dedent(
        """
        import warnings
        warnings.simplefilter("error")
        import antecedent
        from antecedent import preferences, preference_learning, decision, decision_learning
        for name in preferences.__all__:
            getattr(preferences, name)
        for name in preference_learning.__all__:
            getattr(preference_learning, name)
        """
    )
    run = subprocess.run(
        [sys.executable, "-W", "error", "-c", probe], capture_output=True, text=True
    )
    assert run.returncode == 0, run.stderr


def test_preferences_contract_keeps_x_k_as_attribute_k():
    # In the preferences spelling a positional read is the model's k-th attribute (harm is 1).
    by_position = decision.Constraint("cap", decision.x(1), bound=0.5, units="events")
    by_name = pf.constraint(
        "cap", pf.attribute("harm", units="events"), at_most=0.5, units="events"
    )
    positional = _preferences_contract(0.5, 1.5, constraints=(by_position,))
    named = _preferences_contract(0.5, 1.5, constraints=(by_name,))
    assert positional.identity == named.identity
    first = require_success(positional.evaluate(_source()))
    second = require_success(named.evaluate(_source()))
    assert first.to_dict() == second.to_dict()
    assert {a.id: a.admissible for a in first.actions} == {
        a.id: a.admissible for a in second.actions
    }


# --------------------------------------------------------------------------- separation


def test_deciding_with_stated_trade_offs_loads_no_learning():
    probe = textwrap.dedent(
        """
        import sys
        import numpy as np
        import antecedent
        from antecedent import decision
        from antecedent.joint_distribution import (
            DistributionIdentity, JointDistributionArtifact, ScientificQuantity,
        )
        from antecedent.outcomes import require_success

        def q(variable, units, regime):
            return ScientificQuantity(
                variable_id=variable, variable_name=variable, role="outcome", units=units,
                population_id="target", regime_id=regime, horizon=0, functional_id="outcome",
            )

        cols = (q("benefit", "qaly", "do(a=1)"), q("harm", "events", "do(a=1)"),
                q("benefit", "qaly", "do(a=0)"), q("harm", "events", "do(a=0)"))
        identity = DistributionIdentity(
            semantic="interventional_predictive", quantities=cols, alignment="joint",
            source_id="enumerated", provider_id="exact-law", rng_id="deterministic_exact",
            snapshot_id="enumeration-1", causal_contract_id="checked-contract",
        )
        source = JointDistributionArtifact(
            identity, np.array([[1.0, 0.0, 1.0, 0.0], [3.0, 1.0, 1.0, 0.0]]),
            calibration="exact",
        )
        benefit = decision.attribute("benefit", units="qaly")
        harm = decision.attribute("harm", units="events")
        w = decision.tradeoff("w", units="qaly/event", lower=0.5, upper=4.0, assumption="a")
        contract = decision.Contract(
            value=decision.Value(benefit - w * harm, units="qaly"),
            actions=(
                decision.Action("treat", inputs={"benefit": cols[0], "harm": cols[1]}),
                decision.Action("wait", inputs={"benefit": cols[2], "harm": cols[3]}),
            ),
            target_population="target",
        )
        assert require_success(contract.evaluate(source)).kind == "preference_dependent"
        assert callable(contract.compile)
        loaded = sorted(m for m in sys.modules if "learning" in m and m.startswith("antecedent"))
        assert "antecedent.preference_learning" not in sys.modules, loaded
        assert "antecedent.decision_learning" not in sys.modules, loaded
        """
    )
    run = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr


@pytest.mark.parametrize("module", ["antecedent.decision", "antecedent.preferences"])
def test_importing_a_deciding_module_loads_no_learning(module):
    probe = (
        "import sys\n"
        f"import {module}\n"
        "assert 'antecedent.preference_learning' not in sys.modules\n"
        "assert 'antecedent.decision_learning' not in sys.modules\n"
    )
    run = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
