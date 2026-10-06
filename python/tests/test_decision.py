"""Decision contracts evaluated on aligned joint draws.

The fixture is the one `crates/antecedent-design/tests/decision_contract_eval.rs`
asserts: the risky action's utility is ``P * Q``; with the enumerated rows below
``E[P] * E[Q] = 4.5`` but ``E[P * Q] = 2``, so only genuine joint rows choose the
safe action.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import tempfile
import textwrap

import numpy as np
import pytest
from antecedent import decision
from antecedent.errors import CausalUnsupportedError, CausalValueError
from antecedent.joint_distribution import (
    DistributionIdentity,
    JointDistributionArtifact,
    ScientificQuantity,
)

P = [1.0, 3.0, 2.0, 0.0]
Q = [4.0, 0.0, 2.0, 6.0]


def _quantity(variable: str, regime: str) -> ScientificQuantity:
    return ScientificQuantity(
        variable_id=variable,
        variable_name=variable,
        role="outcome",
        units="units",
        population_id="target",
        regime_id=regime,
        horizon=0,
        functional_id="outcome",
    )


COLUMNS = (_quantity("p", "do(a=1)"), _quantity("q", "do(a=1)"), _quantity("safe", "do(a=0)"))


def _source(
    p: list[float] = P,
    q: list[float] = Q,
    *,
    alignment: str = "joint",
    semantic: str = "interventional_predictive",
    calibration: str = "exact",
) -> JointDistributionArtifact:
    identity = DistributionIdentity(
        semantic=semantic,  # type: ignore[arg-type]
        quantities=COLUMNS,
        alignment=alignment,  # type: ignore[arg-type]
        source_id="enumerated",
        provider_id="exact-law",
        rng_id="deterministic_exact",
        snapshot_id="enumeration-1",
        causal_contract_id="checked-contract",
    )
    draws = np.array([[a, b, 3.0] for a, b in zip(p, q, strict=True)], dtype=np.float64)
    return JointDistributionArtifact(
        identity,
        draws,
        calibration=calibration,  # type: ignore[arg-type]
    )


def _contract(criterion: decision.Criterion | None = None, **kwargs: object) -> decision.Contract:
    fields: dict[str, object] = {
        "actions": (
            decision.Action(
                "risky", inputs=(COLUMNS[0], COLUMNS[1]), utility=decision.x(0) * decision.x(1)
            ),
            decision.Action("safe", inputs=(COLUMNS[2],), utility=decision.x(0), kind="policy"),
        ),
        "utility_units": "units",
        "criterion": criterion or decision.Criterion.expected_utility(),
        "target_population": "target",
    }
    fields.update(kwargs)
    return decision.Contract(**fields)  # type: ignore[arg-type]


def test_joint_rows_choose_the_safe_action_and_explain_why():
    result = _contract().evaluate(_source())
    risky, safe = result.actions
    assert risky.expected_utility == pytest.approx(2.0)
    assert safe.expected_utility == pytest.approx(3.0)
    assert result.verdict == decision.Verdict("uniquely_optimal", ("safe",))
    assert result.selected == ("safe",)
    # E[max(PQ, 3)] = 3.5, so perfect information is worth 0.5.
    assert result.evpi == pytest.approx(0.5)
    assert risky.expected_regret == pytest.approx(1.5)
    assert "'safe' ranks first by posterior expected utility" in result.explain()
    assert "perfect information would be worth 0.5" in result.explain()
    assert result.contract_identity == _contract().identity

    # The same marginals paired differently flip the choice.
    flipped = _contract().evaluate(_source(q=[0.0, 6.0, 2.0, 4.0]))
    assert flipped.actions[0].expected_utility == pytest.approx(5.5)
    assert flipped.selected == ("risky",)


def test_expressions_use_ordinary_operators_and_refuse_nonsense():
    expr = decision.maximum(decision.x(0) * decision.x(1) - 1.0, 0.0)
    assert isinstance(expr, decision.Expr)
    assert isinstance(-decision.x(0) + 2 * decision.x(1), decision.Expr)
    with pytest.raises(CausalValueError):
        decision.x(-1)
    with pytest.raises(CausalValueError):
        decision.x(0) + "a"  # type: ignore[operator]


def test_independent_marginals_and_wrong_meanings_refuse_with_registered_codes():
    with pytest.raises(CausalUnsupportedError) as marginals:
        _contract().evaluate(_source(alignment="independent_marginals"))
    assert marginals.value.reason_code == "joint_law_required"
    assert isinstance(marginals.value, decision.DecisionRefusal)

    with pytest.raises(decision.DecisionRefusal) as meaning:
        _contract().evaluate(_source(semantic="causal_functional_posterior"))
    assert meaning.value.reason_code == "distribution_meaning_mismatch"
    assert meaning.value.offending == "risky[0]"

    wrong_units = dataclasses.replace(COLUMNS[0], units="kg")
    other = _contract(
        actions=(
            decision.Action("risky", inputs=(wrong_units, COLUMNS[1]), utility=decision.x(0)),
            decision.Action("safe", inputs=(COLUMNS[2],), utility=decision.x(0)),
        )
    )
    with pytest.raises(decision.DecisionRefusal) as missing:
        other.evaluate(_source())
    assert missing.value.reason_code == "quantity_semantics_mismatch"
    assert missing.value.detail == "decision.quantity_not_found"

    with pytest.raises(decision.DecisionRefusal) as structure:
        _contract(decision.Criterion.maximin_over_structures()).evaluate(_source())
    assert structure.value.reason_code == "route_not_supported"

    with pytest.raises(decision.DecisionRefusal) as invalid:
        _contract(actions=_contract().actions[:1]).evaluate(_source())
    assert invalid.value.reason_code == "decision_contract_unsatisfied"


def test_hard_constraints_exclude_and_never_penalize():
    cap = decision.Constraint(
        "q-cap", decision.x(1), bound=5.0, units="units", applies_to=("risky",)
    )
    result = _contract(constraints=(cap,)).evaluate(_source())
    risky = result.actions[0]
    assert not risky.admissible
    assert risky.exclusions[0].probability == pytest.approx(0.75)
    assert risky.exclusions[0].required == 1.0
    # No penalty was subtracted from the excluded action's utility.
    assert risky.expected_utility == pytest.approx(2.0)
    assert result.selected == ("safe",)
    assert "excluded by a hard constraint: 'risky'" in result.explain()

    everything = decision.Constraint("none", decision.x(0), bound=-1.0, units="units")
    nothing = _contract(constraints=(everything,)).evaluate(_source())
    assert nothing.verdict.kind == "no_admissible_action"
    assert nothing.evpi is None
    assert "No action satisfies every hard constraint" in nothing.explain()


def test_each_criterion_has_its_own_route_and_sampling_noise_is_not_hidden():
    source = _source()
    threshold = _contract(decision.Criterion.threshold_probability(3.0)).evaluate(source)
    assert threshold.actions[0].value == pytest.approx(0.5)
    assert threshold.actions[1].value == pytest.approx(1.0)
    quantile = _contract(decision.Criterion.quantile(0.25)).evaluate(source)
    assert quantile.actions[0].value == pytest.approx(0.0)
    assert _contract(decision.Criterion.expected_loss()).evaluate(source).selected == ("risky",)
    assert _contract(decision.Criterion.regret()).evaluate(source).actions[0].value == (
        pytest.approx(3.0)
    )
    assert _contract(decision.Criterion.expected_regret()).evaluate(source).actions[
        1
    ].value == pytest.approx(0.5)
    # The same rows read as a sample, not an exact law, cannot separate the actions.
    sampled = _contract().evaluate(_source(calibration="unmeasured"))
    assert sampled.verdict.kind == "indistinguishable"
    assert sampled.selected == ("safe", "risky")
    assert "cannot be separated from 'risky'" in sampled.explain()


def test_contract_identity_ignores_declaration_order_and_tracks_semantics():
    base = _contract()
    reordered = _contract(actions=tuple(reversed(base.actions)))
    assert reordered.identity == base.identity
    edits = [
        _contract(structural_policy="maximin"),
        _contract(decision.Criterion.expected_loss()),
        _contract(utility_units="usd"),
        _contract(
            actions=(
                dataclasses.replace(base.actions[0], utility=decision.x(0)),
                base.actions[1],
            )
        ),
    ]
    assert all(edit.identity != base.identity for edit in edits)


def test_contract_and_result_round_trip_and_replay_in_a_fresh_process():
    contract = _contract(
        constraints=(
            decision.Constraint(
                "q-cap",
                decision.x(1),
                bound=5.0,
                units="units",
                min_probability=0.75,
                applies_to=("risky",),
            ),
        )
    )
    source = _source()
    result = contract.evaluate(source)
    contract_bytes = contract.export()
    result_bytes = result.export()

    loaded = decision.Contract.load(contract_bytes, expected_identity=contract.identity)
    assert loaded == contract
    with pytest.raises(Exception, match="differs"):
        decision.Contract.load(contract_bytes, expected_identity="0" * 64)
    replayed = decision.replay(result_bytes, contract=contract, source=source)
    assert replayed.selected == result.selected
    assert replayed.evpi == result.evpi

    other = dataclasses.replace(contract, criterion=decision.Criterion.expected_loss())
    with pytest.raises(decision.DecisionRefusal) as wrong_contract:
        decision.replay(result_bytes, contract=other, source=source)
    assert wrong_contract.value.reason_code == "decision_contract_unsatisfied"
    with pytest.raises(decision.DecisionRefusal):
        decision.replay(result_bytes, contract=contract, source=_source(q=[0.0, 6.0, 2.0, 4.0]))

    script = textwrap.dedent(
        """
        import sys
        from antecedent import decision
        from antecedent.joint_distribution import (
            DistributionIdentity, JointDistributionArtifact, ScientificQuantity,
        )
        import numpy as np

        def q(v, regime):
            return ScientificQuantity(v, v, "outcome", "units", "target", regime, 0, "outcome")

        cols = (q("p", "do(a=1)"), q("q", "do(a=1)"), q("safe", "do(a=0)"))
        identity = DistributionIdentity(
            "interventional_predictive", cols, "joint", "enumerated", "exact-law",
            "deterministic_exact", "enumeration-1", "checked-contract",
        )
        draws = np.array(
            [[1.0, 4.0, 3.0], [3.0, 0.0, 3.0], [2.0, 2.0, 3.0], [0.0, 6.0, 3.0]]
        )
        # The consumer rebuilds the source from its own constants.
        source = JointDistributionArtifact(identity, draws, calibration="exact")
        contract = decision.Contract(
            actions=(
                decision.Action("risky", cols[:2], decision.x(0) * decision.x(1)),
                decision.Action("safe", cols[2:], decision.x(0), kind="policy"),
            ),
            utility_units="units",
            criterion=decision.Criterion.expected_utility(),
            target_population="target",
            constraints=(
                decision.Constraint(
                    "q-cap", decision.x(1), 5.0, "units", 0.75, ("risky",)
                ),
            ),
        )
        retained = sys.argv[3]
        stored = decision.Contract.load(open(sys.argv[1], "rb").read(), expected_identity=retained)
        assert stored == contract
        result = decision.replay(open(sys.argv[2], "rb").read(), contract=contract, source=source)
        assert result.selected == ("safe",)
        assert abs(result.evpi - 0.5) < 1e-12
        """
    )
    with (
        tempfile.NamedTemporaryFile(suffix=".bin") as c_file,
        tempfile.NamedTemporaryFile(suffix=".bin") as r_file,
    ):
        c_file.write(contract_bytes)
        c_file.flush()
        r_file.write(result_bytes)
        r_file.flush()
        done = subprocess.run(
            [sys.executable, "-c", script, c_file.name, r_file.name, contract.identity],
            capture_output=True,
            text=True,
            check=False,
        )
    assert done.returncode == 0, done.stderr
    assert json.loads(json.dumps(result.selected)) == ["safe"]
