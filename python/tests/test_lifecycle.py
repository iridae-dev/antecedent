"""The 2.3 acceptance lifecycle: identify, bind a foreign result, decide, hand off.

Identify a response curve, derive what an external provider must supply, bind an
attested provider's grid, inspect and export the claim, decide between two
actions that read the claim's coordinates, and let a fresh-process consumer
reload the claim under its own retained identity and recompute the decision.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import tempfile
import textwrap

import antecedent as ac
import pytest
from antecedent import decision, external
from antecedent.joint_distribution import ScientificQuantity

GRID = [0.0, 1.0, 2.0]
EDGES = [("x", "a"), ("x", "y"), ("a", "y")]
NAMES = ["x", "a", "y"]


def _spec() -> external.ExternalSpec:
    # (a) identify the response curve.
    ident = ac.identify(graph=EDGES, names=NAMES, query=ac.ResponseCurve("a", "y", grid=GRID))
    # (b) derive what an identified contract requires of an external result.
    return external.response(
        ident,
        outcome_units="mmHg",
        population="target",
        require_evidence=("factor:z",),
        require_assumptions=("ignorability",),
    )


def _provider() -> external.ProviderObject:
    return external.ProviderObject(
        provider_id="lab",
        object_id="curve",
        version="v3",
        snapshot="snap-9",
        request="req-1",
        meaning="interventional_predictive",
        capabilities=("mean",),
    )


def _claim(spec: external.ExternalSpec) -> external.BoundExternalClaim:
    # (c) bind an attested grid. Closed form: E[Y | do(a)] = 1 + 2a.
    return spec.bind(
        external.Response(
            provider=_provider(),
            values=[1.0, 3.0, 5.0],
            evidence=("factor:z",),
            assumptions=("ignorability",),
            attested_by="lab",
        )
    )


def _contract(claim: external.BoundExternalClaim) -> decision.Contract:
    # (e) two actions reading the claim's do(a=0) and do(a=2) coordinates.
    wait, _, treat = claim.quantities
    utility = decision.x(0) * 2.0 - 1.0
    return decision.Contract(
        actions=(
            decision.Action("wait", inputs=(wait,), utility=utility),
            decision.Action("treat", inputs=(treat,), utility=utility),
        ),
        utility_units="utility",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
    )


def test_identify_bind_inspect_decide_and_hand_off_to_a_fresh_process():
    spec = _spec()
    claim = _claim(spec)

    # (d) the claim is inspectable and exportable, and is never native.
    inspection = claim.inspect()
    assert inspection.native is False
    assert [link.id for link in inspection.lineage][-1] == "claim"
    data = claim.export()
    assert data
    assert {q.functional_id for q in claim.quantities} == {"mean"}
    assert [q.regime_id for q in claim.quantities] == ["do(a=0)", "do(a=1)", "do(a=2)"]

    # (e) decide on the claim: 2 * 1 - 1 = 1 and 2 * 5 - 1 = 9.
    contract = _contract(claim)
    result = contract.evaluate(claim)
    wait, treat = result.actions
    assert wait.expected_utility == pytest.approx(1.0)
    assert treat.expected_utility == pytest.approx(9.0)
    assert treat.value == pytest.approx(9.0)
    assert result.verdict == decision.Verdict("uniquely_optimal", ("treat",))
    assert result.selected == ("treat",)
    # A mean carries no joint law, no sampling error and no regret.
    assert result.evpi is None
    assert treat.standard_error is None
    assert treat.expected_regret is None
    assert result.n_draws == 0
    assert "external mean grid" in result.explain()
    assert "not estimated natively" in result.explain()

    # Every stage behind the decision is named, including the foreign provider.
    links = result.lineage
    assert links[-1].id == "decision_result"
    assert f"decision:{contract.identity}" in [link.id for link in links]
    assert "provider:external:lab/curve@v3#snap-9" in [link.id for link in links]
    # Every link, including the two appended after the claim, has a Merkle digest
    # that chains to its parents' digests.
    by_id = {link.id: link for link in links}
    for link in links:
        assert len(link.digest) == 64, link.id
        assert [by_id[p].digest for p in link.parents] == list(link.parent_digests), link.id
    stages = result.stages_behind()
    assert {
        "causal_contract",
        "evidence",
        "external_provider",
        "decision_contract",
        "claim",
    } <= stages
    assert "distribution_artifact" not in stages
    assert result.stages_behind(f"decision:{contract.identity}") == {"decision_contract"}

    # (f) a fresh-process consumer rebuilds everything from constants, loads the
    # claim only under its own retained identity and recomputes the decision.
    script = textwrap.dedent(
        """
        import json, sys
        import antecedent as ac
        from antecedent import decision, external

        ident = ac.identify(
            graph=[("x", "a"), ("x", "y"), ("a", "y")],
            names=["x", "a", "y"],
            query=ac.ResponseCurve("a", "y", grid=[0.0, 1.0, 2.0]),
        )
        spec = external.response(
            ident,
            outcome_units="mmHg",
            population="target",
            require_evidence=("factor:z",),
            require_assumptions=("ignorability",),
        )
        expected = json.loads(sys.argv[2])
        claim = spec.load(open(sys.argv[1], "rb").read(), expected=expected)
        wait_q, _, treat_q = claim.quantities
        utility = decision.x(0) * 2.0 - 1.0
        contract = decision.Contract(
            actions=(
                decision.Action("wait", inputs=(wait_q,), utility=utility),
                decision.Action("treat", inputs=(treat_q,), utility=utility),
            ),
            utility_units="utility",
            criterion=decision.Criterion.expected_utility(),
            target_population="target",
        )
        assert contract.identity == sys.argv[3]
        result = contract.evaluate(claim)
        assert result.selected == ("treat",)
        assert abs(result.actions[0].expected_utility - 1.0) < 1e-12
        assert abs(result.actions[1].expected_utility - 9.0) < 1e-12
        assert result.evpi is None
        assert "external_provider" in result.stages_behind()
        """
    )
    with tempfile.NamedTemporaryFile(suffix=".bin") as handle:
        handle.write(data)
        handle.flush()
        done = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                handle.name,
                json.dumps(claim.identity),
                contract.identity,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    assert done.returncode == 0, done.stderr

    # A claim loaded under a changed identity does not reach a decision.
    changed = {**claim.identity, "snapshot_id": "other-snapshot"}
    with pytest.raises(Exception, match="differs"):
        spec.load(data, expected=changed)


def test_a_mean_grid_refuses_what_only_a_distribution_can_answer():
    claim = _claim(_spec())
    base = _contract(claim)
    wait, _, treat = claim.quantities

    squared = decision.Action("treat", inputs=(treat,), utility=decision.x(0) * decision.x(0))
    nonlinear = decision.Contract(
        actions=(base.actions[0], squared),
        utility_units="utility",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
    )
    with pytest.raises(decision.DecisionRefusal) as nonlinear_refusal:
        nonlinear.evaluate(claim)
    assert nonlinear_refusal.value.reason_code == "decision_contract_unsatisfied"
    assert nonlinear_refusal.value.detail == "decision_evaluation.mean_source_insufficient"
    assert nonlinear_refusal.value.supplied == "mean"
    assert nonlinear_refusal.value.remedy is not None

    cap = decision.Constraint("cap", decision.x(0), bound=10.0, units="mmHg", min_probability=0.9)
    constrained = decision.Contract(
        actions=base.actions,
        utility_units="utility",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
        constraints=(cap,),
    )
    with pytest.raises(decision.DecisionRefusal) as constraint_refusal:
        constrained.evaluate(claim)
    assert constraint_refusal.value.detail == "decision_evaluation.mean_source_insufficient"

    # An outcome-law input cannot be read from a mean grid.
    outcome = decision.Contract(
        actions=(
            decision.Action(
                "wait",
                inputs=(_as_outcome(wait),),
                utility=decision.x(0),
            ),
            decision.Action("treat", inputs=(_as_outcome(treat),), utility=decision.x(0)),
        ),
        utility_units="utility",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
    )
    with pytest.raises(decision.DecisionRefusal) as quantity_refusal:
        outcome.evaluate(claim)
    assert quantity_refusal.value.reason_code == "quantity_semantics_mismatch"

    # A mean-sourced decision is not replayable, so it is never silently exported.
    result = base.evaluate(claim)
    with pytest.raises(decision.DecisionRefusal) as export_refusal:
        result.export()
    assert export_refusal.value.reason_code == "route_not_supported"
    assert export_refusal.value.detail == "decision_evaluation.mean_source_not_replayable"


def _as_outcome(quantity: ScientificQuantity) -> ScientificQuantity:
    return dataclasses.replace(quantity, functional_id="outcome")


# Not covered: the lifecycle's final step, "rank a study", because expected
# value of sample information and design ranking (TODO B0.2) do not exist yet.
