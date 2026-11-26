"""The versioned ledger reader and selected-pool composition on both sides of a revision."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from antecedent import _native, decision
from antecedent import composition_bundle as cb
from antecedent import evidence_ledger as el
from antecedent.joint_distribution import ScientificQuantity
from antecedent.outcomes import require_success

LEDGER = (
    Path(__file__).resolve().parents[2]
    / "spec"
    / "conformance"
    / "evidence_ledger__pooled.artifact"
)
LEDGER_ID = "b36d763a3eb209258f2c4fe925b31cbbf0dc65f028aa0a58b75cf2173e19c3c2"
POINT = "8f93fa7182ab27a50a9b0988634e7289d3c0a778ab67288ef681a207929060ca"
RANGE = "b3ea42845c01939f6437498e593b572288ffd6177e6832566a52b3478ea6dde5"
QUANTITY = ScientificQuantity(
    "pooled_effect",
    "pooled_effect",
    "outcome",
    "unit",
    "target",
    "contrast:treated-control",
    0,
    "mean_difference",
)


def _contract() -> decision.Contract:
    return decision.Contract(
        actions=(
            decision.Action("treat", inputs=(QUANTITY,), utility=decision.x(0) - 1.25),
            decision.Action("wait", inputs=(QUANTITY,), utility=decision.x(0) * 0.0),
        ),
        utility_units="utility",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
    )


def _bundle(ledger: object, **selections: cb.LedgerPooledResult) -> cb.Bundle:
    contract = _contract()
    builder = (
        cb.Bundle.builder()
        .add_artifact(ledger, node_id="ledger")
        .add_artifact(contract.export(artifact_id="contract"), node_id="contract")
    )
    for terminal, selection in selections.items():
        selected = f"{terminal}-selection"
        builder.add_artifact(selection, node_id=selected)
        builder.add_artifact(
            cb.ledger_pooled_decision(contract, selection, artifact_id="decision"),
            node_id=terminal,
        )
        builder.connect("ledger", selected).connect(selected, terminal).connect(
            "contract", terminal
        )
    return builder.build()


def _consume(bundle: cb.Bundle) -> cb.ConsumedBundle:
    return require_success(
        cb.consume_bundle(bytes(bundle.export()), expected_identity=bundle.identity)
    )


def test_ledger_reference_verifies_supplied_bytes_and_refuses_edited_history() -> None:
    ledger = LEDGER.read_bytes()
    assert _native.producer_artifact_identity("ledger", ledger) == LEDGER_ID
    bundle = cb.Bundle.builder().add_producer_reference("history", "ledger", ledger).build()
    exported = bytes(bundle.export())

    def consume(producers: dict[str, bytes]) -> cb.ConsumedNode:
        result = require_success(
            cb.consume_bundle(exported, expected_identity=bundle.identity, producers=producers)
        )
        return result.node("history")

    assert isinstance(consume({}).status, cb.ReferenceUnresolved)
    node = consume({"ledger": ledger})
    assert node.verified and not node.replayed
    assert node.identity == LEDGER_ID
    edited = LEDGER.with_name("evidence_ledger__head_edited.artifact").read_bytes()
    failed = consume({"ledger": edited})
    assert isinstance(failed.status, cb.Failed)
    assert failed.status.stage == "swapped_evidence"


def test_inherited_point_ledger_bundle_recomputes_in_a_fresh_process(tmp_path: Path) -> None:
    ledger = LEDGER.read_bytes()
    assert cb.detect_kind(ledger) == "evidence_ledger"
    selected = require_success(
        cb.select_ledger_pooled_result(
            ledger, expected_ledger=LEDGER_ID, result=POINT, quantity=QUANTITY
        )
    )
    assert selected.estimate == pytest.approx(10.0 / 7.0)
    assert selected.quantity == QUANTITY
    assert selected.method == "fixed_effect_inverse_variance"
    bundle = _bundle(ledger, decision=selected)
    consumed = _consume(bundle)
    assert consumed.all_replayed
    assert consumed.value("decision", "treat.expected_utility") == pytest.approx(5.0 / 28.0)
    plan = require_success(
        cb.compile_ledger_pooled_decision_bundle_recomputation(bundle, "decision")
    )
    run = require_success(plan.execute())
    path = tmp_path / "ledger-decision.bin"
    path.write_bytes(bytes(run.export()))
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json, pathlib, sys
from antecedent import composition_bundle as cb
from antecedent.outcomes import require_success
receipt = require_success(cb.consume_bundle_recomputation(
    pathlib.Path(sys.argv[1]).read_bytes(), route="ledger_pooled_decision",
    expected_bundle=sys.argv[2], expected_program=sys.argv[3],
))
print(json.dumps([receipt.replayed, receipt.decision_fingerprint]))
""",
            str(path),
            bundle.identity,
            plan.program_identity,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(process.stdout) == [True, run.decision_fingerprint]
    with pytest.raises(cb.UnsupportedLawRefusal):
        require_success(
            cb.select_ledger_pooled_result(
                ledger, expected_ledger=LEDGER_ID, result=RANGE, quantity=QUANTITY
            )
        )


def _study(label: str, *observations: int) -> el.SourceDeclaration:
    return el.SourceDeclaration(
        label, "observation_set", observations=el.rows("ledger-pool", observations)
    )


def test_revised_pool_refuses_old_decision_and_reuses_the_unrelated_branch() -> None:
    ledger = el.EvidenceLedger()
    a, b, c, d = [ledger.add_source(_study(str(i), i)) for i in range(4)]
    point = require_success(
        ledger.pool_summaries("old", [el.StudySummary(a, 1.0, 1.0), el.StudySummary(b, 3.0, 4.0)])
    )
    other = require_success(
        ledger.pool_summaries("other", [el.StudySummary(c, 2.0, 1.0), el.StudySummary(d, 4.0, 1.0)])
    )

    def select(artifact, result):
        return require_success(
            cb.select_ledger_pooled_result(
                artifact,
                expected_ledger=artifact.expectation.identity,
                result=result.result,
                quantity=QUANTITY,
            )
        )

    original = ledger.export()
    first, independent = select(original, point), select(original, other)
    before = _bundle(original, decision=first, other=independent)
    assert _consume(before).value("decision", "treat.expected_utility") == pytest.approx(0.15)
    revised = require_success(
        ledger.revise_source(el.SourceRevision.update(a, _study("updated", 0, 4)))
    )
    assert revised.created_source is not None
    fresh = require_success(
        ledger.pool_summaries(
            "replacement",
            [el.StudySummary(revised.created_source, 0.0, 1.0), el.StudySummary(b, 3.0, 4.0)],
            replaces=point.result,
        )
    )
    updated = ledger.export()
    old_decision = _bundle(updated, decision=first, other=independent)
    consumed = _consume(old_decision)
    assert isinstance(consumed.node("decision-selection").status, cb.Failed)
    assert consumed.node("other").replayed
    valid_plan = require_success(
        cb.compile_ledger_pooled_decision_bundle_recomputation(old_decision, "other")
    )
    assert len(require_success(valid_plan.execute()).dependencies) == 4
    assert select(updated, other).identity == independent.identity
    after = _bundle(updated, decision=select(updated, fresh), other=independent)
    assert _consume(after).value("decision", "treat.expected_utility") == pytest.approx(-0.65)
    assert _consume(after).node("other").identity == _consume(before).node("other").identity

    withdrawal = require_success(el.SourceRevision.withdraw(b, "consent withdrawn"))
    report = require_success(ledger.revise_source(withdrawal))
    assert report.revalidation[fresh.result] == "withdrawn"
    withdrawn = _bundle(ledger.export(), decision=select(updated, fresh), other=independent)
    replay = _consume(withdrawn)
    assert isinstance(replay.node("decision-selection").status, cb.Failed)
    assert isinstance(replay.node("decision").status, cb.Failed)
    assert replay.node("other").replayed
    assert replay.value("other", "treat.expected_utility") == pytest.approx(1.75)
