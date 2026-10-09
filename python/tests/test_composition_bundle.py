"""C3: the portable composed result built, exported and consumed from Python.

The decision fixture is the enumerated one of ``test_decision.py``: four equally
likely rows, ``risky = E[P * Q] = 2`` and ``safe = max(3, 0) = 3``. The point-only
fixture is the closed-form external claim ``E[Y | do(a)] = 1 + 2a`` on
``a = 0, 1, 2``: ``wait`` reads ``a = 0`` and is worth ``2 * 1 - 1 = 1``, ``treat``
reads ``a = 2`` and is worth ``2 * 5 - 1 = 9``. Rust verifies the same bundles in
``crates/antecedent-design/tests/composition_verifiers.rs``.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path

import antecedent as ac
import numpy as np
import pytest
from antecedent import composition_bundle as cb
from antecedent import decision, external
from antecedent.errors import CausalUnsupportedError
from antecedent.joint_distribution import (
    DistributionIdentity,
    JointDistributionArtifact,
)

from _refusal import assert_registered_refusal

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_cross_surface_fixtures as gen  # noqa: E402

IDENTITY_LEN = 64


# ------------------------------------------------------------------------ fixtures


def _law(columns: tuple[object, ...] = gen.COLUMNS) -> JointDistributionArtifact:
    identity = DistributionIdentity(
        semantic="interventional_predictive",
        quantities=columns,  # type: ignore[arg-type]
        alignment="joint",
        source_id="enumerated",
        provider_id="exact-law",
        rng_id="deterministic_exact",
        snapshot_id="enumeration-1",
        causal_contract_id="checked-contract",
    )
    draws = np.array([[a, b, 3.0] for a, b in zip(gen.P, gen.Q, strict=True)], dtype=np.float64)
    return JointDistributionArtifact(identity, draws, calibration="exact")


def _claim(
    *, evidence: str = "factor:z", request: str = "req-1", snapshot: str = "snap-9"
) -> external.BoundExternalClaim:
    query = ac.ResponseCurve("a", "y", grid=gen.GRID)
    ident = ac.identify(graph=gen.EDGES, names=gen.NAMES, query=query)
    spec = external.response(
        ident,
        outcome_units="mmHg",
        population="target",
        graph_id="graph-1",
        contract_id="checked-contract",
        require_evidence=(evidence,),
        require_assumptions=("ignorability",),
    )
    return spec.bind(
        gen.external_response(
            provider=gen.external_provider(request=request, snapshot=snapshot),
            evidence=(evidence,),
        )
    )


def _mean_contract(claim: external.BoundExternalClaim) -> decision.Contract:
    q = claim.quantities
    return decision.Contract(
        actions=(
            decision.Action("wait", inputs=(q[0],), utility=decision.x(0) * 2.0 - 1.0),
            decision.Action("treat", inputs=(q[2],), utility=decision.x(0) * 2.0 - 1.0),
        ),
        utility_units="utility",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
    )


def _joint_bytes(law: JointDistributionArtifact | None = None) -> dict[str, bytes]:
    contract = gen.decision_contract()
    law = law or gen.decision_source()
    result = contract.evaluate(law)
    return {
        "contract": contract.export(artifact_id="contract"),
        "law": law.export("law"),
        "result": result.export(artifact_id="result"),
    }


def _joint_builder(parts: dict[str, bytes], *, reverse: bool = False) -> cb.BundleBuilder:
    builder = cb.Bundle.builder()
    order = [
        ("contract", "decision_contract"),
        ("law", "distribution"),
        ("result", "decision_result"),
    ]
    for node_id, kind in reversed(order) if reverse else order:
        builder.add_artifact(kind, parts[node_id], node_id=node_id)
    return builder.connect("law", "result").connect("contract", "result")


def _joint_bundle(parts: dict[str, bytes] | None = None) -> cb.Bundle:
    return _joint_builder(parts or _joint_bytes()).build()


def _consume(bundle: cb.Bundle, supplied: cb.SuppliedSources | None = None) -> cb.ConsumedBundle:
    return cb.consume_bundle(bundle.export(), expected_identity=bundle.identity, supplied=supplied)


def _failed_stage(consumed: cb.ConsumedBundle, node_id: str) -> str:
    status = consumed.node(node_id).status
    assert isinstance(status, cb.Failed), status
    return status.stage


# ------------------------------------------------------------------------- tests


def test_c3_python_built_bundle_is_consumed_in_process_and_reports_the_decision() -> None:
    bundle = _joint_bundle()
    assert len(bundle.identity) == IDENTITY_LEN
    assert {n.id: n.kind for n in bundle.nodes} == {
        "contract": "decision_contract",
        "law": "distribution",
        "result": "decision_result",
    }
    assert all(n.embedded for n in bundle.nodes)
    assert {(e.source, e.target) for e in bundle.edges} == {
        ("law", "result"),
        ("contract", "result"),
    }

    consumed = _consume(bundle)
    consumed.require_verified()
    assert consumed.all_verified
    assert consumed.identity == bundle.identity
    assert consumed.value("result", "risky.expected_utility") == pytest.approx(2.0)
    assert consumed.value("result", "safe.expected_utility") == pytest.approx(3.0)
    assert consumed.claim_label == "joint_draw"
    assert consumed.node("law").facts["law"] == "joint_draw"
    assert consumed.node("result").facts["requires_law"] == "joint_draw"
    assert [e.upstream_digest for e in consumed.edges] == [e.upstream_digest for e in bundle.edges]

    # The node's identity is the one the artifact's own consumer recomputed.
    law = cb.describe_artifact("distribution", _joint_bytes()["law"])
    assert consumed.node("law").identity == law.identity
    assert cb.detect_kind(_joint_bytes()["result"]) == "decision_result"
    assert cb.detect_kind(b"not a container") is None


def test_c3_fresh_interpreter_consumes_the_exported_bundle(tmp_path: Path) -> None:
    bundle = _joint_bundle()
    path = tmp_path / "bundle.bin"
    path.write_bytes(bundle.export())
    script = textwrap.dedent(
        """
        import json, sys
        from antecedent import composition_bundle as cb

        data = open(sys.argv[1], "rb").read()
        consumed = cb.consume_bundle(data, expected_identity=sys.argv[2])
        consumed.require_verified()
        print(json.dumps({
            "verified": consumed.all_verified,
            "safe": consumed.value("result", "safe.expected_utility"),
            "risky": consumed.value("result", "risky.expected_utility"),
            "label": consumed.claim_label,
        }))
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", script, str(path), bundle.identity],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(done.stdout)
    assert report == {"verified": True, "safe": 3.0, "risky": 2.0, "label": "joint_draw"}

    # A fresh interpreter holding another retained identity refuses the same bytes.
    refused = subprocess.run(
        [sys.executable, "-c", script, str(path), "0" * IDENTITY_LEN],
        capture_output=True,
        text=True,
        check=False,
    )
    assert refused.returncode != 0
    assert "expected_identity_mismatch" in refused.stderr


def test_c3_order_invariance_and_deterministic_derived_ids() -> None:
    parts = _joint_bytes()
    forward = _joint_builder(parts).build()
    reverse = _joint_builder(parts, reverse=True).build()
    assert forward.identity == reverse.identity
    assert forward.export("same") == reverse.export("same")

    derived = cb.Bundle.builder().add_artifact("auto", parts["law"])
    again = cb.Bundle.builder().add_artifact("distribution", parts["law"])
    assert derived.last_node_id == again.last_node_id
    assert derived.last_node_id.startswith("distribution:")


def _wrong_identity() -> None:
    bundle = _joint_bundle()
    cb.consume_bundle(bundle.export(), expected_identity="0" * IDENTITY_LEN)


def _flipped_byte() -> None:
    bundle = _joint_bundle()
    data = bytearray(bundle.export())
    data[len(data) // 2] ^= 0xFF
    cb.consume_bundle(bytes(data), expected_identity=bundle.identity)


def _unknown_kind() -> None:
    cb.Bundle.builder().add_artifact("execution_or_fit", b"bytes")


def _wrong_artifact_kind() -> None:
    cb.Bundle.builder().add_artifact("decision_contract", _joint_bytes()["law"])


def _unknown_edge() -> None:
    _joint_builder(_joint_bytes()).connect("law", "missing")


def _cycle() -> None:
    _joint_builder(_joint_bytes()).connect("result", "law").build()


@pytest.mark.parametrize(
    ("case", "refusal", "stage", "code"),
    [
        (
            _wrong_identity,
            cb.ExpectedIdentityMismatchRefusal,
            "expected_identity_mismatch",
            "external_binding_mismatch",
        ),
        (_flipped_byte, cb.CompositionBundleRefusal, "invalid_container", "invalid_argument"),
        (_unknown_kind, cb.UnknownNodeKindRefusal, "unknown_node_kind", "invalid_argument"),
        (
            _wrong_artifact_kind,
            cb.SwappedEvidenceRefusal,
            "swapped_evidence",
            "external_binding_mismatch",
        ),
        (
            _unknown_edge,
            cb.EdgeDigestMismatchRefusal,
            "edge_digest_mismatch",
            "external_binding_mismatch",
        ),
        (
            _cycle,
            cb.EdgeDigestMismatchRefusal,
            "edge_digest_mismatch",
            "external_binding_mismatch",
        ),
    ],
)
def test_c3_container_and_graph_refusals_are_typed_with_registered_codes(
    case: Callable[[], None], refusal: type[cb.CompositionBundleRefusal], stage: str, code: str
) -> None:
    with pytest.raises(refusal) as caught:
        case()
    error = caught.value
    assert isinstance(error, CausalUnsupportedError)
    assert error.bundle_stage == stage
    assert error.detail == f"composition_bundle.{stage}"
    assert error.reason_code == code
    assert_registered_refusal(error)


def test_c3_tamper_table_names_the_failed_stage_on_the_dependent_node() -> None:
    parts = _joint_bytes()

    # A quantity swapped upstream: the result was computed from other coordinates.
    renamed = tuple(
        dataclasses.replace(q, variable_id="p2") if i == 0 else q for i, q in enumerate(gen.COLUMNS)
    )
    swapped = _consume(_joint_bundle({**parts, "law": _law(renamed).export("law")}))
    assert swapped.node("law").verified
    assert _failed_stage(swapped, "result") == "tampered_quantity"
    assert swapped.value("result", "safe.expected_utility") is None
    with pytest.raises(cb.TamperedQuantityRefusal) as tampered:
        swapped.require_verified()
    assert tampered.value.offending == "result"
    assert tampered.value.reason_code == "quantity_semantics_mismatch"

    # Another decision contract upstream of the result.
    other = decision.Contract(
        actions=(
            decision.Action("only", inputs=(gen.COLUMNS[2],), utility=decision.x(0)),
            decision.Action("other", inputs=(gen.COLUMNS[2],), utility=decision.x(0) + 1.0),
        ),
        utility_units="units",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
    )
    mismatched = _consume(
        _joint_bundle({**parts, "contract": other.export(artifact_id="contract")})
    )
    assert _failed_stage(mismatched, "result") == "graph_or_snapshot_mismatch"

    # A result with its source left out refuses to stand on nothing.
    alone = cb.Bundle.builder()
    alone.add_artifact("decision_contract", parts["contract"], node_id="contract")
    alone.add_artifact("decision_result", parts["result"], node_id="result")
    alone.connect("contract", "result")
    assert _failed_stage(_consume(alone.build()), "result") == "tampered_quantity"


def test_c3_joint_requirement_over_a_mean_only_claim_is_unsupported_law() -> None:
    parts = _joint_bytes()
    claim = _claim(snapshot="enumeration-1")
    builder = _joint_builder(parts).add_artifact("external_claim", claim, node_id="claim")
    consumed = _consume(builder.connect("claim", "result").build())
    assert consumed.node("claim").facts["law"] == "mean_only"
    assert _failed_stage(consumed, "result") == "unsupported_law"
    assert consumed.node("result").claim_label is None
    with pytest.raises(cb.UnsupportedLawRefusal) as refused:
        consumed.require_verified()
    assert refused.value.reason_code == "joint_law_required"
    assert refused.value.detail == "composition_bundle.unsupported_law"


def test_c3_external_mean_decision_is_exportable_as_a_point_only_attested_result() -> None:
    claim = _claim()
    contract = _mean_contract(claim)
    result = cb.mean_decision(contract, claim)
    assert cb.detect_kind(result) == "decision_result"

    builder = cb.Bundle.builder()
    builder.add_artifact("external_claim", claim, node_id="claim")
    builder.add_artifact(
        "decision_contract", contract.export(artifact_id="contract"), node_id="contract"
    )
    builder.add_artifact("decision_result", result, node_id="result")
    bundle = builder.connect("claim", "result").connect("contract", "result").build()
    consumed = _consume(bundle)
    consumed.require_verified()
    assert consumed.claim_label == "point_only_attested"
    assert consumed.node("result").claim_label == "point_only_attested"
    assert consumed.value("result", "wait.expected_utility") == pytest.approx(1.0)
    assert consumed.value("result", "treat.expected_utility") == pytest.approx(9.0)
    assert consumed.node("claim").facts["law"] == "mean_only"
    assert consumed.node("claim").facts["trust"] == "externally_attested"
    assert "requires_law" not in consumed.node("contract").facts

    # The decision is the Rust evaluation of the same means, not a stored claim.
    assert contract.evaluate(claim).selected == ("treat",)

    # The same means under another request still carry the stored decision: it rests on
    # the digest of the means, and the request is a separate, checked fact.
    changed = _claim(request="req-2")
    changed_bundle = (
        cb.Bundle.builder()
        .add_artifact("external_claim", changed, node_id="claim")
        .add_artifact(
            "decision_contract", contract.export(artifact_id="contract"), node_id="contract"
        )
        .add_artifact("decision_result", result, node_id="result")
        .connect("claim", "result")
        .connect("contract", "result")
        .build()
    )
    assert _consume(changed_bundle).node("result").verified

    # A functional a mean cannot answer is refused when the result is made.
    squared = dataclasses.replace(
        contract,
        actions=(
            decision.Action(
                "treat", inputs=(claim.quantities[2],), utility=decision.x(0) * decision.x(0)
            ),
            contract.actions[0],
        ),
    )
    with pytest.raises(cb.UnsupportedLawRefusal) as insufficient:
        cb.mean_decision(squared, claim)
    assert_registered_refusal(insufficient.value)


def test_c3_swapped_evidence_fails_the_declared_relationship() -> None:
    first = _claim(request="req-a", evidence="factor:z")

    def build(second: external.BoundExternalClaim) -> cb.ConsumedBundle:
        builder = cb.Bundle.builder()
        builder.add_artifact("external_claim", first, node_id="a")
        builder.add_artifact("external_claim", second, node_id="b")
        builder.relate("b", "a", "independent")
        assert builder.last_node_id == "relation:a:b"
        return _consume(builder.build())

    build(_claim(request="req-b", evidence="factor:w")).require_verified()
    overlapping = build(_claim(request="req-b", evidence="factor:z"))
    assert _failed_stage(overlapping, "relation:a:b") == "swapped_evidence"
    with pytest.raises(cb.SwappedEvidenceRefusal):
        overlapping.require_verified()


def test_c3_reference_nodes_are_unresolved_then_resolved_and_keep_the_inspected_result() -> None:
    parts = _joint_bytes()
    builder = _joint_builder(parts)
    builder.add_reference(
        "causal",
        "causal_contract",
        identity="program-1",
        requires=cb.DataRequirement("enumeration-1", "snapshot-digest"),
        facts={"graph_or_snapshot": "enumeration-1|checked-contract"},
        inspected={"identified_effect": 2.0},
    )
    bundle = builder.connect("causal", "law").build()
    assert {n.id: n.embedded for n in bundle.nodes}["causal"] is False

    without = _consume(bundle)
    status = without.node("causal").status
    assert isinstance(status, cb.ReferenceUnresolved)
    assert status.requires == cb.DataRequirement("enumeration-1", "snapshot-digest")
    assert not without.all_verified
    assert without.node("causal").inspected == {"identified_effect": 2.0}
    assert without.value("causal", "identified_effect") is None
    # The inspected decision survives the unavailable reference.
    assert without.node("result").verified
    assert without.value("result", "safe.expected_utility") == pytest.approx(3.0)
    with pytest.raises(cb.CallbackUnavailableRefusal) as unavailable:
        without.require_verified()
    assert unavailable.value.reason_code == "external_capability_missing"
    assert unavailable.value.offending == "causal"

    supplied = cb.SuppliedSources().with_data("enumeration-1", "snapshot-digest")
    resolved = _consume(bundle, supplied)
    resolved.require_verified()
    assert isinstance(resolved.node("causal").status, cb.Verified)

    wrong = _consume(bundle, cb.SuppliedSources().with_data("enumeration-1", "another-digest"))
    assert _failed_stage(wrong, "causal") == "graph_or_snapshot_mismatch"


def test_c3_changed_provider_request_fails_the_supplied_reference() -> None:
    builder = cb.Bundle.builder()
    builder.add_reference(
        "provider",
        "external_claim",
        identity="claim-id",
        requires=cb.ProviderRequirement("lab", "snap-9", "fp-1"),
        facts={"law": "mean_only"},
    )
    bundle = builder.build()
    unresolved = _consume(bundle)
    assert isinstance(unresolved.node("provider").status, cb.ReferenceUnresolved)
    exact = _consume(bundle, cb.SuppliedSources().with_provider("lab", "snap-9", "fp-1"))
    exact.require_verified()
    changed = _consume(bundle, cb.SuppliedSources().with_provider("lab", "snap-9", "fp-CHANGED"))
    assert _failed_stage(changed, "provider") == "provider_request_changed"
    with pytest.raises(cb.ProviderRequestChangedRefusal):
        changed.require_verified()


def test_c3_describe_reports_facts_read_from_the_decoded_artifact() -> None:
    claim = _claim()
    described = cb.describe_artifact("external_claim", claim)
    assert described.kind == "external_claim"
    assert described.facts["law"] == "mean_only"
    assert described.facts["trust"] == "externally_attested"
    assert described.facts["request_fingerprint"] == "req-1"
    assert described.facts["graph_or_snapshot"] == "snap-9|checked-contract"
    assert described.values["value.2"] == pytest.approx(5.0)
    with pytest.raises(cb.CompositionBundleRefusal):
        cb.describe_artifact("distribution", claim)
    with pytest.raises(CausalUnsupportedError):
        cb.describe_artifact("decision_contract", b"junk")
