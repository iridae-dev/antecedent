"""C3: composition bundles built on one language surface and consumed on the other.

``py_composition_bundle.*`` is written by ``generate_cross_surface_bundle_fixtures.py``
and consumed by ``crates/antecedent-design/tests/cross_surface_bundle.rs``;
``rust_composition_bundle.bin`` is written by that file's ignored
``regenerate_rust_fixtures`` and consumed here. A missing Rust-built fixture fails
loudly; the comparison against the committed Python-built identity skips with the
generation command while that fixture is absent.

The expected bundle identity is built in-process from the shared constants of the
generator (two enumerated decisions, values derived by hand there), never read back
from the Rust bytes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from antecedent import composition_bundle as cb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_cross_surface_bundle_fixtures as bundle_gen  # noqa: E402

FIXTURES = bundle_gen.FIXTURE_DIR


def _flip_middle_byte(data: bytes) -> bytes:
    middle = len(data) // 2
    return data[:middle] + bytes([data[middle] ^ 0xFF]) + data[middle + 1 :]


def _assert_hand_derived_truth(consumed: cb.ConsumedBundle) -> None:
    consumed.require_verified()
    assert consumed.all_verified
    for (node, key), want in bundle_gen.EXPECTED_VALUES.items():
        assert consumed.value(node, key) == pytest.approx(want, abs=1e-9)
    assert consumed.node("result").claim_label == "joint_draw"
    assert consumed.node("mean_result").claim_label == "point_only_attested"
    # A mixed bundle is labelled by its weakest decision.
    assert consumed.claim_label == "point_only_attested"
    assert consumed.node("law").facts["law"] == "joint_draw"
    assert consumed.node("claim").facts["law"] == "mean_only"
    assert consumed.node("claim").facts["trust"] == "externally_attested"
    assert consumed.node("result").facts["requires_law"] == "joint_draw"
    assert "requires_law" not in consumed.node("mean_contract").facts
    assert {n.id: n.kind for n in consumed.nodes} == bundle_gen.NODE_KINDS
    assert {(e.source, e.target) for e in consumed.edges} == set(bundle_gen.EDGES)


def _rust_bundle_bytes() -> bytes:
    path = FIXTURES / bundle_gen.RUST_BUNDLE
    if not path.is_file():
        pytest.fail(
            f"missing cross-surface fixture {path}; generate it with: {bundle_gen.RUST_COMMAND}"
        )
    return path.read_bytes()


def test_c3_xsurface_rust_built_bundle_verifies_under_the_python_declared_identity() -> None:
    expected = bundle_gen.build_bundle()
    consumed = cb.consume_bundle(_rust_bundle_bytes(), expected_identity=expected.identity)
    assert consumed.identity == expected.identity
    _assert_hand_derived_truth(consumed)
    # The Merkle digests of every node agree with the Python-assembled twin.
    assert {n.id: n.chain_digest for n in consumed.nodes} == {
        n.id: n.chain_digest for n in expected.nodes
    }
    assert [e.upstream_digest for e in consumed.edges] == [
        e.upstream_digest for e in expected.edges
    ]


def test_c3_xsurface_rust_built_bundle_refuses_another_identity_and_changed_bytes() -> None:
    data = _rust_bundle_bytes()
    identity = bundle_gen.build_bundle().identity
    with pytest.raises(cb.ExpectedIdentityMismatchRefusal) as wrong:
        cb.consume_bundle(data, expected_identity="0" * 64)
    assert wrong.value.detail == "composition_bundle.expected_identity_mismatch"
    with pytest.raises(cb.CompositionBundleRefusal):
        cb.consume_bundle(_flip_middle_byte(data), expected_identity=identity)


def test_c3_xsurface_python_bundle_identity_equals_the_committed_fixture_constant() -> None:
    path = FIXTURES / bundle_gen.PY_IDENTITY
    if not path.is_file():
        pytest.skip(f"missing {path}; generate it with: {bundle_gen.PY_COMMAND}")
    committed = json.loads(path.read_text(encoding="utf-8"))
    bundle = bundle_gen.build_bundle()
    assert bundle.identity == committed["bundle_identity"]
    assert bundle_gen.identity_document(bundle) == path.read_bytes()
    bin_path = FIXTURES / bundle_gen.PY_BUNDLE
    if not bin_path.is_file():
        pytest.skip(f"missing {bin_path}; generate it with: {bundle_gen.PY_COMMAND}")
    # The committed bytes are consumed under the in-process identity.
    consumed = cb.consume_bundle(bin_path.read_bytes(), expected_identity=bundle.identity)
    _assert_hand_derived_truth(consumed)


def test_c3_xsurface_python_bundle_is_deterministic_and_order_invariant() -> None:
    parts = bundle_gen.bundle_parts()
    forward = bundle_gen.build_bundle()
    reverse = cb.Bundle.builder()
    for node_id, kind in reversed(list(bundle_gen.NODE_KINDS.items())):
        reverse.add_artifact(kind, parts[node_id], node_id=node_id)
    for upstream, dependent in reversed(bundle_gen.EDGES):
        reverse.connect(upstream, dependent)
    assert reverse.build().identity == forward.identity
    assert len(forward.identity) == 64  # noqa: PLR2004
