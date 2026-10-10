"""Artifacts built on one language surface and consumed on the other.

Fixtures live in ``conformance/cross_surface/``. Files named ``rust_*`` are
written by the ignored Rust tests (``regenerate_rust_fixtures``); files named
``py_*`` are written by ``generate_cross_surface_fixtures.py`` and are consumed
by the Rust tests. A missing fixture fails loudly: it is never skipped.

Every Rust-built artifact below is loaded under an identity the consumer holds
independently of the bytes. The expected claim identity is obtained by binding
the same closed-form Response in Python (``claim.identity``) rather than from the
``blake3`` package, which is not a required dependency; Rust holds the
constants-built twin of that identity in ``cross_surface_external_claim.rs``.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest
from antecedent import decision
from antecedent.errors import CausalSerializationError, CausalUnsupportedError
from antecedent.extensibility import ProviderTrust
from antecedent.joint_distribution import JointDistributionArtifact

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_cross_surface_fixtures as gen  # noqa: E402

# A corrupt or mismatched artifact is refused as a serialization error or a coded refusal.
REFUSED = (CausalSerializationError, CausalUnsupportedError)

RUST_CMD = (
    "ANTECEDENT_WRITE_FIXTURES=1 cargo test -p antecedent-io --test cross_surface_external_claim "
    "-- --ignored regenerate_rust_fixtures && "
    "ANTECEDENT_WRITE_FIXTURES=1 cargo test -p antecedent-design --test cross_surface_decision "
    "-- --ignored regenerate_rust_fixtures"
)
PY_CMD = "python python/tests/generate_cross_surface_fixtures.py"
SUPPORT_LABELS = {
    "supported",
    "weak_overlap",
    "extrapolative",
    "outside_empirical_support",
    "missing_evidence",
}


def _read(name: str) -> bytes:
    path = gen.FIXTURE_DIR / name
    if not path.is_file():
        command = RUST_CMD if name.startswith("rust_") else PY_CMD
        pytest.fail(f"missing cross-surface fixture {path}; generate it with: {command}")
    return path.read_bytes()


def _flip_last_byte(data: bytes) -> bytes:
    return data[:-1] + bytes([data[-1] ^ 0xFF])


# --- Rust builds, Python consumes -------------------------------------------------


def test_rust_external_claim_loads_under_the_python_declared_identity():
    spec = gen.external_spec()
    expected = gen.external_claim().identity_fields
    data = _read("rust_external_claim.bin")

    claim = spec.load(data, expected_identity=expected)
    assert np.allclose(claim.values, gen.EXTERNAL_VALUES, atol=1e-12, rtol=0)
    assert claim.native is False
    assert claim.trust is ProviderTrust.EXTERNALLY_ATTESTED
    assert claim.support == gen.SUPPORT
    assert claim.support_status == "outside_empirical_support"
    assert claim.provenance_label == "external:lab/curve@v3#snap-9"
    assert claim.identity_fields == expected
    assert {"causal_contract", "evidence", "external_provider"} <= claim.stages_behind()
    assert [link.id for link in claim.lineage][-1] == "claim"
    # Independent constants, not read back from the artifact.
    assert (claim.identity_fields["graph_id"], claim.identity_fields["causal_contract_id"]) == (
        "graph-1",
        "checked-contract",
    )
    assert claim.identity_fields["evidence_ids"] == ["factor:z"]
    assert claim.identity_fields["assumption_ids"] == ["ignorability"]
    assert [q.regime_id for q in claim.quantities] == ["do(a=0)", "do(a=1)", "do(a=2)"]

    fields = ("provider_id", "object_id", "version_id", "snapshot_id", "request_id")
    assert [claim.identity_fields[f] for f in fields] == ["lab", "curve", "v3", "snap-9", "req-1"]


def test_rust_external_claim_refuses_a_different_identity_or_changed_bytes():
    spec = gen.external_spec()
    expected = gen.external_claim().identity_fields
    data = _read("rust_external_claim.bin")

    with pytest.raises(Exception, match="differs|verification receipt"):
        spec.load(data, expected_identity={**expected, "snapshot_id": "other-snapshot"})
    with pytest.raises(Exception, match="differs|verification receipt"):
        spec.load(data, expected_identity={**expected, "trust": "verified_extension"})
    with pytest.raises(REFUSED):
        spec.load(_flip_last_byte(data), expected_identity=expected)
    with pytest.raises(REFUSED):
        spec.load(data[:-5], expected_identity=expected)


def test_rust_decision_contract_loads_under_the_python_declared_identity():
    contract = gen.decision_contract()
    data = _read("rust_decision_contract.bin")

    loaded = decision.Contract.load(data, expected_identity=contract.identity)
    assert loaded.identity == contract.identity
    assert [a.id for a in loaded.actions] == ["risky", "safe"]
    assert loaded.criterion == decision.Criterion.expected_utility()
    assert loaded.constraints[0].id == "q-cap"
    assert loaded.constraints[0].min_probability == 0.75

    with pytest.raises(Exception, match="differs|verification receipt"):
        decision.Contract.load(data, expected_identity="0" * 64)
    with pytest.raises(REFUSED):
        decision.Contract.load(_flip_last_byte(data), expected_identity=contract.identity)


def test_rust_source_and_result_replay_under_python_declarations():
    contract = gen.decision_contract()
    source_bytes = _read("rust_decision_source.bin")
    result_bytes = _read("rust_decision_result.bin")

    source = JointDistributionArtifact.load(source_bytes, expected_identity=gen.source_identity())
    assert source.shape == (4, 3)
    assert source.calibration == "exact"
    assert np.asarray(source)[:, 0].tolist() == gen.P
    assert np.asarray(source)[:, 1].tolist() == gen.Q
    # The Rust-built source holds the same aligned rows as the Python-built one.
    assert decision.source_digest(source) == decision.source_digest(gen.decision_source())

    replayed = decision.replay(result_bytes, contract=contract, source=source)
    assert replayed.selected == ("safe",)
    assert replayed.verdict == decision.Verdict("uniquely_optimal", ("safe",))
    assert replayed.evpi == pytest.approx(0.5, abs=1e-12)
    risky, safe = replayed.actions
    assert risky.expected_utility == pytest.approx(2.0, abs=1e-12)
    assert safe.expected_utility == pytest.approx(3.0, abs=1e-12)
    assert replayed.contract_identity == contract.identity
    assert replayed.source_digest == decision.source_digest(source)

    with pytest.raises(REFUSED):
        JointDistributionArtifact.load(
            _flip_last_byte(source_bytes), expected_identity=gen.source_identity()
        )
    with pytest.raises(REFUSED):
        decision.replay(_flip_last_byte(result_bytes), contract=contract, source=source)
    other = dataclasses.replace(contract, criterion=decision.Criterion.expected_loss())
    with pytest.raises(decision.DecisionRefusal) as wrong:
        decision.replay(result_bytes, contract=other, source=source)
    assert wrong.value.reason_code == "decision_contract_unsatisfied"


# --- Python-built fixtures stay loadable on the Python surface --------------------


def test_committed_python_fixtures_still_load_and_match_their_identities():
    """Guards a stale ``py_*`` fixture before Rust is asked to consume it."""
    import json

    claim_identity = json.loads(_read("py_external_claim.identity.json"))
    assert claim_identity == gen.external_claim().identity_fields
    claim = gen.external_spec().load(
        _read("py_external_claim.bin"), expected_identity=claim_identity
    )
    assert np.allclose(claim.values, gen.EXTERNAL_VALUES, atol=1e-12, rtol=0)

    contract = gen.decision_contract()
    source = JointDistributionArtifact.load(
        _read("py_decision_source.bin"), expected_identity=gen.source_identity()
    )
    decision.Contract.load(_read("py_decision_contract.bin"), expected_identity=contract.identity)
    replayed = decision.replay(_read("py_decision_result.bin"), contract=contract, source=source)
    assert replayed.selected == ("safe",)
    identities = json.loads(_read("py_decision.identities.json"))
    assert identities["contract_identity"] == contract.identity
    assert identities["source_digest"] == decision.source_digest(source)


# --- Shared vocabulary ------------------------------------------------------------


def test_trust_and_support_vocabulary_matches_the_rust_wire_names():
    attested = gen.external_claim()
    assert attested.identity_fields["trust"] == "externally_attested"
    assert attested.identity_fields["verification"] is None
    assert attested.trust.value == "externally_attested"
    assert set(attested.identity_fields["point_status"]) <= SUPPORT_LABELS
    assert attested.identity_fields["point_status"] == list(gen.SUPPORT)
    assert attested.identity_fields["provider_meaning"] == "interventional_predictive"
    assert attested.identity_fields["identification"] == "nonparametrically_identified"

    undeclared = gen.external_spec().bind(gen.external_response(support=None))
    assert undeclared.identity_fields["point_status"] == ["missing_evidence"] * 3
    assert undeclared.support_status == "missing_evidence"

    probes = tuple(
        gen.external.VerificationProbe(kind, 1.0, 1.0, 0.0)
        for kind in ("shape", "support", "moments", "known_truth")
    )
    verified = gen.external_spec().bind(gen.external_response(attested_by=None, probes=probes))
    assert verified.identity_fields["trust"] == "verified_extension"
    assert verified.trust is ProviderTrust.VERIFIED_EXTENSION
    assert [p["kind"] for p in verified.identity_fields["verification"]] == [
        "shape",
        "support",
        "moments",
        "known_truth",
    ]


def test_refusal_details_match_the_strings_the_rust_tests_assert():
    spec = gen.external_spec()
    wrong_units = tuple(
        dataclasses.replace(q, units="kPa") if i == 1 else q for i, q in enumerate(spec.quantities)
    )
    with pytest.raises(CausalUnsupportedError) as units:
        spec.bind(gen.external_response(quantities=wrong_units))
    assert isinstance(units.value, gen.external.ExternalRefusal)
    assert units.value.reason_code == "quantity_semantics_mismatch"
    assert units.value.detail == "external_response_binding.coordinate_units"
    assert (units.value.offending, units.value.expected, units.value.supplied) == (
        "coordinate[1]",
        "mmHg",
        "kPa",
    )
    assert units.value.stage == "bind"

    marginals = dataclasses.replace(gen.source_identity(), alignment="independent_marginals")
    draws = np.array([[a, b, 3.0] for a, b in zip(gen.P, gen.Q, strict=True)], dtype=np.float64)
    independent = JointDistributionArtifact(marginals, draws, calibration="exact")
    with pytest.raises(CausalUnsupportedError) as joint:
        gen.decision_contract().evaluate(independent)
    assert isinstance(joint.value, decision.DecisionRefusal)
    assert joint.value.reason_code == "joint_law_required"
    assert joint.value.detail == "decision_evaluation.joint_law_required"
    assert joint.value.stage == "evaluate"
    assert joint.value.offending == "risky"
    assert joint.value.expected == "joint"
    assert joint.value.supplied == "independent_marginals"
    assert joint.value.remedy is not None


# ---- F21: the two-coordinate aligned law, built on each surface and read on the other ----


def test_f21_rust_built_joint_law_reports_the_enumerated_truth():
    """Means (1/2, 1), covariance 1/2 and E[XY] = 1, under an identity built from constants."""
    expected = gen.joint_law_identity()
    loaded = JointDistributionArtifact.load(_read("rust_joint_law.bin"), expected_identity=expected)
    assert loaded.mean(0) == pytest.approx(0.5, abs=1e-12)
    assert loaded.mean(1) == pytest.approx(1.0, abs=1e-12)
    assert loaded.covariance(0, 1) == pytest.approx(0.5, abs=1e-12)
    assert loaded.joint_product_expectation(0, 1) == pytest.approx(1.0, abs=1e-12)
    assert loaded.identity == expected
    assert loaded.trust == "unverified"
    assert loaded.calibration == "exact"
    # The Python twin carries byte-for-byte the same draws and identity.
    assert np.array_equal(np.asarray(loaded), np.asarray(gen.joint_law()))


def test_f21_changed_semantic_id_wrong_meaning_and_marginals_refuse_on_this_surface_too():
    data = _read("rust_joint_law.bin")
    for changed in (
        gen.joint_law_identity(second="z"),
        gen.joint_law_identity(semantic="bootstrap"),
        gen.joint_law_identity(semantic="causal_functional_posterior"),
    ):
        with pytest.raises(REFUSED, match="identity_expected"):
            JointDistributionArtifact.load(data, expected_identity=changed)
    marginals = JointDistributionArtifact(
        gen.joint_law_identity(alignment="independent_marginals"),
        np.array([[0.0, 0.0], [1.0, 2.0]], dtype=np.float64),
        calibration="exact",
    )
    with pytest.raises(REFUSED, match="aligned_joint_draws.marginals_not_joint"):
        marginals.covariance(0, 1)


def test_f21_both_directions_use_one_wire_format():
    python_bytes = _read("py_joint_law.bin")
    rust_bytes = _read("rust_joint_law.bin")
    expected = gen.joint_law_identity()
    from_python = JointDistributionArtifact.load(python_bytes, expected_identity=expected)
    from_rust = JointDistributionArtifact.load(rust_bytes, expected_identity=expected)
    assert np.array_equal(np.asarray(from_python), np.asarray(from_rust))
    assert from_python.identity == from_rust.identity
    with pytest.raises(REFUSED):
        JointDistributionArtifact.load(_flip_last_byte(rust_bytes), expected_identity=expected)
