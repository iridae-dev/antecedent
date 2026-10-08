"""2.3 A exit gate, boxes 3 and 4: the aligned joint draw and the external finite response.

Both boxes read ONLY the committed cross-surface fixtures in ``conformance/cross_surface/``
(``py_*`` built by ``generate_cross_surface_fixtures.py``, ``rust_*`` built by the ignored Rust
regeneration tests) and public modules. A missing fixture fails loudly. The Rust twins are
``crates/antecedent-io/tests/a_exit_cross_surface.rs`` (``a_exit_joint_*``, ``a_exit_external_*``)
and the existing ``cross_surface_*.rs`` tests; this file composes ``test_cross_surface.py``,
``test_joint_distribution_artifact.py`` and ``test_external_binding.py`` into one gate-shaped test
per box.

Hand-derived joint truth. Two equally weighted aligned draws (0, 0) and (1, 2): means (1/2, 1),
``E[XY] = (0 + 2) / 2 = 1`` and ``Cov = 1 - 1/2 * 1 = 1/2``. Pairing the marginals independently
gives the four pairs {0, 1} x {0, 2} and ``E[XY] = (0 + 0 + 0 + 2) / 4 = 1/2``: a different
value, which is why independently paired marginals are refused.

Hand-derived nonlinear utility (the decision fixture). Four equally likely rows
``p = (1, 3, 2, 0)``, ``q = (4, 0, 2, 6)``: ``risky = E[p q] = (4 + 0 + 4 + 0) / 4 = 2``,
``safe = 3``, ``E[max(risky, safe)] = (4 + 3 + 4 + 3) / 4 = 7/2`` so ``EVPI = 7/2 - 3 = 1/2``;
``safe`` is uniquely optimal. The constraint ``P(q <= 5) = 3/4 >= 3/4`` holds.

Hand-derived external truth: ``E[Y | do(a)] = 1 + 2a`` on ``a = 0, 1, 2`` is ``(1, 3, 5)``, the
third coordinate declared outside empirical support.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from antecedent import decision, external
from antecedent.errors import CausalSerializationError, CausalUnsupportedError
from antecedent.extensibility import ProviderTrust
from antecedent.joint_distribution import JointDistributionArtifact

from _refusal import assert_registered_refusal

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_cross_surface_fixtures as gen  # noqa: E402

REFUSED = (CausalSerializationError, CausalUnsupportedError)
TOLERANCE = 1e-12


def _read(name: str) -> bytes:
    path = gen.FIXTURE_DIR / name
    if not path.is_file():
        pytest.fail(
            f"missing cross-surface fixture {path}; regenerate Python-built fixtures with "
            "python python/tests/generate_cross_surface_fixtures.py"
        )
    return path.read_bytes()


def _flip_last_byte(data: bytes) -> bytes:
    return data[:-1] + bytes([data[-1] ^ 0xFF])


# ------------------------------------------------------------------------- box 3: joint draw


def test_a_exit_joint_draw_round_trips_and_reproduces_covariance_utility_and_summary() -> None:
    expected = gen.joint_law_identity()
    from_python = JointDistributionArtifact.load(
        _read("py_joint_law.bin"), expected_identity=expected
    )
    from_rust = JointDistributionArtifact.load(
        _read("rust_joint_law.bin"), expected_identity=expected
    )
    for law in (from_python, from_rust):
        assert law.mean(0) == pytest.approx(0.5, abs=TOLERANCE)
        assert law.mean(1) == pytest.approx(1.0, abs=TOLERANCE)
        assert law.covariance(0, 1) == pytest.approx(0.5, abs=TOLERANCE)
        assert law.joint_product_expectation(0, 1) == pytest.approx(1.0, abs=TOLERANCE)
        assert law.identity == expected
        assert law.trust == "unverified"
        assert law.calibration == "exact"
        assert law.shape == (2, 2)
    assert np.array_equal(np.asarray(from_python), np.asarray(from_rust))
    assert np.asarray(from_python).tolist() == [[0.0, 0.0], [1.0, 2.0]]
    # Re-export on this surface: the bytes a consumer receives round-trip again.
    again = JointDistributionArtifact.load(from_rust.export("again"), expected_identity=expected)
    assert np.array_equal(np.asarray(again), np.asarray(from_python))

    # Nonlinear utility: the decision twin of the committed fixtures, built on each surface.
    contract = gen.decision_contract()
    py_source = JointDistributionArtifact.load(
        _read("py_decision_source.bin"), expected_identity=gen.source_identity()
    )
    rust_source = JointDistributionArtifact.load(
        _read("rust_decision_source.bin"), expected_identity=gen.source_identity()
    )
    assert decision.source_digest(py_source) == decision.source_digest(rust_source)
    assert np.asarray(py_source)[:, 0].tolist() == gen.P
    assert np.asarray(py_source)[:, 1].tolist() == gen.Q
    twins = ((py_source, "py_decision_result.bin"), (rust_source, "rust_decision_result.bin"))
    for source, result_name in twins:
        computed = contract.evaluate(source)
        assert computed.selected == ("safe",)
        assert computed.verdict == decision.Verdict("uniquely_optimal", ("safe",))
        assert computed.evpi == pytest.approx(0.5, abs=TOLERANCE)
        risky, safe = computed.actions
        assert risky.expected_utility == pytest.approx(2.0, abs=TOLERANCE)
        assert safe.expected_utility == pytest.approx(3.0, abs=TOLERANCE)
        # Published summary: the stored result replays to the same verdict from stored inputs.
        replayed = decision.replay(_read(result_name), contract=contract, source=source)
        assert replayed.selected == computed.selected
        assert replayed.evpi == pytest.approx(computed.evpi, abs=TOLERANCE)
        assert replayed.contract_identity == contract.identity
        assert replayed.source_digest == decision.source_digest(source)
    identities = json.loads(_read("py_decision.identities.json"))
    assert identities["contract_identity"] == contract.identity
    assert identities["source_digest"] == decision.source_digest(py_source)


def test_a_exit_joint_draw_refuses_independent_marginals_and_wrong_meaning() -> None:
    expected = gen.joint_law_identity()
    # Independently paired marginals: E[XY] over {0, 1} x {0, 2} is 1/2, not the joint 1.
    paired = np.array([[x, y] for x in (0.0, 1.0) for y in (0.0, 2.0)])
    assert float(np.mean(paired[:, 0] * paired[:, 1])) == pytest.approx(0.5, abs=TOLERANCE)
    marginals = JointDistributionArtifact(
        gen.joint_law_identity(alignment="independent_marginals"), paired, calibration="exact"
    )
    assert marginals.mean(0) == pytest.approx(0.5, abs=TOLERANCE)
    with pytest.raises(REFUSED, match="aligned_joint_draws.marginals_not_joint"):
        marginals.covariance(0, 1)
    with pytest.raises(REFUSED, match="aligned_joint_draws.marginals_not_joint"):
        marginals.joint_product_expectation(0, 1)

    # A nonlinear decision refuses the same law with a typed, registered refusal.
    decision_marginals = dataclasses.replace(
        gen.source_identity(), alignment="independent_marginals"
    )
    draws = np.array([[a, b, 3.0] for a, b in zip(gen.P, gen.Q, strict=True)], dtype=np.float64)
    with pytest.raises(decision.DecisionRefusal) as joint:
        gen.decision_contract().evaluate(
            JointDistributionArtifact(decision_marginals, draws, calibration="exact")
        )
    assert joint.value.reason_code == "joint_law_required"
    assert joint.value.detail == "decision_evaluation.joint_law_required"
    assert joint.value.expected == "joint"
    assert joint.value.supplied == "independent_marginals"
    assert_registered_refusal(joint.value)

    # Wrong distribution meaning, a changed quantity identity or changed bytes refuse on load.
    for name in ("py_joint_law.bin", "rust_joint_law.bin"):
        data = _read(name)
        for changed in (
            gen.joint_law_identity(semantic="bootstrap"),
            gen.joint_law_identity(semantic="causal_functional_posterior"),
            gen.joint_law_identity(second="z"),
        ):
            with pytest.raises(REFUSED, match="identity_expected"):
                JointDistributionArtifact.load(data, expected_identity=changed)
        with pytest.raises(REFUSED):
            JointDistributionArtifact.load(_flip_last_byte(data), expected_identity=expected)
        with pytest.raises(REFUSED):
            JointDistributionArtifact.load(data[:-5], expected_identity=expected)


# ----------------------------------------------------------------------- box 4: external response


def test_a_exit_external_response_binds_inspects_exports_and_shows_support_and_provenance() -> None:
    claim = gen.external_claim()
    assert np.allclose(claim.values, gen.EXTERNAL_VALUES, atol=TOLERANCE, rtol=0)
    assert claim.native is False
    assert claim.trust is ProviderTrust.EXTERNALLY_ATTESTED
    assert claim.trust is not ProviderTrust.NATIVE_LICENSED
    assert claim.uncertainty_method is None
    # Coordinate-level support, not one pooled flag.
    assert claim.support == ("supported", "supported", "outside_empirical_support")
    assert claim.support_status == "outside_empirical_support"
    assert [q.regime_id for q in claim.quantities] == ["do(a=0)", "do(a=1)", "do(a=2)"]
    assert {q.units for q in claim.quantities} == {"mmHg"}
    # Full provenance: contract, evidence, provider execution and the claim, with Merkle digests.
    assert claim.provenance_label == "external:lab/curve@v3#snap-9"
    inspection = claim.inspect()
    assert inspection.native is False
    assert [link.id for link in inspection.lineage] == [
        "contract:checked-contract",
        "evidence:factor:z",
        "provider:external:lab/curve@v3#snap-9",
        "claim",
    ]
    by_id = {link.id: link for link in inspection.lineage}
    for link in inspection.lineage:
        assert len(link.digest) == 64
        assert link.parent_digests == tuple(by_id[parent].digest for parent in link.parents)
    assert {"causal_contract", "evidence", "external_provider"} <= claim.stages_behind()
    assert "trust: externally_attested" in str(inspection)

    # Export and consume under the identity the consumer holds, on both surfaces' bytes.
    spec = gen.external_spec()
    data = claim.export(artifact_id="claim")
    for blob in (data, _read("py_external_claim.bin"), _read("rust_external_claim.bin")):
        loaded = spec.load(blob, expected=claim.identity)
        assert np.allclose(loaded.values, gen.EXTERNAL_VALUES, atol=TOLERANCE, rtol=0)
        assert loaded.support == claim.support
        assert loaded.identity == claim.identity
        assert [link.id for link in loaded.lineage] == [link.id for link in claim.lineage]
    fields = ("provider_id", "object_id", "version_id", "snapshot_id", "request_id")
    assert [claim.identity[f] for f in fields] == ["lab", "curve", "v3", "snap-9", "req-1"]
    assert claim.identity["evidence_ids"] == ["factor:z"]
    assert claim.identity["assumption_ids"] == ["ignorability"]
    with pytest.raises(Exception, match="differs|verification receipt"):
        spec.load(data, expected={**claim.identity, "snapshot_id": "other-snapshot"})
    with pytest.raises(REFUSED):
        spec.load(_flip_last_byte(data), expected=claim.identity)


def test_a_exit_external_response_refuses_observational_mismatched_and_unverified() -> None:
    spec = gen.external_spec()

    # Observational law offered for do(a=1) with no checked equivalence.
    one = dataclasses.replace(spec, quantities=(spec.quantities[1],))
    observational = (dataclasses.replace(one.quantities[0], regime_id=external.OBSERVATIONAL),)
    with pytest.raises(CausalUnsupportedError) as law:
        one.bind(gen.external_response(quantities=observational, values=[3.0], support=None))
    assert isinstance(law.value, external.ExternalRefusal)
    assert law.value.detail == "external_response_binding.unchecked_observational_law"
    assert law.value.remedy
    assert_registered_refusal(law.value)

    # Mismatched coordinate: kPa for mmHg at coordinate 1 is refused, never converted.
    wrong_units = tuple(
        dataclasses.replace(q, units="kPa") if i == 1 else q for i, q in enumerate(spec.quantities)
    )
    with pytest.raises(CausalUnsupportedError) as units:
        spec.bind(gen.external_response(quantities=wrong_units))
    assert isinstance(units.value, external.ExternalRefusal)
    assert units.value.reason_code == "quantity_semantics_mismatch"
    assert units.value.detail == "external_response_binding.coordinate_units"
    assert (units.value.offending, units.value.expected, units.value.supplied) == (
        "coordinate[1]",
        "mmHg",
        "kPa",
    )
    assert units.value.stage == "bind"
    assert_registered_refusal(units.value)
    with pytest.raises(CausalUnsupportedError) as short:
        spec.bind(gen.external_response(values=[1.0, 3.0], support=None))
    assert short.value.detail == "external_response_binding.dimension"
    with pytest.raises(CausalUnsupportedError) as graph:
        spec.bind(gen.external_response(graph_id="graph:other"))
    assert graph.value.reason_code == "external_binding_mismatch"

    # Unverified request: an unattested, unprobed response never binds; missing or failing
    # probes name the property; a foreign meaning is refused.
    with pytest.raises(CausalUnsupportedError) as unattested:
        spec.bind(gen.external_response(attested_by=None))
    assert unattested.value.reason_code == "invalid_argument"
    assert_registered_refusal(unattested.value)
    probes = tuple(
        external.VerificationProbe(kind, 1.0, 1.0, 0.0)
        for kind in ("shape", "support", "moments", "known_truth")
    )
    with pytest.raises(CausalUnsupportedError) as missing:
        spec.bind(gen.external_response(attested_by=None, probes=probes[:2]))
    assert missing.value.reason_code == "external_verification_failed"
    assert missing.value.detail == "external_object_verification.missing_probe"
    assert missing.value.offending == "known_truth"
    failing = (*probes[:3], external.VerificationProbe("known_truth", 1.5, 1.0, 1e-9))
    with pytest.raises(CausalUnsupportedError) as failed:
        spec.bind(gen.external_response(attested_by=None, probes=failing))
    assert failed.value.detail == "external_object_verification.failed_probe"
    with pytest.raises(CausalUnsupportedError) as meaning:
        spec.bind(
            gen.external_response(provider=gen.external_provider(meaning="posterior_predictive"))
        )
    assert meaning.value.reason_code == "distribution_meaning_mismatch"
    # The probes that do pass give the stronger, still non-native, trust.
    verified = spec.bind(gen.external_response(attested_by=None, probes=probes))
    assert verified.trust is ProviderTrust.VERIFIED_EXTENSION
    assert verified.native is False
