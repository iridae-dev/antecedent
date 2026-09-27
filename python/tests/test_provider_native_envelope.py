"""Native host envelope for separately installed Python providers."""

from __future__ import annotations

import hashlib
import json

import antecedent
import numpy as np
import pytest
from antecedent.extensibility import (
    CausalProviderSpec,
    ProviderExecution,
    ProviderQuery,
    ProviderRegistry,
    ProviderTrust,
    ProviderVerificationFixture,
)


class External:
    spec = CausalProviderSpec(
        query_family="effect", identification_requirements=("caller_identified",),
        observed_distributions=("Y,T",), nuisance_functions=(),
        support_conditions=("positive_action_support",),
        data_dependence=("caller_supplied_rows",), inference_claims=("point_only",),
        influence_function=None, fold_policy="provider_owned", output_shape=(1,),
        uncertainty_semantics="point_only", artifact_codec="external-receipt-v1",
        deterministic=True, provenance={"package": "independent-provider"},
    )

    def execute(self, request):
        value = float(request["effect"])
        return ProviderExecution(
            estimate=[value], uncertainty=None,
            assumptions=("caller identified",), support_status="caller_asserted",
            provenance={"version": "0.1"}, artifact=f"external:{value}".encode(),
        )


def test_native_envelope_retains_opaque_artifact_and_refuses_tampering():
    from antecedent import _native

    registry = ProviderRegistry()
    registry.register("external", External())
    result = registry.execute("external", {"effect": 2.0})
    assert result.trust is ProviderTrust.EXTERNALLY_ATTESTED
    header_json, external = _native.open_provider_result(result.host_artifact)
    header = json.loads(header_json)
    assert header["provider"] == "external"
    assert header["trust"] == "externally_attested"
    assert header["estimate"] == [2.0]
    assert external == result.artifact == b"external:2.0"
    changed = bytearray(result.host_artifact)
    changed[-1] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        _native.open_provider_result(bytes(changed))


def test_verified_envelope_requires_host_receipt_for_exact_request():
    from antecedent import _native

    registry = ProviderRegistry()
    registry.register("external", External())
    receipt = b"external:2.0"
    report = registry.verify("external", [ProviderVerificationFixture(
        name="known-effect", request={"effect": 2.0}, expected_estimate=[2.0],
        expected_uncertainty=None, expected_assumptions=("caller identified",),
        expected_support_status="caller_asserted",
        expected_provenance={"version": "0.1"},
        artifact_digest=hashlib.sha256(receipt).hexdigest(),
        artifact_decoder=lambda raw: raw.decode(),
        expected_decoded_artifact="external:2.0",
    )], evidence_origin="independent-known-effect/v1")
    exact = registry.execute("external", {"effect": 2.0})
    assert exact.trust is ProviderTrust.VERIFIED_EXTENSION
    with pytest.raises(ValueError, match="host verification"):
        _native.open_provider_result(exact.host_artifact)
    header_json, external = _native.open_provider_result(
        exact.host_artifact,
        verified_spec_digest=report.spec_digest,
        verified_request_digest=report.verified_request_digests[0],
        verified_evidence_digest=report.evidence_digest,
    )
    assert json.loads(header_json)["trust"] == "verified_extension"
    assert external == receipt
    with pytest.raises(ValueError, match="host verification"):
        _native.open_provider_result(
            exact.host_artifact, verified_spec_digest=report.spec_digest,
            verified_request_digest="0" * 64,
            verified_evidence_digest=report.evidence_digest,
        )
    unseen = registry.execute("external", {"effect": 3.0})
    assert unseen.trust is ProviderTrust.EXTERNALLY_ATTESTED
    assert json.loads(_native.open_provider_result(unseen.host_artifact)[0])["trust"] == "externally_attested"


def test_installed_entry_point_provider_analyze_needs_no_rebuild(monkeypatch):
    import antecedent.extensibility as extension

    class EntryPoint:
        value = "external_package:create_provider"

        def load(self):
            return External

    class EntryPoints:
        def select(self, *, group, name):
            assert (group, name) == ("antecedent.providers", "external-native")
            return [EntryPoint()]

    monkeypatch.setattr(extension.metadata, "entry_points", lambda: EntryPoints())
    name = "external-native"
    extension.providers.load_entry_point(name)
    analyzed = antecedent.analyze(
        {"y": np.asarray([0.0, 1.0])}, query=ProviderQuery(name, {"effect": 2.0})
    )
    assert analyzed.provider_result.artifact == b"external:2.0"
    assert analyzed.export() == b"external:2.0"
    host = analyzed.export_host()
    assert json.loads(antecedent._native.open_provider_result(host)[0])["provenance"]["entry_point"] == EntryPoint.value
