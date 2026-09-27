"""Host verification of externally supplied provider fixtures."""

from __future__ import annotations

import hashlib
import json

import pytest
from antecedent.extensibility import (
    CausalProviderSpec,
    ProviderExecution,
    ProviderRegistry,
    ProviderTrust,
    ProviderVerificationFixture,
)
from antecedent.results.provider import ProviderAnalysisResult


def _spec(*, deterministic=True, shape=(1,)):
    return CausalProviderSpec(
        query_family="effect",
        identification_requirements=("randomized_assignment",),
        observed_distributions=("Y,T",),
        nuisance_functions=(),
        support_conditions=("both_arms",),
        data_dependence=("caller_supplied_rows",),
        inference_claims=("point_only",),
        influence_function=None,
        fold_policy="none",
        output_shape=shape,
        uncertainty_semantics="point_only",
        artifact_codec="json-effect-v1",
        deterministic=deterministic,
        provenance={"package": "external-test"},
    )


class Provider:
    spec = _spec()

    def execute(self, request):
        estimate = float(request["known_effect"])
        artifact = json.dumps({"effect": estimate}, sort_keys=True).encode()
        return ProviderExecution(
            [estimate],
            None,
            ("randomized assignment",),
            "both_arms",
            {"version": "1"},
            artifact,
        )


def _fixture(value=2.0):
    artifact = json.dumps({"effect": value}, sort_keys=True).encode()
    return ProviderVerificationFixture(
        name=f"known-truth-{value}",
        request={"known_effect": value},
        expected_estimate=[value],
        expected_uncertainty=None,
        expected_assumptions=("randomized assignment",),
        expected_support_status="both_arms",
        expected_provenance={"package": "external-test", "version": "1"},
        artifact_digest=hashlib.sha256(artifact).hexdigest(),
        artifact_decoder=json.loads,
        expected_decoded_artifact={"effect": value},
    )


def test_verified_provider_promotion_records_evidence_and_shared_result():
    registry = ProviderRegistry()
    registry.register("external", Provider())
    report = registry.verify(
        "external", [_fixture(2.0), _fixture(3.0)], evidence_origin="independent-fixtures/v1"
    )
    assert len(report.evidence_digest) == 64
    assert report.deterministic_replay_checked
    assert registry.verification_report("external") == report
    result = registry.execute("external", {"known_effect": 2.0})
    assert result.trust is ProviderTrust.VERIFIED_EXTENSION
    assert result.provenance["verification_evidence_digest"] == report.evidence_digest
    unseen = registry.execute("external", {"known_effect": 4.0})
    assert unseen.trust is ProviderTrust.EXTERNALLY_ATTESTED
    assert unseen.provenance["trust_boundary"] == "externally_attested"
    assert "verification_evidence_digest" not in unseen.provenance
    view = ProviderAnalysisResult(
        provider_name="external", query_family="effect", provider_result=result
    )
    assert view.support == ("verified_extension",)
    assert view.calibration.status == "unavailable"
    assert view.calibration.reason == "extension_fixture_verification_does_not_calibrate_inference"
    assert view.to_dict()["provider_result"]["support_status"] == "verified_extension"
    with pytest.raises(ValueError, match="only externally attested"):
        registry.verify("external", [_fixture()], evidence_origin="independent-fixtures/v1")


@pytest.mark.parametrize(
    "change,match",
    [
        ({"expected_estimate": [9.0]}, "estimate differs"),
        ({"expected_uncertainty": [0.1]}, "uncertainty differs"),
        ({"expected_assumptions": ("false",)}, "assumptions differ"),
        ({"expected_support_status": "native_licensed"}, "support differs"),
        ({"expected_provenance": {"version": "false"}}, "provenance"),
        ({"artifact_digest": "0" * 64}, "artifact digest differs"),
        ({"expected_decoded_artifact": {"effect": 9.0}}, "round trip differs"),
    ],
)
def test_mismatched_fixture_refuses_promotion(change, match):
    from dataclasses import replace

    registry = ProviderRegistry()
    registry.register("external", Provider())
    with pytest.raises(ValueError, match=match):
        registry.verify(
            "external", [replace(_fixture(), **change)], evidence_origin="independent/v1"
        )
    assert (
        registry.execute("external", {"known_effect": 2.0}).trust
        is ProviderTrust.EXTERNALLY_ATTESTED
    )
    with pytest.raises(KeyError, match="no verification report"):
        registry.verification_report("external")


def test_deterministic_replay_and_spec_tampering_are_refused():
    from dataclasses import replace

    class Drifting(Provider):
        calls = 0

        def execute(self, request):
            self.calls += 1
            raw = super().execute(request)
            return ProviderExecution(
                raw.estimate,
                raw.uncertainty,
                raw.assumptions,
                raw.support_status,
                {"version": str(self.calls)},
                raw.artifact,
            )

    registry = ProviderRegistry()
    registry.register("drifting", Drifting())
    with pytest.raises(ValueError, match="deterministic replay differs"):
        registry.verify(
            "drifting",
            [replace(_fixture(), expected_provenance={"package": "external-test"})],
            evidence_origin="independent/v1",
        )
    assert (
        registry.execute("drifting", {"known_effect": 2.0}).trust
        is ProviderTrust.EXTERNALLY_ATTESTED
    )

    provider = Provider()
    registry.register("stable", provider)
    registry.verify("stable", [_fixture()], evidence_origin="independent/v1")
    provider.spec = _spec(shape=(2,))
    with pytest.raises(ValueError, match="spec changed"):
        registry.execute("stable", {"known_effect": 2.0})


def test_native_registration_and_unverified_artifact_refused():
    registry = ProviderRegistry()
    with pytest.raises(ValueError, match="native licensed"):
        registry.register("native", Provider(), trust=ProviderTrust.NATIVE_LICENSED)
    with pytest.raises(ValueError, match="separate evidence gate"):
        registry.register("verified", Provider(), trust=ProviderTrust.VERIFIED_EXTENSION)
    registry.register("external", Provider())
    from dataclasses import replace

    with pytest.raises(ValueError, match="unverified artifact"):
        registry.verify(
            "external",
            [replace(_fixture(), artifact_digest=None, artifact_decoder=None)],
            evidence_origin="independent/v1",
        )


def test_point_only_uncertainty_and_non_deterministic_contract():
    from dataclasses import replace

    class IncorrectUncertainty(Provider):
        def execute(self, request):
            raw = super().execute(request)
            return ProviderExecution(
                raw.estimate,
                [0.1],
                raw.assumptions,
                raw.support_status,
                raw.provenance,
                raw.artifact,
            )

    registry = ProviderRegistry()
    registry.register("incorrect", IncorrectUncertainty())
    with pytest.raises(ValueError, match="point-only provider returned uncertainty"):
        registry.verify(
            "incorrect",
            [replace(_fixture(), expected_uncertainty=[0.1])],
            evidence_origin="independent/v1",
        )

    class NonDeterministic(Provider):
        spec = _spec(deterministic=False)

    registry.register("nondeterministic", NonDeterministic())
    report = registry.verify("nondeterministic", [_fixture()], evidence_origin="independent/v1")
    assert not report.deterministic_replay_checked
