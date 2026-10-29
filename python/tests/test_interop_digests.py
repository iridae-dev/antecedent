"""One canonical digest for Python identities, and requests hashed by their content."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from antecedent import _native, artifacts
from antecedent._digest import canonical_digest
from antecedent.errors import CausalSerializationError
from antecedent.extensibility import (
    CausalProviderSpec,
    ProviderExecution,
    ProviderRefusal,
    ProviderRegistry,
    ProviderTrust,
    ProviderVerificationFixture,
    UnknownProviderError,
)
from antecedent.outcomes import require_success

from _refusal import assert_registered_refusal


def _spec() -> CausalProviderSpec:
    return CausalProviderSpec(
        query_family="effect",
        identification_requirements=("identified",),
        observed_distributions=("rows",),
        nuisance_functions=(),
        support_conditions=("overlap",),
        data_dependence=("rows",),
        inference_claims=("point",),
        influence_function=None,
        fold_policy="provider_owned",
        output_shape=(1,),
        uncertainty_semantics="point_only",
        artifact_codec="bytes",
        deterministic=True,
        provenance={"package": "fixture"},
    )


class Mean:
    spec = _spec()

    def execute(self, request):
        data = request["data"]
        column = np.asarray(data["y"] if not hasattr(data, "to_numpy") else data["y"].to_numpy())
        return ProviderExecution(
            estimate=[float(column.mean())],
            uncertainty=None,
            assumptions=("fixture",),
            support_status="caller_asserted",
            provenance={"version": "1"},
            artifact=b"receipt",
        )


def _registry() -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register("mean", Mean())
    return registry


def _header(result) -> dict:
    return json.loads(_native.open_provider_result(result.host_artifact)[0])


def test_the_digest_is_independent_of_key_order_and_separates_domains_and_types() -> None:
    a = canonical_digest("antecedent.test.v1", {"b": [1, 2.5], "a": "x"})
    assert a == canonical_digest("antecedent.test.v1", {"a": "x", "b": [1, 2.5]})
    assert a != canonical_digest("antecedent.test.v2", {"a": "x", "b": [1, 2.5]})
    assert canonical_digest("d", [1]) != canonical_digest("d", [1.0])
    assert canonical_digest("d", {"a": np.float64(1.5)}) == canonical_digest("d", {"a": 1.5})
    with pytest.raises(TypeError, match="no canonical form"):
        canonical_digest("d", {"a": object()})
    with pytest.raises(ValueError, match="JSON"):
        canonical_digest("d", [float("nan")])


def test_two_datasets_that_print_alike_have_different_request_digests() -> None:
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"y": rng.normal(size=20_000), "t": rng.integers(0, 2, 20_000)})
    other = frame.copy()
    other.loc[10_000, "y"] += 1e-6
    assert repr(frame) == repr(other), "the repr elides the row that differs"
    registry = _registry()
    first = require_success(registry.execute("mean", {"data": frame}))
    second = require_success(registry.execute("mean", {"data": other}))
    again = require_success(registry.execute("mean", {"data": frame.copy()}))
    assert _header(first)["request_digest"] != _header(second)["request_digest"]
    assert _header(first)["request_digest"] == _header(again)["request_digest"]
    assert first.provenance["request_digest_method"] == "canonical_json_v2"
    # The same columns as a dict of arrays, a different container, are another request form.
    columns = {name: frame[name].to_numpy() for name in frame.columns}
    assert require_success(registry.execute("mean", {"data": columns})).estimate.shape == (1,)


def test_array_requests_are_hashed_by_dtype_shape_and_bytes() -> None:
    registry = _registry()
    base = np.arange(6.0).reshape(2, 3)

    def digest(value) -> str:
        return _header(
            require_success(registry.execute("mean", {"data": {"y": [1.0]}, "x": value}))
        )["request_digest"]

    assert digest(base) == digest(base.copy())
    assert digest(base) == digest(base.astype(">f8")), "byte order is not identity"
    assert digest(base) != digest(base.reshape(3, 2))
    assert digest(base) != digest(base.astype(np.float32))
    changed = base.copy()
    changed[1, 2] += 1e-12
    assert digest(base) != digest(changed)
    assert digest(np.array(["a", "b"], dtype=object)) != digest(np.array(["a", "c"], dtype=object))


def test_a_request_with_no_canonical_form_is_refused_never_hashed_by_repr() -> None:
    registry = _registry()

    class Opaque:
        def __repr__(self) -> str:
            return "Opaque()"

    with pytest.raises(ProviderRefusal) as caught:
        require_success(registry.execute("mean", {"data": {"y": [1.0]}, "handle": Opaque()}))
    error = caught.value
    assert isinstance(error, ValueError)
    assert error.detail == "provider.request_not_canonical"
    assert error.reason_code == "invalid_argument" and error.remedy
    assert error.offending == "request"
    assert_registered_refusal(error)
    # A mapping cannot pose as the digest of an array.
    with pytest.raises(ProviderRefusal, match="reserved"):
        require_success(
            registry.execute(
                "mean", {"data": {"y": [1.0]}, "x": {"__antecedent_array__": {"blake3": "0" * 64}}}
            )
        )
    # Non-finite numbers in a list are refused, not hashed.
    with pytest.raises(ProviderRefusal, match="canonical digest"):
        require_success(registry.execute("mean", {"data": {"y": [1.0]}, "x": [float("nan")]}))


def test_registry_refusals_carry_reason_codes_and_remedies_and_keep_their_classes() -> None:
    registry = _registry()
    with pytest.raises(UnknownProviderError) as unknown:
        require_success(registry.execute("absent", {}))
    error = unknown.value
    assert isinstance(error, KeyError) and isinstance(error, ProviderRefusal)
    assert error.detail == "provider.not_registered" and error.remedy
    assert "provider 'absent' is not explicitly registered" in str(error)
    with pytest.raises(ProviderRefusal) as duplicate:
        registry.register("mean", Mean())
    assert isinstance(duplicate.value, ValueError)
    assert duplicate.value.detail == "provider.already_registered"
    with pytest.raises(ProviderRefusal) as native:
        registry.register("other", Mean(), trust=ProviderTrust.NATIVE_LICENSED)
    assert native.value.reason_code == "external_verification_failed"
    assert native.value.detail == "provider.native_licence_not_available"
    wrong = ProviderVerificationFixture(
        name="truth",
        request={"data": {"y": [1.0, 3.0]}},
        expected_estimate=[5.0],
        expected_uncertainty=None,
        expected_assumptions=("fixture",),
        expected_support_status="caller_asserted",
        expected_provenance={"version": "1"},
        artifact_digest=hashlib.sha256(b"receipt").hexdigest(),
    )
    with pytest.raises(ProviderRefusal) as failed:
        registry.verify("mean", [wrong], evidence_origin="independent/v1")
    assert failed.value.reason_code == "external_verification_failed"
    assert failed.value.detail == "external_object_verification.failed_probe"
    assert failed.value.offending == "truth" and failed.value.remedy
    assert (
        require_success(registry.execute("mean", {"data": {"y": [1.0]}})).trust
        is ProviderTrust.EXTERNALLY_ATTESTED
    )


def test_a_provider_result_loads_through_artifacts_as_its_declared_header() -> None:
    registry = _registry()
    result = require_success(registry.execute("mean", {"data": {"y": [1.0, 3.0]}}))
    loaded = artifacts.loads(result.host_artifact)
    assert loaded.payload_kind == "provider_result"
    assert loaded.artifact_id == "provider-result:mean"
    assert loaded.to_dict()["payload"]["provider"] == "mean"
    assert loaded.to_dict()["payload"]["estimate"] == [2.0]
    assert loaded.to_dict()["payload"]["trust"] == "externally_attested"
    assert loaded.to_dict()["payload"]["external_artifact_digest"] == _native.blake3_digest_hex(
        b"receipt"
    )
    assert result.host_artifact.startswith(b"CAUSAL\0\0"), "new results use the shared container"
    tampered = bytearray(result.host_artifact)
    tampered[-1] ^= 1
    with pytest.raises(CausalSerializationError, match=r"checksum mismatch for section `provider_result\.artifact`"):
        artifacts.loads(bytes(tampered))
