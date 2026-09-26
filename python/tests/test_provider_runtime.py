from __future__ import annotations

import antecedent
import numpy as np
import pytest
from antecedent.extensibility import (
    CausalProviderSpec,
    ProviderExecution,
    ProviderQuery,
    ProviderRegistry,
    ProviderTrust,
    providers,
)


def _spec(shape=(2,)):
    return CausalProviderSpec(
        query_family="effect",
        identification_requirements=("caller_identified",),
        observed_distributions=("Y,T,W",),
        nuisance_functions=(),
        support_conditions=("overlap",),
        data_dependence=("caller_supplied",),
        inference_claims=("point_only",),
        influence_function=None,
        fold_policy="provider_owned",
        output_shape=shape,
        uncertainty_semantics="point_only",
        artifact_codec="provider_specific",
        deterministic=True,
        provenance={"package": "test-provider"},
    )


class _Provider:
    spec = _spec()

    def execute(self, request):
        assert request["query"] == "effect"
        with pytest.raises(TypeError, match="does not support item assignment"):
            request["query"] = "mutated"
        return ProviderExecution(
            estimate=[1.0, 2.0], uncertainty=None,
            assumptions=("unconfounded",), support_status="caller_asserted",
            provenance={"version": "0.1"},
        )


def test_registry_requires_explicit_registration_and_validates_output():
    registry = ProviderRegistry()
    with pytest.raises(KeyError, match="not explicitly registered"):
        registry.execute("x", {})
    registry.register("x", _Provider())
    result = registry.execute("x", {"query": "effect"})
    np.testing.assert_array_equal(result.estimate, [1.0, 2.0])
    assert not result.estimate.flags.writeable
    assert result.trust is ProviderTrust.EXTERNALLY_ATTESTED
    assert result.provenance["registry_name"] == "x"
    assert result.uncertainty_semantics == "point_only"


def test_registry_rejects_native_self_promotion_and_wrong_shape():
    registry = ProviderRegistry()
    with pytest.raises(ValueError, match="native licensed"):
        registry.register("x", _Provider(), trust=ProviderTrust.NATIVE_LICENSED)

    class WrongShape(_Provider):
        def execute(self, request):
            return ProviderExecution([1.0], None, ("a",), "supported", {"version": "x"})

    registry.register("y", WrongShape())
    with pytest.raises(ValueError, match="does not match declared"):
        registry.execute("y", {})


def test_analyze_dispatches_explicit_provider_query_to_shared_result_envelope():
    name = "runtime-test-analyze-provider"
    providers.register(name, _Provider())
    result = antecedent.analyze(
        {"x": [1.0, 2.0]},
        query=ProviderQuery(name, {"query": "effect"}),
    )
    assert result.answer.kind == "structured"
    assert result.query_family == "effect"
    assert result.provider_result.trust is ProviderTrust.EXTERNALLY_ATTESTED
    assert result.provenance["registry_name"] == name
    assert "native licensed causal claim" in result.claim()
    assert result.calibration.reason == "attested_not_reverifiable"
