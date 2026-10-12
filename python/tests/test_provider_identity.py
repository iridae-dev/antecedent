"""One provider identity: derived from the registry, not restated by each surface."""

from __future__ import annotations

import json

import pytest
from antecedent.extensibility import (
    CausalProviderSpec,
    ExecutableProvider,
    ProviderCapability,
    ProviderExecution,
    ProviderRegistry,
    ProviderVerificationFixture,
    UncertaintySemantics,
)
from antecedent.joint_distribution import QuantityCondition, ScientificQuantity


def _spec(**changes):
    values = {
        "query_family": "effect",
        "identification_requirements": ("randomized_assignment",),
        "observed_distributions": ("Y,T",),
        "nuisance_functions": (),
        "support_conditions": ("both_arms",),
        "data_dependence": ("caller_supplied_rows",),
        "inference_claims": ("point_only",),
        "influence_function": None,
        "fold_policy": "none",
        "output_shape": (1,),
        "uncertainty_semantics": "point_only",
        "artifact_codec": "json-effect-v1",
        "deterministic": True,
        "provenance": {"package": "external-test"},
        "capabilities": ("mean",),
        "meaning": "interventional_predictive",
    }
    values.update(changes)
    return CausalProviderSpec(**values)


class _Provider:
    spec = _spec()

    def execute(self, request):
        value = float(request["known_effect"])
        return ProviderExecution([value], None, ("randomized",), "both_arms", {"version": "1"})


def _quantity() -> ScientificQuantity:
    return ScientificQuantity(
        variable_id="y",
        variable_name="y",
        role="outcome",
        units="mmHg",
        population_id="target",
        regime_id="do(a=1)",
        horizon=0,
        functional_id="mean",
        conditioning=(QuantityCondition("a", "1"),),
        transform_id="identity",
    )


def test_spec_types_capabilities_and_uncertainty_semantics():
    spec = _spec()
    assert spec.capabilities == (ProviderCapability.MEAN,)
    assert all(isinstance(c, ProviderCapability) for c in spec.capabilities)
    assert spec.uncertainty_semantics is UncertaintySemantics.POINT_ONLY
    assert spec.uncertainty_method is None
    declared = _spec(uncertainty_semantics="subject_score_standard_error")
    assert declared.uncertainty_semantics == "subject_score_standard_error"
    assert declared.uncertainty_method == "subject_score_standard_error"
    with pytest.raises(ValueError, match="unknown provider capability 'teleport'"):
        _spec(capabilities=("mean", "teleport"))
    with pytest.raises(ValueError, match="repeat"):
        _spec(capabilities=("mean", ProviderCapability.MEAN))
    with pytest.raises(ValueError, match="one string"):
        _spec(capabilities="mean")
    with pytest.raises(ValueError, match="meaning must be one of"):
        _spec(meaning="mostly_causal")


def test_provider_object_is_derived_from_the_registered_provider():
    registry = ProviderRegistry()
    provider: ExecutableProvider = _Provider()
    registry.register("external", provider)
    request = {"known_effect": 2.0}
    obj = registry.provider_object("external", request)
    assert obj.provider_id == "external" and obj.object_id == "effect"
    assert obj.capabilities == ("mean",) and obj.meaning == "interventional_predictive"
    assert obj.request == obj.snapshot and len(obj.request) == 64
    assert obj.version == registry.provider_object("external", {"known_effect": 3.0}).version
    assert obj.request != registry.provider_object("external", {"known_effect": 3.0}).request
    pinned = registry.provider_object("external", request, snapshot="rows-1", meaning="bootstrap")
    assert (pinned.snapshot, pinned.meaning) == ("rows-1", "bootstrap")

    # The same identity is what a verification receipt records for the request.
    receipt = registry.verify(
        "external",
        [
            ProviderVerificationFixture(
                name="known",
                request=request,
                expected_estimate=[2.0],
                expected_uncertainty=None,
                expected_assumptions=("randomized",),
                expected_support_status="both_arms",
                expected_provenance={"version": "1"},
            )
        ],
        evidence_origin="independent/v1",
    )
    (verified,) = receipt.requests
    assert (verified.provider, verified.query_family) == (obj.provider_id, obj.object_id)
    assert (verified.spec_digest, verified.request_digest) == (obj.version, obj.request)
    assert verified.snapshot == obj.snapshot


def test_provider_object_needs_declared_meaning_and_capabilities():
    registry = ProviderRegistry()

    class NoMeaning:
        spec = _spec(meaning=None)
        execute = _Provider.execute

    class NoCapabilities:
        spec = _spec(capabilities=())
        execute = _Provider.execute

    registry.register("no-meaning", NoMeaning())
    registry.register("no-capabilities", NoCapabilities())
    with pytest.raises(ValueError, match="no distribution meaning"):
        registry.provider_object("no-meaning", {})
    assert registry.provider_object("no-meaning", {}, meaning="bootstrap").meaning == "bootstrap"
    with pytest.raises(ValueError, match="no capabilities"):
        registry.provider_object("no-capabilities", {})
    with pytest.raises(KeyError, match="not explicitly registered"):
        registry.provider_object("absent", {})


def test_distribution_identity_takes_provider_and_snapshot_from_the_object():
    registry = ProviderRegistry()
    registry.register("external", _Provider())
    obj = registry.provider_object("external", {"known_effect": 2.0}, snapshot="rows-1")
    identity = obj.distribution_identity(
        [_quantity()], source_id="study-1", rng_id="seed-7", causal_contract_id="contract-1"
    )
    assert identity.provider_id == "external" and identity.snapshot_id == "rows-1"
    assert identity.semantic == "interventional_predictive"
    assert json.dumps(identity._wire())  # serializable on the shared wire
