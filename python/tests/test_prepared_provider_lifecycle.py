"""Provider journeys retain owned requests without inventing native authority."""

import traceback
from dataclasses import replace

import antecedent as ac
import numpy as np
import pytest
from antecedent.errors import CausalUnsupportedError, CausalValueError
from antecedent.execution import CachePolicy, Execution, ResourceLimits
from antecedent.extensibility import (
    CausalProviderSpec,
    ProviderExecution,
    ProviderQuery,
    ProviderTrust,
    providers,
)


@pytest.fixture
def provider(monkeypatch):
    for name in ("_providers", "_verification", "_entry_points"):
        monkeypatch.setattr(providers, name, {})

    class Provider:
        spec = CausalProviderSpec(
            query_family="owned_mean",
            identification_requirements=("caller identified",),
            observed_distributions=("x",),
            nuisance_functions=(),
            support_conditions=("caller asserted",),
            data_dependence=("supplied rows",),
            inference_claims=("point_only",),
            influence_function=None,
            fold_policy="provider_owned",
            output_shape=(),
            uncertainty_semantics="point_only",
            artifact_codec="opaque",
            deterministic=True,
            provenance={"package": "lifecycle-test"},
        )

        def __init__(self):
            self.calls = []

        def execute(self, request):
            self.calls.append(request)
            value = np.mean(request["data"]["x"]) + request.get("offsets", (0,))[0]
            return ProviderExecution(
                value, None, ("caller identified",), "caller asserted", {"version": "1"}
            )

    instance = Provider()
    providers.register("prepared-test", instance)
    return instance


def question(offsets=None):
    return ProviderQuery("prepared-test", {"offsets": [0] if offsets is None else offsets})


def test_prepare_owns_rows_and_request_then_retains_completed_study(provider):
    rows = {"x": np.array([1.0, 3.0])}
    offsets = [2.0]
    study = ac.prepare(rows, query=question(offsets))
    assert provider.calls == []
    assert study.inspect().stage == "prepared"
    assert "externally classified" in study.report()
    rows["x"][:] = 100
    offsets[0] = 500
    first = ac.estimate(study)
    assert first.accepted
    assert float(first.value.estimate) == 4.0
    assert first.study.inspect().stage == "executed"
    assert study.inspect().stage == "prepared"
    assert first.value.provider_result.trust is ProviderTrust.EXTERNALLY_ATTESTED
    assert first.value.claim_id is None
    assert first.value.program_id is None
    assert first.value.host_artifact is not None
    assert first.inspect().identification.available is False
    assert float(ac.estimate(study).value.estimate) == 4.0


def test_rebind_and_reconfigure_are_nonexecuting_immutable_branches(provider):
    study = ac.prepare({"x": [1.0, 3.0]}, query=question())
    changed = {"x": np.array([10.0, 20.0])}
    rebound = ac.rebind(study, data=changed)
    configured = ac.reconfigure(study, query=question([7.0]))
    assert provider.calls == []
    changed["x"][:] = 200
    assert float(ac.estimate(rebound).value.estimate) == 15.0
    assert float(ac.estimate(configured).value.estimate) == 9.0
    assert float(ac.estimate(study).value.estimate) == 2.0


def test_analyze_uses_the_same_provider_preparation(provider):
    result = ac.analyze({"x": [2.0, 4.0]}, query=question())
    assert result.study is not None
    rebound = ac.rebind(result.study, data={"x": [6.0, 8.0]})
    assert float(ac.estimate(rebound).value.estimate) == 7.0
    assert float(result.value.estimate) == 3.0


@pytest.mark.parametrize("mutation", ["spec", "identity", "callback"])
def test_preparation_rejects_provider_contract_mutation(provider, mutation):
    study = ac.prepare({"x": [1.0, 3.0]}, query=question())
    original_report = study.inspect().to_dict()
    if mutation == "spec":
        provider.spec = replace(provider.spec, query_family="changed")
    elif mutation == "identity":
        providers._providers["prepared-test"] = (object(), ProviderTrust.EXTERNALLY_ATTESTED)
    else:
        provider.execute = lambda request: None
    assert study.inspect().to_dict() == original_report
    with pytest.raises(CausalValueError, match="changed"):
        ac.estimate(study)
    assert provider.calls == []


def test_scientific_looking_callback_error_keeps_identity_and_traceback(provider):
    original = CausalUnsupportedError(
        "provider callback failed", reason_code="effect_not_identified"
    )

    def provider_callback_raises(request):
        raise original

    provider.execute = provider_callback_raises
    study = ac.prepare({"x": [1.0, 3.0]}, query=question())
    with pytest.raises(CausalUnsupportedError) as caught:
        ac.estimate(study)
    assert caught.value is original
    assert original._antecedent_operational_callback
    assert any(
        frame.name == "provider_callback_raises"
        for frame in traceback.extract_tb(original.__traceback__)
    )


@pytest.mark.parametrize(
    "policy",
    [
        Execution(seed=9),
        Execution(threads=2),
        Execution(limits=ResourceLimits(memory_bytes=100_000)),
        Execution(cache=CachePolicy.disabled()),
        Execution(kernels="scalar"),
        Execution(on_progress=lambda *_: None),
        Execution(on_stage=lambda *_: None),
    ],
)
def test_provider_execution_controls_are_explicitly_inapplicable(provider, policy):
    study = ac.prepare({"x": [1.0, 3.0]}, query=question())
    with pytest.raises(CausalUnsupportedError, match="does not apply"):
        ac.reconfigure(study, execution=policy)
    assert provider.calls == []


def test_provider_request_cannot_replace_owned_data(provider):
    with pytest.raises(CausalValueError, match="must not contain data"):
        ac.prepare({"x": [1.0]}, query=ProviderQuery("prepared-test", {"data": {}}))
    assert provider.calls == []
