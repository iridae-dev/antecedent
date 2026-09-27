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


def test_point_only_provider_cannot_return_uncertainty_at_execution():
    class FalseUncertainty(_Provider):
        def execute(self, request):
            return ProviderExecution([1.0, 2.0], [0.1, 0.1],
                                     ("caller_asserted",), "both_arms", {"version": "1"})

    registry = ProviderRegistry()
    registry.register("point-only", FalseUncertainty())
    with pytest.raises(ValueError, match="point-only provider returned uncertainty"):
        registry.execute("point-only", {})


def test_provider_reported_support_cannot_become_the_host_trust_status():
    class SelfPromoting(_Provider):
        def execute(self, request):
            return ProviderExecution(
                [1.0, 2.0], None, ("caller_asserted",), "native_licensed", {"version": "x"}
            )

    registry = ProviderRegistry()
    registry.register("self-promoting", SelfPromoting())
    from antecedent.results.provider import ProviderAnalysisResult

    result = ProviderAnalysisResult(
        provider_name="self-promoting",
        query_family="effect",
        provider_result=registry.execute("self-promoting", {}),
    )
    assert result.support == ("externally_attested",)
    assert result.inspect().support.summary == "externally_attested"
    dumped = result.to_dict()["provider_result"]
    assert dumped["support_status"] == "externally_attested"
    assert dumped["provider_reported_support_status"] == "native_licensed"


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
    assert result.support == ("externally_attested",)
    assert result.inspect().support.summary == "externally_attested"
    assert result.to_dict()["provider_result"]["support_status"] == "externally_attested"
    assert "native licensed causal claim" in result.claim()
    assert result.calibration.reason == "attested_not_reverifiable"


def test_installed_entry_point_load_is_explicit_and_preserves_artifact(monkeypatch):
    import antecedent.extensibility as extension

    calls = []

    class EntryPoint:
        value = "external_package.provider:create_provider"

        def load(self):
            calls.append("load")

            class External(_Provider):
                def execute(self, request):
                    return ProviderExecution(
                        estimate=[1.0, 2.0],
                        uncertainty=None,
                        assumptions=("caller asserted exchangeability",),
                        support_status="caller_asserted",
                        provenance={"version": "0.2", "entry_point": "spoofed"},
                        artifact=b"external portable receipt",
                    )

            return External

    class EntryPoints:
        def select(self, *, group, name):
            assert group == "antecedent.providers"
            assert name == "external"
            return [EntryPoint()]

    monkeypatch.setattr(extension.metadata, "entry_points", lambda: EntryPoints())
    registry = ProviderRegistry()
    assert calls == []  # No plugin import or execution during registry construction.
    registry.load_entry_point("external")
    assert calls == ["load"]
    result = registry.execute("external", {"query": "effect"})
    assert result.artifact == b"external portable receipt"
    assert result.provenance["entry_point"] == EntryPoint.value
    assert result.provenance["trust_boundary"] == "externally_attested"
    assert result.trust is ProviderTrust.EXTERNALLY_ATTESTED


def test_entry_point_load_refuses_missing_duplicate_and_invalid_factories(monkeypatch):
    import antecedent.extensibility as extension

    class EntryPoints:
        def __init__(self, matches):
            self.matches = matches

        def select(self, *, group, name):
            return self.matches

    class EntryPoint:
        value = "external_package.provider:factory"

        def load(self):
            return object()  # Installed entry points must expose a factory.

    registry = ProviderRegistry()
    monkeypatch.setattr(extension.metadata, "entry_points", lambda: EntryPoints([]))
    with pytest.raises(KeyError, match="not installed"):
        registry.load_entry_point("missing")
    monkeypatch.setattr(extension.metadata, "entry_points", lambda: EntryPoints([EntryPoint()] * 2))
    with pytest.raises(ValueError, match="ambiguous"):
        registry.load_entry_point("duplicate")
    monkeypatch.setattr(extension.metadata, "entry_points", lambda: EntryPoints([EntryPoint()]))
    with pytest.raises(TypeError, match="must expose a factory"):
        registry.load_entry_point("invalid")
    with pytest.raises(KeyError, match="not explicitly registered"):
        registry.get("invalid")


def test_separately_installed_entry_point_loads_without_rebuilding(tmp_path, monkeypatch):
    """Exercise real importlib distribution discovery, not a patched registry."""
    import importlib

    module = tmp_path / "antecedent_external_fixture.py"
    module.write_text(
        """\
from antecedent.extensibility import CausalProviderSpec, ProviderExecution

class ExternalProvider:
    spec = CausalProviderSpec(
        query_family="effect",
        identification_requirements=("caller_identified",),
        observed_distributions=("Y,T",),
        nuisance_functions=(),
        support_conditions=("overlap",),
        data_dependence=("caller_supplied",),
        inference_claims=("point_only",),
        influence_function=None,
        fold_policy="provider_owned",
        output_shape=(1,),
        uncertainty_semantics="point_only",
        artifact_codec="provider_specific",
        deterministic=True,
        provenance={"package": "external-fixture"},
    )

    def execute(self, request):
        return ProviderExecution(
            estimate=[float(request["value"])], uncertainty=None,
            assumptions=("caller_asserted",), support_status="caller_asserted",
            provenance={"version": "0.1"}, artifact=b"external-receipt",
        )

def create_provider():
    return ExternalProvider()
""",
        encoding="utf-8",
    )
    dist = tmp_path / "antecedent_external_fixture-0.1.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: antecedent-external-fixture\nVersion: 0.1\n",
        encoding="utf-8",
    )
    (dist / "entry_points.txt").write_text(
        "[antecedent.providers]\nexternal-fixture = antecedent_external_fixture:create_provider\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    registry = ProviderRegistry()
    registry.load_entry_point("external-fixture")
    result = registry.execute("external-fixture", {"value": 3.0})
    assert result.estimate.tolist() == [3.0]
    assert result.artifact == b"external-receipt"
    assert result.trust is ProviderTrust.EXTERNALLY_ATTESTED
    assert result.provenance["entry_point"] == "antecedent_external_fixture:create_provider"


def test_separately_packaged_provider_upgrades_in_place_without_rebuilding(tmp_path, monkeypatch):
    """A separately packaged provider is installed, exercised with a host-artifact
    round-trip, then upgraded in place to a new version that changes the estimate
    and now reports uncertainty -- all without rebuilding Antecedent. The upgrade's
    estimate, version/provenance, uncertainty, and provider-owned fold policy are
    reflected in the fresh result and the round-tripped host envelope."""
    import importlib
    import json
    import sys

    module_name = "antecedent_upgrade_fixture"

    def install(version: str, *, multiplier: int, with_uncertainty: bool) -> None:
        inference = '("pointwise_95",)' if with_uncertainty else '("point_only",)'
        semantics = "subject_score_standard_error" if with_uncertainty else "point_only"
        uncertainty = "[abs(value) * 0.1 + 0.5]" if with_uncertainty else "None"
        (tmp_path / f"{module_name}.py").write_text(
            f"""\
from antecedent.extensibility import CausalProviderSpec, ProviderExecution

class UpgradeProvider:
    spec = CausalProviderSpec(
        query_family="effect",
        identification_requirements=("caller_identified",),
        observed_distributions=("Y,T",),
        nuisance_functions=(),
        support_conditions=("overlap",),
        data_dependence=("caller_supplied",),
        inference_claims={inference},
        influence_function=None,
        fold_policy="provider_owned",
        output_shape=(1,),
        uncertainty_semantics="{semantics}",
        artifact_codec="provider_specific",
        deterministic=True,
        provenance={{"package": "upgrade-fixture", "version": "{version}"}},
    )

    def execute(self, request):
        value = float(request["effect"]) * {multiplier}
        return ProviderExecution(
            estimate=[value], uncertainty={uncertainty},
            assumptions=("caller_asserted",), support_status="caller_asserted",
            provenance={{"version": "{version}"}}, artifact=b"upgrade-receipt-{version}",
        )

def create_provider():
    return UpgradeProvider()
""",
            encoding="utf-8",
        )
        for stale in tmp_path.glob(f"{module_name}-*.dist-info"):
            for child in stale.iterdir():
                child.unlink()
            stale.rmdir()
        dist = tmp_path / f"{module_name}-{version}.dist-info"
        dist.mkdir()
        (dist / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {module_name.replace('_', '-')}\nVersion: {version}\n",
            encoding="utf-8",
        )
        (dist / "entry_points.txt").write_text(
            f"[antecedent.providers]\nupgrade-fixture = {module_name}:create_provider\n",
            encoding="utf-8",
        )
        sys.modules.pop(module_name, None)
        importlib.invalidate_caches()

    monkeypatch.syspath_prepend(str(tmp_path))

    # v0.1 -- installed as a separate distribution, point-only, loaded without a rebuild.
    install("0.1", multiplier=1, with_uncertainty=False)
    first_registry = ProviderRegistry()
    first_registry.load_entry_point("upgrade-fixture")
    first = first_registry.execute("upgrade-fixture", {"effect": 3.0})
    assert first.estimate.tolist() == [3.0]
    assert first.uncertainty is None
    assert first.provenance["version"] == "0.1"
    assert first.provenance["entry_point"] == f"{module_name}:create_provider"
    assert first.trust is ProviderTrust.EXTERNALLY_ATTESTED
    assert first_registry.get("upgrade-fixture").spec.fold_policy == "provider_owned"
    assert first.artifact == b"upgrade-receipt-0.1"
    host_first = json.loads(antecedent._native.open_provider_result(first.host_artifact)[0])
    assert host_first["provenance"]["version"] == "0.1"

    # v0.2 -- upgraded in place: new version, doubled estimate, now reports a standard
    # error. A fresh registry over the reinstalled distribution needs no Antecedent rebuild.
    install("0.2", multiplier=2, with_uncertainty=True)
    second_registry = ProviderRegistry()
    second_registry.load_entry_point("upgrade-fixture")
    second = second_registry.execute("upgrade-fixture", {"effect": 3.0})
    assert second.estimate.tolist() == [6.0]
    assert second.uncertainty is not None
    assert second.uncertainty_semantics == "subject_score_standard_error"
    assert second.provenance["version"] == "0.2"
    assert second.trust is ProviderTrust.EXTERNALLY_ATTESTED
    assert second_registry.get("upgrade-fixture").spec.fold_policy == "provider_owned"
    assert second.artifact == b"upgrade-receipt-0.2"
    host_second = json.loads(antecedent._native.open_provider_result(second.host_artifact)[0])
    assert host_second["provenance"]["version"] == "0.2"
