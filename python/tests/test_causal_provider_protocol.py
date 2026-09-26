from __future__ import annotations

import antecedent
from antecedent.extensibility import CausalProvider, ProviderTrust


def _data() -> dict[str, list[float]]:
    z = [0.0, 1.0, 2.0]
    return {"z": z, "t": z, "y": z}


def test_econml_handoff_implements_declarative_provider_contract_only():
    graph = antecedent.Dag.from_edges(
        ["z", "t", "y"], [("z", "t"), ("z", "y"), ("t", "y")]
    )
    identified = antecedent.identify(
        graph=graph, query=antecedent.AverageEffect("t", "y")
    )
    provider = antecedent.handoff.econml(identified)
    assert isinstance(provider, CausalProvider)
    assert not hasattr(provider, "estimate")
    spec = provider.spec
    assert spec.query_family == "point_identified_adjustment_estimand"
    assert spec.fold_policy == "external_learner_owned_not_recorded_by_handoff"
    assert spec.uncertainty_semantics == "externally_attested_not_reverifiable"
    assert "handoff_does_not_own_or_record_training_rows" in spec.data_dependence
    assert "interval_calibration_is_not_reverified" in spec.inference_claims
    assert spec.provenance["adapter"] == "antecedent.handoff.EconMLSpec"
    assert ProviderTrust.EXTERNALLY_ATTESTED.value == "externally_attested"
    assert provider.columns(_data())["W"].shape == (3, 1)
