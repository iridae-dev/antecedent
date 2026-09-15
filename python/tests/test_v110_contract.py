"""1.10 contracts-first: identities and transformation preview."""

from __future__ import annotations

import numpy as np
from antecedent.estimation import PreparedAnalysis
from antecedent.query import AverageEffect


def _data() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(11)
    z = rng.normal(size=80)
    t = (rng.uniform(size=80) < 1 / (1 + np.exp(-0.4 + 0.9 * z))).astype(float)
    y = 2.0 * t + z + rng.normal(scale=0.4, size=80)
    return {"t": t, "y": y, "z": z}


def test_prepared_contract_has_domain_separated_identities() -> None:
    prepared = PreparedAnalysis.prepare(
        _data(),
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=AverageEffect("t", "y"),
        refute="none",
        bootstrap=0,
    )
    contract = prepared.contract()
    assert contract["accepted_version"] == "1"
    assert contract["accepted_variable_binding"] in {"explicit", "unbound"}
    assert contract["target"] != contract["program"]
    assert "identification_product" in contract
    assert contract["identification_status"] == "nonparametrically_identified"
    assert contract["matrix_coordinate"].startswith("AverageEffect:Dag:explicit:Frequentist:")
    assert contract["empirical_support"].startswith("unavailable:")
    assert "assumptions" in contract
    assert contract["uncertainty"].startswith("unavailable:")
    refused = prepared.preview_transform("average_unweighted_class")
    assert refused["refused"] == "true"
    preview = prepared.preview_transform("compatible_data_replace")
    assert preview["intent"] == "compatible_data_replace"
    assert preview["refused"] == "false"


def test_same_shape_refresh_changes_only_data_identity() -> None:
    data = _data()
    prepared = PreparedAnalysis.prepare(
        data,
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=AverageEffect("t", "y"),
        refute="none",
        bootstrap=0,
    )
    before = prepared.contract()
    preview = prepared.preview_transform("compatible_data_replace")
    assert preview["input_data_snapshot"] == before["data_snapshot"]
    prepared.refresh({**data, "y": data["y"] + data["t"]})
    after = prepared.contract()
    for layer in (
        "target",
        "identification",
        "identification_product",
        "program",
        "inference_binding",
        "observation",
    ):
        assert before[layer] == after[layer], layer
    assert before["data_snapshot"] != after["data_snapshot"]
    assert preview["input_program"] == after["program"]
    assert preview["input_data_snapshot"] != after["data_snapshot"]


def test_contracted_artifact_is_independently_accepted() -> None:
    from antecedent import artifacts

    data = _data()
    prepared = PreparedAnalysis.prepare(
        data,
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=AverageEffect("t", "y"),
        refute="none",
        bootstrap=0,
    )
    prepared.estimate(data)
    contract = prepared.contract()
    encoded = prepared.export_contracted_artifact()
    loaded = artifacts.loads(encoded)
    assert loaded.payload_kind == "analysis_result"
    assert loaded.contract is not None
    accepted = artifacts.accept(encoded)
    assert accepted["accepts_as_verified_program"] == "true"
    assert accepted["program"] == contract["program"]
    assert accepted["target"] == contract["target"]
    for key in (
        "treatment",
        "outcome",
        "control",
        "active",
        "population",
        "temporal_coordinates",
        "variable_names",
    ):
        assert accepted[key] == contract[key], key
    assert accepted["treatment"] == "0"
    assert accepted["outcome"] == "1"
    assert accepted["control"] == "set:0=0"
    assert accepted["active"] == "set:0=1"
    assert accepted["population"] == "all_observed"
    assert accepted["temporal_coordinates"] == "none"
    assert accepted["variable_names"] == "t,y,z"
