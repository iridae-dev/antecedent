"""Readable 2.3 results: explain / to_dict / repr, results re-exports, identity keyword."""

from __future__ import annotations

import json

import pytest
from antecedent import repair, results
from antecedent.errors import CausalTypeError, CausalValueError

from test_external_binding import _response, _spec_with_premises
from test_repair import backdoor, candidates

# ------------------------------------------------------------------ results re-exports


@pytest.mark.parametrize(
    "name",
    [
        "SupportReport",
        "SupportStatus",
        "SupportDiagnostic",
        "ResponseView",
        "ResponseUncertainty",
        "UncertaintyKind",
        "IntervalInterpretation",
        "response_coordinates",
    ],
)
def test_response_support_surface_is_exported_from_results(name):
    assert name in results.__all__
    assert hasattr(results, name)


def test_results_all_has_no_duplicates_and_resolves():
    assert len(results.__all__) == len(set(results.__all__))
    for name in results.__all__:
        assert hasattr(results, name), name


def test_response_coordinates_helper_is_the_coordinates_module_function():
    from antecedent.results import coordinates

    assert results.response_coordinates is coordinates.response_coordinates


# ------------------------------------------------------------------ RepairResult


def test_repair_result_is_readable():
    result = repair.repair(backdoor(), candidates())
    assert repr(result).startswith("<RepairResult backdoor outcome=")
    text = result.explain()
    assert result.family in text
    assert "not an estimate" in text
    assert "'none'" in text  # the inference claim is always none
    data = result.to_dict()
    assert data["outcome"] == result.outcome
    assert data["family"] == "backdoor"
    assert data["inference_claim"] == "none"
    assert len(data["table"]) == len(result.table)
    assert len(data["ranked"]) == len(result.ranked)
    assert data["obligations"][0]["kind"] == "provide_joint_law"
    json.dumps(data)  # JSON-safe


def test_repair_rows_have_short_reprs_and_dicts():
    result = repair.repair(backdoor(), candidates())
    row = result.table[0]
    assert repr(row).startswith("<CandidateOutcome ")
    assert row.to_dict()["classification"] == row.classification
    assert result.best is not None
    assert result.best.to_dict()["derivation"]["verified"] is True


def test_repair_result_survives_export_and_consume():
    result = repair.repair(backdoor(), candidates())
    again = repair.consume(result.export())
    assert again.to_dict() == result.to_dict()


# ------------------------------------------------------------------ external claim


def test_bound_claim_is_readable_and_carries_its_trust_label():
    claim = _spec_with_premises().bind(_response())
    assert repr(claim).startswith("<BoundExternalClaim ")
    text = claim.explain()
    assert "externally_attested" in text
    assert "natively estimated: False" in text
    assert "missing_evidence" in text
    data = claim.to_dict()
    assert data["provider_trust"] == "externally_attested"
    assert data["native"] is False
    assert data["values"] == [1.0, 3.0, 5.0]
    assert data["lineage"][-1]["id"] == "claim"
    json.dumps(data)


def test_claim_inspection_is_readable():
    inspection = _spec_with_premises().bind(_response()).inspect()
    assert repr(inspection).startswith("<ClaimInspection ")
    assert "externally_attested" in inspection.explain()
    assert inspection.to_dict()["provider_trust"] == "externally_attested"


# ------------------------------------------------------------------ identity convention


def test_claim_identity_is_a_string_and_fields_are_structured():
    claim = _spec_with_premises().bind(_response())
    assert isinstance(claim.identity, str)
    fields = claim.identity_fields
    assert isinstance(fields, dict)
    assert fields["provider_id"] == "lab"
    assert json.loads(claim.identity) == fields
    assert claim.identity == _spec_with_premises().bind(_response()).identity


def test_load_takes_expected_identity_as_string_or_mapping():
    spec = _spec_with_premises()
    claim = spec.bind(_response())
    data = claim.export()
    assert spec.load(data, expected_identity=claim.identity).values.tolist() == [1.0, 3.0, 5.0]
    assert spec.load(data, expected_identity=claim.identity_fields).identity == claim.identity
    with pytest.raises(TypeError, match="expected"):
        spec.load(data, expected=claim.identity)  # type: ignore[call-arg]
    with pytest.raises(CausalTypeError):
        spec.load(data, expected_identity=3)  # type: ignore[arg-type]
    with pytest.raises(CausalValueError):
        spec.load(data, expected_identity="not json")
    changed = {**claim.identity_fields, "snapshot_id": "other-snapshot"}
    with pytest.raises(Exception, match="differs"):
        spec.load(data, expected_identity=changed)


def test_consume_functions_use_expected_identity():
    import inspect

    from antecedent import (
        categorical_treatment,
        latent_class,
        nonlinear_mediation,
        temporal,
        temporal_counterfactual,
        vector_treatment,
    )

    for function in (
        categorical_treatment.consume_categorical_effects,
        latent_class.consume_latent_class_artifact,
        nonlinear_mediation.consume_mediation_artifact,
        temporal.consume_effect_constancy_artifact,
        temporal_counterfactual.consume_temporal_counterfactual_artifact,
        vector_treatment.consume_joint_effects,
    ):
        parameters = inspect.signature(function).parameters
        assert "expected_identity" in parameters, function.__name__
        assert "expected" not in parameters, function.__name__
