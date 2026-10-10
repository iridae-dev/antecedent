"""Typed support labels retain causal error semantics across result constructors."""

import json
from typing import get_args

import pytest
from antecedent.errors import CausalValueError
from antecedent.results.response import SupportReport, SupportStatus
from pydantic import ValidationError


@pytest.mark.parametrize("field", ["status", "point_status"])
@pytest.mark.parametrize("invalid", ["maybe", 7, ["supported"]])
@pytest.mark.parametrize("entry", ["constructor", "python", "json"])
def test_invalid_support_labels_preserve_causal_errors(field, invalid, entry):
    data = {"status": "supported", "query_region": {}}
    data[field] = [invalid] if field == "point_status" else invalid
    with pytest.raises(CausalValueError, match="unknown support status"):
        if entry == "constructor":
            SupportReport(**data)
        elif entry == "python":
            SupportReport.model_validate(data)
        else:
            SupportReport.model_validate_json(json.dumps(data))


def test_string_validation_preserves_causal_errors():
    with pytest.raises(CausalValueError, match="unknown support status"):
        SupportReport.model_validate_strings({"status": "maybe", "query_region": {}})


def test_support_labels_have_enum_schema_and_json_round_trip():
    schema = SupportReport.model_json_schema()["properties"]
    labels = list(get_args(SupportStatus))
    assert schema["status"]["enum"] == labels
    assert schema["point_status"]["anyOf"][0]["items"]["enum"] == labels
    for status in labels:
        report = SupportReport(status, {"treatment": (0.0, 1.0)}, point_status=labels)
        restored = SupportReport.model_validate_json(report.model_dump_json())
        assert restored.status == status
        assert list(restored.point_status) == labels
        assert restored.to_dict() == report.to_dict()


def test_schema_errors_are_not_misreported_as_causal_domain_errors():
    with pytest.raises(ValidationError, match="query_region"):
        SupportReport.model_validate({"status": "supported"})
    with pytest.raises(ValidationError, match="Invalid JSON"):
        SupportReport.model_validate_json("not JSON")
