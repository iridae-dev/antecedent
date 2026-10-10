"""Structured exceptions retain their concrete family and fields in transit."""

import copy
import pickle
import subprocess
import sys

import pytest
from antecedent.composition_bundle import CompositionBundleRefusal
from antecedent.errors import CausalValueError, StructuredRefusal
from antecedent.external import ExternalRefusal
from antecedent.repair import RepairRefusal


def test_errors_namespace_discovers_original_external_refusal_class():
    from antecedent import errors, external

    assert "ExternalRefusal" in dir(errors)
    assert errors.ExternalRefusal is external.ExternalRefusal


@pytest.mark.parametrize(
    "first", ["antecedent.errors", "antecedent.external", "antecedent.composition_bundle"]
)
def test_external_refusal_import_orders_have_no_cycle(first):
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                f"import {first}\n"
                "import antecedent.errors as errors\n"
                "import antecedent.external as external\n"
                "assert 'ExternalRefusal' in dir(errors)\n"
                "assert errors.ExternalRefusal is external.ExternalRefusal\n"
            ),
        ],
        check=True,
    )


@pytest.mark.parametrize("code", ["invalid_argument", "external_capability_missing"])
@pytest.mark.parametrize(
    "family", [StructuredRefusal, ExternalRefusal, CompositionBundleRefusal, RepairRefusal]
)
@pytest.mark.parametrize("method", ["copy", "deepcopy", "pickle"])
def test_refusal_reconstruction_retains_class_and_metadata(code, family, method):
    fields = {
        "code": code,
        "detail": "example.missing",
        "stage": "bind",
        "message": "the required input is absent",
        "offending": "coordinate[1]",
        "remedy": "supply the required input",
        "expected": "coordinate",
        "supplied": "none",
    }
    if family is RepairRefusal:
        original = family(
            fields["message"], reason_code=code, detail=fields["detail"], remedy=fields["remedy"]
        )
    else:
        original = family(fields)
    if method == "copy":
        restored = copy.copy(original)
    elif method == "deepcopy":
        restored = copy.deepcopy(original)
    else:
        restored = pickle.loads(pickle.dumps(original))
    assert isinstance(restored, family)
    assert type(restored) is type(original)
    assert isinstance(restored, CausalValueError) == (code == "invalid_argument")
    assert restored.args == original.args
    assert restored.__dict__ == original.__dict__
    assert restored.code == code
    assert str(restored) == str(original)
