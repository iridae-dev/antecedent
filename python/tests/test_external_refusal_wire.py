"""The structured external-refusal wire carries ``capability`` from every binding.

The decision, composition, composition-bundle, design-ranking and inverse-query bindings
once wrote seven refusal fields and silently dropped the missing operation, so
``ExternalRefusal.capability`` was always ``None`` whatever the native refusal said. They now
write the same ``RefusalWire`` the program-claims binding does.
"""

from __future__ import annotations

import json

from antecedent import _native
from antecedent.external import ExternalRefusal

WIRE_FIELDS = {
    "code",
    "stage",
    "detail",
    "offending",
    "expected",
    "supplied",
    "capability",
    "remedy",
}


def _refusal_from_native() -> dict:
    value, refusal = _native.decision_contract_normalize("{")
    assert value is None
    assert refusal is not None
    return json.loads(refusal)


def test_a_decision_binding_refusal_carries_the_capability_field() -> None:
    wire = _refusal_from_native()
    assert set(wire) == WIRE_FIELDS
    assert "capability" in wire
    assert wire["capability"] is None  # a malformed declaration names no missing operation


def test_the_python_refusal_reads_the_capability_from_the_wire() -> None:
    wire = _refusal_from_native()
    assert ExternalRefusal(wire).capability is None
    named = {**wire, "code": "external_capability_missing", "capability": "cdf"}
    assert ExternalRefusal(named).capability == "cdf"
