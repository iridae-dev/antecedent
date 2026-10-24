"""The host-integration contract: schema drift, stable codes, and Rust/Python parity.

The schema is generated from the registries and Rust enums that own each vocabulary.
These tests fail when the committed file drifts from its generator, when the installed
package disagrees with it, when a refusal code or exception family is not registered,
or when Python and Rust would classify the same failure differently.
"""

from __future__ import annotations

import ast
import inspect
import subprocess
import sys

import antecedent
import pytest
from antecedent import _native, errors, host

from _repo_text import REPO_ROOT, load_json, load_toml, read_text

SCHEMA_PATH = REPO_ROOT / "crates" / "antecedent" / "schemas" / "host_contract.schema.json"
REGISTRY = load_toml(REPO_ROOT / "parity" / "reason_codes.toml")["code"]
PACKAGE = REPO_ROOT / "python" / "antecedent"


def test_the_committed_schema_matches_its_generator():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "generate_host_schema.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_the_package_serves_the_committed_bytes():
    assert host.schema_text() == read_text(SCHEMA_PATH)
    assert host.schema() == load_json(SCHEMA_PATH)
    assert load_json(SCHEMA_PATH)["x-contract-version"] == host.CONTRACT_VERSION


def test_refusal_codes_are_the_registered_runtime_refusals():
    registered = {row["id"] for row in REGISTRY if "runtime_refusal" in row.get("applies_to", [])}
    assert host.refusal_codes() == registered
    assert host.refusal_codes() == frozenset(_native.runtime_refusal_codes())
    assert host.reason_codes() == {row["id"] for row in REGISTRY}


def _code_literals(source: str) -> list[str]:
    """String constants given as ``reason_code=`` / ``code=`` or under a ``"code"`` key."""
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.keyword) and node.arg in {"reason_code", "code"}:
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                found.append(node.value.value)
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if (
                    isinstance(key, ast.Constant)
                    and key.value in {"code", "reason_code"}
                    and isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                ):
                    found.append(value.value)
    return found


def test_every_python_exception_reason_is_a_registered_refusal_code():
    roots = [PACKAGE, REPO_ROOT / "examples" / "python"]
    unregistered = {
        (str(path.relative_to(REPO_ROOT)), code)
        for root in roots
        for path in sorted(root.rglob("*.py"))
        for code in _code_literals(read_text(path))
        if code not in host.refusal_codes()
    }
    assert not unregistered, sorted(unregistered)


def test_an_unregistered_reason_code_cannot_be_raised():
    with pytest.raises(ValueError, match="unregistered runtime reason code"):
        errors.CausalUnsupportedError("refused", reason_code="no_such_code_" + "anywhere")


def _exception_classes() -> list[type[BaseException]]:
    classes: dict[str, type[BaseException]] = {}
    for module in (errors, _native):
        for _, value in inspect.getmembers(module, inspect.isclass):
            if issubclass(value, antecedent.CausalError):
                classes[f"{value.__module__}.{value.__qualname__}"] = value
    return list(classes.values())


def test_every_exception_family_is_registered_and_resolves():
    families = {family.python_class: family for family in host.error_families()}
    codes = {family.code for family in host.error_families()}
    assert len(codes) == len(host.error_families())
    # Every family the registry names exists, with the parent the registry states.
    for family in host.error_families():
        cls = getattr(errors, family.python_class, None) or getattr(_native, family.python_class)
        assert issubclass(cls, antecedent.CausalError)
        if family.parent is not None:
            parent = next(f for f in host.error_families() if f.code == family.parent)
            assert issubclass(cls, getattr(errors, parent.python_class))
        if family.narrows is not None:
            assert family.narrows in codes
    # Every exception class the package defines classifies to a registered family.
    for cls in _exception_classes():
        assert host.error_family(cls) in codes, cls
    assert {"CausalError", "CausalCancelledError", "CausalUnsupportedError"} <= set(families)


def test_exception_families_classify_like_the_rust_boundary():
    expected = {
        "CausalIdentifyError": "identify",
        "CausalEstimateError": "estimate",
        "CausalDataError": "data",
        "CausalCompileError": "compile",
        "CausalResourceError": "resource",
        "CausalCancelledError": "cancelled",
        "CausalUnsupportedError": "unsupported",
        "CausalSerializationError": "serialization",
        "CausalReviewError": "review_required",
        "EffectNotIdentified": "not_identified",
        "CausalTypeError": "validate",
        "CausalValueError": "validate",
        "StructuredRefusal": "unsupported",
        "ReviewRequired": "review_required",
    }
    for name, family in expected.items():
        cls = getattr(errors, name, None) or getattr(_native, name)
        assert host.error_family(cls) == family, name
    assert host.error_family(ValueError("not ours")) == "error"


def test_a_refusal_record_carries_a_registered_code_and_its_family():
    error = errors.CausalUnsupportedError(
        "the attestation cannot be re-verified", reason_code="attested_not_reverifiable"
    )
    record = host.refusal_record(error)
    assert record["status"] == "unsupported"
    assert record["family"] == "unsupported"
    assert record["reason_code"] == "attested_not_reverifiable"
    assert "reason=attested_not_reverifiable" in record["message"]
    uncoded = host.refusal_record(errors.CausalResourceError("scratch budget exceeded"))
    assert uncoded["reason_code"] is None
    assert uncoded["family"] == "resource"


@pytest.mark.parametrize(
    ("claim", "status"),
    [
        (("point", "identified", "supported", "evaluated"), "supported"),
        (("point", "identified", "unknown", "evaluated"), "supported"),
        (("response", "identified", "supported", "evaluated"), "supported"),
        (("bounds", "identified", "supported", "evaluated"), "partial"),
        (("incomplete", "unknown", "unknown", "unknown"), "partial"),
        (("point", "unknown", "supported", "evaluated"), "partial"),
        (("point", "identified", "supported", "unknown"), "partial"),
        (("associational", "not_claimed", "unknown", "evaluated"), "supported"),
        (("associational", "not_claimed", "supported", "evaluated"), "supported"),
        (("associational", "not_claimed", "unknown", "unknown"), "partial"),
        (("associational", "identified", "supported", "evaluated"), "partial"),
        (("associational", "not_claimed", "contradicted", "evaluated"), "unsupported"),
        (("point", "not_claimed", "supported", "evaluated"), "partial"),
        (("refusal", "unknown", "unknown", "unknown"), "unsupported"),
        (("point", "identified", "contradicted", "evaluated"), "unsupported"),
        (("point", "identified", "outside_scope", "evaluated"), "unsupported"),
        (("point", "identified", "supported", "not_a_status"), "unsupported"),
    ],
)
def test_claim_status_separates_supported_partial_and_unsupported(claim, status):
    kind, identification, support, evaluated = claim
    wire = {
        "kind": kind,
        "identification_domain": identification,
        "support_domain": support,
        "evaluated_domain": evaluated,
    }
    record = {
        "kind": kind,
        "domains": {
            "identification": identification,
            "support": support,
            "evaluated": evaluated,
        },
    }
    assert host.claim_status(wire) == status
    assert host.claim_status(record) == status


def test_an_associational_record_is_noncausal_and_a_point_record_is_causal():
    association = {
        "kind": "associational",
        "domains": {
            "identification": "not_claimed",
            "support": "unknown",
            "evaluated": "evaluated",
        },
    }
    record = host.claim_record(association)
    assert record["causal"] is False
    assert record["status"] == "supported"
    point = {
        "kind": "point",
        "domains": {"identification": "identified", "support": "unknown", "evaluated": "evaluated"},
    }
    assert host.claim_record(point)["causal"] is True


def test_operation_status_follows_applicability_then_readiness():
    assert host.operation_status("licensed", "executable") == "supported"
    for readiness in (
        "review_required",
        "assumption_missing",
        "empirical_check_pending",
        "binding_missing",
        "runtime_unavailable",
        "unknown_support",
    ):
        assert host.operation_status("licensed", readiness) == "partial", readiness
    assert host.operation_status("licensed", "empirical_support_failed") == "unsupported"
    assert host.operation_status("unlicensed", "executable") == "unsupported"
    assert host.operation_status("impossible") == "unsupported"
    assert host.operation_status("unknown") == "partial"
    assert host.operation_status("not_a_status") == "unsupported"


def test_the_schema_lists_the_statuses_the_rule_returns():
    statuses = set(host.schema()["$defs"]["host_status"]["enum"])
    assert statuses == {"supported", "partial", "unsupported"}
