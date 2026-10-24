"""Python parity for plan inspection and execution-receipt facts.

The static-mean inspection fixture lives in ``crates/antecedent-core/src/inspect.rs``.
Execution-receipt cases follow ``crates/antecedent-core/src/execution_facts.rs``.
Physical identity, the stored scientific digest, and the numerical record stay separate.
"""

from __future__ import annotations

import pytest
from antecedent import _native, errors

_LIMIT = (1 << 64) - 1


def _counts(
    scans: int = 2,
    copies: int = 3,
    fits: int = 5,
    tests: int = 7,
    cache_hits: int = 11,
) -> _native.ExecutionCounts:
    return _native.ExecutionCounts(scans, copies, fits, tests, cache_hits)


def _resources(
    threads: int = 4, memory_bytes: int = 1024, spill_bytes: int = 64
) -> _native.ExecutionResources:
    return _native.ExecutionResources(threads, memory_bytes, spill_bytes)


def _facts(
    *,
    counts: _native.ExecutionCounts | None = None,
    retention: str = "full_reusable",
    resources: _native.ExecutionResources | None = None,
    backend: str = "cpu",
    kernel: str = "default",
    status: str = "completed",
    scientific: bytes | None = None,
    contract: bool = False,
) -> _native.ExecutionFacts:
    return _native.ExecutionFacts(
        _counts() if counts is None else counts,
        retention,
        _resources() if resources is None else resources,
        backend,
        kernel,
        status,
        scientific,
        contract,
    )


def test_missing_target_and_blank_declarations_are_refused():
    with pytest.raises(errors.CausalValueError, match="scientific target is missing"):
        _native.InspectionInputs().inspect()
    with pytest.raises(errors.CausalValueError, match="declaration name must be non-blank"):
        _native.InspectionInputs().with_evidence([" "])
    with pytest.raises(errors.CausalValueError, match="unknown assumption"):
        _native.InspectionInputs().with_assumptions(["not_an_assumption"])


def test_counts_record_what_was_passed():
    record = _facts(
        retention="disk_backed", backend="portable", kernel="scalar_only", scientific=None
    )
    assert record.counts().scans() == 2
    assert record.counts().copies() == 3
    assert record.counts().fits() == 5
    assert record.counts().tests() == 7
    assert record.counts().cache_hits() == 11
    assert record.resources().threads() == 4
    assert record.resources().memory_bytes() == 1024
    assert record.resources().spill_bytes() == 64
    assert record.retention() == "disk-backed output"
    assert record.backend_name() == "portable"
    assert record.kernel() == "scalar_only"
    assert record.status() == "completed"
    assert record.scientific_identity() is None
    assert record.scientific_contract_changed() is False
    assert record.publishes_claim() is True
    assert len(record.physical_identity()) == 32
    assert (
        record.physical_identity()
        == _facts(
            retention="disk_backed", backend="portable", kernel="scalar_only"
        ).physical_identity()
    )


def test_overflow_refuses():
    one = _counts(1, 1, 1, 1, 1)
    for scans, copies, fits, tests, hits in (
        (_LIMIT, 0, 0, 0, 0),
        (0, _LIMIT, 0, 0, 0),
        (0, 0, _LIMIT, 0, 0),
        (0, 0, 0, _LIMIT, 0),
        (0, 0, 0, 0, _LIMIT),
    ):
        with pytest.raises(errors.CausalValueError, match="overflowed u64"):
            _counts(scans, copies, fits, tests, hits).try_add(one)

    unchanged = _counts(_LIMIT, 1, 2, 3, 4)
    added = unchanged.try_add(_counts(0, 0, 0, 0, 0))
    assert added.scans() == _LIMIT
    assert added.copies() == 1
    assert unchanged.scans() == _LIMIT

    spill = _resources(1, 2, _LIMIT)
    with pytest.raises(errors.CausalValueError, match="overflowed u64"):
        spill.try_add(_resources(0, 0, 1))
    assert spill.spill_bytes() == _LIMIT

    digest = bytes([3]) * 32
    record = _facts(counts=_counts(_LIMIT, 0, 0, 0, 0), resources=spill, scientific=digest)
    with pytest.raises(errors.CausalValueError, match="overflowed u64") as caught:
        record.try_add_counts(_counts(1, 0, 0, 0, 0))
    assert type(caught.value) is errors.CausalValueError
    assert record.counts().scans() == _LIMIT
    assert record.scientific_identity() == digest
    with pytest.raises(errors.CausalValueError, match="overflowed u64"):
        record.try_add_resources(_resources(0, 0, 1))
    assert record.resources().spill_bytes() == _LIMIT
    assert record.scientific_identity() == digest


def test_cancelled_status_does_not_publish():
    digest = bytes([5]) * 32
    completed = _facts(scientific=digest)
    assert completed.publishes_claim() is True
    cancelled = completed.rebuild(
        _counts(), "full_reusable", _resources(), "cpu", "default", "cancelled"
    )
    assert cancelled.status() == "cancelled"
    assert cancelled.publishes_claim() is False
    assert cancelled.counts().scans() == 2
    assert cancelled.counts().copies() == 3
    assert cancelled.scientific_identity() == digest
    flagged = cancelled.declare_numerical_policy_changes_scientific_contract(True)
    assert flagged.scientific_contract_changed() is True
    assert flagged.publishes_claim() is False
    failed = flagged.rebuild(_counts(), "summary_only", _resources(), "cpu", "default", "failed")
    assert failed.publishes_claim() is False
    assert failed.status() == "failed"
    assert failed.retention() == "summary-only"
    assert failed.scientific_identity() == digest
    assert failed.scientific_contract_changed() is True


def test_scan_count_change_leaves_scientific_digest_unchanged():
    digest = bytes([7]) * 32
    original = _facts(scientific=digest)
    rebuilt = original.rebuild(
        _counts(scans=9), "full_reusable", _resources(), "cpu", "default", "completed"
    )
    assert rebuilt.counts().scans() == 9
    assert rebuilt.scientific_identity() == digest
    assert original.scientific_identity() == rebuilt.scientific_identity()
    assert rebuilt.physical_identity() != original.physical_identity()
    other = bytes([8]) * 32
    replaced = original.with_scientific_identity(other)
    assert replaced.scientific_identity() == other
    assert replaced.physical_identity() == original.physical_identity()
    assert replaced.counts().scans() == original.counts().scans()


def test_numerical_policy_flag_is_the_only_scientific_contract_change():
    digest = bytes([9]) * 32
    original = _facts(scientific=digest)
    assert original.scientific_contract_changed() is False
    rebuilt = original.rebuild(
        _counts(40, 1, 1, 1, 1),
        "summary_only",
        _resources(8, 2048, 128),
        "other",
        "scalar_only",
        "cancelled",
    )
    assert rebuilt.kernel() != original.kernel()
    assert rebuilt.counts().scans() != original.counts().scans()
    assert rebuilt.retention() != original.retention()
    assert rebuilt.resources().spill_bytes() != original.resources().spill_bytes()
    assert rebuilt.publishes_claim() is False
    assert rebuilt.scientific_contract_changed() is False
    assert rebuilt.scientific_identity() == digest
    assert rebuilt.physical_identity() != original.physical_identity()
    flagged = rebuilt.declare_numerical_policy_changes_scientific_contract(True)
    assert flagged.scientific_contract_changed() is True
    assert flagged.scientific_identity() == digest
    assert flagged.physical_identity() == rebuilt.physical_identity()
    cleared = flagged.declare_numerical_policy_changes_scientific_contract(False)
    assert cleared.scientific_contract_changed() is False
    assert cleared.scientific_identity() == digest
    explicit = _facts(scientific=digest, contract=True)
    assert explicit.scientific_contract_changed() is True
    assert original.scientific_contract_changed() is False
    assert explicit.physical_identity() == original.physical_identity()
    assert explicit.scientific_identity() == original.scientific_identity()


def test_execution_standing_keeps_integrity_apart_from_the_scientific_digest():
    digest = bytes([7]) * 32
    facts = _facts(scientific=digest)
    standing = facts.standing()
    assert standing.integrity() == facts.physical_identity()
    assert standing.semantic_verification() == digest
    replay = standing.numerical_replay()
    assert replay.counts().scans() == facts.counts().scans()
    assert replay.resources().spill_bytes() == facts.resources().spill_bytes()
    assert replay.retention() == facts.retention()
    assert replay.backend_name() == facts.backend_name()
    assert replay.kernel() == facts.kernel()
    assert replay.status() == facts.status()
    assert replay.contract_changed() is False
    assert facts.standing().semantic_verification() == digest
    bare = _facts(retention="disk_backed", backend="portable", kernel="scalar_only")
    assert bare.standing().semantic_verification() is None
    assert bare.standing().integrity() != standing.integrity()


def test_scientific_identity_must_be_32_bytes():
    with pytest.raises(errors.CausalValueError, match="32 bytes"):
        _facts(scientific=b"\x03" * 31)
