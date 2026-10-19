"""Python parity for plan inspection and execution-receipt facts.

The cases follow the Rust fixtures in ``crates/antecedent-core/src/inspect.rs``
and ``crates/antecedent-core/src/execution_facts.rs``. Integrity, semantic
verification, numerical replay, and resumability stay separate. Loading these
records does not resume a live study. Family interchange is not bound.
"""

from __future__ import annotations

import struct

import pytest
from antecedent import _native, errors

_LIMIT = (1 << 64) - 1


def _level_bits(level: float) -> int:
    return struct.unpack(">Q", struct.pack(">d", level))[0]


def _held(*, uncertainty: bool) -> _native.InspectionInputs:
    inputs = (
        _native.InspectionInputs()
        .with_static_mean_target()
        .with_assumptions(["positivity", "consistency"])
        .with_realization("score_frequentist")
        .with_evidence(["standardize"])
        .with_reuse("frozen_nuisance", True)
        .with_materialization("design_matrix", 4096)
        .with_resources(10, 20, 30)
        .with_backend("scalar_only")
        .with_read_operators([(7, 1, 2, 3)])
    )
    if uncertainty:
        inputs = inputs.with_uncertainty(True, 0.95)
    return inputs


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


def test_complete_inspection_round_trips_fields():
    held = _held(uncertainty=True)
    before = held.target_identity()
    plan = held.inspect()
    assert before is not None
    assert plan.target_identity() == before
    assert plan.target_inference() == "frequentist"
    assert plan.assumptions() == ["positivity", "consistency"]
    assert plan.realization() == "score_frequentist"
    assert plan.evidence() == ["standardize"]
    assert plan.uncertainty_joint() is True
    assert plan.uncertainty_level() == 0.95
    assert plan.uncertainty_level_bits() == _level_bits(0.95)
    assert plan.reuse_name() == "frozen_nuisance"
    assert plan.reuse_frozen_fit() is True
    assert plan.materialization_name() == "design_matrix"
    assert plan.materialization_copy_bytes() == 4096
    assert plan.memory_soft_bytes() == 10
    assert plan.memory_hard_bytes() == 20
    assert plan.estimated_bytes() == 30
    assert plan.backend() == "scalar_only"
    assert plan.operators() == [("read", [7], 1, 2, 3)]
    assert plan.standing().semantic_verification() == before


def test_missing_uncertainty_demand_is_refused():
    with pytest.raises(
        errors.CausalUnsupportedError, match="uncertainty demand is missing"
    ) as caught:
        _held(uncertainty=False).inspect()
    assert type(caught.value) is errors.CausalUnsupportedError

    with pytest.raises(errors.CausalUnsupportedError, match="scientific target is missing"):
        _native.InspectionInputs().inspect()
    with pytest.raises(errors.CausalUnsupportedError, match="declaration name must be non-blank"):
        _held(uncertainty=True).with_evidence([" "])
    with pytest.raises(errors.CausalValueError, match="unknown assumption"):
        _held(uncertainty=True).with_assumptions(["not_an_assumption"])


def test_inspect_accepts_no_fit_or_scan_callback():
    called: list[str] = []

    def callback() -> None:
        called.append("called")

    plan = _held(uncertainty=True).inspect()
    assert plan.realization() == "score_frequentist"
    with pytest.raises(
        errors.CausalUnsupportedError, match="does not resume a live study"
    ) as caught:
        _native.resume_live_study(callback)
    assert called == []
    assert type(caught.value) is errors.CausalUnsupportedError


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
    assert record.physical_identity() == _facts(
        retention="disk_backed", backend="portable", kernel="scalar_only"
    ).physical_identity()


def test_overflow_refuses():
    one = _counts(1, 1, 1, 1, 1)
    for scans, copies, fits, tests, hits in (
        (_LIMIT, 0, 0, 0, 0),
        (0, _LIMIT, 0, 0, 0),
        (0, 0, _LIMIT, 0, 0),
        (0, 0, 0, _LIMIT, 0),
        (0, 0, 0, 0, _LIMIT),
    ):
        with pytest.raises(errors.CausalUnsupportedError, match="overflowed u64"):
            _counts(scans, copies, fits, tests, hits).try_add(one)

    unchanged = _counts(_LIMIT, 1, 2, 3, 4)
    added = unchanged.try_add(_counts(0, 0, 0, 0, 0))
    assert added.scans() == _LIMIT
    assert added.copies() == 1
    assert unchanged.scans() == _LIMIT

    spill = _resources(1, 2, _LIMIT)
    with pytest.raises(errors.CausalUnsupportedError, match="overflowed u64"):
        spill.try_add(_resources(0, 0, 1))
    assert spill.spill_bytes() == _LIMIT

    digest = bytes([3]) * 32
    record = _facts(counts=_counts(_LIMIT, 0, 0, 0, 0), resources=spill, scientific=digest)
    with pytest.raises(errors.CausalUnsupportedError, match="overflowed u64") as caught:
        record.try_add_counts(_counts(1, 0, 0, 0, 0))
    assert type(caught.value) is errors.CausalUnsupportedError
    assert record.counts().scans() == _LIMIT
    assert record.scientific_identity() == digest
    with pytest.raises(errors.CausalUnsupportedError, match="overflowed u64"):
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
    failed = flagged.rebuild(
        _counts(), "summary_only", _resources(), "cpu", "default", "failed"
    )
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


def test_standings_distinguish_integrity_semantic_verification_numerical_replay_and_resumability():
    plan = _held(uncertainty=True).inspect()
    inspected = plan.standing()
    assert inspected.semantic_verification() == plan.target_identity()
    with pytest.raises(errors.CausalUnsupportedError, match="payload checksum") as checksum:
        inspected.integrity()
    assert type(checksum.value) is errors.CausalUnsupportedError
    with pytest.raises(errors.CausalUnsupportedError, match="does not numerically replay"):
        inspected.numerical_replay()
    with pytest.raises(errors.CausalUnsupportedError, match="does not resume a live study"):
        inspected.resumability()

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
    assert replay.invoked_callback() is False
    assert replay.reexecuted() is False
    with pytest.raises(
        errors.CausalUnsupportedError, match="does not resume a live study"
    ) as resume:
        standing.resumability()
    assert type(resume.value) is errors.CausalUnsupportedError
    assert facts.standing().semantic_verification() == digest
    bare = _facts(retention="disk_backed", backend="portable", kernel="scalar_only")
    assert bare.standing().semantic_verification() is None
    assert bare.standing().integrity() != standing.integrity()


def test_family_interchange_consumer_is_refused():
    with pytest.raises(errors.CausalUnsupportedError, match="no bound consumer") as caught:
        _native.consume_interchange_family("adjustment")
    assert type(caught.value) is errors.CausalUnsupportedError
    assert "adjustment" in str(caught.value)
    with pytest.raises(errors.CausalValueError, match="32 bytes"):
        _facts(scientific=b"\x03" * 31)
