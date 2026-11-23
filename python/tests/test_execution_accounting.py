"""Execution facts keep measured counts, admission reservations and plan estimates apart."""

from antecedent import _native


def _facts() -> _native.ExecutionFacts:
    return _native.ExecutionFacts(
        _native.ExecutionCounts(2, 3, 5, 7, 11),
        "full_reusable",
        _native.ExecutionResources(4, 1024, 64),
        "cpu",
        "default",
        "completed",
        None,
        False,
    )


def test_a_caller_declared_record_reports_no_measurement():
    facts = _facts()
    assert not facts.is_native
    assert facts.observed_allocations is None
    assert facts.observed_allocated_bytes is None
    by_name = {name: (basis, value) for name, basis, value in facts.accounting()}
    for name in ("scans", "copies", "copied_bytes", "allocations", "allocated_bytes"):
        assert by_name[name] == ("measured", None), name
    for name in ("peak_admitted_bytes", "retained_output_bytes"):
        assert by_name[name][0] == "admission_reservation", name
        assert by_name[name][1] is None, name


def test_plan_estimates_are_labelled_as_estimates_and_never_fill_a_measurement():
    accounting = _facts().accounting(
        planned_peak_memory_bytes=4096, planned_copy_bytes=512, planned_worker_threads=2
    )
    by_name = {name: (basis, value) for name, basis, value in accounting}
    assert by_name["planned_peak_memory_bytes"] == ("estimate", 4096)
    assert by_name["planned_copy_bytes"] == ("estimate", 512)
    assert by_name["planned_workspace_bytes"] == ("estimate", None)
    assert by_name["planned_worker_threads"] == ("estimate", 2)
    assert by_name["copies"] == ("measured", None)
    bases = {basis for _, basis, _ in accounting}
    assert bases == {"measured", "admission_reservation", "estimate"}
