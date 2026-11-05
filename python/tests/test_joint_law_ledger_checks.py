"""The propagation rule's dependence decisions from the evidence ledger and from real alignment.

Reference. Whether two inputs share evidence is a count of common observations; the ledger rows
registered here are the ranges ``[0, 50)``, ``[50, 100)``, ``[25, 75)`` and a source with no
computable rows. Disjoint ranges share nothing, ``[25, 75)`` shares ``25`` rows with each of the
others, and the unknown source's overlap cannot be decided, so a declared independence over it is
refused rather than accepted on distinct spellings. Row pairing is decided by alignment
identities: two columns of one joint artifact carry one identity and pair; columns of different
artifacts carry different identities and refuse; independently produced marginals and a summary
carry none and refuse. These are counts and equalities of identities, so no numerical tolerance
is involved.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from antecedent import _reason_details as reason_details
from antecedent import evidence_ledger as el
from antecedent import joint_law_propagation as jl
from antecedent.errors import CausalTypeError, CausalValueError

from test_joint_law_propagation import R, _artifact, _draws


def source(ledger, label, start, stop):
    return ledger.add_source(
        el.SourceDeclaration(
            label, "observation_set", observations=el.rows("study", range(start, stop))
        )
    )


def moments(first, second):
    return [jl.MomentInput("a", first, 1.0), jl.MomentInput("b", second, 2.0)]


def declared_independent(inputs, ledger=None):
    return jl.check_propagation(
        "sum", inputs, dependence="declared_independent", ledger=ledger
    )


def test_a_declared_independence_is_decided_from_the_rows_not_the_spellings():
    ledger = el.EvidenceLedger()
    first = source(ledger, "first", 0, 50)
    disjoint = source(ledger, "disjoint", 50, 100)
    overlapping = source(ledger, "overlapping", 25, 75)
    vendor = ledger.add_source(el.SourceDeclaration("vendor claim", "external_claim"))

    plan = declared_independent(moments(first, disjoint), ledger)
    assert plan.route == "linear_moments"
    assert plan.shared_evidence == ()
    assert plan.label == "point_only"

    # Every spelling differs, so without the ledger the independence is accepted ...
    assert declared_independent(moments(first, overlapping)).route == "linear_moments"
    # ... with it, the 25 shared rows refute it.
    with pytest.raises(R) as shared:
        declared_independent(moments(first, overlapping), ledger)
    assert shared.value.reason_code == "evidence_double_counted"
    assert shared.value.remedy

    # An overlap the ledger cannot compute is unknown dependence, never independence.
    with pytest.raises(R) as unknown:
        declared_independent(moments(first, vendor), ledger)
    assert unknown.value.reason_code == "sampling_dependence_unknown"
    assert unknown.value.detail == reason_details.joint_law_propagation.EVIDENCE_OVERLAP_UNKNOWN

    # A declared covariance does not need the ledger to be independent of the evidence.
    covariance = jl.check_propagation(
        "sum",
        moments(first, overlapping),
        dependence="covariance",
        covariance_id="cov-ab",
        ledger=ledger,
    )
    assert covariance.route == "linear_moments"
    assert covariance.covariance_id == "cov-ab"
    assert set(covariance.shared_evidence) == {first, overlapping}


def test_the_same_source_twice_is_one_source_counted_twice():
    ledger = el.EvidenceLedger()
    first = source(ledger, "first", 0, 50)
    with pytest.raises(R) as error:
        declared_independent(moments(first, first), ledger)
    assert error.value.reason_code == "evidence_double_counted"


def test_inputs_and_options_are_typed():
    ledger = el.EvidenceLedger()
    a = source(ledger, "a", 0, 10)
    b = source(ledger, "b", 10, 20)
    with pytest.raises(CausalTypeError, match="EvidenceLedger"):
        declared_independent(moments(a, b), ledger=object())
    with pytest.raises(CausalTypeError, match="PropagationDraws or MomentInput"):
        jl.check_propagation("sum", [object()], dependence="unknown")  # type: ignore[list-item]
    with pytest.raises(CausalValueError):
        jl.check_propagation("sum", moments(a, b), dependence="no_such_dependence")
    with pytest.raises(CausalValueError):
        jl.check_propagation("sum", moments(a, b), dependence="covariance")
    with pytest.raises(CausalValueError):
        jl.check_propagation("no_such_operation", moments(a, b))


def test_rows_pair_only_under_one_alignment_identity():
    rng = np.random.default_rng(7)
    x, y = rng.standard_normal(500), rng.standard_normal(500)
    left = _artifact("snap-left", x, y)
    right = _artifact("snap-right", y, x)
    lx, ly = _draws(left)
    rx, _ = _draws(right)
    rx = dataclasses.replace(rx, id="right_x")
    marginals = _artifact("snap-marginals", x, y, alignment="independent_marginals")
    mx, my = _draws(marginals)

    # Two columns of one artifact share one identity and pair.
    assert jl.check_rows_pairable([lx, ly]) is None
    plan = jl.check_propagation(
        "nonlinear_functional", [lx, ly], dependence="aligned_joint_law"
    )
    assert plan.route == "aligned_joint_law"
    assert plan.alignment_id == lx.alignment_id == ly.alignment_id

    # Columns of different artifacts have different identities.
    assert lx.alignment_id != rx.alignment_id
    with pytest.raises(R) as mismatch:
        jl.check_rows_pairable([lx, rx])
    assert mismatch.value.reason_code == "joint_law_required"
    assert mismatch.value.detail == reason_details.joint_law_propagation.ALIGNMENT_MISMATCH
    with pytest.raises(R):
        jl.check_propagation("nonlinear_functional", [lx, rx], dependence="aligned_joint_law")

    # Independently produced marginals and a summary have nothing to pair.
    for unpaired in ([mx, my], [lx, my], [lx, jl.MomentInput("summary", "snap-left", 1.0)]):
        with pytest.raises(R) as refused:
            jl.check_rows_pairable(unpaired)
        assert refused.value.reason_code == "joint_law_required"
        assert refused.value.detail == reason_details.joint_law_propagation.INDEX_PAIRING_REFUSED
    # A covariance or a declared independence is never a route to row pairing.
    with pytest.raises(R):
        jl.check_propagation(
            "nonlinear_functional", [mx, my], dependence="covariance", covariance_id="cov"
        )
