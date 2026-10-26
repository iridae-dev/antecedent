"""Canonical proof pages preserve native scientific records and owner coordinates."""

from __future__ import annotations

import gc
import json
from dataclasses import FrozenInstanceError

import antecedent as ant
import pytest
from antecedent.errors import (
    CausalStateError,
    CausalTypeError,
    CausalUnsupportedError,
    CausalValueError,
)
from antecedent.execution_attempt import observe_native_attempts
from antecedent.identify import PreparedIdentification
from antecedent.outcomes import Refusal
from antecedent.prepared import InspectionPage

from test_public_prepared_identification import _study


def _proof(view):
    if view.kind == "proof":
        return view
    for child in view.children():
        selected = _proof(child)
        if selected is not None:
            return selected
    return None


def _records(page):
    records = []
    while page is not None:
        records.extend(page.to_dict()["items"])
        page = page.next()
    return records


def test_canonical_detail_pages_preserve_known_adjustment_assumptions_and_derivation():
    proof = _proof(_study().identification)
    assert proof is not None
    wire = proof.to_dict()["identification"]
    estimands = proof.estimands(limit=1)
    assert isinstance(estimands, InspectionPage)
    assert estimands.total == len(wire["estimands"])
    assert estimands.items[0]["index"] == 0
    assert estimands.items[0]["method"] == wire["estimands"][0]["method"]
    assert estimands.items[0]["adjustment_count"] == 1
    assert estimands.items[0]["claim_status"] == wire["status"]
    adjustment = proof.adjustment(estimand=0, limit=1)
    assert adjustment.available and adjustment.reason is None
    assert adjustment.metadata["estimand"] == 0
    assert adjustment.items == ({"name": "z", "variable": 0, "offset": 0, "node": 0},)
    assumptions = proof.assumptions(limit=1)
    assert assumptions.metadata["scope"] == "proof"
    assert assumptions.metadata["status"] == wire["status"]
    assert _records(assumptions) == wire["required_assumptions"]
    assert assumptions.total > 0
    derivation = proof.derivation(limit=1)
    assert _records(derivation) == wire["derivation"]
    assert derivation.total > 0
    assert all(step["rule"] and step["detail"] for step in wire["derivation"])
    # Exact estimand selection is retained in the page's bound native reader.
    selected = proof.assumptions(estimand=0, limit=1)
    assert selected.metadata["scope"] in {"estimand", "estimand_shared_native_proof"}
    assert selected.metadata["estimand"] == 0
    assert selected.metadata["status"] == wire["status"]
    assert _records(selected) == wire["required_assumptions"]


def test_detail_page_reads_and_repr_do_no_fitting_or_full_materialization(monkeypatch):
    proof = _proof(_study().identification)
    assert proof is not None

    def forbidden(*args, **kwargs):
        pytest.fail("page inspection must not execute or materialize a full proof")

    monkeypatch.setattr(PreparedIdentification, "to_dict", forbidden)
    monkeypatch.setattr(ant.Execution, "_call", forbidden)

    def read():
        for page in (
            proof.estimands(limit=1),
            proof.adjustment(estimand=0, limit=1),
            proof.assumptions(limit=1),
            proof.derivation(limit=1),
        ):
            assert isinstance(page, InspectionPage)
            assert len(repr(page)) < 300
            assert len(page._repr_html_()) < 500
            assert len(page.items) <= 1
        return proof.inspect()

    observed = observe_native_attempts(read)
    observed.unwrap()
    for operation in ("glm_fit", "posterior_fit", "posterior_draw", "score_construction"):
        assert observed.report.operations[operation].attempted == 0


@pytest.mark.parametrize("method", ["estimands", "adjustment", "assumptions", "derivation"])
def test_detail_pages_preserve_native_order_and_lifetime_after_source_drop(method):
    study = _study()
    proof = _proof(study.identification)
    assert proof is not None
    selected = {"estimand": 0} if method in {"adjustment", "assumptions"} else {}
    expected = _records(getattr(proof, method)(limit=256, **selected))
    page = getattr(proof, method)(limit=1, **selected)
    with pytest.raises(FrozenInstanceError):
        page.limit = 2
    if page.items:
        with pytest.raises(TypeError, match="does not support item assignment"):
            page.items[0]["fabricated"] = True
    with pytest.raises(TypeError, match="does not support item assignment"):
        page.metadata["fabricated"] = True
    del study, proof
    gc.collect()
    assert _records(page) == expected


@pytest.mark.parametrize("method", ["estimands", "adjustment", "assumptions", "derivation"])
def test_composite_nodes_require_explicit_native_proof_selection(method):
    root = _study().identification
    with pytest.raises(CausalUnsupportedError, match="proof_selection_required") as caught:
        getattr(root, method)()
    assert caught.value.reason_code == "option_not_applicable"
    proof = _proof(root)
    assert proof is not None
    assert getattr(proof, method)().available


@pytest.mark.parametrize("method", ["estimands", "adjustment", "assumptions", "derivation"])
def test_detail_page_bounds_and_zero_length_navigation(method):
    proof = _proof(_study().identification)
    assert proof is not None
    read = getattr(proof, method)
    page = read(limit=1)
    assert page.total is not None and page.total > 0
    assert read(offset=page.total + 1).items == ()
    assert read(offset=page.total + 1).next() is None
    empty = read(limit=0)
    assert empty.items == ()
    with pytest.raises(CausalValueError, match="positive limit"):
        empty.next()
    assert empty.next(limit=1).items == page.items
    with pytest.raises(CausalValueError, match="at most 256"):
        read(limit=257)
    with pytest.raises(CausalTypeError, match="integer"):
        read(offset=True)
    with pytest.raises(CausalValueError, match="nonnegative"):
        read(offset=-1)


def test_native_ambiguity_and_unavailability_are_preserved_without_selection_guessing():
    # Protocol boundary fixture only: actual scientific records above come from
    # Rust. Native ambiguous-alternative semantics also have a Rust regression in
    # alternative_estimands_require_selection_and_empty_claim_vector_has_native_shared_semantics.
    calls = []
    original = CausalUnsupportedError("identification.estimand_selection_required")

    class Native:
        def adjustment_json(self, *, estimand, offset, limit):
            calls.append((estimand, offset, limit))
            if estimand is None:
                raise original
            return json.dumps(
                {
                    "available": False,
                    "reason": "temporal_namespace_not_retained",
                    "estimand": estimand,
                }
            )

        def assumptions_json(self, *, estimand, offset, limit):
            calls.append((estimand, offset, limit))
            return json.dumps(
                {
                    "offset": offset,
                    "total": 2,
                    "items": [{"index": offset}],
                    "next_offset": offset + 1 if offset == 0 else None,
                    "scope": "estimand",
                    "estimand": estimand,
                    "status": "native",
                }
            )

    proof = PreparedIdentification.from_native(Native(), ant.Execution())
    with pytest.raises(CausalUnsupportedError) as caught:
        proof.adjustment()
    assert caught.value is original
    absent = proof.adjustment(estimand=1)
    assert not absent.available and absent.total is None
    assert absent.reason == "temporal_namespace_not_retained"
    assert absent.metadata["estimand"] == 1
    assert absent.next() is None
    selected = proof.assumptions(estimand=1, limit=1)
    assert selected.next().metadata["estimand"] == 1
    assert calls[-2:] == [(1, 0, 1), (1, 1, 1)]
    for estimand in (True, -1):
        with pytest.raises((CausalTypeError, CausalValueError)):
            proof.adjustment(estimand=estimand)


def test_absent_proof_pages_preserve_original_refusal_and_missing_adapter_is_an_error():
    original = CausalUnsupportedError("original scientific refusal")
    refusal = Refusal(code="effect_not_identified", message=str(original), _error=original)
    view = PreparedIdentification.unavailable("preparation_refused", refusal=refusal)
    with pytest.raises(CausalUnsupportedError) as caught:
        view.assumptions()
    assert caught.value is original
    with pytest.raises(CausalStateError, match="no retained proof"):
        PreparedIdentification.unavailable("not_executed").derivation()
    with pytest.raises(CausalStateError, match="detail-page adapter"):
        PreparedIdentification.from_native(object(), ant.Execution()).estimands()
