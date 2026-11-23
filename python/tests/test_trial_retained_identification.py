"""Actual native trial proof, owner lifetime, and explicit JSON boundaries."""

import gc

import antecedent as ant
import numpy as np
import pytest
from antecedent import _native, transport
from antecedent.errors import CausalResourceError, CausalValueError
from antecedent.learners import Linear
from antecedent.outcomes import require_success
from antecedent.transport import advanced


def fixture(standardized=False):
    treatment = [False, False, True, True] * 30 + [False] * 80
    z = [-1.0, 1.0, -1.0, 1.0] * 30 + [-1.0, 1.0] * 40
    # A strided float64 input must be copied directly into the one owned vector.
    storage = np.empty(400)
    storage[::2] = z
    storage[1::2] = 777.0
    covariates = {"z": storage[::2]} if standardized else {}
    outcome = [
        z[i] + 2.0 * treatment[i] if standardized else 1.0 + 2.0 * treatment[i] for i in range(120)
    ] + [0.0] * 80
    data = transport.TrialAipwData(
        covariates,
        outcome,
        treatment,
        [True] * 120 + [False] * 80,
        [0.5] * 200,
        "independent_samples",
    )
    names = ["z", "a", "y"] if standardized else ["a", "y"]
    edges = [("z", "y"), ("a", "y")] if standardized else [("a", "y")]
    query = advanced.TrialAipwQuery(
        ant.Admg.from_edges(names, edges),
        advanced.SelectionDiagram("trial", "target", ["z"] if standardized else []),
        "a",
        "y",
    )
    study = ant.prepare(
        data,
        query=query,
        provider=transport.TrialAipw(outcome=Linear(), folds=3),
        inference=transport.TransportInference(bootstrap=0),
        execution=ant.Execution(seed=8),
    )
    return study, data


@pytest.mark.parametrize("standardized", [False, True])
def test_retained_trial_identification_preserves_actual_formula_and_coordinates(standardized):
    study, _ = fixture(standardized)
    proof = study.identification
    assert proof.kind == "transport"
    assert proof.status == "identified"
    assert proof.status_unavailable is None
    assert proof.child_count == 0
    assert proof.children() == ()
    formula = proof.estimands(limit=1).items[0]
    assert formula["kind"] == "transport_formula"
    assert formula["formula_kind"] == ("standardize" if standardized else "direct")
    assert formula["functional_arena"] is None
    assert [item["name"] for item in proof.adjustment().items] == (["z"] if standardized else [])
    assert proof.assumptions().total == proof.summary.metadata["premise_count"]
    assert proof.derivation().metadata["meaning"] == "checked_native_transport_certificate_premises"
    assert require_success(study.estimate()).estimate == pytest.approx(2.0, abs=1e-10)


def test_trial_proof_inspection_never_materializes_fits_or_reprepares(monkeypatch):
    from antecedent._prepared_identification import PreparedIdentification
    from antecedent.estimation import _PreparedEngine as Engine

    study, _ = fixture(True)

    def forbidden(*args, **kwargs):
        raise AssertionError("proof reading executed or materialized scientific work")

    with monkeypatch.context() as context:
        context.setattr(PreparedIdentification, "to_dict", forbidden)
        context.setattr(Engine, "estimate", forbidden)
        context.setattr(_native, "prepare_learned_trial", forbidden)
        proof = study.identification
        assert proof.inspect().status == "identified"
        assert "transport" in repr(proof)
        assert "identified" in proof._repr_html_()
        assert proof.adjustment(limit=1).items[0]["name"] == "z"
        assert study.assumption_obligations().items[0]["id"] == "transport.trial_aipw"
        assert not study.uncertainty_components().available
        assert not study.evidence_obligations().available
        assert (
            study.evidence_obligations().reason == "trial_evidence_obligation_ledger_not_retained"
        )
        assert study.evidence_obligations().total is None
        assert not study.evidence_actions().available
    with pytest.raises(CausalValueError, match="256"):
        proof.assumptions(limit=257)
    with pytest.raises(CausalValueError, match="index must be 0"):
        proof.adjustment(estimand=1)


def test_trial_proof_and_bound_page_pin_the_original_native_owner():
    study, data = fixture(True)
    proof = study.identification
    page = proof.assumptions(limit=1)
    total = page.total
    branch = ant.reconfigure(study, inference=transport.TransportInference(bootstrap=5))
    assert proof.status == branch.identification.status == "identified"
    assert proof.adjustment().items[0]["name"] == "z"
    del study, data, proof, branch
    gc.collect()
    count = 0
    while page is not None:
        count += len(page.items)
        assert all(item["scope"] == "transport_certificate" for item in page.items)
        page = page.next(limit=1)
    assert count == total


def test_trial_explicit_proof_materialization_has_exact_output_admission():
    study, _ = fixture(True)
    proof = study.identification
    native = proof._native
    raw = native.materialize_json()
    length = len(raw.encode("utf-8"))
    exact = ant.Execution(limits=ant.ResourceLimits(memory_bytes=length))
    assert exact._call(native.materialize_json) == raw
    short = ant.Execution(limits=ant.ResourceLimits(memory_bytes=length - 1))
    with pytest.raises(CausalResourceError):
        short._call(native.materialize_json)
    full = proof.to_dict(execution=exact)
    assert full["summary"]["formula_kind"] == "standardize"
    assert "transportable" in full["transport"]
    assert set(full["names"]) == {"z", "a", "y"}
