"""Scientific labels and explicit fallback on the immutable public lifecycle."""

import antecedent as ant
import numpy as np
import pytest
from antecedent.errors import CausalValueError
from antecedent.results import CausalResponseView


def observations():
    rng = np.random.default_rng(23)
    a = rng.normal(size=400)
    return {"a": a, "y": 2 * a + rng.normal(scale=0.2, size=len(a))}


def test_labels_bind_the_native_claim_without_implicit_artifact_encoding(monkeypatch):
    study = ant.prepare(
        observations(),
        graph=[("a", "y")],
        query=ant.ResponseCurve("a", "y", grid=[-0.5, 0.0, 0.5]),
        outcome_units="mmHg",
        dose_units="mg",
        quantity_population="trial",
    )
    with monkeypatch.context() as patch:

        def unexpected(*args, **kwargs):
            raise AssertionError("label binding serialized an artifact")

        patch.setattr(CausalResponseView, "export", unexpected)
        outcome = ant.estimate(study)
        assert outcome.accepted
        assert {quantity.units for quantity in outcome.value.quantities} == {"mmHg"}
        assert {quantity.population_id for quantity in outcome.value.quantities} == {"trial"}
        assert outcome.value.program_binding is not None
        assert "executed" in outcome.inspect().stage
    from antecedent import artifacts

    encoded = artifacts.loads(bytes(outcome.export()))
    assert bytes(encoded.contract["claim"]["claim_id"]).hex() == outcome.value.claim_id


def test_refused_preparation_retains_an_explicit_separate_association():
    graph = ant.Admg.from_edges(["a", "y"], [("a", "y")], [("a", "y")])
    prepared = ant.prepare(
        observations(),
        graph=graph,
        query=ant.AverageEffect("a", "y"),
        on_refusal="describe",
    )
    assert prepared.refusal is not None
    assert prepared.inspect().stage == "prepared"
    outcome = ant.estimate(prepared)
    assert not outcome.accepted
    assert outcome.refusal.code == prepared.refusal.code
    assert outcome.associational is not None and outcome.associational.accepted
    assert outcome.associational.answer.kind == "associational"
    with pytest.raises(CausalValueError, match="select quantity"):
        outcome.export()
    assert outcome.export(quantity="refusal").kind == "scientific_refusal_v1"
    association = ant.load(outcome.export(quantity="associational"))
    assert association.answer.kind == "associational"
