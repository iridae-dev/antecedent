"""Immutable nuisance summaries expose bounded common outcome inspection."""

import pytest
from antecedent import nuisance_scores as ns
from antecedent.outcomes import Outcome


def _reading(values=None, *, uncertainty="nuisance_posterior_laplace_not_an_effect_posterior"):
    return ns.NuisanceScoreProduct(
        estimator_id="bayesian.propensity_posterior",
        target="propensity_score_nuisance",
        uncertainty=uncertainty,
        licensing="unlicensed",
        calibration="unmeasured",
        values={"row0.mean": 0.3} if values is None else values,
        assumptions=("declared treatment model",),
        request_digest="native-request-digest",
        artifact=b"retained encoding",
        diagnostics=("native diagnostic",),
    )


def test_nuisance_reading_owns_immutable_values_and_materializes_explicitly():
    source = {"row0.mean": 0.3}
    product = _reading(source)
    source["row0.mean"] = 0.9
    assert product["row0.mean"] == 0.3
    with pytest.raises(TypeError, match="does not support item assignment"):
        product.values["row0.mean"] = 0.8
    materialized = product.to_dict()
    materialized["values"]["row0.mean"] = 0.7
    materialized["assumptions"].append("another model")
    assert product["row0.mean"] == 0.3
    assert product.assumptions == ("declared treatment model",)
    assert product.artifact == b"retained encoding"
    assert "artifact" not in materialized


@pytest.mark.parametrize(
    ("uncertainty", "available"),
    [
        ("nuisance_posterior_laplace_not_an_effect_posterior", True),
        ("penalized_mode_no_uncertainty", False),
    ],
)
def test_nuisance_common_reading_distinguishes_model_uncertainty_from_effects(
    uncertainty,
    available,
):
    product = _reading(uncertainty=uncertainty)
    outcome = Outcome(value=product)
    inspection = outcome.inspect()
    assert outcome.answer.kind == "structured"
    assert outcome.answer.value is None and outcome.answer.bounds is None
    assert outcome.answer.detail == "propensity_score_nuisance"
    assert outcome.uncertainty.available is available
    assert outcome.uncertainty.summary == uncertainty
    assert outcome.uncertainty.payload["target"] == "propensity_score_nuisance"
    assert not inspection.identification.available
    assert inspection.identification.reason == "nuisance_not_causal_effect"
    assert not outcome.support.available
    assert outcome.support.payload == {"licensing": "unlicensed", "calibration": "unmeasured"}
    assert outcome.assumptions.payload["declarations"] == product.assumptions
    assert outcome.diagnostics == ("native diagnostic",)
    assert outcome.provenance == product.provenance
    assert outcome.identities == {"request_digest": "native-request-digest"}
    assert outcome.capabilities == ("inspect", "export", "to_dict")
    with pytest.raises(TypeError, match="does not support item assignment"):
        inspection.assumptions.payload["declarations"] = ()


def test_nuisance_inspection_and_display_do_not_materialize_values_or_execute(monkeypatch):
    product = _reading({f"row{i}.mean": 0.3 for i in range(10_000)})

    def forbidden(*_args, **_kwargs):
        raise AssertionError("inspection must not fit, decode, or materialize")

    monkeypatch.setattr(ns, "_consume", forbidden)
    monkeypatch.setattr(ns.json, "dumps", forbidden)
    monkeypatch.setattr(ns.NuisanceScoreProduct, "to_dict", forbidden)
    inspection = product.inspect()
    assert inspection.details["value_count"] == 10_000
    assert "row9999" not in repr(product)
    assert "row9999" not in product._repr_html_()
    assert len(repr(product)) < 500
    assert len(product._repr_html_()) < 2_000
