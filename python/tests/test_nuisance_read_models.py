"""Immutable nuisance summaries expose bounded common outcome inspection."""

import gc

import pytest
from antecedent import nuisance_scores as ns
from antecedent.outcomes import Outcome, require_success

from test_prepared_nuisance_score_lifecycle import _data, _study, _temporal_study


def _reading(*, posterior=True, rows=()):
    if posterior:
        return require_success(ns.treatment_score_posterior(_study(), draws=8, summary_rows=rows))
    study, _ = _temporal_study()
    return require_success(ns.temporal_treatment_score(study, [[0]], penalty=2))


def test_nuisance_reading_owns_immutable_values_and_materializes_explicitly():
    source = _data()
    study = _study(source)
    product = require_success(ns.treatment_score_posterior(study, draws=8, summary_rows=[0]))
    before = product["row0.mean"]
    assumptions = product.assumptions
    source["t"][:] = 1 - source["t"]
    source["z"][:] = 90
    assert product["row0.mean"] == before
    with pytest.raises(TypeError, match="does not support item assignment"):
        product.values["row0.mean"] = 0.8
    materialized = product.to_dict()
    materialized["values"]["row0.mean"] = 0.7
    materialized["assumptions"].append("another model")
    assert product["row0.mean"] == before
    assert product.assumptions == assumptions
    assert "artifact" not in materialized
    view = product.values
    del product, study
    gc.collect()
    assert view["row0.mean"] == before


@pytest.mark.parametrize("posterior", [True, False])
def test_nuisance_common_reading_distinguishes_model_uncertainty_from_effects(posterior):
    product = _reading(posterior=posterior)
    outcome = Outcome(value=product)
    inspection = outcome.inspect()
    assert outcome.answer.kind == "structured"
    assert outcome.answer.value is None and outcome.answer.bounds is None
    assert outcome.answer.detail == "propensity_score_nuisance"
    assert outcome.uncertainty.available is posterior
    assert outcome.uncertainty.summary == product.uncertainty
    assert outcome.uncertainty.payload["target"] == "propensity_score_nuisance"
    assert not inspection.identification.available
    assert inspection.identification.reason == "nuisance_not_causal_effect"
    assert not outcome.support.available
    assert outcome.support.payload == {"licensing": "unlicensed", "calibration": "unmeasured"}
    assert outcome.assumptions.payload["declarations"] == product.assumptions
    assert outcome.diagnostics == product.diagnostics
    assert outcome.provenance == product.provenance
    assert outcome.identities == {"request_digest": product.request_digest}
    assert outcome.capabilities == ("inspect", "export", "to_dict")
    with pytest.raises(TypeError, match="does not support item assignment"):
        inspection.assumptions.payload["declarations"] = ()


def test_nuisance_inspection_and_display_do_not_materialize_values_or_execute(monkeypatch):
    product = _reading(rows=[0] * 2500)
    assert len(product.values) >= 10_000
    assert not product._native.encoding_retained()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("inspection must not fit, decode, export or materialize")

    monkeypatch.setattr(ns, "_consume", forbidden)
    monkeypatch.setattr(ns.json, "dumps", forbidden)
    monkeypatch.setattr(ns.NuisanceScoreProduct, "to_dict", forbidden)
    monkeypatch.setattr(ns.NuisanceScoreProduct, "export", forbidden)
    inspection = product.inspect()
    assert inspection.details["value_count"] == len(product.values)
    assert "row2499" not in repr(product)
    assert "row2499" not in product._repr_html_()
    assert len(repr(product)) < 500
    assert len(product._repr_html_()) < 2_000
    assert not product._native.encoding_retained()
