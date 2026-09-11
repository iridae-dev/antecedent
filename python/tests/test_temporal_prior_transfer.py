"""Named source/target prior transfer on licensed Bayesian temporal cells."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent
from antecedent.errors import CausalUnsupportedError
from antecedent.estimation import PreparedAnalysis
from antecedent.intervention import Sequence, Set

_ROOT = Path(__file__).resolve().parents[2]
_PIN = json.loads(
    (_ROOT / "conformance" / "bayesian" / "temporal_prior_transfer" / "expected.json").read_text()
)
_EDGES = [("pressure", 1, "defect", 0)]
_EDGES_W = [("pressure", 1, "defect", 0), ("w", 0, "defect", 0)]
_TRUTH = float(_PIN["true_effect"])
_ATOL = float(_PIN["atol"])
_N = int(_PIN["n"])
_DRAWS = int(_PIN["n_draws"])
_SEED = int(_PIN["seed"])


def _series(n: int = _N, noise: float = 0.05, seed: int = _SEED) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    pressure = np.sin(0.04 * np.arange(n))
    defect = np.zeros(n)
    for t in range(1, n):
        defect[t] = 0.9 * pressure[t - 1] + noise * rng.uniform(-1.0, 1.0)
    return {"pressure": pressure, "defect": defect}


def _series_w(n: int = _N, seed: int = _SEED) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + 17)
    w = rng.uniform(-1.0, 1.0, n)
    pressure = np.sin(0.04 * np.arange(n)) + 0.4 * w
    defect = np.zeros(n)
    for t in range(1, n):
        defect[t] = 0.2 * pressure[t - 1] + 0.35 * w[t] + 0.25 * rng.uniform(-1.0, 1.0)
    return {"pressure": pressure, "defect": defect, "w": w}


def _pulse():
    return antecedent.PulseEffect(
        treatment="pressure",
        outcome="defect",
        treatment_lag=1,
        horizon_steps=1,
        active_level=1.0,
    )


def _sustained():
    return antecedent.SustainedEffect(
        treatment="pressure",
        outcome="defect",
        treatment_lag=1,
        horizon_steps=1,
        active_level=1.0,
    )


def _curve():
    return antecedent.ResponseCurve(
        "pressure",
        "defect",
        grid=_PIN["grid"],
        horizons=_PIN["horizons"],
    )


def _source_artifact(query, data: dict[str, np.ndarray]) -> bytes:
    prepared = PreparedAnalysis.prepare(
        data,
        graph=_EDGES,
        query=query,
        inference=antecedent.Bayesian(n_draws=_DRAWS, backend="conjugate"),
        refute=False,
        seed=_SEED,
    )
    result = prepared.estimate(data, seed=_SEED)
    assert result.posterior is not None
    return bytes(prepared.export_artifact())


def _meta(artifact_id: str, query_kind: str, outcome: str = "defect", mapping=None):
    return antecedent.priors.PriorSourceMeta(
        artifact_id=artifact_id,
        estimand=antecedent.priors.EstimandFingerprint(
            query_kind=query_kind, treatment="pressure", outcome=outcome
        ),
        identification="NonparametricallyIdentified",
        design=(
            antecedent.priors.DesignVariable(name="pressure", role="treatment"),
            antecedent.priors.DesignVariable(name="defect", role="outcome"),
        ),
        declared_mapping=mapping,
    )


def test_fixture_names_source_target_and_filter():
    assert _PIN["compatibility_filter"] == "PriorCatalog.filter_compatible"
    assert (
        _PIN["source_cells"]["same_design_pulse"]
        == "PulseEffect × TemporalDag × explicit × Bayesian × none"
    )
    assert (
        _PIN["target_cells"]["same_design_response_curve"]
        == "ResponseCurve × TemporalDag × explicit × Bayesian × none"
    )
    assert _PIN["incompatible"]["reason_code"] == "estimand_mismatch"


@pytest.mark.parametrize(
    ("pin", "source_query", "target_query", "query_kind"),
    [
        ("same_design_pulse", _pulse(), _pulse(), "pulse"),
        ("same_design_sustained", _sustained(), _sustained(), "sustained"),
    ],
)
def test_same_design_transfer_on_staged_path(pin, source_query, target_query, query_kind):
    data = _series()
    artifact = _source_artifact(source_query, data)
    catalog = antecedent.priors.PriorCatalog.from_sources(
        [antecedent.priors.PriorSource(meta=_meta("match", query_kind), artifact=artifact)]
    )
    reports = catalog.compatible_with(query=target_query, variables=["pressure", "defect"])
    assert reports[0].is_usable
    source = catalog.require_usable(query=target_query, variables=["pressure", "defect"])
    assert source.meta.artifact_id == "match"
    assert _PIN["source_cells"][pin].startswith(source_query.__class__.__name__)
    assert _PIN["target_cells"][pin].startswith(target_query.__class__.__name__)
    prepared = PreparedAnalysis.prepare(
        data,
        graph=_EDGES,
        query=target_query,
        inference=antecedent.Bayesian(
            n_draws=_DRAWS,
            backend="conjugate",
            prior_from=source.artifact,
        ),
        refute=False,
        seed=_SEED,
    )
    result = prepared.estimate(data, seed=_SEED)
    assert result.posterior is not None
    assert abs(result.posterior.effect_mean - _TRUTH) < _ATOL


def test_same_design_response_curve_uses_declared_identical_mapping():
    data = _series()
    artifact = _source_artifact(_pulse(), data)
    mapping = antecedent.priors.PriorMapping.identical()
    catalog = antecedent.priors.PriorCatalog.from_sources(
        [
            antecedent.priors.PriorSource(
                meta=_meta("match", "pulse", mapping=mapping), artifact=artifact
            )
        ]
    )
    source = catalog.require_usable(query=_curve(), variables=["pressure", "defect"])
    prepared = PreparedAnalysis.prepare(
        data,
        graph=_EDGES,
        query=_curve(),
        inference=antecedent.Bayesian(
            n_draws=_DRAWS,
            backend="conjugate",
            prior_from=source.artifact,
            mapping=mapping,
        ),
        refute=False,
        seed=_SEED,
    )
    result = prepared.estimate(data, seed=_SEED)
    assert result.response is not None
    values = np.asarray([row[0] for row in result.response.values], dtype=float)
    assert np.isfinite(values).all()


@pytest.mark.parametrize(
    ("pin", "target_query", "edges"),
    [
        ("mapped_effect_pulse", _pulse(), _EDGES_W),
        ("mapped_effect_sustained", _sustained(), _EDGES_W),
        ("mapped_effect_response_curve", _curve(), _EDGES_W),
    ],
)
def test_mapped_effect_transfer_on_staged_path(pin, target_query, edges):
    source = _source_artifact(_pulse(), _series())
    mapping = antecedent.priors.PriorMapping.effect_functional("ate")
    catalog = antecedent.priors.PriorCatalog.from_sources(
        [
            antecedent.priors.PriorSource(
                meta=_meta("mapped", "pulse", mapping=mapping), artifact=source
            )
        ]
    )
    chosen = catalog.require_usable(query=target_query, variables=["pressure", "defect", "w"])
    assert chosen.meta.artifact_id == "mapped"
    assert _PIN["target_cells"][pin]
    data = _series_w()
    prepared = PreparedAnalysis.prepare(
        data,
        graph=edges,
        query=target_query,
        inference=antecedent.Bayesian(
            n_draws=_DRAWS,
            backend="conjugate",
            prior_from=chosen.artifact,
            mapping=mapping,
        ),
        refute=False,
        seed=_SEED,
    )
    result = prepared.estimate(data, seed=_SEED)
    if hasattr(result, "posterior") and result.posterior is not None:
        assert np.isfinite(result.posterior.effect_mean)
    else:
        assert result.response is not None


def test_incompatible_catalog_fails_closed():
    artifact = _source_artifact(_pulse(), _series())
    catalog = antecedent.priors.PriorCatalog.from_sources(
        [
            antecedent.priors.PriorSource(
                meta=_meta("wrong_outcome", "pulse", outcome="other"), artifact=artifact
            )
        ]
    )
    reports = catalog.compatible_with(query=_pulse(), variables=["pressure", "defect"])
    assert reports[0].status == "rejected"
    assert reports[0].reason is not None
    assert reports[0].reason.get("code") == _PIN["incompatible"]["reason_code"]
    with pytest.raises(CausalUnsupportedError, match="incompatible"):
        catalog.require_compatible(query=_pulse(), variables=["pressure", "defect"])


def test_sequence_refuses_transfer_without_new_filter():
    artifact = _source_artifact(_pulse(), _series())
    query = antecedent.InterventionResponse(
        "defect",
        intervention=Sequence([Set("pressure", 1.0), Set("pressure", 1.0)]),
        horizons=[1, 2],
        policy="pulse",
        treatment_lag=1,
    )
    with pytest.raises(
        (CausalUnsupportedError, antecedent.CausalError),
        match=_PIN["sequence_refuses"]["message_contains"],
    ):
        PreparedAnalysis.prepare(
            _series(),
            graph=_EDGES,
            query=query,
            inference=antecedent.Bayesian(
                n_draws=_DRAWS, backend="conjugate", prior_from=artifact
            ),
            refute=False,
            seed=_SEED,
        ).estimate(_series(), seed=_SEED)
