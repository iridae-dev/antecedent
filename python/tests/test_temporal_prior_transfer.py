"""Named source/target prior transfer on licensed Bayesian temporal cells."""

from __future__ import annotations

import antecedent
import numpy as np
import pytest
from antecedent.errors import CausalUnsupportedError
from antecedent.intervention import Sequence, Set
from antecedent.outcomes import require_success

from _repo_text import REPO_ROOT, load_json

_ROOT = REPO_ROOT
_PIN = load_json(_ROOT / "conformance" / "bayesian" / "temporal_prior_transfer" / "expected.json")
_PULSE_CELL = "PulseEffect \u00d7 TemporalDag \u00d7 explicit \u00d7 Bayesian \u00d7 none"
_CURVE_CELL = "ResponseCurve \u00d7 TemporalDag \u00d7 explicit \u00d7 Bayesian \u00d7 none"
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
    prepared = antecedent.prepare(
        data,
        graph=_EDGES,
        query=query,
        inference=antecedent.Bayesian(n_draws=_DRAWS, backend="conjugate"),
        refute=False,
        execution=antecedent.Execution(seed=_SEED),
    )
    result = require_success(antecedent.estimate(prepared))
    assert result.posterior is not None
    return bytes(result.export(quantity="posterior"))


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
    assert _PIN["source_cells"]["same_design_pulse"] == _PULSE_CELL
    assert _PIN["target_cells"]["same_design_response_curve"] == _CURVE_CELL
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
    prepared = antecedent.prepare(
        data,
        graph=_EDGES,
        query=target_query,
        inference=antecedent.Bayesian(
            n_draws=_DRAWS,
            backend="conjugate",
            prior_from=source.artifact,
        ),
        refute=False,
        execution=antecedent.Execution(seed=_SEED),
    )
    result = require_success(antecedent.estimate(prepared))
    assert result.posterior is not None
    assert abs(result.posterior.effect_mean - _TRUTH) < _ATOL
    # Noise half-width 0.05 over 240 rows: the posterior sits much closer than the fixture band.
    assert abs(result.posterior.effect_mean - _TRUTH) < 0.02


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
    prepared = antecedent.prepare(
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
        execution=antecedent.Execution(seed=_SEED),
    )
    result = require_success(antecedent.estimate(prepared))
    assert result.response is not None
    values = np.asarray([row[0] for row in result.response.values], dtype=float)
    assert np.isfinite(values).all()
    # The curve at grid [0, 1] has intercept 0 and slope 0.9 (defect_t = 0.9 pressure_{t-1} plus
    # noise of half-width 0.05), and the source posterior centres on the same law.
    assert values == pytest.approx([0.0, 0.9], abs=0.02)


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
    prepared = antecedent.prepare(
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
        execution=antecedent.Execution(seed=_SEED),
    )
    result = require_success(antecedent.estimate(prepared))
    baseline = require_success(
        antecedent.estimate(
            antecedent.prepare(
                data,
                graph=edges,
                query=target_query,
                inference=antecedent.Bayesian(n_draws=_DRAWS, backend="conjugate"),
                refute=False,
                execution=antecedent.Execution(seed=_SEED),
            )
        )
    )
    # The target data have effect 0.2 and the source posterior centres on 0.9. A transferred
    # effect prior pulls the posterior strictly between the two, away from the data-only fit.
    if getattr(result, "posterior", None) is not None:
        informed, flat = result.posterior.effect_mean, baseline.posterior.effect_mean
        assert flat == pytest.approx(0.2, abs=0.05)
        assert flat + 0.1 < informed < _TRUTH, (flat, informed)
    else:
        assert result.response is not None
        informed = np.asarray([row[0] for row in result.response.values], dtype=float)
        flat = np.asarray([row[0] for row in baseline.response.values], dtype=float)
        assert np.isfinite(informed).all()
        assert np.abs(informed - flat).max() > 0.1, (flat, informed)


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
        CausalUnsupportedError,
        match=_PIN["sequence_refuses"]["message_contains"],
    ):
        require_success(
            antecedent.estimate(
                antecedent.prepare(
                    _series(),
                    graph=_EDGES,
                    query=query,
                    inference=antecedent.Bayesian(
                        n_draws=_DRAWS, backend="conjugate", prior_from=artifact
                    ),
                    refute=False,
                    execution=antecedent.Execution(seed=_SEED),
                )
            )
        )
