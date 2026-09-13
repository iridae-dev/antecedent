"""1.8 Bayesian remainder: staged cells and named fixtures."""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import antecedent as ac
import numpy as np
import pytest
from antecedent.estimation import PreparedAnalysis

ROOT = Path(__file__).resolve().parents[2]


def _load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def test_functional_validation_fixture_is_named():
    pin = _load("conformance/estimate/functional_validation/expected.json")
    assert pin["path_effect"] == pytest.approx(0.3)


def test_static_kinds_and_derivatives_fixtures_are_named():
    kinds = _load("conformance/estimate/staged_static_kinds/expected.json")
    deriv = _load("conformance/response/staged_derivatives/expected.json")
    admg = _load("conformance/estimate/admg_frontdoor_functional/expected.json")
    dist = _load("conformance/estimate/interventional_distribution/expected.json")
    path = _load("conformance/context/path_specific_natural/expected.json")
    assert kinds["direct"] == pytest.approx(1.8)
    assert deriv["average"] == pytest.approx(2.0)
    assert admg["frequentist"]["expected_ate"] == pytest.approx(0.3)
    assert dist["mean"] == pytest.approx(0.7)
    assert path["ate"] == pytest.approx(0.3)


def test_prior_transfer_fixtures_are_named():
    from antecedent.priors import (
        DesignVariable,
        EstimandFingerprint,
        PriorCatalog,
        PriorSource,
        PriorSourceMeta,
    )

    effect = _load("conformance/bayesian/static_effect_prior_transfer/expected.json")
    response = _load("conformance/bayesian/static_response_prior_transfer/expected.json")
    mediation = _load("conformance/bayesian/static_mediation_prior_transfer/expected.json")
    mixtures = _load("conformance/bayesian/known_truth_mixtures/expected.json")
    assert effect["compatibility_filter"] == "PriorCatalog.filter_compatible"
    assert response["target_cells"]["missing_mapping"] == "typed_refuse"
    assert mediation["compatibility_filter"] == "outcome-mechanism ATE/Δ hydrate"
    assert "static_average_effect" in mixtures
    catalog = PriorCatalog.from_sources(
        [
            PriorSource(
                meta=PriorSourceMeta(
                    artifact_id="source",
                    estimand=EstimandFingerprint("ate", "t", "y"),
                    identification="NonparametricallyIdentified",
                    design=(
                        DesignVariable("t", "treatment"),
                        DesignVariable("y", "outcome"),
                    ),
                )
            ),
            PriorSource(
                meta=PriorSourceMeta(
                    artifact_id="wrong",
                    estimand=EstimandFingerprint("ate", "t", "other"),
                    identification="NonparametricallyIdentified",
                )
            ),
        ]
    )

    class _Query:
        kind = "average"
        treatment = "t"
        outcome = "y"

    reports = catalog.compatible_with(query=_Query(), variables=["t", "y"])
    assert reports[0].is_usable
    assert reports[1].status == "rejected"
    assert (reports[1].reason or {}).get("code") == "estimand_mismatch"


def test_bayesian_path_prepare_estimate():
    t = np.array([0.0] * 100 + [1.0] * 100)
    m = np.array([0.0] * 50 + [1.0] * 50 + [0.0] * 20 + [1.0] * 80)
    y = np.array(
        [0.0] * 40
        + [1.0] * 10
        + [0.0] * 10
        + [1.0] * 40
        + [0.0] * 10
        + [1.0] * 10
        + [0.0] * 10
        + [1.0] * 70
    )
    data = {"t": t, "m": m, "y": y}
    dag = ac.Dag.from_edges(["t", "m", "y"], [("t", "m"), ("m", "y")])
    prepared = PreparedAnalysis.prepare(
        data,
        graph=dag,
        query=ac.PathSpecificEffect("t", "y", path_nodes=["m"]),
        inference=ac.Bayesian(n_draws=64),
        refute="none",
    )
    result = prepared.estimate(data)
    assert result.estimate.estimator_id == "functional.effect"
    assert any("exec.identify.cached" in item for item in result.diagnostics)
    assert result.posterior is not None
    assert result.identification
    assert result.assumptions is not None


def test_bayesian_mediation_four_axes():
    kinds = _load("conformance/estimate/staged_static_kinds/expected.json")
    i = np.arange(500, dtype=float)
    a = np.sin(i * 0.71)
    m = 2 * a + np.cos(i * 1.13)
    y = 3 * a + 4 * m + 0.1 * np.sin(i * 0.31)
    data = {"a": a, "m": m, "y": y}
    dag = ac.Dag.from_edges(["a", "m", "y"], [("a", "m"), ("a", "y"), ("m", "y")])
    prepared = PreparedAnalysis.prepare(
        data,
        graph=dag,
        query=ac.MediationEffect(
            "a",
            "y",
            mediators=["m"],
            contrast="natural_direct",
            control_level=kinds["control"],
            active_level=kinds["active"],
        ),
        inference=ac.Bayesian(n_draws=64),
        refute="none",
    )
    result = prepared.estimate(data)
    assert result.estimate.estimator_id == "mediation.linear"
    assert result.posterior is not None
    assert result.identification
    assert result.assumptions
    assert result.mediation is not None
    assert result.effect == pytest.approx(kinds["direct"], abs=0.2)


def test_bayesian_counterfactual_unit_posterior():
    kinds = _load("conformance/estimate/staged_static_kinds/expected.json")
    i = np.arange(500, dtype=float)
    a = np.sin(i * 0.71)
    m = 2 * a + np.cos(i * 1.13)
    y = 3 * a + 4 * m + 0.1 * np.sin(i * 0.31)
    data = {"a": a, "m": m, "y": y}
    dag = ac.Dag.from_edges(["a", "m", "y"], [("a", "m"), ("a", "y"), ("m", "y")])
    query = ac.Counterfactual(
        "a",
        "y",
        control_level=kinds["control"],
        active_level=kinds["active"],
    )
    bayes = PreparedAnalysis.prepare(
        data,
        graph=dag,
        query=query,
        inference=ac.Bayesian(n_draws=64),
        refute="none",
    ).estimate(data)
    freq = PreparedAnalysis.prepare(
        data,
        graph=dag,
        query=query,
        inference=ac.Frequentist(),
        refute="none",
    ).estimate(data)
    assert bayes.unit_effects is not None
    assert freq.unit_effects is not None
    assert bayes.mean_ite == pytest.approx(kinds["counterfactual_mean"], abs=0.25)
    assert float(np.mean(bayes.unit_effects)) == pytest.approx(bayes.mean_ite, abs=0.05)
    assert not np.allclose(bayes.unit_effects, freq.unit_effects)


def test_bayesian_mediation_mapped_prior_is_used():
    from antecedent.priors import PriorMapping

    pin = _load("conformance/bayesian/static_mediation_prior_transfer/expected.json")
    kinds = _load("conformance/estimate/staged_static_kinds/expected.json")
    i = np.arange(500, dtype=float)
    a = np.sin(i * 0.71)
    m = 2 * a + np.cos(i * 1.13)
    y = 3 * a + 4 * m + 0.1 * np.sin(i * 0.31)
    data = {"a": a, "m": m, "y": y}
    dag = ac.Dag.from_edges(["a", "m", "y"], [("a", "m"), ("a", "y"), ("m", "y")])
    source = ac.analyze(
        data,
        graph=dag,
        query=ac.AverageEffect(
            "a",
            "y",
            control_level=kinds["control"],
            active_level=kinds["active"],
        ),
        inference=ac.Bayesian(n_draws=64, backend="conjugate"),
        refute=False,
        seed=int(pin["seed"]),
        return_posterior_artifact=True,
    )
    assert source.posterior is not None
    artifact = bytes(source.posterior.artifact)
    transferred = ac.analyze(
        data,
        graph=dag,
        query=ac.MediationEffect(
            "a",
            "y",
            mediators=["m"],
            contrast="natural_direct",
            control_level=kinds["control"],
            active_level=kinds["active"],
        ),
        inference=ac.Bayesian(
            n_draws=64,
            backend="conjugate",
            prior_from=artifact,
            mapping=PriorMapping.effect_functional("ate"),
        ),
        refute=False,
        seed=int(pin["seed"]),
    )
    isotropic = ac.analyze(
        data,
        graph=dag,
        query=ac.MediationEffect(
            "a",
            "y",
            mediators=["m"],
            contrast="natural_direct",
            control_level=kinds["control"],
            active_level=kinds["active"],
        ),
        inference=ac.Bayesian(n_draws=64, backend="conjugate"),
        refute=False,
        seed=int(pin["seed"]),
    )
    assert transferred.posterior is not None
    text = " ".join(str(a) for a in transferred.assumptions)
    assert "mapped ATE/Δ prior hydrated onto outcome-mechanism" in text
    assert "implied NDE/ATE mean" in text
    assert source.effect == pytest.approx(kinds["total"], abs=0.25)
    assert isotropic.effect == pytest.approx(kinds["direct"], abs=kinds["tolerance"])
    assert pin["compatibility_filter"] == "outcome-mechanism ATE/Δ hydrate"


def test_staged_native_signatures_match_type_stubs():
    from antecedent import _native

    tree = ast.parse((ROOT / "python/antecedent/_native.pyi").read_text(encoding="utf-8"))
    prepared = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PreparedAnalysis"
    )
    for method in prepared.body:
        if isinstance(method, ast.FunctionDef) and method.name.startswith("prepare"):
            declared = {arg.arg for arg in method.args.args + method.args.kwonlyargs}
            actual = set(
                inspect.signature(getattr(_native.PreparedAnalysis, method.name)).parameters
            )
            assert declared == actual, method.name
