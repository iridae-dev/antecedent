"""1.2 licensing evidence: independent posterior moments and actual artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import antecedent as ac
import numpy as np
import pytest
from antecedent import artifacts
from antecedent.estimation import PreparedAnalysis
from antecedent.inference import decode_posterior_artifact, encode_posterior_artifact

FIXTURE = json.loads(
    (
        Path(__file__).parents[2] / "conformance/bayesian/release_12_moments/expected.json"
    ).read_text()
)


def prepare(data, graph, query, *, accepted, scale=10.0, suite="none", bayesian=True):
    return PreparedAnalysis.prepare(
        data,
        query=query,
        graph=ac.AcceptedGraph(graph) if accepted else graph,
        inference=ac.Bayesian(backend="conjugate", n_draws=8192, prior_scale=scale)
        if bayesian
        else ac.Frequentist(),
        refute=suite,
        latency=None,
        bootstrap=0,
    )


def check_moments(mean, variance, expected):
    assert mean == pytest.approx(expected[0], abs=0.08 * np.sqrt(expected[1]))
    assert variance == pytest.approx(expected[1], rel=0.06)


@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("scale", [0.1, 10.0])
@pytest.mark.parametrize("family", ["conditional", "mediation", "window", "response"])
def test_posterior_moments_and_artifact(accepted, scale, family):
    source = {k: np.array(v) for k, v in FIXTURE["data"].items()}
    if family == "conditional":
        data = {k: source[k] for k in ["t", "w", "y"]}
        graph = [("t", "y"), ("w", "y")]
        query = ac.ConditionalEffect("t", "y", "w")
    elif family == "mediation":
        data = {"t": source["t"], "m": source["m"], "y": source["o"]}
        graph = [("t", 1, "m", 0), ("t", 1, "y", 0), ("m", 0, "y", 0)]
        query = ac.TemporalMediationEffect("t", "m", "y")
    elif family == "window":
        data = {"t": source["t"], "y": source["s"]}
        graph = [("t", 1, "y", 0), ("t", 2, "y", 0)]
        query = ac.SustainedEffect("t", "y", window=(-2, -1))
    else:
        data = {"t": source["t"], "y": source["r"]}
        graph = [("t", "y")]
        query = ac.ResponseCurve("t", "y", grid=[0, 1])
    prepared = prepare(data, graph, query, accepted=accepted, scale=scale)
    result = prepared.estimate(data, seed=1201)
    assert prepared.evidence_status == "licensed"
    assert prepared.structure_source == ("accepted" if accepted else "explicit")
    encoded = prepared.export_artifact()
    query_art = artifacts.loads(prepared.export_artifact(payload="query"))
    assert (
        artifacts.loads(
            artifacts.dumps(
                "query",
                query_art.payload,
                variable_names=query_art.variable_names,
                artifact_id=query_art.artifact_id,
            )
        )
        == query_art
    )
    expected = FIXTURE["posterior"][str(scale)][family]
    if family == "response":
        art = artifacts.loads(encoded)
        assert art.payload_kind == "response_result"
        again = artifacts.loads(
            artifacts.dumps(
                art.payload_kind,
                art.payload,
                variable_names=art.variable_names,
                artifact_id=art.artifact_id,
            )
        )
        assert again == art
        assert result.response is not None
        for mean, lo, hi, pin in zip(
            result.response.values,
            result.uncertainty.lower,
            result.uncertainty.upper,
            expected,
            strict=True,
        ):
            # NIG marginal coefficients have t_(n+.002) tails. Its .975
            # quantile at n=96 is 1.984984 (rounded below 1e-6).
            variance = ((hi[0] - lo[0]) / (2 * 1.984984)) ** 2 * (96.002 / (96.002 - 2))
            check_moments(mean[0], variance, pin)
        assert "response.bayesian" in str(art.payload)
    else:
        art = decode_posterior_artifact(encoded)
        again = decode_posterior_artifact(encode_posterior_artifact(art))
        assert list(again.draws) == list(art.draws)
        assert list(again.quantity_names) == list(art.quantity_names)
        effect_index = list(art.quantity_names).index(
            "mediation"
            if family == "mediation"
            else "sustained_window"
            if family == "window"
            else "ate"
        )
        check_moments(art.mean[effect_index], art.sd[effect_index] ** 2, expected)
        assert result.posterior is not None
        assert art.mean[effect_index] == result.posterior.effect_mean
        if family == "mediation":
            draws = np.asarray(art.draws).reshape((-1, art.n_draws))
            assert np.allclose(draws[1], draws[2] + draws[3], rtol=0, atol=1e-14)


@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("suite", ["none", "cheap", "full"])
@pytest.mark.parametrize("bayesian", [False, True])
def test_mediation_validation_targets_and_prior_grid(accepted, suite, bayesian):
    source = {k: np.array(v) for k, v in FIXTURE["data"].items()}
    data = {"t": source["t"], "m": source["m"], "y": source["o"]}
    query = ac.TemporalMediationEffect("t", "m", "y")
    graph = [("t", 1, "m", 0), ("t", 1, "y", 0), ("m", 0, "y", 0)]
    prepared = prepare(data, graph, query, accepted=accepted, suite=suite, bayesian=bayesian)
    result = prepared.estimate(data, seed=1201)
    # Validation checks the fixed finite sample, whose posterior moments are pinned independently.
    assert result.ate == pytest.approx(FIXTURE["posterior"]["10.0"]["mediation"][0], abs=0.02)
    if suite != "none":
        reports = {r.refuter: r for r in result.validation.reports}
        assert abs(reports["mediation.placebo_mediator"].refuted_ate) < 0.05
        assert reports["mediation.random_common_cause"].refuted_ate == pytest.approx(
            result.ate, abs=0.02
        )
        if suite == "full":
            assert "mediation.contiguous_window" in reports
            if bayesian:
                assert result.validation.prior_sensitivity is not None
    else:
        assert not result.validation.reports


@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("intervention", [False, True])
def test_temporal_response_surface_artifact(accepted, intervention):
    pin = json.loads(
        (
            Path(__file__).parents[2] / "conformance/bayesian/response_surfaces/expected.json"
        ).read_text()
    )
    t = np.resize([0.0, 1.0, 0.0, -1.0], 1200)
    y = 1 + 2 * np.r_[0, t[:-1]] + 3 * np.r_[0, 0, t[:-2]]
    data = {"t": t, "y": y}
    graph = [("t", 1, "y", 0), ("t", 2, "y", 0)]
    if intervention:
        query = ac.InterventionResponse(
            "y", intervention=ac.intervention.Set("t", 1), horizons=[1, 2], treatment_lag=1
        )
    else:
        query = ac.ResponseCurve("t", "y", grid=[0, 1], horizons=[1, 2], treatment_lag=1)
    prepared = prepare(data, graph, query, accepted=accepted)
    result = prepared.estimate(data, seed=1201)
    expected = pin["temporal_mean"][2:] if intervention else pin["temporal_mean"]
    assert np.asarray(result.response.values).flatten() == pytest.approx(
        expected, abs=pin["temporal_tolerance"]
    )
    art = artifacts.loads(prepared.export_artifact())
    assert "response.temporal.bayesian" in str(art.payload)
    assert (
        artifacts.loads(
            artifacts.dumps(
                art.payload_kind,
                art.payload,
                variable_names=art.variable_names,
                artifact_id=art.artifact_id,
            )
        )
        == art
    )


@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("suite", ["none", "cheap", "full"])
def test_functional_validation_uses_table_and_accepted_structure(accepted, suite):
    rng = np.random.default_rng(1202)
    t = np.repeat([0.0, 1.0], 500)
    y = np.r_[np.zeros(400), np.ones(100), np.zeros(150), np.ones(350)]
    order = rng.permutation(1000)
    t, y = t[order], y[order]
    data = {"t": t, "y": y}
    prepared = prepare(
        data,
        [("t", "y")],
        ac.InterventionalDistribution("y", interventions={"t": 1}),
        accepted=accepted,
        suite=suite,
        bayesian=False,
    )
    result = prepared.estimate(data)
    assert result.ate == pytest.approx(0.7, abs=0.05)
    query_art = artifacts.loads(prepared.export_artifact(payload="query"))
    assert (
        artifacts.loads(
            artifacts.dumps(
                "query",
                query_art.payload,
                variable_names=query_art.variable_names,
                artifact_id=query_art.artifact_id,
            )
        )
        == query_art
    )

    if suite != "none":
        reports = {r.refuter: r for r in result.validation.reports}
        assert reports["distribution.normalization"].passed
        # A nonzero table distance verifies row masks affect the evaluator.
        assert 0 < reports["distribution.subset_tv"].refuted_ate < 0.1


def test_shift_support_assesses_shifted_distribution_not_only_its_mean():
    source = {k: np.array(v) for k, v in FIXTURE["data"].items()}
    data = {"t": source["t"], "y": source["r"]}
    query = ac.InterventionResponse("y", intervention=ac.intervention.Shift("t", 0.1))
    prepared = prepare(data, [("t", "y")], query, accepted=False)
    prepared.estimate(data)
    payload = artifacts.loads(prepared.export_artifact()).payload
    assert payload["support"]["status"] == "outside_empirical_support"
    assert payload["support"]["query_region"]["maxima"][0] == pytest.approx(source["t"].max() + 0.1)


@pytest.mark.parametrize("suite", ["cheap", "full"])
@pytest.mark.parametrize("sustained", [False, True])
def test_dbn_validation_and_mass_survive_posterior_export(suite, sustained):
    from known_truth import BAYES, TEMPORAL, temporal_posterior, white_noise_pulse_series

    data = white_noise_pulse_series(int(TEMPORAL["n"]), int(TEMPORAL["seed"]))
    query_type = ac.SustainedEffect if sustained else ac.PulseEffect
    prepared = PreparedAnalysis.prepare(
        data,
        query=query_type("pressure", "defect"),
        discovery=temporal_posterior(),
        inference=BAYES,
        refute=suite,
        latency=None,
    )
    result = prepared.estimate(data, seed=17)
    assert result.posterior is not None
    assert result.validation.reports
    art = decode_posterior_artifact(prepared.export_artifact())
    assert art.unidentified_mass == pytest.approx(result.posterior.unidentified_mass)
    assert art.identification == "GraphDependent"
    if suite == "full":
        assert result.validation.prior_sensitivity is not None


def test_binary_mediator_support_flags_disjoint_ranges():
    rng = np.random.default_rng(1213)
    t = rng.integers(0, 2, 800).astype(float)
    m = 10 * np.r_[0, t[:-1]] + rng.normal(scale=0.1, size=800)
    y = 0.2 * np.r_[0, t[:-1]] + 0.5 * m + rng.normal(scale=0.1, size=800)
    data = {"t": t, "m": m, "y": y}
    prepared = prepare(
        data,
        [("t", 1, "m", 0), ("t", 1, "y", 0), ("m", 0, "y", 0)],
        ac.TemporalMediationEffect("t", "m", "y"),
        accepted=False,
        suite="cheap",
        bayesian=False,
    )
    reports = {r.refuter: r for r in prepared.estimate(data).validation.reports}
    assert not reports["mediation.binary_mediator_support"].passed


def test_incompatible_explicit_estimator_is_not_silently_replaced():
    data = {k: np.array(FIXTURE["data"][k]) for k in ["t", "w", "y"]}
    with pytest.raises(ac.errors.CausalUnsupportedError, match="conditional.bayesian"):
        PreparedAnalysis.prepare(
            data,
            query=ac.ConditionalEffect("t", "y", "w"),
            graph=[("t", "y"), ("w", "y")],
            inference=ac.Bayesian(),
            estimator="conditional.linear.adjustment",
        )


@pytest.mark.parametrize("window", [False, True])
def test_composed_hmc_refuses_without_derived_chain_diagnostics(window):
    source = {k: np.array(v) for k, v in FIXTURE["data"].items()}
    if window:
        data = {"t": source["t"], "y": source["s"]}
        graph = [("t", 1, "y", 0), ("t", 2, "y", 0)]
        query = ac.SustainedEffect("t", "y", window=(-2, -1))
    else:
        data = {"t": source["t"], "m": source["m"], "y": source["o"]}
        graph = [("t", 1, "m", 0), ("t", 1, "y", 0), ("m", 0, "y", 0)]
        query = ac.TemporalMediationEffect("t", "m", "y")
    prepared = PreparedAnalysis.prepare(
        data,
        query=query,
        graph=graph,
        inference=ac.Bayesian(backend="hmc"),
        refute="none",
        latency=None,
    )
    with pytest.raises(
        ac.errors.CausalEstimateError, match="derived-contrast chain diagnostics"
    ):
        prepared.estimate(data)


@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("suite", ["none", "cheap", "full"])
def test_path_functional_and_query_artifact(accepted, suite):
    t = np.resize([0.0, 1.0], 200)
    data = {"t": t, "m": t.copy(), "y": t.copy()}
    prepared = prepare(
        data,
        [("t", "m"), ("m", "y")],
        ac.PathSpecificEffect("t", "y", path_nodes=["m"]),
        accepted=accepted,
        suite=suite,
        bayesian=False,
    )
    result = prepared.estimate(data)
    assert result.ate == pytest.approx(1.0)
    assert len(result.validation.reports) == {"none": 0, "cheap": 1, "full": 2}[suite]
    artifact = artifacts.loads(prepared.export_artifact(payload="query"))
    assert (
        artifacts.loads(
            artifacts.dumps(
                "query",
                artifact.payload,
                variable_names=artifact.variable_names,
                artifact_id=artifact.artifact_id,
            )
        )
        == artifact
    )
