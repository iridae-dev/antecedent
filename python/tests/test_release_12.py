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


def _path_specific_chain():
    t_vals: list[float] = []
    m_vals: list[float] = []
    y_vals: list[float] = []
    for t, m, y, count in (
        (0.0, 0.0, 0.0, 40),
        (0.0, 0.0, 1.0, 10),
        (0.0, 1.0, 0.0, 10),
        (0.0, 1.0, 1.0, 40),
        (1.0, 0.0, 0.0, 10),
        (1.0, 0.0, 1.0, 10),
        (1.0, 1.0, 0.0, 10),
        (1.0, 1.0, 1.0, 70),
    ):
        t_vals.extend([t] * count)
        m_vals.extend([m] * count)
        y_vals.extend([y] * count)
    return {
        "t": np.asarray(t_vals, dtype=np.float64),
        "m": np.asarray(m_vals, dtype=np.float64),
        "y": np.asarray(y_vals, dtype=np.float64),
    }


@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("scale", [0.1, 10.0])
@pytest.mark.parametrize(
    "family",
    ["conditional", "mediation", "window", "response", "stationary_window", "confounded_mediation"],
)
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
    elif family == "confounded_mediation":
        data = {"z": source["z"], "t": source["tc"], "m": source["mc"], "y": source["oc"]}
        graph = [
            ("z", 0, "t", 0),
            ("z", 1, "m", 0),
            ("z", 1, "y", 0),
            ("t", 1, "m", 0),
            ("t", 1, "y", 0),
            ("m", 0, "y", 0),
        ]
        query = ac.TemporalMediationEffect("t", "m", "y")
    elif family == "stationary_window":
        data = {"t": source["t"], "m": source["m"], "y": source["u"]}
        graph = [("t", 1, "m", 0), ("m", 1, "y", 0), ("m", 2, "y", 0)]
        query = ac.SustainedEffect("t", "y", window=(-3, -2))
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
            if family in {"mediation", "confounded_mediation"}
            else "sustained_window"
            if family in {"window", "stationary_window"}
            else "ate"
        )
        check_moments(art.mean[effect_index], art.sd[effect_index] ** 2, expected)
        assert result.posterior is not None
        assert art.mean[effect_index] == result.posterior.effect_mean
        if family in {"mediation", "confounded_mediation"}:
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
    with pytest.raises(ac.errors.CausalEstimateError, match="derived-contrast chain diagnostics"):
        prepared.estimate(data)


@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("suite", ["none", "cheap", "full"])
def test_path_functional_and_query_artifact(accepted, suite):
    data = _path_specific_chain()
    prepared = prepare(
        data,
        [("t", "m"), ("m", "y")],
        ac.PathSpecificEffect("t", "y", path_nodes=["m"]),
        accepted=accepted,
        suite=suite,
        bayesian=False,
    )
    result = prepared.estimate(data)
    assert result.ate == pytest.approx(0.3)
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


@pytest.mark.parametrize("temporal", [False, True])
@pytest.mark.parametrize("intervention", [False, True])
@pytest.mark.parametrize("entry", ["prepare", "analyze"])
@pytest.mark.parametrize("field", ["observation", "assumptions", "population"])
def test_bayesian_response_preserves_or_refuses_observation_contract(
    temporal, intervention, entry, field
):
    from antecedent import observation, population

    t = np.linspace(-2, 2, 120)
    data = {
        "t": t,
        "y": np.minimum(1 + t, 1.5),
        "event": (t < 0.5).astype(float),
        "c": np.full(len(t), 1.5),
    }
    options = {"horizons": [1]} if temporal else {}
    if field == "observation":
        options["observation"] = observation.RightCensored("y", "y", "c", "event")
    elif field == "assumptions":
        options["observation_assumptions"] = [observation.IndependentGiven([])]
    else:
        options["target_population"] = population.Treated()
    query = (
        ac.InterventionResponse("y", intervention=ac.intervention.Set("t", 1), **options)
        if intervention
        else ac.ResponseCurve("t", "y", grid=[0, 1], **options)
    )
    graph = [("t", 1, "y", 0)] if temporal else [("t", "y")]
    call = PreparedAnalysis.prepare if entry == "prepare" else ac.analyze
    with pytest.raises(
        ac.errors.CausalUnsupportedError, match="observations|observation_assumptions|AllObserved"
    ):
        call(data, query=query, graph=graph, inference=ac.Bayesian(), refute="none")


@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("suite", ["none", "cheap", "full"])
@pytest.mark.parametrize("bayesian", [False, True])
def test_mediation_graph_confounder_adjusts_estimate_and_refuters(accepted, suite, bayesian):
    rng = np.random.default_rng(120)
    n = 4000
    z = rng.normal(size=n)
    t = z + rng.normal(size=n)
    m = np.r_[0, t[:-1]] + 10 * np.r_[0, z[:-1]] + rng.normal(size=n)
    y = 2 * m + 0.25 * np.r_[0, t[:-1]] + 5 * np.r_[0, z[:-1]] + rng.normal(size=n)
    data = {"z": z, "t": t, "m": m, "y": y}
    graph = [
        ("z", 0, "t", 0),
        ("z", 1, "m", 0),
        ("z", 1, "y", 0),
        ("t", 1, "m", 0),
        ("t", 1, "y", 0),
        ("m", 0, "y", 0),
    ]
    prepared = prepare(
        data,
        graph,
        ac.TemporalMediationEffect("t", "m", "y"),
        accepted=accepted,
        suite=suite,
        bayesian=bayesian,
    )
    result = prepared.estimate(data, seed=120)
    assert result.ate == pytest.approx(2.0, abs=0.15)
    if suite != "none":
        reports = {report.refuter: report for report in result.validation.reports}
        assert reports["mediation.random_common_cause"].refuted_ate == pytest.approx(
            result.ate, abs=0.02
        )
        if suite == "full":
            assert reports["mediation.contiguous_window"].refuted_ate == pytest.approx(2.0, abs=0.2)


@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("suite", ["cheap", "full"])
@pytest.mark.parametrize("supported", [False, True])
def test_continuous_conditional_validation_checks_conditional_support(accepted, suite, supported):
    rng = np.random.default_rng(1218)
    w = rng.normal(size=1000)
    t = rng.normal(size=1000) if supported else w + rng.normal(scale=0.01, size=1000)
    y = 2 * t + 0.5 * w + 0.5 * t * w + rng.normal(size=1000)
    data = {"t": t, "w": w, "y": y}
    graph = [("w", "t"), ("w", "y"), ("t", "y")]
    prepared = prepare(
        data, graph, ac.ConditionalEffect("t", "y", "w"), accepted=accepted, suite=suite
    )
    result = prepared.estimate(data)
    reports = {report.refuter: report for report in result.validation.reports}
    support = reports["overlap.continuous_support"]
    assert support.informative
    assert support.passed == supported
    # Same polarity as binary overlap.assessment: comparison is unsupported mass.
    assert (support.comparison <= 0.5) == supported
    assert "posterior_predictive" in reports
    if suite == "full":
        assert result.validation.prior_sensitivity is not None


@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("family", ["conditional", "pulse", "sustained"])
def test_query_export_preserves_outer_kind_and_original_variable_ids(accepted, family):
    rng = np.random.default_rng(22)
    t, w = rng.normal(size=(2, 300))
    if family == "conditional":
        data = {"t": t, "w": w, "y": 2 * t + 0.5 * t * w + rng.normal(size=300)}
        graph = [("t", "y"), ("w", "y")]
        query = ac.ConditionalEffect("t", "y", "w")
    else:
        data = {"t": t, "y": 2 * np.r_[0, t[:-1]] + 3 * np.r_[0, 0, t[:-2]] + rng.normal(size=300)}
        graph = [("t", 1, "y", 0), ("t", 2, "y", 0)]
        query = (
            ac.PulseEffect("t", "y")
            if family == "pulse"
            else ac.SustainedEffect("t", "y", window=(-2, -1))
        )
    prepared = prepare(data, graph, query, accepted=accepted)
    prepared.estimate(data)
    artifact = artifacts.loads(prepared.export_artifact(payload="query"))
    payload = artifact.payload
    if family == "conditional":
        assert set(payload) == {"conditional_effect"}
        q = payload["conditional_effect"]["inner"]["average_effect"]
        assert q["effect_modifiers"] == [1]
        assert q["treatment"] == 0 and q["outcome"] == 2
    else:
        q = payload["temporal_effect"]
        assert q["treatment"] == 0 and q["outcome"] == 1
        assert q["policy"] == (
            {"pulse": {"at": -1}} if family == "pulse" else {"sustained": {"from": -2, "until": -1}}
        )
        assert q["horizon_steps"] == 1


@pytest.mark.parametrize("accepted", [False, True])
def test_prepared_response_accepts_explicit_complete_all_observed(accepted):
    from antecedent import observation, population

    source = {k: np.asarray(v) for k, v in FIXTURE["data"].items()}
    data = {"t": source["t"], "y": source["r"]}
    query = ac.ResponseCurve(
        "t",
        "y",
        grid=[0, 1],
        observation=observation.Complete(),
        target_population=population.AllRows(),
    )
    p = prepare(data, [("t", "y")], query, accepted=accepted)
    p.estimate(data)
    q = artifacts.loads(p.export_artifact(payload="query")).payload["response"]
    assert q["observation"] == "complete"
    assert q["target_population"] == "all_observed"


@pytest.mark.parametrize("backend", ["conjugate", "laplace"])
def test_stationary_sustained_intervals_cover_shared_mechanism_truth(backend):
    # Independent Gaussian innovations, one stationary a in two causal paths.
    # The old independent-time-copy fit covered only 83% at the nominal 95% level.
    graph = [("t", 1, "m", 0), ("m", 1, "y", 0), ("m", 2, "y", 0)]
    covered, points, sds = [], [], []
    for seed in range(200):
        rng = np.random.default_rng(1400 + seed)
        n = 300
        t = rng.normal(size=n)
        m = np.r_[0, t[:-1]] + rng.normal(size=n)
        y = np.r_[0, m[:-1]] + np.r_[0, 0, m[:-2]] + rng.normal(scale=0.01, size=n)
        data = {"t": t, "m": m, "y": y}
        p = PreparedAnalysis.prepare(
            data,
            query=ac.SustainedEffect("t", "y", window=(-3, -2)),
            graph=graph,
            inference=ac.Bayesian(backend=backend, n_draws=4096),
            refute="none",
            latency=None,
        )
        result = p.estimate(data, seed=123)
        draws = np.asarray(decode_posterior_artifact(p.export_artifact()).draws)
        lower, upper = np.quantile(draws, [0.025, 0.975])
        covered.append(lower <= 2 <= upper)
        points.append(result.ate)
        sds.append(result.posterior.effect_sd)
    # A finite Monte Carlo gate, not a blanket coverage guarantee.
    assert 0.91 <= np.mean(covered) <= 0.995
    assert np.mean(sds) == pytest.approx(np.std(points, ddof=1), rel=0.15)
