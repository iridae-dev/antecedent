"""Cross-surface slice cases on the public Python API.

Scientific inputs and hand values are `conformance/cross_surface/slice_cases.json`.
Each case drops the producer and reads an independently loaded artifact.
Specified feedback has no Python binding; the Rust contract test covers it.
"""

from __future__ import annotations

import json

import pytest
from antecedent import (
    Admg,
    AverageEffect,
    Bayesian,
    Frequentist,
    PulseEffect,
    ResponseCurve,
    analyze,
    load,
)
from antecedent import evidence_ledger as ledger_api
from antecedent.inference import MeasuredInference
from antecedent.transport import advanced as transport

from _repo_text import REPO_ROOT, load_json
from test_observation_recovery import REQUESTS, decide, observed_law, truth

CASES = load_json(REPO_ROOT / "conformance" / "cross_surface" / "slice_cases.json")


def _gaussian_cell(horizon: bool, bayesian: bool):
    gaussian = CASES["gaussian"]
    if horizon:
        data = {"t": gaussian["horizon_treatment"], "y": gaussian["horizon_outcome"]}
        graph = [("t", 1, "y", 0)]
        query = PulseEffect("t", "y", treatment_lag=1, horizon_steps=1)
        tolerance = 1e-4 if bayesian else 1e-8
        label = "gaussian-horizon-bayesian" if bayesian else "gaussian-horizon-frequentist"
    else:
        data = {"t": gaussian["treatment"], "y": gaussian["outcome"]}
        graph = [("t", "y")]
        query = AverageEffect("t", "y")
        tolerance = 1e-6 if bayesian else 1e-12
        label = "gaussian-static-bayesian" if bayesian else "gaussian-static-frequentist"
    inference = (
        Bayesian(
            n_draws=gaussian["posterior_draws"],
            prior_scale=gaussian["prior_scale"],
            backend="conjugate",
        )
        if bayesian
        else Frequentist()
    )
    result = analyze(
        data,
        graph=graph,
        query=query,
        inference=inference,
        bootstrap=0,
        refute="none",
        seed=1,
    )
    assert result.estimate.ate == pytest.approx(gaussian["effect"], abs=tolerance)
    if bayesian:
        assert result.posterior is not None
        assert result.posterior.effect_mean == pytest.approx(gaussian["effect"], abs=0.05)
        assert result.posterior.interval_type == "equal_tailed_95"
    else:
        assert result.posterior is None
    artifact = result.export(artifact_id=label)
    del result
    loaded = load(artifact)
    assert loaded.acceptance.status == "verified"
    assert loaded.decoded_artifact.payload["estimate"] == pytest.approx(
        gaussian["effect"], abs=tolerance
    )


def test_four_way_gaussian_cells_reload_from_independent_artifacts():
    for horizon in (False, True):
        for bayesian in (False, True):
            _gaussian_cell(horizon, bayesian)


def _discrete_program(regime: str):
    discrete = CASES["finite_discrete"]
    graph = Admg.from_edges(["t", "z", "y"], [("z", "t"), ("z", "y"), ("t", "y")])
    catalog = transport.EvidenceCatalog(
        regimes=[
            transport.EvidenceRegime(regime, discrete["population"], measured=["t", "z", "y"])
        ]
    )
    query = transport.TransportQuery(
        ResponseCurve("t", "y", grid=[0.0, 1.0]),
        transport.SelectionDiagram("source", discrete["population"], []),
        catalog=catalog,
    )
    identification = transport.identify(graph=graph, query=query)
    return identification, transport.reload_lowered_program(identification), catalog


def _law(counts, regime: str, snapshot: str):
    discrete = CASES["finite_discrete"]
    total = sum(counts)
    return transport.ExactDiscreteLaw(
        discrete["population"],
        regime,
        (("t", (0.0, 1.0)), ("z", (0.0, 1.0)), ("y", (0.0, 1.0))),
        tuple(count / total for count in counts),
        snapshot,
        empirical_counts=tuple(counts) if snapshot == discrete["snapshot"] else None,
    )


def _contrast(program, catalog, law) -> float:
    active = program.evaluate_exact(catalog, (law,), {"t": 1.0, "y": 1.0})
    control = program.evaluate_exact(catalog, (law,), {"t": 0.0, "y": 1.0})
    return active - control


def test_finite_discrete_adjustment_reloads_from_the_law():
    discrete = CASES["finite_discrete"]
    counts = discrete["counts"]
    for regime in (str(discrete["static_regime"]), str(discrete["horizon_regime"])):
        identification, program, catalog = _discrete_program(regime)
        law = _law(counts, regime, discrete["snapshot"])
        assert _contrast(program, catalog, law) == pytest.approx(discrete["contrast"])
        wire = program.to_wire_json()
        del program
        restored = transport.restore_lowered_program(identification, wire)
        assert _contrast(restored, catalog, law) == pytest.approx(discrete["contrast"])
    mixed = 0.0
    regime = str(discrete["static_regime"])
    identification, program, catalog = _discrete_program(regime)
    for index, draw in enumerate(discrete["posterior_draws"]):
        law = _law(draw["counts"], regime, f"draw-{index}")
        contrast = _contrast(program, catalog, law)
        assert contrast == pytest.approx(draw["contrast"])
        mixed += draw["weight"] * contrast
    assert mixed == pytest.approx(discrete["posterior_mean"])
    del identification


def _sources(session):
    mixed = CASES["mixed_evidence"]
    sources = []
    for index, (start, end) in enumerate(mixed["observation_ranges"]):
        sources.append(
            session.add_source(
                ledger_api.SourceDeclaration(
                    f"study_{index}",
                    "observation_set",
                    observations=ledger_api.rows("cohort", range(start, end)),
                )
            )
        )
    return [
        ledger_api.StudySummary(source, estimate, variance)
        for source, estimate, variance in zip(
            sources, mixed["estimates"], mixed["variances"], strict=True
        )
    ]


def test_mixed_evidence_pools_and_reloads_from_the_ledger():
    mixed = CASES["mixed_evidence"]
    session = ledger_api.EvidenceLedger()
    pooled = session.pool_summaries("pool", _sources(session))
    assert isinstance(pooled.value, ledger_api.FixedEffect)
    assert pooled.value.estimate == pytest.approx(mixed["pooled"])
    assert pooled.value.variance == pytest.approx(mixed["pooled_variance"])
    declared = sum(end - start for start, end in mixed["observation_ranges"])
    assert pooled.mass.distinct_observations == declared
    exported = session.export("mixed-evidence")
    del session
    consumed = ledger_api.consume_evidence_ledger_artifact(exported.artifact, exported.identity)
    replay = consumed.synthesis_of(pooled.result)
    assert replay.values[0] == pytest.approx(mixed["pooled"])
    assert replay.values[1] == pytest.approx(mixed["pooled_variance"])


def test_reused_evidence_keeps_one_observation_count():
    session = ledger_api.EvidenceLedger()
    pooled = session.pool_summaries("pool", _sources(session))
    first = session.reuse_result("cited_once", pooled.result)
    second = session.reuse_result("cited_again", pooled.result)
    assert first != second
    counted = pooled.mass.distinct_observations
    for result_id in (pooled.result, first, second):
        assert session.result(result_id).mass.distinct_observations == counted
    exported = session.export("reused-evidence")
    del session
    consumed = ledger_api.consume_evidence_ledger_artifact(exported.artifact, exported.identity)
    labels = {record.label: record.mass.distinct_observations for record in consumed.results}
    assert labels["pool"] == counted
    assert labels["cited_once"] == counted
    assert labels["cited_again"] == counted


def _panel():
    temporal = CASES["temporal"]
    rows = []
    for unit in range(temporal["units"]):
        shock = 0.2 if unit % 2 else -0.2
        time = 0
        for s0 in range(2):
            for a1 in range(2):
                for l2 in range(2):
                    for a2 in range(2):
                        mean = 0.3 + 0.1 * l2 + 0.1 * s0 + 0.2 * s0 * l2 + 0.05 * (a1 + a2)
                        rows.append((unit, time, s0, a1, l2, a2, mean + shock))
                        time += 1
    return transport.TemporalUnitPanel.from_rows(temporal["snapshot"], rows)


def test_temporal_dependence_reloads_from_the_measured_artifact():
    temporal = CASES["temporal"]
    result = transport.temporal_dependent_interval(
        _panel(),
        sequence=tuple(temporal["sequence"]),
        target_law=transport.InitialStateLaw(
            temporal["target_snapshot"],
            {level: mass for level, mass in temporal["target_law"]},
        ),
        seed=temporal["seed"],
        replicates=temporal["replicates"],
        min_units=temporal["min_units"],
    )
    assert result.scalar("response").point == pytest.approx(temporal["point"], abs=1e-12)
    artifact = result.export()
    expected = result.expected_identity
    del result
    loaded = MeasuredInference.load(artifact, expected=expected)
    assert loaded.scalar("response").point == pytest.approx(temporal["point"], abs=1e-12)


def test_recovered_feedback_reloads_the_observation_law():
    recovery = CASES["recovery"]
    law = observed_law(snapshot=recovery["python_snapshot"])
    assert law.population == recovery["population"]
    assert law.snapshot_identity == recovery["python_snapshot"]
    builder = decide()
    prepared = builder.prepare_exact(law, REQUESTS)
    del builder
    live = json.loads(prepared.estimate())
    artifact = prepared.export()
    del prepared
    consumed = json.loads(transport.consume_observation_recovery_artifact(artifact))
    assert consumed["recovered"]["probabilities"] == pytest.approx(
        live["recovered"]["probabilities"]
    )
    assert consumed["recovered"]["probabilities"] == pytest.approx(truth())
    assert consumed["recovered"]["origin"] == "recovered"


def test_reduced_retention_serves_only_the_summary():
    from antecedent import _native
    from antecedent.errors import CausalStateError

    name = CASES["retention"]["reduced"]
    mode = _native.RetentionMode.summary_only()
    assert mode.as_str() == name
    mode.require(_native.RetainedOperation.reuse_summary())
    with pytest.raises(CausalStateError, match="summary-only cannot serve row reconstruction"):
        mode.require(_native.RetainedOperation.row_reconstruction())
