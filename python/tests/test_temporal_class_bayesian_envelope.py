"""1.7 Bayesian temporal-class envelope pins."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

antecedent = pytest.importorskip("antecedent")


_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PIN = json.loads(
    (_ROOT / "conformance" / "bayesian" / "temporal_class_envelope" / "expected.json").read_text()
)
_TRANSFER = json.loads(
    (
        _ROOT / "conformance" / "bayesian" / "temporal_class_prior_transfer" / "expected.json"
    ).read_text()
)
_OBS = json.loads(
    (
        _ROOT / "conformance" / "response" / "temporal_class_observation" / "expected.json"
    ).read_text()
)


def _series(pin: dict, *, two_lag: bool = False) -> dict[str, np.ndarray]:
    n = int(pin["n"])
    z = np.array([0.0 if i % 2 == 0 else 1.0 for i in range(n)], dtype=np.float64)
    t = 0.3 + 0.4 * z + 0.05 * np.sin(np.arange(n, dtype=np.float64) * 0.017)
    y = np.zeros(n, dtype=np.float64)
    if two_lag:
        y[2:] = 1.0 + 2.0 * t[1:-1] + 3.0 * t[:-2]
    else:
        y[1:] = 1.0 + 2.0 * t[:-1] + 0.5 * z[:-1]
    return {"t": t, "y": y, "z": z}


def _pulse(pin: dict) -> antecedent.PulseEffect:
    spec = pin["query"]
    return antecedent.PulseEffect(
        treatment=spec["treatment"],
        outcome=spec["outcome"],
        treatment_lag=abs(int(spec["treatment_offset"])),
        horizon_steps=int(spec["horizon_steps"]),
        active_level=float(spec["active_level"]),
    )


def _cpdag():
    return antecedent.graph.TemporalCpdag.from_lagged_edges(
        _PIN["columns"],
        [tuple(edge) for edge in _PIN["cpdag"]["directed"]],
        [tuple(edge) for edge in _PIN["cpdag"]["undirected"]],
    )


def test_temporal_class_bayesian_pulse_without_prior_is_identified_set() -> None:
    data = _series(_PIN)
    result = antecedent.analyze(
        data,
        graph=_cpdag(),
        query=_pulse(_PIN),
        inference=antecedent.Bayesian(n_draws=64, backend="conjugate"),
        refute=False,
        bootstrap=0,
        seed=7,
    )
    assert result.posterior is None or result.posterior.n_draws is None
    assert result.ate is None or not np.isfinite(result.ate)
    assert result.structural_weight_basis == "completion_enumeration"
    assert any(
        "estimate.temporal_class.enumeration_not_probability" in diagnostic
        for diagnostic in result.diagnostics
    )


def test_temporal_class_bayesian_pulse_with_class_prior_mixes() -> None:
    data = _series(_PIN)
    prior = antecedent.ClassPrior(ordered=tuple(_PIN["class_prior_ordered"]))
    result = antecedent.analyze(
        data,
        graph=_cpdag(),
        query=_pulse(_PIN),
        inference=antecedent.Bayesian(n_draws=64, backend="conjugate"),
        class_prior=prior,
        refute=False,
        bootstrap=0,
        seed=7,
    )
    assert result.ate is not None and np.isfinite(result.ate)
    assert result.structural_weight_basis == "caller_supplied_class_prior"
    assert result.structural_unidentified_mass == pytest.approx(0.0, abs=1e-9)
    prepared = antecedent.estimation.PreparedAnalysis.prepare(
        data,
        graph=_cpdag(),
        query=_pulse(_PIN),
        inference=antecedent.Bayesian(n_draws=64, backend="conjugate"),
        class_prior=prior,
        refute=False,
        bootstrap=0,
        seed=7,
        latency="interactive",
    )
    click = prepared.estimate(data, seed=7)
    assert any(diagnostic.startswith("exec.identify.cached") for diagnostic in click.diagnostics)


def test_temporal_class_identify_exposes_completion_keys() -> None:
    identified = antecedent.identify(graph=_cpdag(), query=_pulse(_PIN))
    assert identified.status == "PartiallyIdentified"
    assert identified.completion_keys
    assert _PIN["cpdag"]["identification"]["status"] == "PartiallyIdentified"


def test_temporal_class_observation_fixture_names_licensed_pairs() -> None:
    assert _OBS["case"] == "temporal_class_observation"
    assert "Selected x OutcomeIndependentGiven" in _OBS["licensed_pairs"]


def test_temporal_class_prior_transfer_fixture_names_filter() -> None:
    assert _TRANSFER["compatibility_filter"] == "PriorCatalog.filter_compatible"
    assert _TRANSFER["conflict_does_not_flip_identification"] is True


@pytest.mark.parametrize(
    "inference", [antecedent.Frequentist(), antecedent.Bayesian(n_draws=64, backend="conjugate")]
)
def test_class_curve_executes_without_collapsing_graph(inference) -> None:
    from antecedent import artifacts

    query = antecedent.ResponseCurve("t", "y", grid=[0.0, 1.0], horizons=[1, 2])
    prepared = antecedent.estimation.PreparedAnalysis.prepare(
        _series(_PIN),
        graph=_cpdag(),
        query=query,
        inference=inference,
        refute=False,
        bootstrap=0,
        seed=17,
    )
    result = prepared.estimate(_series(_PIN), seed=17)
    assert result.envelope.weight_basis == "completion_enumeration"
    assert len(result.envelope.atom_keys) == 4
    decoded = artifacts.loads(prepared.export_artifact())
    atoms = decoded.payload["structural_response"]["atoms"]
    assert len(atoms) == 4
    assert all(atom["response"] is not None for atom in atoms)


def test_class_curve_prior_survives_artifact_roundtrip() -> None:
    from antecedent import artifacts

    query = antecedent.ResponseCurve("t", "y", grid=[0.0, 1.0], horizons=[1, 2])
    prepared = antecedent.estimation.PreparedAnalysis.prepare(
        _series(_PIN),
        graph=_cpdag(),
        query=query,
        inference=antecedent.Bayesian(n_draws=64, backend="conjugate"),
        class_prior=antecedent.ClassPrior.from_ordered([0.2, 0.8]),
        refute=False,
        bootstrap=0,
        seed=17,
    )
    result = prepared.estimate(_series(_PIN), seed=17)
    assert result.envelope.weight_basis == "caller_supplied_class_prior"
    payload = artifacts.loads(prepared.export_artifact()).payload
    structural = payload["structural_response"]
    assert structural["conditional_on_identified"] is not None
    assert structural["identified_mass"] == pytest.approx(1.0)
    restored = artifacts.loads(
        artifacts.dumps(
            "analysis_result",
            payload,
            artifact_id="roundtrip",
            variable_names=list(_PIN["columns"]),
        )
    )
    assert restored.payload == payload


@pytest.mark.parametrize("masses", [[float("inf")], [float("nan")], [1e308, 1e308]])
def test_class_prior_rejects_nonfinite_total(masses) -> None:
    with pytest.raises(ValueError):
        antecedent.ClassPrior.from_ordered(masses)


@pytest.mark.parametrize("key", [1.5, True, "1", -1, 2**64])
def test_class_prior_does_not_coerce_invalid_completion_keys(key) -> None:
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        antecedent.ClassPrior.from_pairs([(key, 1.0)])


def test_prepared_scalar_keeps_atom_posterior_artifacts() -> None:
    from antecedent import artifacts

    prepared = antecedent.estimation.PreparedAnalysis.prepare(
        _series(_PIN),
        graph=_cpdag(),
        query=_pulse(_PIN),
        inference=antecedent.Bayesian(n_draws=64, backend="conjugate"),
        refute=False,
        bootstrap=0,
    )
    prepared.estimate(_series(_PIN))
    payload = artifacts.loads(prepared.export_artifact()).payload
    assert payload["posterior_artifact"] is None
    assert all(atom["posterior_artifact"] for atom in payload["structural_response"]["atoms"])


def test_temporal_class_identify_preserves_response_schedule() -> None:
    from antecedent.intervention import Sequence, Set

    query = antecedent.InterventionResponse(
        "y",
        intervention=Sequence([Set("t", 1.0), Set("t", 1.0)]),
        horizons=[1, 2],
        treatment_lag=2,
    )
    identified = antecedent.identify(graph=_cpdag(), query=query)
    assert len(identified.completion_keys) == 2
    assert "sequential" in json.dumps(identified.certificate)
    # Sequence ends at -treatment_lag, preserving both preceding interventions.
    assert [node["offset"] for node in identified.certificate["treatments"]] == [-3, -2]
    assert identified.certificate["outcome"]["offset"] == 0


def test_class_mediation_preserves_functional_and_atom_posterior() -> None:
    from antecedent import artifacts

    index = np.arange(300, dtype=np.float64)
    treatment = np.sin(index * 0.31)
    mediator = 0.8 * np.roll(treatment, 1) + 0.1 * np.sin(index * 0.51)
    outcome = 0.25 * np.roll(treatment, 1) + 0.55 * mediator + 0.01 * np.cos(index * 0.19)
    data = {"t": treatment, "m": mediator, "y": outcome}
    graph = antecedent.graph.TemporalCpdag.from_lagged_edges(
        list(data),
        [("t", 1, "m", 0), ("t", 1, "y", 0), ("m", 0, "y", 0)],
        [],
    )
    query = antecedent.TemporalMediationEffect("t", "m", "y", horizons=[1])
    identified = antecedent.identify(graph=graph, query=query)
    assert len(identified.completion_keys) == 1
    assert "temporal_mediation" in json.dumps(identified.certificate)
    assert identified.certificate["treatments"][0]["offset"] == -1
    assert identified.certificate["outcome"]["offset"] == 0
    prepared = antecedent.estimation.PreparedAnalysis.prepare(
        data,
        graph=graph,
        query=query,
        inference=antecedent.Bayesian(n_draws=64, backend="conjugate"),
        class_prior=antecedent.ClassPrior.from_ordered([1.0]),
        refute=False,
        bootstrap=0,
    )
    result = prepared.estimate(data)
    assert result.ate is None
    payload = artifacts.loads(prepared.export_artifact()).payload
    atom = payload["structural_response"]["atoms"][0]
    assert atom["posterior_artifact"]
    assert payload["structural_response"]["conditional_on_identified"] is not None
