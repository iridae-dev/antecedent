"""1.4 numeric evidence for licensed CPDAG ATE cells."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np
import pytest

antecedent = pytest.importorskip("antecedent")


_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PIN = json.loads(
    (_ROOT / "conformance" / "estimate" / "cpdag_ate_envelope" / "expected.json").read_text()
)


def _expand_contingency(pin: dict[str, Any]) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {name: [] for name in pin["columns"]}
    for cell in pin["contingency_table"]:
        count = int(cell["count"])
        assert count > 0
        for name in pin["columns"]:
            values[name].extend([float(cell[name])] * count)
    lengths = {len(column) for column in values.values()}
    assert len(lengths) == 1
    return {name: np.asarray(column, dtype=np.float64) for name, column in values.items()}


def _query(pin: dict[str, Any]) -> antecedent.AverageEffect:
    spec = pin["query"]
    return antecedent.AverageEffect(
        treatment=spec["treatment"],
        outcome=spec["outcome"],
        control_level=float(spec["control_level"]),
        active_level=float(spec["active_level"]),
    )


def _cpdag(*, accepted: bool):
    spec = _PIN["graph"]
    graph = antecedent.Cpdag.from_directed_undirected(
        _PIN["columns"],
        [tuple(edge) for edge in spec["directed_edges"]],
        [tuple(edge) for edge in spec["undirected_edges"]],
    )
    if not accepted:
        return graph
    return antecedent.AcceptedGraph.from_graph(graph, algorithm_id="fixture.cpdag-envelope")


_VALIDATIONS = [
    pytest.param(False, None, id="none"),
    pytest.param("cheap", "overlap+evalue", id="cheap"),
    pytest.param("full", "validation.full", id="full"),
]


def _assert_common_cell_contract(
    fresh,
    click,
    prepared,
    *,
    accepted: bool,
    validation_suite: str | None,
    identifier: str,
    estimator: str,
    status: str,
    expected_ate: float,
    tolerance: float,
) -> None:
    expected_source = "accepted" if accepted else "explicit"
    assert prepared.structure_source == expected_source
    assert prepared.evidence_status == "licensed"
    assert fresh.evidence_status == click.evidence_status == "licensed"
    assert fresh.plan.validation_suite == click.plan.validation_suite == validation_suite
    assert fresh.plan.identifier == click.plan.identifier == identifier
    assert fresh.plan.estimator == click.plan.estimator == estimator
    assert fresh.estimate.estimator_id == click.estimate.estimator_id == estimator
    assert fresh.identification.status == click.identification.status == status
    assert fresh.ate == pytest.approx(expected_ate, abs=tolerance)
    assert click.ate == pytest.approx(expected_ate, abs=tolerance)
    assert all(
        not diagnostic.startswith("exec.identify.cached") for diagnostic in fresh.diagnostics
    )
    assert any(diagnostic.startswith("exec.identify.cached") for diagnostic in click.diagnostics)
    if validation_suite is None:
        assert not fresh.validation.reports
        assert not click.validation.reports
        assert fresh.validation.prior_predictive is None
        assert click.validation.prior_predictive is None
    else:
        assert fresh.validation.ran and click.validation.ran
        assert fresh.validation.reports and click.validation.reports
        if estimator == "bayesian.gcomp":
            assert fresh.validation.prior_predictive is not None
            assert fresh.validation.posterior_predictive is not None
            assert click.validation.prior_predictive is not None
            assert click.validation.posterior_predictive is not None
            if validation_suite == "validation.full":
                assert fresh.validation.prior_sensitivity is not None
                assert click.validation.prior_sensitivity is not None


@pytest.mark.parametrize("accepted", [False, True], ids=["explicit", "accepted"])
@pytest.mark.parametrize("validation, validation_suite", _VALIDATIONS)
@pytest.mark.parametrize("inference_name", ["frequentist", "bayesian"])
def test_cpdag_ate_envelope_numeric_pin(
    accepted: bool,
    validation,
    validation_suite: str | None,
    inference_name: str,
) -> None:
    data = _expand_contingency(_PIN)
    graph = _cpdag(accepted=accepted)
    query = _query(_PIN)
    section = _PIN[inference_name]
    if inference_name == "bayesian":
        inference = antecedent.Bayesian(
            backend=section["backend"],
            n_draws=int(section["n_draws"]),
            prior_scale=float(section["prior_scale"]),
        )
        seed = int(section["seed"])
    else:
        inference = None
        seed = 1

    fresh = antecedent.analyze(
        data,
        graph=graph,
        query=query,
        inference=inference,
        refute=validation,
        bootstrap=0,
        seed=seed,
    )
    prepared = antecedent.estimation.PreparedAnalysis.prepare(
        data,
        graph=graph,
        query=query,
        inference=inference,
        refute=validation,
        bootstrap=0,
        seed=seed,
        latency="interactive",
    )
    click = prepared.estimate(data, seed=seed)

    _assert_common_cell_contract(
        fresh,
        click,
        prepared,
        accepted=accepted,
        validation_suite=validation_suite,
        identifier=_PIN["identification"]["identifier"],
        estimator=section["estimator"],
        status=_PIN["identification"]["status"],
        expected_ate=float(section["expected_ate"]),
        tolerance=float(section["absolute_tolerance"]),
    )
    if inference_name == "frequentist":
        mass = _PIN["identification"]
        envelope_fact = (
            f"identified_mass={mass['identified_mass']:g}, "
            f"unidentified_mass={mass['unidentified_mass']:g}, "
            f"cases={mass['completion_count']}"
        )
        assert any(envelope_fact in diagnostic for diagnostic in fresh.diagnostics)
        assert any(envelope_fact in diagnostic for diagnostic in click.diagnostics)
        assert fresh.posterior is click.posterior is None
    else:
        for result in (fresh, click):
            assert result.posterior is not None
            assert result.posterior.backend == section["posterior_backend"]
            assert result.posterior.n_draws == section["n_draws"]
            assert result.posterior.unidentified_mass == pytest.approx(
                _PIN["identification"]["unidentified_mass"], abs=1e-15
            )
            assert result.posterior.effect_mean == pytest.approx(
                section["expected_ate"], abs=section["absolute_tolerance"]
            )


def test_cpdag_same_schema_refresh_reuses_identification() -> None:
    data = _expand_contingency(_PIN)
    prepared = antecedent.estimation.PreparedAnalysis.prepare(
        data,
        graph=_cpdag(accepted=False),
        query=_query(_PIN),
        refute=False,
        bootstrap=0,
        seed=1,
        latency="interactive",
    )
    click = prepared.estimate(data, seed=1)
    refreshed = prepared.refresh(data, seed=1)
    expected = float(_PIN["frequentist"]["expected_ate"])
    tolerance = float(_PIN["frequentist"]["absolute_tolerance"])
    assert click.ate == pytest.approx(expected, abs=tolerance)
    assert refreshed.ate == pytest.approx(expected, abs=tolerance)
    assert any(diagnostic.startswith("exec.identify.cached") for diagnostic in click.diagnostics)
    assert any(
        diagnostic.startswith("exec.identify.cached") for diagnostic in refreshed.diagnostics
    )


def test_cpdag_numeric_pin_matches_recorded_functionals() -> None:
    data = _expand_contingency(_PIN)
    t, y, z = (data[name] for name in ("t", "y", "z"))
    conditional = [
        float(np.mean(y[(t == 1.0) & (z == z_level)]))
        - float(np.mean(y[(t == 0.0) & (z == z_level)]))
        for z_level in (0.0, 1.0)
    ]
    assert conditional == pytest.approx([0.4, 0.4], abs=1e-15)
    unadjusted = float(np.mean(y[t == 1.0])) - float(np.mean(y[t == 0.0]))
    assert unadjusted == pytest.approx(0.52, abs=1e-15)
    effects = _PIN["identification"]["completion_effects"]
    assert effects == pytest.approx([conditional[0], unadjusted], abs=1e-15)
    assert sum(effects) / len(effects) == pytest.approx(
        _PIN["frequentist"]["expected_ate"], abs=1e-15
    )
