"""1.1 numeric evidence for the already-licensed PAG and ADMG ATE cells.

The earlier fixtures certified generalized-adjustment/general-ID identification.
These tests consume frozen empirical laws and also pin the effect number returned by
every explicit/accepted and validation variant licensed in ``support_licensed.toml``.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np
import pytest

antecedent = pytest.importorskip("antecedent")


_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load_pin(name: str) -> dict[str, Any]:
    path = _ROOT / "conformance" / "estimate" / name / "expected.json"
    return json.loads(path.read_text())


_PAG_PIN = _load_pin("pag_ate_envelope")
_ADMG_PIN = _load_pin("admg_frontdoor_functional")


def _expand_contingency(pin: dict[str, Any]) -> dict[str, np.ndarray]:
    """Expand the fixture's exact integer empirical law in its frozen column order."""
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


def _pag(*, accepted: bool):
    spec = _PAG_PIN["graph"]
    graph = antecedent.Pag.from_marked_edges(
        _PAG_PIN["columns"], [tuple(edge) for edge in spec["marked_edges"]]
    )
    if not accepted:
        return graph
    return antecedent.AcceptedGraph.from_graph(graph, algorithm_id="fixture.pag-envelope")


def _admg(*, accepted: bool):
    spec = _ADMG_PIN["graph"]
    graph = antecedent.Admg.from_edges(
        _ADMG_PIN["columns"],
        [tuple(edge) for edge in spec["directed_edges"]],
        bidirected=[tuple(edge) for edge in spec["bidirected_edges"]],
    )
    if not accepted:
        return graph
    return antecedent.AcceptedGraph.from_graph(graph, algorithm_id="fixture.frontdoor")


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


@pytest.mark.parametrize("accepted", [False, True], ids=["explicit", "accepted"])
@pytest.mark.parametrize("validation, validation_suite", _VALIDATIONS)
@pytest.mark.parametrize("inference_name", ["frequentist", "bayesian"])
def test_pag_ate_envelope_numeric_pin(
    accepted: bool,
    validation,
    validation_suite: str | None,
    inference_name: str,
) -> None:
    """Pin all 12 licensed PAG ATE structure/inference/validation cells."""
    data = _expand_contingency(_PAG_PIN)
    graph = _pag(accepted=accepted)
    query = _query(_PAG_PIN)
    section = _PAG_PIN[inference_name]
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
        identifier=_PAG_PIN["identification"]["identifier"],
        estimator=section["estimator"],
        status=_PAG_PIN["identification"]["status"],
        expected_ate=float(section["expected_ate"]),
        tolerance=float(section["absolute_tolerance"]),
    )
    if inference_name == "frequentist":
        mass = _PAG_PIN["identification"]
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
                _PAG_PIN["identification"]["unidentified_mass"], abs=1e-15
            )
            assert result.posterior.effect_mean == pytest.approx(
                section["expected_ate"], abs=section["absolute_tolerance"]
            )


@pytest.mark.parametrize("accepted", [False, True], ids=["explicit", "accepted"])
@pytest.mark.parametrize("validation, validation_suite", _VALIDATIONS)
def test_admg_frontdoor_functional_effect_numeric_pin(
    accepted: bool,
    validation,
    validation_suite: str | None,
) -> None:
    """Pin all six licensed Frequentist ADMG ATE structure/validation cells."""
    data = _expand_contingency(_ADMG_PIN)
    graph = _admg(accepted=accepted)
    query = _query(_ADMG_PIN)
    section = _ADMG_PIN["frequentist"]
    seed = 1

    fresh = antecedent.analyze(
        data,
        graph=graph,
        query=query,
        refute=validation,
        bootstrap=0,
        seed=seed,
    )
    prepared = antecedent.estimation.PreparedAnalysis.prepare(
        data,
        graph=graph,
        query=query,
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
        identifier=_ADMG_PIN["identification"]["identifier"],
        estimator=section["estimator"],
        status=_ADMG_PIN["identification"]["status"],
        expected_ate=float(section["expected_ate"]),
        tolerance=float(section["absolute_tolerance"]),
    )
    assert fresh.posterior is click.posterior is None


def test_numeric_pin_laws_match_their_recorded_functionals() -> None:
    """Keep the JSON pins analytic rather than turning them into magic outputs."""
    pag_data = _expand_contingency(_PAG_PIN)
    pag_t, pag_y, pag_z = (pag_data[name] for name in ("t", "y", "z"))
    conditional_effects = [
        float(np.mean(pag_y[(pag_t == 1.0) & (pag_z == z_level)]))
        - float(np.mean(pag_y[(pag_t == 0.0) & (pag_z == z_level)]))
        for z_level in (0.0, 1.0)
    ]
    assert conditional_effects == pytest.approx([0.4, 0.4], abs=1e-15)
    unadjusted_effect = float(np.mean(pag_y[pag_t == 1.0])) - float(np.mean(pag_y[pag_t == 0.0]))
    assert unadjusted_effect == pytest.approx(0.52, abs=1e-15)
    pag_effects = _PAG_PIN["identification"]["completion_effects"]
    assert pag_effects == pytest.approx(
        [conditional_effects[0], unadjusted_effect, conditional_effects[1]], abs=1e-15
    )
    assert sum(pag_effects) / len(pag_effects) == pytest.approx(
        _PAG_PIN["frequentist"]["expected_ate"], abs=1e-15
    )

    data = _expand_contingency(_ADMG_PIN)
    t, m, y = (data[name] for name in ("t", "m", "y"))
    p_t = {level: float(np.mean(t == level)) for level in (0.0, 1.0)}
    g = {
        m_level: sum(
            p_t[t_level] * float(np.mean(y[(m == m_level) & (t == t_level)]))
            for t_level in (0.0, 1.0)
        )
        for m_level in (0.0, 1.0)
    }

    def response(t_level: float) -> float:
        selected = t == t_level
        return sum(float(np.mean(m[selected] == m_level)) * g[m_level] for m_level in (0.0, 1.0))

    functional = response(1.0) - response(0.0)
    assert functional == pytest.approx(_ADMG_PIN["frequentist"]["expected_ate"], abs=1e-15)
