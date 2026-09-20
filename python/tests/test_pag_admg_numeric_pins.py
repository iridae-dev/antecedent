"""1.1 numeric evidence for the already-licensed PAG and ADMG ATE cells.

The earlier fixtures certified generalized-adjustment/general-ID identification.
These tests consume frozen empirical laws and also pin the effect number returned by
every explicit/accepted and validation variant licensed in ``support_licensed.toml``.
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pytest

from _repo_text import load_json

antecedent = pytest.importorskip("antecedent")


_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load_pin(name: str) -> dict[str, Any]:
    path = _ROOT / "conformance" / "estimate" / name / "expected.json"
    return load_json(path)


_PAG_PIN = _load_pin("pag_ate_envelope")
_ADMG_PIN = _load_pin("admg_frontdoor_functional")
_ADMG_DIST_PIN = _load_pin("admg_frontdoor_distribution")


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
    # One-call analysis now executes its retained preparation as well.
    assert any(diagnostic.startswith("exec.identify.cached") for diagnostic in fresh.diagnostics)
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
def test_pag_ate_envelope_numeric_pin(
    accepted: bool, validation, validation_suite: str | None, inference_name: str
) -> None:
    """Visible mixed MAGs retain all twelve licensed estimation/validation cells."""
    from test_mag_adjustment_visibility import _case

    data, graph = _case(True)
    if accepted:
        graph = antecedent.AcceptedGraph.from_graph(graph)
    query = antecedent.AverageEffect("t", "y")
    inference = antecedent.Bayesian(prior_scale=100.0) if inference_name == "bayesian" else None
    kwargs = dict(
        graph=graph, query=query, inference=inference, refute=validation, bootstrap=0, seed=1
    )
    fresh = antecedent.analyze(data, **kwargs)
    prepared = antecedent.estimation.PreparedAnalysis.prepare(data, **kwargs)
    click = prepared.estimate(data, seed=1)
    _assert_common_cell_contract(
        fresh,
        click,
        prepared,
        accepted=accepted,
        validation_suite=validation_suite,
        identifier="generalized.adjustment",
        estimator="bayesian.gcomp" if inference_name == "bayesian" else "linear.adjustment.ate",
        status="NonparametricallyIdentified",
        expected_ate=2.0,
        tolerance=0.05,
    )


def test_old_invisible_pag_has_no_adjustment_certificate():
    identified = antecedent.identify(graph=_pag(accepted=False), query=_query(_PAG_PIN))
    assert identified.status == "NotIdentified"
    assert identified.certificate["identified_weight"] == 0.0
    assert identified.certificate["unidentified_weight"] == 3.0


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


def test_pag_same_schema_refresh_reuses_identification() -> None:
    from test_mag_adjustment_visibility import _case

    data, graph = _case(True)
    prepared = antecedent.estimation.PreparedAnalysis.prepare(
        data,
        graph=graph,
        query=antecedent.AverageEffect("t", "y"),
        refute=False,
        bootstrap=0,
        seed=1,
        latency="interactive",
    )
    click = prepared.estimate(data, seed=1)
    refreshed = prepared.refresh(data, seed=1)
    expected = 2.0
    tolerance = 1e-10
    assert click.ate == pytest.approx(expected, abs=tolerance)
    assert refreshed.ate == pytest.approx(expected, abs=tolerance)
    assert any(diagnostic.startswith("exec.identify.cached") for diagnostic in click.diagnostics)
    assert any(
        diagnostic.startswith("exec.identify.cached") for diagnostic in refreshed.diagnostics
    )


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
    # These are observational contrasts, not identified effects for the MAG.

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


def _admg_distribution(*, accepted: bool):
    spec = _ADMG_DIST_PIN["graph"]
    graph = antecedent.Admg.from_edges(
        _ADMG_DIST_PIN["columns"],
        [tuple(edge) for edge in spec["directed_edges"]],
        bidirected=[tuple(edge) for edge in spec["bidirected_edges"]],
    )
    if not accepted:
        return graph
    return antecedent.AcceptedGraph.from_graph(graph, algorithm_id="fixture.frontdoor")


@pytest.mark.parametrize("accepted", [False, True], ids=["explicit", "accepted"])
def test_admg_interventional_distribution_numeric_pin(accepted: bool) -> None:
    """Pin licensed ADMG InterventionalDistribution to the front-door SCM truth.

    conformance/estimate/admg_frontdoor_distribution: P(Y=1|do(T=1))=0.625 and
    P(Y=1|do(T=0))=0.375 on the exact 800-row law.
    """
    data = _expand_contingency(_ADMG_DIST_PIN)
    graph = _admg_distribution(accepted=accepted)
    tolerance = float(_ADMG_DIST_PIN["frequentist"]["absolute_tolerance"])
    truth = {float(row["t"]): float(row["p_y1"]) for row in _ADMG_DIST_PIN["truth"]}
    for level, expected in truth.items():
        query = antecedent.InterventionalDistribution("y", interventions={"t": level})
        fresh = antecedent.analyze(
            data, graph=graph, query=query, refute=False, bootstrap=0, seed=1
        )
        prepared = antecedent.estimation.PreparedAnalysis.prepare(
            data, graph=graph, query=query, refute=False, bootstrap=0, seed=1
        )
        click = prepared.estimate(data, seed=1)
        assert fresh.ate == pytest.approx(expected, abs=tolerance)
        assert click.ate == pytest.approx(expected, abs=tolerance)
        one = next(a for a in fresh.estimate.distribution if a.outcomes[0][1] == 1.0)
        assert one.probability == pytest.approx(expected, abs=tolerance)


@pytest.mark.parametrize("refute", ["cheap", "full"])
def test_admg_interventional_distribution_refuses_cheap_full(refute: str) -> None:
    """Cheap/full stay closed on ADMG InterventionalDistribution."""
    from antecedent.errors import CausalUnsupportedError

    data = _expand_contingency(_ADMG_DIST_PIN)
    graph = _admg_distribution(accepted=False)
    query = antecedent.InterventionalDistribution("y", interventions={"t": 1.0})
    with pytest.raises(CausalUnsupportedError):
        antecedent.analyze(data, graph=graph, query=query, refute=refute)


@pytest.mark.parametrize("accepted", [False, True], ids=["explicit", "accepted"])
def test_admg_response_pins_against_distribution(accepted: bool) -> None:
    """Licensed ADMG InterventionResponse / ResponseCurve match distribution means."""
    data = _expand_contingency(_ADMG_PIN)
    graph = _admg(accepted=accepted)
    tolerance = float(_ADMG_PIN["frequentist"]["absolute_tolerance"])
    dist_means = {}
    for level in (0.0, 1.0):
        dist = antecedent.analyze(
            data,
            graph=graph,
            query=antecedent.InterventionalDistribution("y", interventions={"t": level}),
            refute=False,
            bootstrap=0,
            seed=1,
        )
        ir = antecedent.analyze(
            data,
            graph=graph,
            query=antecedent.InterventionResponse(
                "y", intervention=antecedent.intervention.Set("t", level)
            ),
            refute=False,
            bootstrap=0,
            seed=1,
        )
        prepared = antecedent.estimation.PreparedAnalysis.prepare(
            data,
            graph=graph,
            query=antecedent.InterventionResponse(
                "y", intervention=antecedent.intervention.Set("t", level)
            ),
            refute=False,
            bootstrap=0,
            seed=1,
        )
        click = prepared.estimate(data, seed=1)
        dist_means[level] = float(dist.ate)
        assert ir.ate == pytest.approx(dist_means[level], abs=tolerance)
        assert click.ate == pytest.approx(dist_means[level], abs=tolerance)
        assert ir.response is not None
    curve = antecedent.analyze(
        data,
        graph=graph,
        query=antecedent.ResponseCurve("t", "y", grid=[0.0, 1.0]),
        refute=False,
        bootstrap=0,
        seed=1,
    )
    assert curve.response is not None
    values = list(curve.response.values)
    assert values[0][0] == pytest.approx(dist_means[0.0], abs=tolerance)
    assert values[1][0] == pytest.approx(dist_means[1.0], abs=tolerance)
    ate = antecedent.analyze(
        data,
        graph=graph,
        query=_query(_ADMG_PIN),
        refute=False,
        bootstrap=0,
        seed=1,
    )
    assert (dist_means[1.0] - dist_means[0.0]) == pytest.approx(ate.ate, abs=tolerance)
