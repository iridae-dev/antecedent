"""Numeric evidence for licensed Cpdag/Pag response cells."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import antecedent
import numpy as np
import pytest
from antecedent.outcomes import require_success

from _repo_text import REPO_ROOT, load_json

_ROOT = REPO_ROOT
_PIN = load_json(_ROOT / "conformance" / "response" / "class_aware_envelope" / "expected.json")


def _expand_contingency(pin: dict[str, Any]) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {name: [] for name in pin["columns"]}
    for cell in pin["contingency_table"]:
        count = int(cell["count"])
        for name in pin["columns"]:
            values[name].extend([float(cell[name])] * count)
    return {name: np.asarray(column, dtype=np.float64) for name, column in values.items()}


def _count_derived_intervention_targets(pin: dict[str, Any], level: float) -> tuple[float, float]:
    """The z-adjusted and unadjusted completion targets from the original counts."""
    cells = pin["contingency_table"]
    total = sum(float(cell["count"]) for cell in cells)
    adjusted = 0.0
    selected_count = 0.0
    selected_outcome_sum = 0.0
    for z in (0.0, 1.0):
        stratum = [cell for cell in cells if float(cell["z"]) == z]
        treated = [cell for cell in stratum if float(cell["t"]) == level]
        z_count = sum(float(cell["count"]) for cell in stratum)
        t_count = sum(float(cell["count"]) for cell in treated)
        y_sum = sum(float(cell["count"]) * float(cell["y"]) for cell in treated)
        assert t_count > 0.0
        adjusted += z_count / total * y_sum / t_count
        selected_count += t_count
        selected_outcome_sum += y_sum
    return adjusted, selected_outcome_sum / selected_count


def _continuous_linear(pin: dict[str, Any]) -> dict[str, np.ndarray]:
    n = int(pin["continuous"]["n"])
    z = np.array([0.0 if i % 2 == 0 else 1.0 for i in range(n)], dtype=np.float64)
    t = 0.3 + 0.4 * z + 0.2 * np.sin(np.arange(n, dtype=np.float64) * 0.017)
    y = 0.10 + 0.40 * t + 0.20 * z
    return {"t": t, "y": y, "z": z}


def _cpdag(*, accepted: bool):
    spec = _PIN["cpdag"]["graph"]
    graph = antecedent.Cpdag.from_directed_undirected(
        _PIN["columns"],
        [tuple(edge) for edge in spec["directed_edges"]],
        [tuple(edge) for edge in spec["undirected_edges"]],
    )
    if not accepted:
        return graph
    return antecedent.AcceptedGraph.from_graph(graph, algorithm_id="fixture.cpdag-response")


def _pag(*, accepted: bool):
    spec = _PIN["pag"]["graph"]
    graph = antecedent.Pag.from_marked_edges(
        _PIN["columns"],
        [tuple(edge) for edge in spec["marked_edges"]],
    )
    if not accepted:
        return graph
    return antecedent.AcceptedGraph.from_graph(graph, algorithm_id="fixture.pag-response")


def _curve() -> antecedent.ResponseCurve:
    spec = _PIN["query"]
    return antecedent.ResponseCurve(
        spec["treatment"],
        spec["outcome"],
        grid=[float(v) for v in _PIN["grid"]],
    )


def _intervention(level: float) -> antecedent.InterventionResponse:
    return antecedent.InterventionResponse(
        _PIN["query"]["outcome"],
        intervention=antecedent.intervention.Set(_PIN["query"]["treatment"], level),
    )


def _values(result) -> np.ndarray:
    assert result.response is not None
    return np.asarray(result.response.values, dtype=np.float64).reshape(-1)


@pytest.mark.parametrize("accepted", [False, True], ids=["explicit", "accepted"])
@pytest.mark.parametrize("class_name", ["cpdag", "pag"])
def test_class_aware_intervention_pins(accepted: bool, class_name: str) -> None:
    data = _expand_contingency(_PIN)
    graph = _cpdag(accepted=accepted) if class_name == "cpdag" else _pag(accepted=accepted)
    if class_name == "pag":
        identified = antecedent.identify(graph=graph, query=antecedent.AverageEffect("t", "y"))
        assert identified.status == "NotIdentified"
        assert identified.certificate["identified_weight"] == 0.0
        return
    section = _PIN[class_name]
    high = require_success(
        antecedent.analyze(
            data,
            graph=graph,
            query=_intervention(1.0),
            refute=False,
            inference=antecedent.Frequentist(bootstrap=0),
            execution=antecedent.Execution(seed=1),
        )
    )
    low = require_success(
        antecedent.analyze(
            data,
            graph=graph,
            query=_intervention(0.0),
            refute=False,
            inference=antecedent.Frequentist(bootstrap=0),
            execution=antecedent.Execution(seed=1),
        )
    )
    prepared = antecedent.prepare(
        data,
        graph=graph,
        query=_intervention(1.0),
        refute=False,
        inference=antecedent.Frequentist(bootstrap=0, profile="interactive"),
        execution=antecedent.Execution(seed=1),
    )
    click = require_success(antecedent.estimate(prepared))
    assert prepared.structure_source == ("accepted" if accepted else "explicit")
    assert prepared.evidence_status == "licensed"
    # Both graph completions identify their own target. Their equal-weight mean
    # here is a fixture calculation, not a probability distribution over graphs.
    high_targets = _count_derived_intervention_targets(_PIN, 1.0)
    low_targets = _count_derived_intervention_targets(_PIN, 0.0)
    assert high_targets == pytest.approx((0.6, 0.66), abs=1e-12, rel=0)
    assert low_targets == pytest.approx((0.2, 0.14), abs=1e-12, rel=0)
    high_mean = sum(high_targets) / 2.0
    low_mean = sum(low_targets) / 2.0
    assert high.evidence_status == low.evidence_status == click.evidence_status == "licensed"
    assert high.identification.status == section["identification"]["status"]
    assert float(_values(high)[0]) == pytest.approx(high_mean, abs=1e-5, rel=0)
    assert float(_values(low)[0]) == pytest.approx(low_mean, abs=1e-5, rel=0)
    assert float(_values(high)[0]) - float(_values(low)[0]) == pytest.approx(
        section["ate_contrast"], abs=section["intervention"]["contrast_tolerance"], rel=0
    )
    assert float(_values(click)[0]) == pytest.approx(high_mean, abs=1e-5, rel=0)
    for result, targets in ((high, high_targets), (low, low_targets), (click, high_targets)):
        # Each end of this two-completion identified set is an actual atom;
        # check both, rather than letting an average conceal offsetting errors.
        envelope = result.envelope
        assert envelope is not None
        assert envelope.weight_basis == "completion_enumeration"
        assert envelope.completion_count == 2
        assert envelope.lower[0][0] == pytest.approx(min(targets), abs=1e-5, rel=0)
        assert envelope.upper[0][0] == pytest.approx(max(targets), abs=1e-5, rel=0)


@pytest.mark.parametrize("accepted", [False, True], ids=["explicit", "accepted"])
@pytest.mark.parametrize("class_name", ["cpdag", "pag"])
def test_class_aware_curve_pins(accepted: bool, class_name: str) -> None:
    data = _continuous_linear(_PIN)
    graph = _cpdag(accepted=accepted) if class_name == "cpdag" else _pag(accepted=accepted)
    if class_name == "pag":
        identified = antecedent.identify(graph=graph, query=antecedent.AverageEffect("t", "y"))
        assert identified.status == "NotIdentified"
        assert identified.certificate["identified_weight"] == 0.0
        return
    section = _PIN[class_name]
    query = _curve()
    fresh = require_success(
        antecedent.analyze(
            data,
            graph=graph,
            query=query,
            refute=False,
            inference=antecedent.Frequentist(bootstrap=0),
            execution=antecedent.Execution(seed=1),
        )
    )
    prepared = antecedent.prepare(
        data,
        graph=graph,
        query=query,
        refute=False,
        inference=antecedent.Frequentist(bootstrap=0, profile="interactive"),
        execution=antecedent.Execution(seed=1),
    )
    click = require_success(antecedent.estimate(prepared))
    assert prepared.structure_source == ("accepted" if accepted else "explicit")
    expected = np.asarray(section["curve"]["mean"], dtype=np.float64)
    for result in (fresh, click):
        assert result.evidence_status == "licensed"
        assert result.identification.status == section["identification"]["status"]
        values = _values(result)
        # Regression pin of the extrapolated levels: grid [0, 1] lies outside the observed
        # treatment support [0.1, 0.9]. It is not accuracy evidence, and the contrast is not
        # compared with AverageEffect (a different functional; see the next test).
        assert values == pytest.approx(expected, rel=0, abs=section["curve"]["absolute_tolerance"])
        assert result.response is not None


def _completion_levels(data: dict[str, np.ndarray], a: float) -> tuple[float, float]:
    """Independent closed forms of the two completions' targets at treatment level `a`.

    Adjusting for z: E[Y | do(t=a)] = 0.1 + 0.4 a + 0.2 E[z] with E[z] = mean(z).
    Not adjusting: E[Y | T=a] by the data's own law, z = 1 exactly when t > 0.5, so
    0.1 + 0.4 a + 0.2 * 1[a > 0.5].
    """
    assert np.all(data["t"][data["z"] == 0.0] <= 0.5 + 1e-12)
    assert np.all(data["t"][data["z"] == 1.0] >= 0.5 - 1e-12)
    adjusted = 0.1 + 0.4 * a + 0.2 * float(np.mean(data["z"]))
    unadjusted = 0.1 + 0.4 * a + (0.2 if a > 0.5 else 0.0)
    return unadjusted, adjusted


def test_class_aware_curve_is_the_envelope_of_the_completion_curves_and_flags_extrapolation():
    data = _continuous_linear(_PIN)
    graph = _cpdag(accepted=False)
    result = require_success(
        antecedent.analyze(
            data,
            graph=graph,
            query=antecedent.ResponseCurve("t", "y", grid=[0.2, 0.8]),
            refute=False,
            inference=antecedent.Frequentist(bootstrap=0),
            execution=antecedent.Execution(seed=1),
        )
    )
    # The class disagrees (adjusting for z or not): the answer is not a single curve.
    assert result.answer.kind == "partial"
    assert result.answer.detail == "identified_set"
    levels = _values(result)
    for a, level in zip([0.2, 0.8], levels, strict=True):
        unadjusted, adjusted = _completion_levels(data, a)
        assert abs(unadjusted - adjusted) > 0.09
        # The published levels lie between the completions, strictly inside the envelope.
        assert min(unadjusted, adjusted) - 3e-3 < level < max(unadjusted, adjusted) + 3e-3

    # Grid [0, 1] is extrapolation: the independent AverageEffect completions explain the gap
    # between that curve's contrast and the AverageEffect, which are different functionals.
    t, y, z = data["t"], data["y"], data["z"]
    ones = np.ones_like(t)
    unadjusted_slope = np.linalg.lstsq(np.column_stack([ones, t]), y, rcond=None)[0][1]
    adjusted_slope = np.linalg.lstsq(np.column_stack([ones, t, z]), y, rcond=None)[0][1]
    assert adjusted_slope == pytest.approx(0.4, abs=1e-9)
    ate = require_success(
        antecedent.analyze(
            data,
            graph=graph,
            query=antecedent.AverageEffect("t", "y"),
            refute=False,
            inference=antecedent.Frequentist(bootstrap=0),
            execution=antecedent.Execution(seed=1),
        )
    )
    assert ate.answer.bounds == pytest.approx((adjusted_slope, unadjusted_slope), abs=1e-9)
    pinned = _PIN["cpdag"]["curve"]["mean"]
    curve_contrast = pinned[1] - pinned[0]
    unadjusted_contrast = _completion_levels(data, 1.0)[0] - _completion_levels(data, 0.0)[0]
    assert unadjusted_contrast == pytest.approx(0.6, abs=1e-12)
    # curve contrast = mean of (0.6 unadjusted boundary-line contrast, ~0.4 adjusted line);
    # AverageEffect = mean of (0.737 projection slope, 0.4).
    assert curve_contrast == pytest.approx((unadjusted_contrast + adjusted_slope) / 2, abs=2e-3)
    assert (unadjusted_slope - unadjusted_contrast) / 2 > 0.06


@pytest.mark.parametrize("accepted", [False, True], ids=["explicit", "accepted"])
@pytest.mark.parametrize("class_name", ["cpdag", "pag"])
def test_class_aware_bayesian_intervention_pins(accepted: bool, class_name: str) -> None:
    data = _expand_contingency(_PIN)
    graph = _cpdag(accepted=accepted) if class_name == "cpdag" else _pag(accepted=accepted)
    if class_name == "pag":
        identified = antecedent.identify(graph=graph, query=antecedent.AverageEffect("t", "y"))
        assert identified.status == "NotIdentified"
        assert identified.certificate["identified_weight"] == 0.0
        return
    bayes = _PIN["bayesian"]
    inference = antecedent.Bayesian(
        backend="conjugate",
        n_draws=int(bayes["n_draws"]),
        prior_scale=float(bayes["prior_scale"]),
    )
    high = require_success(
        antecedent.analyze(
            data,
            graph=graph,
            query=_intervention(1.0),
            inference=inference,
            refute=False,
            execution=antecedent.Execution(seed=int(bayes["seed"])),
        )
    )
    low = require_success(
        antecedent.analyze(
            data,
            graph=graph,
            query=_intervention(0.0),
            inference=inference,
            refute=False,
            execution=antecedent.Execution(seed=int(bayes["seed"])),
        )
    )
    prepared = antecedent.prepare(
        data,
        graph=graph,
        query=_intervention(1.0),
        inference=replace(inference, profile="interactive"),
        refute=False,
        execution=antecedent.Execution(seed=int(bayes["seed"])),
    )
    click = require_success(antecedent.estimate(prepared))
    assert high.evidence_status == click.evidence_status == "licensed"
    assert any(str(d).startswith(bayes["diagnostic"]) for d in high.diagnostics)
    expected = bayes["cpdag_ate_contrast" if class_name == "cpdag" else "pag_ate_contrast"]
    contrast = float(_values(high)[0]) - float(_values(low)[0])
    assert contrast == pytest.approx(expected, abs=bayes["contrast_tolerance"])
    assert float(_values(click)[0]) == pytest.approx(float(_values(high)[0]), abs=1e-12)


def test_class_response_strategy_options_are_checked_and_the_class_pair_runs() -> None:
    data = _continuous_linear(_PIN)
    options = dict(
        graph=_cpdag(accepted=False),
        query=_curve(),
        refute=False,
        inference=antecedent.Frequentist(bootstrap=0),
        execution=antecedent.Execution(seed=1),
    )
    accepted = require_success(
        antecedent.analyze(
            data,
            identifier="generalized.adjustment",
            estimator="response.kennedy_dr",
            **options,
        )
    )
    plan = accepted.study.plan_view().statistical
    assert plan.identifier == "generalized.adjustment"
    assert plan.estimator == "response.kennedy_dr"
    expected = _PIN["cpdag"]["curve"]["mean"]
    tolerance = _PIN["cpdag"]["curve"]["absolute_tolerance"]
    assert list(_values(accepted)) == pytest.approx(expected, abs=tolerance)
    refused = (ValueError, antecedent.errors.CausalUnsupportedError)
    with pytest.raises(refused, match="requires identifier"):
        require_success(antecedent.analyze(data, identifier="backdoor.adjustment", **options))
    with pytest.raises(refused, match="requires estimator"):
        require_success(antecedent.analyze(data, estimator="aipw", **options))
