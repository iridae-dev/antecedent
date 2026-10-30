"""Noncausal association through the compiled ``Association`` study.

The known truths are closed forms: the least-squares line, the group means and standard errors of a
conditional law, the log-odds ratio of a two-by-two table, and the Pearson correlation of a
symmetric quadratic. The selection rule is a deterministic function of the outcome scale and the
functional, a series resamples circular blocks under the declared dependent-series design, and
every result is associational: identification not claimed, never an intervention effect, ranking
no action, unlicensed and uncalibrated.
"""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent.descriptive import Association, AssociationEstimate
from antecedent.errors import CausalError, CausalUnsupportedError, CausalValueError
from antecedent.outcomes import require_success

from _sealed_loads import assert_answer_kept


def _line(n: int = 120) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(3)
    x = np.linspace(0.0, 10.0, n)
    return {"x": x, "y": 1.0 + 2.0 * x + rng.uniform(-0.5, 0.5, n)}


def _table(counts: dict[float, int], ones: dict[float, int]) -> dict[str, np.ndarray]:
    x: list[float] = []
    y: list[float] = []
    for level, count in counts.items():
        x += [level] * count
        y += [1.0] * ones[level] + [0.0] * (count - ones[level])
    return {"x": np.asarray(x), "y": np.asarray(y)}


def _series(n: int = 200) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(5)
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.8 * x[t - 1] + rng.normal()
        y[t] = 0.5 * x[t] + 0.6 * y[t - 1] + rng.normal()
    return {"x": x, "y": y}


def _association(result: ant.AnalysisResult) -> AssociationEstimate:
    section = result.association
    assert isinstance(section, AssociationEstimate)
    return section


def test_the_compiled_study_reports_the_least_squares_line_and_is_noncausal() -> None:
    data = _line()
    result = require_success(
        ant.analyze(
            data, query=Association("y", "x"), refute="none", execution=ant.Execution(seed=1)
        )
    )
    section = _association(result)
    x, y = data["x"], data["y"]
    slope = np.cov(x, y, bias=True)[0, 1] / np.var(x)
    assert section.analysis == "linear_regression"
    assert section.terms[1].estimate == pytest.approx(slope, abs=1e-8)
    # HC1 sandwich standard error of the slope, recomputed from the residuals.
    n = len(x)
    centered = x - x.mean()
    residual = y - y.mean() - slope * centered
    hc1 = np.sqrt(n / (n - 2) * np.sum(centered**2 * residual**2) / np.sum(centered**2) ** 2)
    assert section.terms[1].standard_error == pytest.approx(hc1, rel=1e-9)
    assert section.terms[1].standard_error_method == "sandwich_hc1"
    assert section.licensed is False and section.calibration == "unmeasured"
    assert section.ranks_actions is False and section.is_intervention_effect is False
    assert section.refused_after is None
    assert any(premise.startswith("descriptive_only") for premise in section.assumptions)
    answer = result.answer
    assert answer.kind == "associational"
    assert answer.value is None and answer.detail == "noncausal_association"
    assert "Identification is not claimed" in result.claim()


def test_conditional_laws_match_the_group_statistics_and_the_two_by_two_table() -> None:
    data = _table({0.0: 100, 1.0: 100}, {0.0: 20, 1.0: 60})
    conditional = require_success(
        ant.analyze(data, query=Association("y", "x", functional="conditional"), refute="none")
    )
    section = _association(conditional)
    assert section.analysis == "stratified_probability"
    assert [term.estimate for term in section.terms] == pytest.approx([0.2, 0.6])
    assert section.terms[0].standard_error == pytest.approx(np.sqrt(0.2 * 0.8 / 99), abs=1e-12)
    logistic = _association(
        require_success(ant.analyze(data, query=Association("y", "x"), refute="none"))
    )
    assert logistic.analysis == "logistic_regression"
    assert logistic.outcome_kind == "binary"
    expected = np.log(0.6 / 0.4) - np.log(0.2 / 0.8)
    assert logistic.terms[1].estimate == pytest.approx(expected, abs=1e-6)
    assert logistic.terms[1].scale == "log_odds"


def test_the_selection_rule_is_deterministic_and_a_declared_scale_wins() -> None:
    data = _line()
    first = _association(
        require_success(
            ant.analyze(
                data, query=Association("y", "x"), refute="none", execution=ant.Execution(seed=1)
            )
        )
    )
    again = _association(
        require_success(
            ant.analyze(
                data, query=Association("y", "x"), refute="none", execution=ant.Execution(seed=99)
            )
        )
    )
    assert first == again
    assert first.rule == "association_selection_rule"
    counted = _association(
        require_success(
            ant.analyze(
                {"x": data["x"], "y": np.round(np.abs(data["y"]))},
                query=Association("y", "x", outcome_scale="count"),
                refute="none",
            )
        )
    )
    assert counted.analysis == "poisson_regression" and counted.outcome_kind == "count"
    marginal = _association(
        require_success(ant.analyze(data, query=Association("y"), refute="none"))
    )
    assert marginal.analysis == "stratified_mean" and len(marginal.terms) == 1
    assert marginal.terms[0].estimate == pytest.approx(np.mean(data["y"]), abs=1e-9)


def test_correlations_and_a_scored_predictive_model() -> None:
    x = np.linspace(-3.95, 3.95, 80)
    data = {"x": x, "y": x * x}
    pearson = _association(
        require_success(
            ant.analyze(
                data, query=Association("y", "x", functional="pearson_correlation"), refute="none"
            )
        )
    )
    assert pearson.terms[0].estimate == pytest.approx(0.0, abs=1e-9)
    distance = _association(
        require_success(
            ant.analyze(
                data, query=Association("y", "x", functional="distance_correlation"), refute="none"
            )
        )
    )
    # Sample distance correlation (V-statistic) from double-centred distance matrices.
    def centred(v: np.ndarray) -> np.ndarray:
        d = np.abs(v[:, None] - v[None, :])
        return d - d.mean(axis=0) - d.mean(axis=1)[:, None] + d.mean()

    a, b = centred(data["x"]), centred(data["y"])
    dcor = np.sqrt(np.mean(a * b) / np.sqrt(np.mean(a * a) * np.mean(b * b)))
    assert distance.terms[0].estimate == pytest.approx(dcor, abs=1e-9)
    scored = _association(
        require_success(
            ant.analyze(
                _line(),
                query=Association("y", "x", functional="predictive", folds=5),
                refute="none",
                execution=ant.Execution(seed=2),
            )
        )
    )
    assert scored.prediction is not None
    assert scored.prediction.cases == 120 and scored.prediction.score == "log_score_gaussian"
    assert scored.prediction.mean_score < scored.prediction.baseline_score
    # Gaussian log score 0.5 ln(2 pi s2) + 0.5: baseline at the outcome variance, model at
    # the line's residual variance (held-out folds sit within 0.05 of the in-sample values).
    line = _line()
    lx, ly = line["x"], line["y"]
    fit = np.polyfit(lx, ly, 1)
    gaussian = lambda s2: 0.5 * np.log(2 * np.pi * s2) + 0.5  # noqa: E731
    assert scored.prediction.baseline_score == pytest.approx(gaussian(np.var(ly)), abs=0.05)
    assert scored.prediction.mean_score == pytest.approx(
        gaussian(np.mean((ly - np.polyval(fit, lx)) ** 2)), abs=0.05
    )


def test_a_series_resamples_blocks_under_the_declared_dependent_design() -> None:
    result = require_success(
        ant.analyze(
            _series(),
            query=Association("y", "x", temporal=True),
            refute="none",
            inference=ant.Frequentist(bootstrap=60),
            sampling_design="dependent_time_series",
            execution=ant.Execution(seed=4),
        )
    )
    section = _association(result)
    assert section.modality == "series"
    assert section.resampling is not None
    assert section.resampling.method == "block_bootstrap"
    assert section.resampling.unit == "calendar_block"
    assert section.resampling.block_length is not None and section.resampling.block_length > 1
    assert section.terms[1].standard_error_method == "bootstrap_circular_block"
    series = _series()
    centered = series["x"] - series["x"].mean()
    slope = np.sum(centered * (series["y"] - series["y"].mean())) / np.sum(centered**2)
    assert section.terms[1].estimate == pytest.approx(slope, abs=1e-8)
    with pytest.raises(CausalError) as refusal:
        require_success(
            ant.analyze(
                _series(),
                query=Association("y", "x", temporal=True),
                refute="none",
                inference=ant.Frequentist(bootstrap=60),
                sampling_design="independent_subject_trajectories",
            )
        )
    assert "Next attempt" in str(refusal.value)


def test_bayesian_readings_are_conjugate_for_gaussian_outcomes_and_a_bootstrap_otherwise() -> None:
    gaussian = _association(
        require_success(
            ant.analyze(
                _line(),
                query=Association("y", "x"),
                inference=ant.Bayesian(backend="conjugate", n_draws=400),
                refute="none",
                execution=ant.Execution(seed=6),
            )
        )
    )
    assert gaussian.posterior is not None
    assert gaussian.posterior.method == "conjugate_normal_inverse_gamma"
    slope = gaussian.terms[1]
    assert slope.posterior_mean == pytest.approx(slope.estimate, abs=0.02)
    assert slope.posterior_lower is not None and slope.posterior_upper is not None
    assert slope.posterior_lower < slope.posterior_mean < slope.posterior_upper
    binary = _association(
        require_success(
            ant.analyze(
                _table({0.0: 100, 1.0: 100}, {0.0: 25, 1.0: 55}),
                query=Association("y", "x"),
                inference=ant.Bayesian(backend="conjugate", n_draws=400),
                refute="none",
                execution=ant.Execution(seed=6),
            )
        )
    )
    assert binary.posterior is not None and binary.posterior.method == "bayesian_bootstrap"
    assert binary.terms[1].estimate == pytest.approx(
        np.log(0.55 / 0.45) - np.log(0.25 / 0.75), abs=1e-6
    )
    series = _association(
        require_success(
            ant.analyze(
                _series(120),
                query=Association("y", "x", temporal=True),
                inference=ant.Bayesian(backend="conjugate", n_draws=200),
                refute="none",
                execution=ant.Execution(seed=6),
            )
        )
    )
    assert series.posterior is not None and series.posterior.method == "bayesian_bootstrap"


def test_the_refusal_of_a_causal_query_stays_beside_the_description() -> None:
    result = require_success(
        ant.analyze(
            _line(),
            query=Association("y", "x", refused_after="effect_not_identified"),
            refute="none",
        )
    )
    section = _association(result)
    assert section.refused_after == "effect_not_identified"
    assert "effect_not_identified" in result.claim()
    assert result.answer.kind == "associational"


def test_declarations_that_do_not_fit_are_refused_with_the_next_attempt() -> None:
    with pytest.raises(CausalValueError):
        Association("y", "y")
    with pytest.raises(CausalValueError):
        Association("y", ["a", "b"], functional="pearson_correlation")
    with pytest.raises(CausalValueError):
        Association("y", "x", functional="predictive", folds=1)
    data = _line()
    with pytest.raises(CausalUnsupportedError) as graph:
        require_success(
            ant.analyze(data, query=Association("y", "x"), graph=[("x", "y")], refute="none")
        )
    assert graph.value.reason_code == "option_not_applicable"
    with pytest.raises(CausalUnsupportedError) as refute:
        require_success(ant.analyze(data, query=Association("y", "x"), refute="full"))
    assert refute.value.reason_code == "option_not_applicable"
    with pytest.raises(CausalError):
        require_success(
            ant.analyze(
                data,
                query=Association("y", "x"),
                refute="none",
                sampling_design="dependent_time_series",
            )
        )


def test_an_exported_result_loads_with_the_same_associational_answer() -> None:
    result = require_success(
        ant.analyze(
            _line(), query=Association("y", "x"), refute="none", execution=ant.Execution(seed=3)
        )
    )
    loaded = require_success(ant.load(bytes(result.export())))
    assert_answer_kept(loaded)
    assert loaded.answer == result.answer
    assert loaded.answer.kind == "associational"
    assert loaded.answer.structured is not None


def test_nonpredictive_association_rejects_nondefault_folds():
    with pytest.raises(ValueError, match="folds"):
        Association("y", "x", functional="conditional", folds=3)
    assert Association("y", "x", functional="conditional").folds == 5
