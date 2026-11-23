"""One prior currency for evidence-weighted estimation.

A composed prior carries the names of its coordinates through composition (a kernel- or
row-weighted target transport and a power-prior discount selected from the data included), so it
maps by name onto the coefficient columns of a Bayesian g-computation and onto the
outcome-regression columns of the DML / AIPW nuisance. A power-prior discount selected from the data
is a prior object with its selection-valid interval semantics in its provenance. A Beta or Gamma
prior (composed or selected) is the prior of a conditional or exceedance association, whose cells
get the closed-form conjugate posterior.

Known truths: the posterior mean of the named Gaussian prior is the ridge-type closed form
``(X'X + V0^-1)^-1 (X'y + V0^-1 m)``; a named prior equals the positional prior in design order; each
cell of an exceedance association is ``Beta(a + k, b + n - k)`` and each count cell
``Gamma(shape + sum y, rate + n)``.
"""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent.descriptive import Association
from antecedent.errors import CausalError, CausalValueError
from antecedent.estimators import DML, Aipw
from antecedent.outcomes import require_success
from antecedent.priors import (
    BetaHyperparameters,
    BetaSource,
    ComposedCountPrior,
    ExternalPriorSourceSpec,
    GammaHyperparameters,
    GammaSource,
    PriorMapping,
    TransportKernel,
    TransportPolicy,
    compose_count_priors,
    compose_external_priors,
    compose_external_priors_for_target,
    exceedance_posterior,
    gamma_poisson_posterior,
    select_alpha_prior_beta_binomial,
    select_alpha_prior_gamma_poisson,
    select_alpha_prior_linear,
)

QUERY = ant.AverageEffect("t", "y")
GRAPH = [("z", "t"), ("z", "y"), ("t", "y")]
DESIGN = ("intercept", "coef_t", "coef_z")


def _confounded(n: int = 400, seed: int = 3) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (rng.uniform(size=n) < 1 / (1 + np.exp(-0.5 * z))).astype(np.float64)
    y = 1.0 + 2.0 * t + z + 0.3 * rng.normal(size=n)
    return {"z": z, "t": t, "y": y}


def _sources(names: tuple[str, ...] | None = DESIGN) -> list[ExternalPriorSourceSpec]:
    return [
        ExternalPriorSourceSpec("a", (0.8, 1.7, 0.9), (0.5, 0.4, 0.3), coefficient_names=names),
        ExternalPriorSourceSpec("b", (1.2, 2.3, 1.1), (0.6, 0.5, 0.4), coefficient_names=names),
    ]


def test_names_flow_through_composition_and_a_kernel_weighted_target() -> None:
    plain = compose_external_priors(_sources())
    assert plain.coefficient_names == DESIGN
    assert plain.to_native_dict()["coefficient_names"] == list(DESIGN)
    rows = 60
    effects = list(np.linspace(1.5, 2.5, rows))
    distances = list(np.linspace(0.0, 1.0, rows))

    def weighted(bandwidth: float):
        return compose_external_priors_for_target(
            _sources(),
            transport=TransportPolicy.invariant_conditional_outcome(),
            unit_effects=effects,
            kernel=TransportKernel("gaussian", bandwidth),
            unit_distances=distances,
            coef_index=1,
        )

    narrow, wide = weighted(0.8), weighted(2.0)
    assert narrow.coefficient_names == wide.coefficient_names == DESIGN
    assert narrow.target_overlap is not None
    # The kernel is part of the prior's provenance, so its bandwidth changes the identity.
    assert narrow.assumption_ids != wide.assumption_ids or narrow.mean != wide.mean
    assert narrow.to_native_dict()["coefficient_names"] == list(DESIGN)
    # An unnamed composition stays positional.
    assert compose_external_priors(_sources(None)).coefficient_names is None


def test_sources_composed_together_name_their_coordinates_alike() -> None:
    mixed = [*_sources()[:1], *_sources(None)[1:]]
    with pytest.raises(CausalValueError, match="coefficient_names"):
        compose_external_priors(mixed)
    with pytest.raises(CausalValueError, match="unique"):
        ExternalPriorSourceSpec("a", (0.0, 1.0), (1.0, 1.0), coefficient_names=("u", "u"))
    with pytest.raises(CausalValueError, match="names 1 coefficients"):
        ExternalPriorSourceSpec("a", (0.0, 1.0), (1.0, 1.0), coefficient_names=("u",))


def test_a_mapping_renames_coordinates_and_refuses_an_effect_functional() -> None:
    composed = compose_external_priors(_sources())
    renamed = composed.with_coefficient_mapping(
        PriorMapping.named_parameters([("coef_z", "covariate"), ("intercept", "intercept")])
    )
    assert renamed.coefficient_names == ("intercept", "unmapped:coef_t", "covariate")
    assert composed.with_coefficient_mapping(PriorMapping.identical()) is composed
    with pytest.raises(CausalValueError, match="effect-functional"):
        composed.with_coefficient_mapping(PriorMapping.effect_functional())
    with pytest.raises(CausalValueError, match="does not have"):
        composed.with_coefficient_mapping(PriorMapping.named_parameters([("nope", "intercept")]))


def _ridge_effect(data: dict[str, np.ndarray], mean: np.ndarray, v0: np.ndarray) -> float:
    design = np.column_stack([np.ones(len(data["y"])), data["t"], data["z"]])
    precision = design.T @ design + np.diag(1.0 / v0)
    coefficients = np.linalg.solve(precision, design.T @ data["y"] + mean / v0)
    return float(coefficients[1])


def test_a_named_prior_maps_by_name_onto_the_bayesian_design_and_equals_the_closed_form() -> None:
    data = _confounded()
    composed = compose_external_priors(_sources())
    shuffled = compose_external_priors(
        [
            ExternalPriorSourceSpec(
                s.id,
                (s.mean[2], 7.0, s.mean[0], s.mean[1]),
                (s.variance[2], 3.0, s.variance[0], s.variance[1]),
                coefficient_names=("coef_z", "extra", "intercept", "coef_t"),
            )
            for s in _sources()
        ],
        baseline=([0.0] * 4, [100.0] * 4),
    )

    def fit(prior: object) -> float:
        return require_success(
            ant.analyze(
                data,
                graph=GRAPH,
                query=QUERY,
                inference=ant.Bayesian(backend="conjugate", n_draws=4000, prior_from=prior),
                refute=False,
                execution=ant.Execution(seed=7),
            )
        ).effect

    positional = fit(composed)
    closed = _ridge_effect(data, np.asarray(composed.mean), np.asarray(composed.variance))
    assert positional == pytest.approx(closed, abs=0.01)
    assert fit(shuffled) == pytest.approx(closed, abs=0.01)


def test_the_coordinates_of_a_named_prior_are_selected_by_name_in_the_design_order() -> None:
    data = _confounded()
    composed = compose_external_priors(_sources())
    shuffled = compose_external_priors(
        [
            ExternalPriorSourceSpec(
                s.id,
                (s.mean[2], 7.0, s.mean[0], s.mean[1]),
                (s.variance[2], 3.0, s.variance[0], s.variance[1]),
                coefficient_names=("coef_z", "extra", "intercept", "coef_t"),
            )
            for s in _sources()
        ],
        baseline=([0.0] * 4, [100.0] * 4),
    )

    def fit(prior: object) -> float:
        return require_success(
            ant.analyze(
                data,
                graph=GRAPH,
                query=QUERY,
                inference=ant.Bayesian(backend="conjugate", n_draws=4000, prior_from=prior),
                refute=False,
                execution=ant.Execution(seed=7),
            )
        ).effect

    # The coordinate the design lacks is dropped and the order is the design's: the same prior.
    assert fit(shuffled) == pytest.approx(fit(composed), abs=1e-9)


def test_a_named_prior_with_no_coordinate_of_the_design_is_refused() -> None:
    stranger = compose_external_priors(_sources(("a", "b", "c")))
    with pytest.raises(CausalError, match="coefficient_names") as caught:
        require_success(
            ant.analyze(
                _confounded(),
                graph=GRAPH,
                query=QUERY,
                inference=ant.Bayesian(backend="conjugate", prior_from=stranger),
                refute=False,
            )
        )
    message = f"{getattr(caught.value, 'reason_code', '')} {caught.value}"
    assert "quantity_semantics_mismatch" in message


def _outcome_sources(names: tuple[str, ...] | None) -> list[ExternalPriorSourceSpec]:
    # The per-arm outcome regression has `intercept` and `coef_z` (no treatment coefficient).
    return [
        ExternalPriorSourceSpec(
            "a", (0.9, 1.0, 1.8), (0.5, 0.4, 0.5), coefficient_names=names and (*names, "coef_t")
        ),
        ExternalPriorSourceSpec(
            "b", (1.1, 0.8, 2.2), (0.6, 0.5, 0.5), coefficient_names=names and (*names, "coef_t")
        ),
    ]


def test_a_named_composed_prior_regularizes_the_outcome_nuisance() -> None:
    _named_composed_prior_regularizes_the_outcome_nuisance(DML)


def test_a_named_composed_prior_regularizes_the_aipw_outcome_nuisance() -> None:
    _named_composed_prior_regularizes_the_outcome_nuisance(Aipw)


def _named_composed_prior_regularizes_the_outcome_nuisance(estimator: type) -> None:
    data = _confounded(seed=2)
    kwargs = {"bootstrap": 0} if estimator is Aipw else {}
    composed = compose_external_priors(_outcome_sources(("intercept", "coef_z")))
    # The treatment coordinate has no column in the outcome regression: it is dropped.
    result = require_success(
        ant.analyze(
            data,
            graph=GRAPH,
            query=QUERY,
            estimator=estimator(prior_from=composed, **kwargs),
            refute=False,
        )
    )
    assert abs(result.effect - 2.0) < 0.2, result.effect
    text = " ".join(str(a) for a in (result.assumptions or []))
    assert "aipw_outcome_nuisance_prior" in text or "outcome-regression nuisance" in text
    # The same two coordinates, positional: the identical prior, so the identical estimate.
    positional = compose_external_priors(
        [ExternalPriorSourceSpec(s.id, s.mean[:2], s.variance[:2]) for s in _outcome_sources(None)]
    )
    again = require_success(
        ant.analyze(
            data,
            graph=GRAPH,
            query=QUERY,
            estimator=estimator(prior_from=positional, **kwargs),
            refute=False,
        )
    )
    assert again.effect == pytest.approx(result.effect, abs=1e-9)


def test_a_composed_prior_with_unmappable_names_is_refused_with_the_next_attempt() -> None:
    stranger = compose_external_priors(_sources(("a", "b", "c")))
    with pytest.raises(CausalError) as caught:
        require_success(
            ant.analyze(
                _confounded(seed=2),
                graph=GRAPH,
                query=QUERY,
                estimator=DML(prior_from=stranger),
                refute=False,
            )
        )
    message = f"{getattr(caught.value, 'reason_code', '')} {caught.value}"
    assert "coefficient_names" in message and "quantity_semantics_mismatch" in message
    with pytest.raises(CausalValueError, match="effect-functional"):
        DML(
            prior_from=compose_external_priors(_sources()), mapping=PriorMapping.effect_functional()
        )


def test_an_alpha_selected_prior_is_a_prior_object_with_its_selection_in_the_provenance() -> None:
    data = _confounded(n=300, seed=11)
    design = np.column_stack([np.ones(300), data["t"], data["z"]])
    selected = select_alpha_prior_linear(
        _sources(),
        [0.0, 0.0, 0.0],
        [100.0, 100.0, 100.0],
        design.tolist(),
        data["y"].tolist(),
        [0.0, 1.0, 0.0],
        grid=[0.0, 0.25, 0.5, 0.75, 1.0],
        discrepancy_bound=0.5,
        known_residual_variance=0.09,
    )
    report = selected.alpha_selection
    assert report is not None and report.discrepancy_bound == 0.5
    alpha = report.selected_alpha
    # It is the ordinary composition at the selected alpha.
    from antecedent.priors import ExternalPriorWeight

    direct = compose_external_priors(
        [
            ExternalPriorSourceSpec(
                s.id, s.mean, s.variance, ExternalPriorWeight(alpha), coefficient_names=DESIGN
            )
            for s in _sources()
        ],
        baseline=([0.0] * 3, [100.0] * 3),
    )
    assert selected.mean == direct.mean and selected.variance == direct.variance
    ids = selected.to_native_dict()["assumption_ids"]
    assert any(
        i.startswith("power_prior_alpha_selected:criterion=empirical_bayes")
        and "interval=bias_aware" in i
        and "discrepancy_bound=0.5" in i
        for i in ids
    ), ids
    # Usable wherever prior_from is: the Bayesian estimate is the closed form of that prior.
    effect = require_success(
        ant.analyze(
            data,
            graph=GRAPH,
            query=QUERY,
            inference=ant.Bayesian(backend="conjugate", n_draws=4000, prior_from=selected),
            refute=False,
            execution=ant.Execution(seed=5),
        )
    ).effect
    closed = _ridge_effect(data, np.asarray(selected.mean), np.asarray(selected.variance))
    assert effect == pytest.approx(closed, abs=0.01)
    # A prior selected under another interval rule is another prior.
    uniform = select_alpha_prior_linear(
        _sources(),
        [0.0, 0.0, 0.0],
        [100.0, 100.0, 100.0],
        design.tolist(),
        data["y"].tolist(),
        [0.0, 1.0, 0.0],
        grid=[0.0, 0.25, 0.5, 0.75, 1.0],
        known_residual_variance=0.09,
    )
    assert uniform.to_native_dict()["assumption_ids"] != ids


def _cells() -> dict[str, np.ndarray]:
    x = np.repeat([0.0, 1.0], [40, 60])
    y = np.concatenate(
        [
            np.where(np.arange(40) < 15, 0.8, 0.1) + 0.001 * np.arange(40),
            np.where(np.arange(60) < 45, 0.8, 0.1) + 0.001 * np.arange(60),
        ]
    )
    return {"x": x, "y": y}


def test_the_exceedance_of_each_cell_is_the_beta_binomial_posterior() -> None:
    data = _cells()
    prior = BetaHyperparameters(2.0, 3.0)
    result = require_success(
        ant.analyze(
            data,
            query=Association(
                outcome="y", predictors=("x",), functional="exceedance", threshold=0.5
            ),
            inference=ant.Bayesian(backend="conjugate", prior_from=prior),
            refute="none",
            execution=ant.Execution(seed=1),
        )
    )
    section = result.association
    assert section is not None and section.posterior is not None
    assert section.posterior.method == "conjugate_beta_binomial" and section.posterior.draws == 0
    assert section.supported is False and section.calibration == "not_measured"
    for term, level in zip(section.terms, (0.0, 1.0), strict=True):
        outcomes = data["y"][data["x"] == level]
        closed = exceedance_posterior(outcomes, 0.5, prior=prior, level=0.95, predictive_trials=1)
        assert term.posterior_mean == pytest.approx(closed.mean, abs=1e-12)
        assert term.posterior_lower == pytest.approx(closed.lower, abs=1e-9)
        assert term.posterior_upper == pytest.approx(closed.upper, abs=1e-9)
        cell = term.conjugate
        assert cell is not None and cell.family == "beta_binomial"
        assert (cell.posterior_a, cell.posterior_b) == (closed.alpha, closed.beta)
        assert cell.predictive_mean == pytest.approx(closed.predictive.mean, abs=1e-12)
        assert cell.log_marginal_likelihood == pytest.approx(closed.log_marginal_likelihood)


def test_the_rate_of_each_count_cell_is_the_gamma_poisson_posterior() -> None:
    counts = [[0, 1, 1, 2, 0, 3, 1, 0, 2, 1], [4, 2, 5, 3, 6, 4, 3, 5, 2, 4, 3, 5]]
    data = {
        "x": np.repeat([0.0, 1.0], [len(c) for c in counts]),
        "y": np.asarray(sum(counts, []), dtype=np.float64),
    }
    prior = GammaHyperparameters(2.0, 0.5)
    result = require_success(
        ant.analyze(
            data,
            query=Association(
                outcome="y", predictors=("x",), functional="conditional", outcome_scale="count"
            ),
            inference=ant.Bayesian(backend="conjugate", prior_from=prior),
            refute="none",
            execution=ant.Execution(seed=1),
        )
    )
    section = result.association
    assert section is not None and section.posterior is not None
    assert section.posterior.method == "conjugate_gamma_poisson"
    for term, cell_counts in zip(section.terms, counts, strict=True):
        closed = gamma_poisson_posterior(cell_counts, prior=prior, predictive_exposure=1.0)
        assert term.posterior_mean == pytest.approx(closed.mean, abs=1e-12)
        assert term.posterior_lower == pytest.approx(closed.lower, abs=1e-9)
        assert term.posterior_upper == pytest.approx(closed.upper, abs=1e-9)
        cell = term.conjugate
        assert cell is not None and cell.family == "gamma_poisson"
        assert (cell.posterior_a, cell.posterior_b) == (closed.shape, closed.rate)


def test_a_power_prior_discount_selected_from_the_data_is_a_count_prior() -> None:
    _a_beta_binomial_power_prior_discount_is_selected_from_the_data()


def test_a_power_prior_discount_selected_from_event_counts_is_a_gamma_prior() -> None:
    selected = select_alpha_prior_gamma_poisson(
        40, 20.0, [1, 2, 2, 1, 3, 2], base=GammaHyperparameters(1.0, 1.0), grid=[0.0, 1.0]
    )
    assert selected.family == "gamma" and selected.alpha_selection is not None
    alpha = selected.alpha_selection.selected_alpha
    assert alpha in (0.0, 1.0)
    assert selected.hyper == GammaHyperparameters(1.0 + alpha * 40, 1.0 + alpha * 20.0)


def _a_beta_binomial_power_prior_discount_is_selected_from_the_data() -> None:
    data = _cells()
    base = BetaHyperparameters(1.0, 1.0)
    cell = data["y"][data["x"] == 1.0] > 0.5
    selected = select_alpha_prior_beta_binomial(
        30, 50, [bool(o) for o in cell], base=base, grid=[0.0, 0.5, 1.0], discrepancy_bound=0.2
    )
    assert isinstance(selected, ComposedCountPrior) and selected.alpha_selection is not None
    alpha = selected.alpha_selection.selected_alpha
    assert selected.hyper == BetaHyperparameters(1.0 + alpha * 30, 1.0 + alpha * 20)
    ids = selected.to_native_dict()["assumption_ids"]
    assert any("interval=bias_aware" in i and "discrepancy_bound=0.2" in i for i in ids)
    result = require_success(
        ant.analyze(
            data,
            query=Association(
                outcome="y", predictors=("x",), functional="exceedance", threshold=0.5
            ),
            inference=ant.Bayesian(backend="conjugate", prior_from=selected),
            refute="none",
            execution=ant.Execution(seed=1),
        )
    )
    section = result.association
    assert section is not None
    successes = int(cell.sum())
    term = section.terms[1]
    posterior = BetaHyperparameters(
        selected.hyper.alpha + successes, selected.hyper.beta + len(cell) - successes
    )
    assert term.posterior_mean == pytest.approx(
        posterior.alpha / (posterior.alpha + posterior.beta), abs=1e-12
    )


def test_count_sources_compose_by_the_power_prior() -> None:
    beta = compose_count_priors(
        BetaHyperparameters(1.0, 2.0),
        [BetaSource("a", 10, 40, 1.0), BetaSource("b", 6, 10, 0.5)],
    )
    assert beta.hyper == BetaHyperparameters(1.0 + 10 + 3.0, 2.0 + 30 + 2.0)
    gamma = compose_count_priors(GammaHyperparameters(1.0, 1.0), [GammaSource("a", 12, 4.0, 0.5)])
    assert gamma.hyper == GammaHyperparameters(1.0 + 6.0, 1.0 + 2.0)
    with pytest.raises(CausalValueError, match=r"\[0, 1\]"):
        compose_count_priors(BetaHyperparameters(1.0, 1.0), [BetaSource("a", 1, 2, 1.5)])


def test_a_count_prior_belongs_to_the_association_route() -> None:
    with pytest.raises(CausalError, match="Association") as caught:
        require_success(
            ant.analyze(
                _confounded(),
                graph=GRAPH,
                query=QUERY,
                inference=ant.Bayesian(
                    backend="conjugate", prior_from=BetaHyperparameters(2.0, 3.0)
                ),
                refute=False,
            )
        )
    assert getattr(caught.value, "reason_code", None) == "option_not_applicable"
    with pytest.raises(CausalValueError, match="exceedance needs a finite threshold"):
        Association(outcome="y", predictors=("x",), functional="exceedance")
    with pytest.raises(CausalValueError, match="threshold applies only"):
        Association(outcome="y", predictors=("x",), functional="conditional", threshold=0.5)


def test_a_conjugate_association_stays_a_noncausal_unsupported_description() -> None:
    result = require_success(
        ant.analyze(
            _cells(),
            query=Association(
                outcome="y", predictors=("x",), functional="exceedance", threshold=0.5
            ),
            inference=ant.Bayesian(backend="conjugate", prior_from=BetaHyperparameters(1.0, 1.0)),
            refute="none",
            execution=ant.Execution(seed=1),
        )
    )
    section = result.association
    assert section is not None
    assert section.supported is False and section.calibration == "not_measured"
    assert section.ranks_actions is False and section.is_intervention_effect is False
    assert all(term.conjugate is not None for term in section.terms)


def test_a_selected_count_prior_records_its_interval_semantics() -> None:
    cell = _cells()["y"][_cells()["x"] == 0.0] > 0.5
    base = BetaHyperparameters(1.0, 1.0)
    bounded = select_alpha_prior_beta_binomial(
        20, 40, [bool(o) for o in cell], base=base, grid=[0.0, 1.0], discrepancy_bound=0.1
    )
    uniform = select_alpha_prior_beta_binomial(
        20, 40, [bool(o) for o in cell], base=base, grid=[0.0, 1.0]
    )
    assert bounded.alpha_selection is not None and uniform.alpha_selection is not None
    assert "interval=bias_aware" in bounded.alpha_selection.provenance_id()
    assert "interval=uniform" in uniform.alpha_selection.provenance_id()
    assert bounded.to_native_dict()["assumption_ids"] != uniform.to_native_dict()["assumption_ids"]
