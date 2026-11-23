"""Public complete-data exact DAG scoring and its declared structure constraints."""

import antecedent
import numpy as np
import pytest
from antecedent.discovery import EquivalenceClassPosterior, ExactDagPosterior
from antecedent.errors import CausalSerializationError, CausalUnsupportedError, CausalValueError
from antecedent.outcomes import require_success


def test_exact_score_options_validate_and_keep_legacy_wire() -> None:
    assert ExactDagPosterior()._wire() == {"algorithm": "exact_dag_posterior"}
    with pytest.raises(CausalValueError, match="unknown exact DAG score"):
        ExactDagPosterior(score="unknown")  # type: ignore[arg-type]
    with pytest.raises(CausalValueError, match="max_parents"):
        ExactDagPosterior(max_parents=-1)
    search = ExactDagPosterior(score="gaussian_signed_uniform", max_parents=1)
    assert search._wire() == {
        "algorithm": "exact_dag_posterior",
        "score": "gaussian_signed_uniform",
        "max_parents": 1,
    }
    with pytest.raises(CausalUnsupportedError, match="pooling"):
        search.run(imputations=[{"x": [1.0, 2.0]}])


def test_complete_exact_score_matches_independent_quadrature_and_composes() -> None:
    rng = np.random.default_rng(517)
    x = rng.normal(size=24)
    y = 0.8 * x + rng.normal(size=len(x))
    data = {"x": x, "y": y}
    score = ExactDagPosterior(score="gaussian_signed_uniform", max_parents=1)
    points, weights = np.polynomial.legendre.leggauss(64)
    coefficients = np.concatenate((points * 0.25 - 0.75, points * 0.25 + 0.75))
    weights = np.tile(weights * 0.25, 2)

    def bayes_factor(parent: np.ndarray, child: np.ndarray) -> float:
        return float(
            weights
            @ np.exp((parent @ child) * coefficients - (parent @ parent) * coefficients**2 / 2)
        )

    forward, backward = bayes_factor(x, y), bayes_factor(y, x)
    result = score.run(data)
    label = "discovery.gaussian_signed_uniform_unit_noise_marginal"
    assert result.provenance is not None
    assert result.provenance["graph_score"] == label
    assert result.algorithm == "exact_dag_posterior.gaussian_signed_uniform"
    assert (
        EquivalenceClassPosterior(base=score).run(data).accept_posterior()["graph_score"] == label
    )
    assert result.provenance["program"] == result.accept_posterior()["program"]
    assert (
        result.provenance["program"] != ExactDagPosterior().run(data).accept_posterior()["program"]
    )
    with pytest.raises(ValueError, match="pooling"):
        type(result).pool([result, result])
    matrix = np.asarray(result.edge_marginals).reshape(2, 2)
    # The local integral has a declared 1e-8 relative error budget.
    assert matrix[0, 1] == pytest.approx(forward / (1 + forward + backward), abs=2e-8, rel=0)
    assert matrix[1, 0] == pytest.approx(backward / (1 + forward + backward), abs=2e-8, rel=0)
    empty = ExactDagPosterior(score="gaussian_signed_uniform", max_parents=0).run(data)
    assert np.all(np.asarray(empty.edge_marginals) == 0)

    def fit_search(discovery: object, configured: bool):
        prepared = antecedent.prepare(
            data,
            discovery=discovery,
            query=antecedent.AverageEffect(treatment="x", outcome="y"),
            inference=antecedent.Bayesian(backend="conjugate", n_draws=256, prior_scale=1e6),
            refute=False,
            execution=antecedent.Execution(seed=1),
        )
        bound = prepared._engine._native.contract().get("structure_provenance.graph_score")
        assert bound == (label if configured else None)
        fit = require_success(prepared.estimate())
        assert fit.posterior is not None
        return fit

    fitted = fit_search(score, True)
    supplied = fit_search(result, True)
    assert fitted.posterior.effect_mean == pytest.approx(supplied.posterior.effect_mean)
    assert fitted.posterior.unidentified_mass == pytest.approx(supplied.posterior.unidentified_mass)
    legacy = fit_search(ExactDagPosterior(), False)
    assert fitted.program_id and legacy.program_id and fitted.program_id != legacy.program_id
    for producer, other in ((fitted, legacy), (legacy, fitted)):
        artifact = producer.export()
        require_success(antecedent.load(artifact, expected={"program_id": producer.program_id}))
        with pytest.raises(CausalSerializationError, match="program_id"):
            require_success(antecedent.load(artifact, expected={"program_id": other.program_id}))


def test_dynamic_programming_retains_score_and_parent_cap() -> None:
    rng = np.random.default_rng(32)
    data = {f"x{i}": rng.normal(size=12) for i in range(7)}
    search = ExactDagPosterior(score="gaussian_signed_uniform", max_parents=0)
    result = search.run(data)
    assert result.score == search.score
    assert result.max_parents == 0
    assert np.all(np.asarray(result.edge_marginal_matrix()) == 0)
