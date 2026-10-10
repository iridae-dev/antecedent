"""Input validation and NumPy normalization for the typed native objective facade."""

import numpy as np
import pytest
from antecedent import design
from antecedent.errors import CausalDesignError, CausalTypeError, CausalValueError


def test_real_numpy_inputs_retain_native_objective_values():
    objective = design.EffectWidth(np.array([4, 0, 0, 9], dtype=np.float32), np.float32(4), 1, 20)
    assert objective.xtx == (4.0, 0.0, 0.0, 9.0)
    assert isinstance(objective.sigma2, float)
    result = design.rank_designs([design.Sampling(20)], objective=objective)
    assert result.best.score == pytest.approx((2 - np.sqrt(2)) / 3, abs=1e-13)
    models = design.ModelDistinction(
        np.array([10, 20]), np.array([[1, 1], [4, 4]], dtype=np.float32)
    )
    assert models.model_ids == (10, 20)
    assert models.log_likelihoods == ((1.0, 1.0), (4.0, 4.0))
    assert isinstance(design.MeasurementColumn(1, [np.int64(1)], np.float32(2)).self_dot, float)


@pytest.mark.parametrize("invalid", [True, np.bool_(False), 1 + 2j, np.complex64(1)])
def test_numeric_vectors_do_not_coerce_booleans_or_complex_numbers(invalid):
    with pytest.raises(CausalTypeError):
        design.EffectWidth([invalid], 1.0, 0, 1)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), 10**1000])
def test_nonfinite_and_overflowing_numeric_vectors_raise_causal_value_errors(invalid):
    with pytest.raises(CausalValueError, match="finite"):
        design.EffectWidth([invalid], 1.0, 0, 1)


@pytest.mark.parametrize("invalid", [np.array(1.0), np.eye(2), np.ones((1, 1, 1))])
def test_flattened_numeric_inputs_require_exactly_one_dimension(invalid):
    with pytest.raises(CausalValueError, match="1-dimensional"):
        design.EffectWidth(invalid, 1.0, 0, 1)


@pytest.mark.parametrize("field", ["measurement_columns", "environments"])
def test_missing_nested_effect_information_is_a_causal_type_error(field):
    with pytest.raises(CausalTypeError, match="sequence"):
        design.EffectWidth([1.0], 1.0, 0, 1, **{field: None})


@pytest.mark.parametrize("field", ["model_ids", "log_likelihoods"])
def test_missing_nested_model_information_is_a_causal_type_error(field):
    inputs = {"model_ids": [1, 2], "log_likelihoods": [[1.0], [2.0]]}
    inputs[field] = None
    with pytest.raises(CausalTypeError, match="sequence"):
        design.ModelDistinction(**inputs)


@pytest.mark.parametrize("invalid", [None, 3, "ids"])
def test_malformed_nested_unlocks_raise_causal_type_errors(invalid):
    with pytest.raises(CausalTypeError, match="sequence"):
        design.rank_designs(
            [design.Sampling(1)],
            prior=design.StructurePrior.uniform([False]),
            variable_unlocks={0: invalid},
        )


@pytest.mark.parametrize("invalid", [{}, 3, "config"])
def test_malformed_monte_carlo_options_raise_causal_type_errors(invalid):
    with pytest.raises(CausalTypeError, match="MonteCarlo"):
        design.rank_designs(
            [design.Sampling(1)],
            objective=design.GraphEntropy(),
            prior=design.StructurePrior.uniform([False]),
            monte_carlo=invalid,
        )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_callable_utilities_refuse_nonfinite_tables(invalid):
    def utility(actions, outcomes):
        return np.full(len(actions) * len(outcomes), invalid, dtype=np.float64)

    with pytest.raises(CausalDesignError, match="non-finite"):
        design.rank_designs(
            [design.Sampling(1)],
            objective=design.DecisionRegret(
                np.array([0.0, 1.0]),
                utility,
                design.StatePrior.draws([0.25, 0.75]),
                design.BinomialSignal(),
            ),
        )


def test_callable_utilities_retain_original_domain_exception_as_cause():
    original = CausalValueError("utility domain failure")

    def utility(actions, outcomes):
        raise original

    with pytest.raises(CausalDesignError) as caught:
        design.rank_designs(
            [design.Sampling(1)],
            objective=design.DecisionRegret(
                [0.0, 1.0],
                utility,
                design.StatePrior.draws([0.25, 0.75]),
                design.BinomialSignal(),
            ),
        )
    assert caught.value.__cause__ is original
