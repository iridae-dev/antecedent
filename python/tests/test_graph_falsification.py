"""Graph permutation falsification: graph and data only, no fitted mechanisms."""

from __future__ import annotations

import antecedent
import numpy as np
import pytest
from antecedent.gcm import GraphFalsification, evaluate_graph

TRUE_PAIRS = [("x0", "x1"), ("x2", "x3"), ("x4", "x5"), ("x6", "x7")]
CROSSED_PAIRS = [("x0", "x2"), ("x1", "x3"), ("x4", "x6"), ("x5", "x7")]


def _linear_pairs(rows: int = 1500, seed: int = 5) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    data = {}
    for pair in range(4):
        root = rng.normal(size=rows)
        data[f"x{2 * pair}"] = root
        data[f"x{2 * pair + 1}"] = 0.9 * root + 0.45 * rng.normal(size=rows)
    return data


def test_a_correct_graph_is_not_rejected_and_reports_its_evidence():
    result = evaluate_graph(
        _linear_pairs(), TRUE_PAIRS, alpha=0.01, n_permutations=300, permutation_seed=11
    )
    assert isinstance(result, GraphFalsification)
    assert result.outcome == "not_rejected"
    assert result.p_value is not None and result.p_value < 0.05
    assert result.n_implications == len(result.evidence) == 48
    assert result.permutations_completed == 300
    assert result.permutation_seed == 11
    assert result.tests_run > 0 and result.stop is None
    assert any("shuffled-outcome" in text for text in result.limitations)


def test_false_independencies_are_rejected():
    result = evaluate_graph(
        _linear_pairs(), CROSSED_PAIRS, alpha=0.01, n_permutations=300, permutation_seed=11
    )
    assert result.outcome == "rejected"
    assert result.p_value is not None and result.p_value > 0.5
    assert result.n_violations is not None and result.n_violations >= 6


def test_a_complete_graph_is_uninformative():
    rng = np.random.default_rng(2)
    data = {name: rng.normal(size=200) for name in "abcd"}
    edges = [("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d")]
    result = evaluate_graph(data, edges)
    assert result.outcome == "uninformative"
    assert result.uninformative_reason == "no_testable_implication"
    assert result.n_implications == 0


def test_a_test_budget_that_cannot_cover_the_tests_is_incomplete():
    result = evaluate_graph(_linear_pairs(400), TRUE_PAIRS, alpha=0.01, max_ci_tests=5)
    assert result.outcome == "incomplete"
    assert result.stop == "search.operations"
    assert result.p_value is None
    assert result.search_receipt is not None


def test_a_nonlinear_test_sees_the_dependence_partial_correlation_misses():
    rng = np.random.default_rng(9)
    data = {}
    for pair in range(3):
        root = rng.normal(size=200)
        data[f"x{2 * pair}"] = root
        data[f"x{2 * pair + 1}"] = root**2 + 0.25 * rng.normal(size=200)
    wrong = [("x0", "x2"), ("x1", "x4"), ("x3", "x5")]
    options = {
        "alpha": 0.02,
        "n_permutations": 20,
        "min_permutations": 10,
        "max_equivalent_share": 0.5,
        "ci_replicates": 99,
        "permutation_seed": 4,
    }
    linear = evaluate_graph(data, wrong, **options)
    nonlinear = evaluate_graph(data, wrong, ci_test="gpdc", **options)
    assert linear.n_violations is not None and linear.n_violations <= 2
    assert nonlinear.n_violations is not None and nonlinear.n_violations >= 4
    assert nonlinear.outcome == "rejected"


def test_invalid_settings_are_refused():
    with pytest.raises(antecedent.CausalError):
        evaluate_graph(_linear_pairs(100), TRUE_PAIRS, ci_test="no_such_test")


def test_a_cancelled_token_yields_an_incomplete_result():
    token = antecedent.state.CancellationToken()
    token.cancel()
    result = evaluate_graph(
        _linear_pairs(300), TRUE_PAIRS, alpha=0.01, execution=antecedent.Execution(cancel=token)
    )
    assert result.outcome == "incomplete"
    assert result.stop == "search.cancelled"
    assert result.p_value is None
