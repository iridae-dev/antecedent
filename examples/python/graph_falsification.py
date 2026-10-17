"""Falsifying a graph by node permutation, with no fitted model.

Eight variables form four strongly dependent pairs. The graph that joins the pairs is compared
with 300 random relabellings of its own nodes on the same data, through the local Markov
implications it asserts. A graph that pairs the wrong variables is not better than its
relabellings; a complete graph asserts nothing the data could contradict; a nonlinear
dependence needs a nonlinear independence test; and a test budget too small to cover the
implications gives no verdict rather than a guess."""

from __future__ import annotations

import numpy as np
from antecedent.gcm import evaluate_graph

TRUE_PAIRS = [("x0", "x1"), ("x2", "x3"), ("x4", "x5"), ("x6", "x7")]
CROSSED_PAIRS = [("x0", "x2"), ("x1", "x3"), ("x4", "x6"), ("x5", "x7")]


def linear_pairs(rows: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    data = {}
    for pair in range(4):
        root = rng.normal(size=rows)
        data[f"x{2 * pair}"] = root
        data[f"x{2 * pair + 1}"] = 0.9 * root + 0.45 * rng.normal(size=rows)
    return data


data = linear_pairs(1500, 5)
options = {"alpha": 0.01, "n_permutations": 300, "permutation_seed": 11}

# The graph that matches the data has far fewer violated implications than its relabellings.
# Not rejected is not "correct": Markov-equivalent graphs cannot be told apart.
right = evaluate_graph(data, TRUE_PAIRS, **options)
assert right.outcome == "not_rejected"
assert right.p_value is not None and right.p_value < 0.05
assert right.n_implications == len(right.evidence) == 48
assert right.permutations_completed == 300 and right.permutation_seed == 11
assert right.stop is None and right.tests_run > 0
assert any("shuffled-outcome" in text for text in right.limitations)

# Pairing independent variables asserts independencies the data contradicts.
wrong = evaluate_graph(data, CROSSED_PAIRS, **options)
assert wrong.outcome == "rejected"
assert wrong.p_value is not None and wrong.p_value > 0.5
assert wrong.n_violations is not None and wrong.n_violations >= 6
assert any(test.violated and test.p_value < 0.01 for test in wrong.evidence)

# The seeds reproduce the verdict exactly.
again = evaluate_graph(data, TRUE_PAIRS, **options)
assert again.permutation_violations == right.permutation_violations
assert again.p_value == right.p_value

# A complete graph implies no independence, so there is nothing to falsify.
independent = np.random.default_rng(2)
noise = {name: independent.normal(size=200) for name in "abcd"}
complete = [("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d")]
nothing = evaluate_graph(noise, complete)
assert nothing.outcome == "uninformative"
assert nothing.uninformative_reason == "no_testable_implication"

# Partial correlation sees no linear dependence in y = x^2 + noise; the nonlinear gpdc test does.
rng = np.random.default_rng(9)
curved = {}
for pair in range(3):
    root = rng.normal(size=200)
    curved[f"x{2 * pair}"] = root
    curved[f"x{2 * pair + 1}"] = root**2 + 0.25 * rng.normal(size=200)
claims_independence = [("x0", "x2"), ("x1", "x4"), ("x3", "x5")]
nonlinear_options = {
    "alpha": 0.02,
    "n_permutations": 20,
    "min_permutations": 10,
    "max_equivalent_share": 0.5,
    "ci_replicates": 99,
    "permutation_seed": 4,
}
linear_test = evaluate_graph(curved, claims_independence, **nonlinear_options)
nonlinear_test = evaluate_graph(
    curved, claims_independence, ci_test="gpdc", **nonlinear_options
)
assert linear_test.n_violations is not None and linear_test.n_violations <= 2
assert nonlinear_test.n_violations is not None and nonlinear_test.n_violations >= 4
assert nonlinear_test.outcome == "rejected"

# A budget that cannot cover the implications is reported as incomplete, not guessed at.
capped = evaluate_graph(linear_pairs(400, 5), TRUE_PAIRS, alpha=0.01, max_ci_tests=5)
assert capped.outcome == "incomplete" and capped.p_value is None
assert capped.stop == "search.operations" and capped.search_receipt is not None
print(f"matched graph p={right.p_value:.4f}; crossed graph p={wrong.p_value:.3f}")
