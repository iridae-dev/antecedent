"""Information available to a wide-design overlap decision with 400 subjects.

Only P(A0=1 | L0) differs between the two laws. The histories, second action
and outcome have identical *conditional* distributions after that choice.
Thus the full panel's KL divergence equals 400 times the mean Bernoulli KL.
Mapping the panel to any publish/refuse decision can only reduce that KL.
"""

from math import log

import numpy as np

from sequential_assignment_information import expit, logistic_fit, panel


def binary_kl(p, q):
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def refusal_ceiling(rows, safe_slope, unsafe_slope, order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    baseline = nodes * np.sqrt(3)
    divergence = (
        rows
        * (
            weights
            @ binary_kl(expit(safe_slope * baseline), expit(unsafe_slope * baseline))
        )
        / 2
    )
    # For a publication probability >= .95 under the safe law, the KL of
    # the two decisions is at least KL(Bern(.95) || Bern(q)) when q < .95.
    low, high = 0.0, 0.95
    for _ in range(60):
        q = (low + high) / 2
        if binary_kl(0.95, q) > divergence:
            low = q
        else:
            high = q
    return 1 - low, divergence


def fitted_refusals(slope, attempts, seed):
    rng = np.random.default_rng(seed)
    folds = np.arange(400) % 5
    refused = 0
    for _ in range(attempts):
        history, first, second, *_ = panel(rng, 400, slope)
        for fold in range(5):
            train, test = folds != fold, folds == fold
            first_logit = history[test, :2] @ logistic_fit(
                history[train, :2], first[train]
            )
            second_logit = history[test] @ logistic_fit(history[train], second[train])
            if max(np.abs(first_logit).max(), np.abs(second_logit).max()) > log(
                0.95 / 0.05
            ):
                refused += 1
                break
    return refused


if __name__ == "__main__":
    for slope in (1.5, 1.71):
        refusals = fitted_refusals(slope, 4000, seed=0x120A)
        print(f"slope={slope} existing cross-fit refusals={refusals}/4000")
    for slope in (1.71, 1.75, 1.8, 1.9):
        result = refusal_ceiling(400, 1.5, slope, 64)
        assert np.allclose(
            result, refusal_ceiling(400, 1.5, slope, 128), atol=1e-12, rtol=0
        )
        print(
            f"slope={slope} minimum true probability={expit(-slope * np.sqrt(3)):.5f}"
            f" panel KL={result[1]:.5f} maximum refusal with 95% safe publication={result[0]:.5f}"
        )
