"""Independent six-variable reference with the final two cells of one column hidden.

Enumerate the 7**5 directed forests (at most one parent per node, flat graph
weights). Integrate each edge coefficient by Gauss-Legendre quadrature on both
signed intervals and both absent cells by Gauss-Hermite quadrature. The null
graph's independent standard normals provide the integration measure. No fitted
proposal, randomized completion or subset recursion enters the reference.
"""

from itertools import product
from math import log, pi

import numpy as np


def directed_forests():
    forests = []
    for parents in product(
        *(tuple(range(i)) + (6,) + tuple(range(i + 1, 6)) for i in range(6))
    ):
        for start in range(6):
            node = start
            seen = set()
            while node != 6 and node not in seen:
                seen.add(node)
                node = parents[node]
            if node != 6:
                break
        else:
            forests.append(parents)
    assert len(forests) == 7**5
    return np.array(forests)


def reference(forests, order):
    # Small integer inputs shared with the Rust regression.
    values = np.fromfunction(
        lambda r, c: ((r * 7 + c * 3 + r * c) % 17 - 8) / 5, (9, 6)
    )
    values[-2:, -1] = 0.0
    null_log_likelihood = -(values.size - 2) * log(2 * pi) / 2 - np.sum(values**2) / 2
    nodes, weights = np.polynomial.legendre.leggauss(order)
    beta = np.concatenate([nodes * 0.25 - 0.75, nodes * 0.25 + 0.75])
    weights = np.tile(weights * 0.25, 2)
    missing, gh = np.polynomial.hermite.hermgauss(order)
    masses = np.zeros(len(forests))
    for i, j in product(range(order), repeat=2):
        values[-2:, -1] = np.sqrt(2.0) * missing[[i, j]]
        gram = values.T @ values
        ratios = (
            np.exp(
                gram[:, :, None] * beta - gram.diagonal()[:, None, None] * beta**2 / 2
            )
            @ weights
        )
        local = np.vstack((ratios, np.ones(6)))
        masses += (gh[i] * gh[j] / pi) * np.prod(local[forests, np.arange(6)], axis=1)
    probs = masses / masses.sum()
    edge = np.array(
        [np.bincount(parents, weights=probs, minlength=7)[:6] for parents in forests.T]
    ).T
    return edge, null_log_likelihood + log(masses.sum())


if __name__ == "__main__":
    forests = directed_forests()
    low, log_low = reference(forests, 16)
    high, log_high = reference(forests, 32)
    print("quadrature refinement", np.max(np.abs(high - low)), abs(log_high - log_low))
    print("edge probabilities", high.tolist())
    print("log normalizer", log_high)
