"""Independent grid integral of three-option hierarchical predictive study value."""

import json
import math
from pathlib import Path

import numpy as np
from scipy.special import gammaln

COUNTS = [(2, 1, 0), (0, 2, 1), (1, 0, 2), (1, 1, 1), (2, 1, 0), (0, 2, 1), (1, 0, 2)]
MU = math.log(5.0)
SIGMA = 1.0


def evaluate(size):
    u = np.linspace(-7.0, 9.0, size)
    h = u[1] - u[0]
    u2, u3 = np.meshgrid(u, u, indexing="ij")
    base = np.zeros(4)
    outcomes = np.zeros((9, 4))
    for u1 in u:
        alpha = np.stack(
            [np.full_like(u2, math.exp(u1)), np.exp(u2), np.exp(u3)], axis=-1
        )
        kappa = alpha.sum(axis=-1)
        pi = alpha / kappa[..., None]
        v = np.log(kappa)
        logw = (
            math.log(2.0)
            - 0.5 * ((v - MU) / SIGMA) ** 2
            - math.log(SIGMA)
            - 0.5 * math.log(2 * math.pi)
            + np.log(pi).sum(axis=-1)
        )
        for row in COUNTS:
            logw += gammaln(kappa) - gammaln(kappa + sum(row))
            for j, count in enumerate(row):
                logw += gammaln(alpha[..., j] + count) - gammaln(alpha[..., j])
        w = np.exp(logw) * h**3
        features = np.concatenate([np.ones((*pi.shape[:-1], 1)), pi], axis=-1)
        base += np.einsum("ij,ijk->k", w, features)
        weights = [pi[..., j] for j in range(3)]
        weights += [
            pi[..., j] * (kappa * pi[..., j] + 1) / (kappa + 1) for j in range(3)
        ]
        weights += [
            2 * kappa * pi[..., j] * pi[..., other] / (kappa + 1)
            for j in range(3)
            for other in range(j + 1, 3)
        ]
        for j, probability in enumerate(weights):
            outcomes[j] += np.einsum("ij,ijk->k", w * probability, features)
    z = base[0]
    now = base[1:].max() / z
    return [
        float(outcomes[:3, 1:].max(axis=1).sum() / z - now),
        float(outcomes[3:, 1:].max(axis=1).sum() / z - now),
    ]


coarse, fine = evaluate(100), evaluate(140)
assert max(abs(a - b) for a, b in zip(coarse, fine, strict=True)) < 2e-5
Path(__file__).with_suffix(".json").write_text(
    json.dumps(
        {
            "evsi": fine,
            "grid_change": [abs(a - b) for a, b in zip(coarse, fine, strict=True)],
        },
        indent=2,
    )
    + "\n"
)
