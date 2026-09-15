"""Independent numpy reference for the CPDAG envelope ATE and its joint-IF SE.

Each identified completion is an OLS fit of ``y ~ 1 + t + Z_g`` on the expanded
contingency table. The completion's coefficient on ``t`` has the
Frisch-Waugh influence function

    psi_i^g = n * e_i^g * r_i^g / sum_j (r_j^g)^2,

where ``e^g`` are the OLS residuals and ``r^g`` is ``t`` residualized on
``1 + Z_g``. The envelope reports the frozen-weight mixture
``sum_g w_g tau_g / sum_g w_g`` and the SE of the mixture IF
``phi_i = sum_g (w_g / sum w) psi_i^g`` on the shared rows:

    SE^2 = sum_i (phi_i - mean(phi))^2 / (n (n - 1)).

This is written from the formulas above, not translated from the Rust code.

Run: ``python3 conformance/estimate/cpdag_ate_envelope/reference.py``
(add ``--check`` to compare against expected.json).

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def expand(pin: dict) -> dict[str, np.ndarray]:
    columns = pin["columns"]
    data: dict[str, list[float]] = {c: [] for c in columns}
    for cell in pin["contingency_table"]:
        for c in columns:
            data[c].extend([float(cell[c])] * int(cell["count"]))
    return {c: np.asarray(v) for c, v in data.items()}


def completion_fit(data: dict[str, np.ndarray], t: str, y: str, adjust: list[str]):
    n = data[t].shape[0]
    covariates = np.column_stack([np.ones(n)] + [data[z] for z in adjust])
    design = np.column_stack([covariates[:, :1], data[t], covariates[:, 1:]])
    beta, *_ = np.linalg.lstsq(design, data[y], rcond=None)
    resid = data[y] - design @ beta
    gamma, *_ = np.linalg.lstsq(covariates, data[t], rcond=None)
    t_resid = data[t] - covariates @ gamma
    psi = n * resid * t_resid / np.sum(t_resid**2)
    return float(beta[1]), psi


def envelope(pin: dict, atoms: list[dict]) -> tuple[float, float, list[float]]:
    data = expand(pin)
    t = pin["query"]["treatment"]
    y = pin["query"]["outcome"]
    n = data[t].shape[0]
    weights = np.asarray([a["weight"] for a in atoms], dtype=float)
    weights = weights / weights.sum()
    effects = []
    phi = np.zeros(n)
    for w, atom in zip(weights, atoms):
        tau, psi = completion_fit(data, t, y, atom["adjustment_set"])
        effects.append(tau)
        phi += w * psi
    ate = float(np.dot(weights, effects))
    se = float(np.sqrt(np.sum((phi - phi.mean()) ** 2) / (n * (n - 1))))
    return ate, se, effects


def main() -> int:
    pin = json.loads((HERE / "expected.json").read_text())
    freq = pin["frequentist"]
    ate, se, effects = envelope(pin, freq["reference_atoms"])
    print(json.dumps({"ate": ate, "se": se, "completion_effects": effects}, indent=2))
    if "--check" in sys.argv:
        assert abs(ate - freq["expected_ate"]) <= 1e-12, (ate, freq["expected_ate"])
        assert abs(se - freq["expected_se"]) <= 1e-12, (se, freq["expected_se"])
        print("reference matches expected.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
