"""Independent numpy reference for the identified multi-completion PAG envelope.

The frozen input is a contingency table over six binary variables, built from
the law in ``LAW`` (expected counts at ``N`` rows, rounded). Each identified
MAG completion contributes one OLS fit; the envelope is the frozen-weight
mixture over identified completions (equal completion weights, unidentified
mass excluded from the mixture and reported separately).

AverageEffect (``linear.adjustment.ate``): completion ``g`` fits
``y ~ 1 + t + Z_g``. Its coefficient on ``t`` has the Frisch-Waugh influence
``psi_i = n e_i r_i / sum r^2`` with ``r`` = ``t`` residualized on ``1 + Z_g``.

ConditionalEffect (``conditional.linear.adjustment``, one modifier ``x``):
completion ``g`` fits ``y ~ 1 + t + x + t*x + Z_g`` and reports the effect
averaged over the modifier, ``theta = b_t + b_tx * mean(x)``. Its influence
is ``n * g' (X'X)^-1 x_i e_i + b_tx (x_i - mean(x))`` with
``g = e_t + mean(x) e_tx``; the second term is the sampling variation of the
modifier mean that the averaged functional carries.

For either query the envelope SE is the SE of the mixed influence
``phi = sum_g w_g psi^g / sum w`` on the shared rows:
``SE^2 = sum (phi - mean phi)^2 / (n (n - 1))``.

This file is written from the formulas above, not translated from Rust.

Run: ``python3 conformance/estimate/pag_ate_envelope_identified/reference.py``
(``--check`` compares with expected.json; ``--table`` prints the table).

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
COLUMNS = ["t", "y", "z", "m", "v", "x"]
N = 2000


def law(z: int, t: int, v: int, m: int, x: int, y: int) -> float:
    """Joint probability under z -> t -> v, z -> m -> y, t -> y, x -> y."""

    def bern(p: float, value: int) -> float:
        return p if value else 1.0 - p

    p = bern(0.5, z)
    p *= bern(0.3 + 0.4 * z, t)
    p *= bern(0.3 + 0.4 * t, v)
    p *= bern(0.2 + 0.6 * z, m)
    p *= bern(0.5, x)
    p *= bern(0.1 + 0.3 * t + 0.3 * m + 0.1 * x + 0.1 * t * x, y)
    return p


def table() -> list[dict]:
    cells = []
    for z, t, v, m, x, y in itertools.product([0, 1], repeat=6):
        count = int(round(N * law(z, t, v, m, x, y)))
        cells.append({"t": t, "y": y, "z": z, "m": m, "v": v, "x": x, "count": count})
    return cells


def expand(pin: dict) -> dict[str, np.ndarray]:
    data: dict[str, list[float]] = {c: [] for c in pin["columns"]}
    for cell in pin["contingency_table"]:
        for c in pin["columns"]:
            data[c].extend([float(cell[c])] * int(cell["count"]))
    return {c: np.asarray(v) for c, v in data.items()}


def ate_atom(d: dict[str, np.ndarray], adjust: list[str]):
    n = d["t"].shape[0]
    covariates = np.column_stack([np.ones(n)] + [d[z] for z in adjust])
    design = np.column_stack([np.ones(n), d["t"]] + [d[z] for z in adjust])
    beta, *_ = np.linalg.lstsq(design, d["y"], rcond=None)
    resid = d["y"] - design @ beta
    gamma, *_ = np.linalg.lstsq(covariates, d["t"], rcond=None)
    r = d["t"] - covariates @ gamma
    return float(beta[1]), n * resid * r / np.sum(r**2)


def cate_atom(d: dict[str, np.ndarray], modifier: str, adjust: list[str]):
    n = d["t"].shape[0]
    w = d[modifier]
    extra = [z for z in adjust if z != modifier]
    design = np.column_stack([np.ones(n), d["t"], w, d["t"] * w] + [d[z] for z in extra])
    beta, *_ = np.linalg.lstsq(design, d["y"], rcond=None)
    resid = d["y"] - design @ beta
    w_bar = w.mean()
    g = np.zeros(design.shape[1])
    g[1] = 1.0
    g[3] = w_bar
    xtx_inv = np.linalg.inv(design.T @ design)
    psi = n * (design @ (xtx_inv @ g)) * resid + beta[3] * (w - w_bar)
    return float(beta[1] + beta[3] * w_bar), psi


def mixture(d, atoms, fit):
    n = d["t"].shape[0]
    weights = np.asarray([a["weight"] for a in atoms], dtype=float)
    weights /= weights.sum()
    effects, phi = [], np.zeros(n)
    for w, atom in zip(weights, atoms):
        est, psi = fit(atom["adjustment_set"])
        effects.append(est)
        phi += w * psi
    se = float(np.sqrt(np.sum((phi - phi.mean()) ** 2) / (n * (n - 1))))
    return float(np.dot(weights, effects)), se, effects


def main() -> int:
    if "--table" in sys.argv:
        print(json.dumps(table()))
        return 0
    pin = json.loads((HERE / "expected.json").read_text())
    assert pin["contingency_table"] == table(), "contingency table drifted from LAW"
    d = expand(pin)
    atoms = pin["identified_completions"]
    ate, ate_se, ate_effects = mixture(d, atoms, lambda adj: ate_atom(d, adj))
    modifier = pin["conditional"]["modifier"]
    cate, cate_se, cate_effects = mixture(d, atoms, lambda adj: cate_atom(d, modifier, adj))
    out = {
        "ate": ate,
        "ate_se": ate_se,
        "ate_completion_effects": ate_effects,
        "cate": cate,
        "cate_se": cate_se,
        "cate_completion_effects": cate_effects,
    }
    print(json.dumps(out, indent=2))
    if "--check" in sys.argv:
        freq = pin["frequentist"]
        cond = pin["conditional"]["frequentist"]
        tol = freq["absolute_tolerance"]
        assert abs(ate - freq["expected_ate"]) <= tol, (ate, freq["expected_ate"])
        assert abs(ate_se - freq["expected_se"]) <= tol, (ate_se, freq["expected_se"])
        assert abs(cate - cond["expected_ate"]) <= tol, (cate, cond["expected_ate"])
        assert abs(cate_se - cond["expected_se"]) <= tol, (cate_se, cond["expected_se"])
        print("reference matches expected.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
