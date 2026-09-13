"""Independent reference for the two-path path-specific effect fixture.

Known structural causal model (all variables binary, noises independent U(0,1)):

    C = 1[U_c < 0.4]
    T = 1[U_t < 0.3 + 0.4 C]
    M(t) = 1[U_m < 0.2 + 0.4 t + 0.2 C]
    Y(t, m) = 1[U_y < 0.1 + 0.3 m + 0.1 t + 0.2 m t + 0.2 C]

Graph: C -> T, C -> M, C -> Y, T -> M, T -> Y, M -> Y. The query is the
natural path-specific effect of T on Y along the path through M,

    E[Y(T=0, M(T=1))] - E[Y(T=0, M(T=0))],

i.e. T enters M at the active level and Y (directly) at the control level.

Every threshold is a multiple of 0.1, so each indicator is constant on the ten
equal bins of its noise. Enumerating the 10^4 equally likely noise cells gives
(1) the cross-world counterfactual truth straight from the structural
equations, with no g-formula involved, and (2) the frozen observational table:
each cell contributes one observed row, so the counts are the SCM law times
10^4 exactly (divided by their common factor 4: 2,500 rows). The script then
recomputes the edge g-formula

    sum_{c,m} P(c) P(m | T=1, c) E[Y | T=0, m, c]  -  (same with T=0 in P(m|.))

from the expanded table with numpy and checks that it reproduces the
structural truth. The total effect and the swapped-level contrast (natural
direct effect) are recorded as discriminators: an implementation that binds
one treatment level everywhere returns the total effect.

Run: ``python3 conformance/estimate/path_specific_edge_gformula/reference.py``
(add ``--write`` to regenerate expected.json, ``--check`` to compare).

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BINS = 10


def p_c() -> float:
    return 0.4


def p_t(c: int) -> float:
    return 0.3 + 0.4 * c


def p_m(t: int, c: int) -> float:
    return 0.2 + 0.4 * t + 0.2 * c


def p_y(t: int, m: int, c: int) -> float:
    return 0.1 + 0.3 * m + 0.1 * t + 0.2 * m * t + 0.2 * c


def below(u_bin: int, p: float) -> int:
    """1[U < p] for U in bin [u_bin/10, (u_bin+1)/10); p is a multiple of 0.1."""
    return int(u_bin < round(p * BINS))


def structural_cells():
    """Yield (c, t, m(0), m(1), y(t, m) table) for every equally likely noise cell."""
    for uc, ut, um, uy in itertools.product(range(BINS), repeat=4):
        c = below(uc, p_c())
        t = below(ut, p_t(c))
        m_of = {tt: below(um, p_m(tt, c)) for tt in (0, 1)}
        y_of = {(tt, mm): below(uy, p_y(tt, mm, c)) for tt in (0, 1) for mm in (0, 1)}
        yield c, t, m_of, y_of


def structural_truths() -> dict[str, float]:
    cells = list(structural_cells())
    n = len(cells)

    def mean(f) -> float:
        return sum(f(c, t, m, y) for c, t, m, y in cells) / n

    y00 = mean(lambda c, t, m, y: y[(0, m[0])])
    y01 = mean(lambda c, t, m, y: y[(0, m[1])])  # Y(T=0, M(T=1))
    y10 = mean(lambda c, t, m, y: y[(1, m[0])])  # Y(T=1, M(T=0))
    y11 = mean(lambda c, t, m, y: y[(1, m[1])])
    return {
        "path_specific_effect": y01 - y00,
        "total_effect": y11 - y00,
        "natural_direct_effect": y10 - y00,
    }


def contingency_table() -> list[dict]:
    counts: dict[tuple[int, int, int, int], int] = {}
    for c, t, m_of, y_of in structural_cells():
        m = m_of[t]
        y = y_of[(t, m)]
        key = (t, m, y, c)
        counts[key] = counts.get(key, 0) + 1
    # Dividing by the common factor keeps the law exact with fewer rows.
    common = math.gcd(*counts.values())
    return [
        {"t": float(t), "m": float(m), "y": float(y), "c": float(c), "count": n // common}
        for (t, m, y, c), n in sorted(counts.items())
    ]


def expand(table: list[dict]) -> dict[str, np.ndarray]:
    cols: dict[str, list[float]] = {k: [] for k in ("t", "m", "y", "c")}
    for cell in table:
        for k in cols:
            cols[k].extend([cell[k]] * cell["count"])
    return {k: np.asarray(v) for k, v in cols.items()}


def edge_g_formula(data: dict[str, np.ndarray], level_m: float, level_y: float) -> float:
    """sum_{c,m} P(c) P(m | T=level_m, c) E[Y | T=level_y, m, c] from the empirical table."""
    t, m, y, c = data["t"], data["m"], data["y"], data["c"]
    total = 0.0
    for cv in (0.0, 1.0):
        pc = np.mean(c == cv)
        for mv in (0.0, 1.0):
            sel_m = (t == level_m) & (c == cv)
            pm = np.mean(m[sel_m] == mv)
            sel_y = (t == level_y) & (m == mv) & (c == cv)
            total += pc * pm * np.mean(y[sel_y])
    return float(total)


def empirical_contrasts(table: list[dict]) -> dict[str, float]:
    data = expand(table)
    y00 = edge_g_formula(data, 0.0, 0.0)
    return {
        "path_specific_effect": edge_g_formula(data, 1.0, 0.0) - y00,
        "total_effect": edge_g_formula(data, 1.0, 1.0) - y00,
        "natural_direct_effect": edge_g_formula(data, 0.0, 1.0) - y00,
    }


def build() -> dict:
    table = contingency_table()
    truth = structural_truths()
    plug_in = empirical_contrasts(table)
    for key, value in truth.items():
        if abs(value - plug_in[key]) > 1e-12:
            raise SystemExit(f"{key}: structural {value} != table plug-in {plug_in[key]}")
    return {
        "schema_version": 1,
        "case": "two_path_edge_g_formula",
        "columns": ["t", "m", "y", "c"],
        "query": {
            "treatment": "t",
            "outcome": "y",
            "path_nodes": ["m"],
            "control_level": 0.0,
            "active_level": 1.0,
        },
        "graph": {
            "class": "Dag",
            "directed_edges": [
                ["c", "t"],
                ["c", "m"],
                ["c", "y"],
                ["t", "m"],
                ["t", "y"],
                ["m", "y"],
            ],
        },
        "contingency_table": table,
        "identification": {
            "identifier": "path_specific.natural",
            "status": "NonparametricallyIdentified",
            "rule": "path_specific.edge_gformula",
        },
        "truth": {k: round(v, 12) for k, v in truth.items()},
        "frequentist": {
            "estimator": "functional.effect",
            "expected_effect": round(truth["path_specific_effect"], 12),
            "absolute_tolerance": 1e-9,
        },
        "bayesian": {
            "estimator": "functional.effect",
            "posterior_mean_tolerance": 0.02,
            "note": "posterior mean within tolerance of the truth; 90% interval covers it",
        },
    }


def main() -> None:
    expected = build()
    path = HERE / "expected.json"
    text = json.dumps(expected, indent=2) + "\n"
    if "--write" in sys.argv:
        path.write_text(text, encoding="utf-8")
    if "--check" in sys.argv:
        frozen = json.loads(path.read_text(encoding="utf-8"))
        if frozen != expected:
            raise SystemExit("expected.json is stale; rerun with --write")
    print(json.dumps(expected["truth"], indent=2))


if __name__ == "__main__":
    main()
