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

A second case, stored under the ``shared_descendant`` key, pins a graph where
one node lies on both a selected and an unselected path without being a
recanting witness (binary, noises independent U(0,1), thresholds multiples of
1/4):

    C = 1[U_c < 2/4]
    T = 1[U_t < (1 + 2C)/4]
    A(t) = 1[U_a < (1 + 2t)/4]
    B(t) = 1[U_b < (3 - 2t)/4]
    W(a, b, c) = 1[U_w < (2a + b + c)/4]
    Y(w, c) = 1[U_y < (1 + 2w + c)/4]

Graph: C -> T, C -> W, C -> Y, T -> A, T -> B, A -> W, B -> W, W -> Y. The
query selects the paths through A (only T -> A -> W -> Y); T -> B -> W -> Y is
unselected. W is on both, but the two paths leave T through different
children, so the effect is identified:

    E[Y(W(A(1), B(0)))] - E[Y(W(A(0), B(0)))]
      = sum P(c) [P(a | T=1) - P(a | T=0)] P(b | T=0) P(w | a, b, c) P(y | w, c).

The truth again comes from enumerating the 4^6 noise cells of the structural
equations; the table plug-in is only a cross-check.

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


def below(u_bin: int, p: float, bins: int = BINS) -> int:
    """1[U < p] for U in bin [u_bin/bins, (u_bin+1)/bins); p is a multiple of 1/bins."""
    return int(u_bin < round(p * bins))


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


SHARED_BINS = 4
SHARED_COLUMNS = ("t", "a", "b", "w", "y", "c")


def shared_cells():
    """Yield (t, a(.), b(.), w(., .), y(.), c) for every equally likely noise cell."""
    for uc, ut, ua, ub, uw, uy in itertools.product(range(SHARED_BINS), repeat=6):
        c = below(uc, 2 / 4, SHARED_BINS)
        t = below(ut, (1 + 2 * c) / 4, SHARED_BINS)
        a_of = {tt: below(ua, (1 + 2 * tt) / 4, SHARED_BINS) for tt in (0, 1)}
        b_of = {tt: below(ub, (3 - 2 * tt) / 4, SHARED_BINS) for tt in (0, 1)}
        w_of = {
            (aa, bb): below(uw, (2 * aa + bb + c) / 4, SHARED_BINS)
            for aa in (0, 1)
            for bb in (0, 1)
        }
        y_of = {ww: below(uy, (1 + 2 * ww + c) / 4, SHARED_BINS) for ww in (0, 1)}
        yield t, a_of, b_of, w_of, y_of, c


def shared_truths() -> dict[str, float]:
    cells = list(shared_cells())
    n = len(cells)

    def mean(t_a: int, t_b: int) -> float:
        # Y(W(A(t_a), B(t_b))): A answers the active or control arm independently of B.
        return sum(y[w[(a[t_a], b[t_b])]] for _, a, b, w, y, _ in cells) / n

    base = mean(0, 0)
    return {
        "path_specific_effect": mean(1, 0) - base,
        "total_effect": mean(1, 1) - base,
        "complementary_path_effect": mean(0, 1) - base,
    }


def shared_table() -> list[dict]:
    counts: dict[tuple[int, ...], int] = {}
    for t, a_of, b_of, w_of, y_of, c in shared_cells():
        a, b = a_of[t], b_of[t]
        w = w_of[(a, b)]
        key = (t, a, b, w, y_of[w], c)
        counts[key] = counts.get(key, 0) + 1
    common = math.gcd(*counts.values())
    return [
        {**{k: float(v) for k, v in zip(SHARED_COLUMNS, key)}, "count": n // common}
        for key, n in sorted(counts.items())
    ]


def shared_edge_g_formula(table: list[dict], level_a: float, level_b: float) -> float:
    """sum P(c) P(a | T=level_a) P(b | T=level_b) P(w | a, b, c) E[Y | w, c] from the table."""
    cols: dict[str, list[float]] = {k: [] for k in SHARED_COLUMNS}
    for cell in table:
        for k in cols:
            cols[k].extend([cell[k]] * cell["count"])
    t, a, b, w, y, c = (np.asarray(cols[k]) for k in SHARED_COLUMNS)
    total = 0.0
    for cv, av, bv, wv in itertools.product((0.0, 1.0), repeat=4):
        pc = np.mean(c == cv)
        pa = np.mean(a[t == level_a] == av)
        pb = np.mean(b[t == level_b] == bv)
        sel_w = (a == av) & (b == bv) & (c == cv)
        pw = np.mean(w[sel_w] == wv)
        ey = np.mean(y[(w == wv) & (c == cv)])
        total += pc * pa * pb * pw * ey
    return float(total)


def build_shared() -> dict:
    table = shared_table()
    truth = shared_truths()
    base = shared_edge_g_formula(table, 0.0, 0.0)
    plug_in = {
        "path_specific_effect": shared_edge_g_formula(table, 1.0, 0.0) - base,
        "total_effect": shared_edge_g_formula(table, 1.0, 1.0) - base,
        "complementary_path_effect": shared_edge_g_formula(table, 0.0, 1.0) - base,
    }
    for key, value in truth.items():
        if abs(value - plug_in[key]) > 1e-12:
            raise SystemExit(f"shared {key}: structural {value} != table plug-in {plug_in[key]}")
    return {
        "case": "shared_descendant_distinct_children",
        "columns": list(SHARED_COLUMNS),
        "query": {
            "treatment": "t",
            "outcome": "y",
            "path_nodes": ["a"],
            "control_level": 0.0,
            "active_level": 1.0,
        },
        "graph": {
            "class": "Dag",
            "directed_edges": [
                ["c", "t"],
                ["c", "w"],
                ["c", "y"],
                ["t", "a"],
                ["t", "b"],
                ["a", "w"],
                ["b", "w"],
                ["w", "y"],
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
        "shared_descendant": build_shared(),
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
    print(json.dumps(expected["shared_descendant"]["truth"], indent=2))


if __name__ == "__main__":
    main()
