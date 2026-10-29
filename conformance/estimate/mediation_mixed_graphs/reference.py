# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the mediation fixture for mixed graphs and classes.

Two binary structural causal models, every noise independent and uniform on four equally likely
bins, every threshold a multiple of 1/4. Enumerating the noise cells with exact rational
arithmetic gives, from the structural equations alone (no identification formula),

* the total effect       E[Y(1, M(1))] - E[Y(0, M(0))],
* the natural direct     E[Y(1, M(0))] - E[Y(0, M(0))], and
* the pure natural indirect effect E[Y(0, M(1))] - E[Y(0, M(0))],

each by evaluating the outcome equation with the treatment argument and the mediator's treatment
argument set separately on every noise cell; and the observed law, each noise cell contributing
one row, so the contingency table is the law times the number of cells divided by the gcd.

covariate_latent (latent U, columns w, t, m, y; the ADMG has w -> t, w -> m, w -> y, t -> m,
t -> y, m -> y and w <-> y):

    U = 1[Uu < 2/4]      W(u) = 1[Uw < (1 + 2u)/4]      T(w) = 1[Ut < (1 + 2w)/4]
    M(w, t) = 1[Um < (1 + w + t)/4]      Y(w, t, m, u) = 1[Uy < (1 + min(1, t + m) + min(1, u + w))/4]

markov_parents (no latent, columns c, w, t, m, y; the graph has c -> t, w -> m, t -> m, t -> y,
m -> y):

    C = 1[Uc < 2/4]      W = 1[Uw < 2/4]      T(c) = 1[Ut < (1 + 2c)/4]
    M(t, w) = 1[Um < r(t, w)/4], r = {(0,0): 1, (0,1): 2, (1,0): 3, (1,1): 3}      Y(t, m) = 1[Uy < q(t, m)/4], q = {(0,0): 1, (1,0): 2, (0,1): 3, (1,1): 3}

The table-only cross checks apply the identified functionals to the table alone: on markov_parents
NDE = sum_m [E(Y|1,m) - E(Y|0,m)] P(m|0), NIE = sum_m E(Y|0,m) [P(m|1) - P(m|0)] and
TE = E(Y|1) - E(Y|0); on covariate_latent the covariate w is adjusted (u is independent of t and
m given w), NDE = sum_{w,m} P(w) P(m|0,w) [E(Y|1,w,m) - E(Y|0,w,m)], NIE = sum_{w,m} P(w)
E(Y|0,w,m) [P(m|1,w) - P(m|0,w)] and TE = sum_w P(w) [E(Y|1,w) - E(Y|0,w)]. Both must reproduce
the structural truth exactly.

    uv run --frozen --script conformance/estimate/mediation_mixed_graphs/reference.py --check
    uv run --frozen --script conformance/estimate/mediation_mixed_graphs/reference.py --self-test

Standard library only; imports no Antecedent code and no statistics package.

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import copy
import itertools
import json
import math
import sys
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "expected.json"
BINS = 4
MARKOV_M = {(0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 3}
MARKOV_Y = {(0, 0): 1, (1, 0): 2, (0, 1): 3, (1, 1): 3}


def below(bin_: int, quarters: int) -> int:
    return int(bin_ < quarters)


def covariate_cells():
    """Yield (observed root values, m_of(t), y_of(t, m)) per noise cell."""
    for uu, uw, ut, um, uy in itertools.product(range(BINS), repeat=5):
        u = below(uu, 2)
        w = below(uw, 1 + 2 * u)
        t = below(ut, 1 + 2 * w)
        m_of = {tt: below(um, 1 + w + tt) for tt in (0, 1)}
        y_of = {
            (tt, mm): below(uy, 1 + min(1, tt + mm) + min(1, u + w)) for tt in (0, 1) for mm in (0, 1)
        }
        yield {"w": w, "t": t}, m_of, y_of


def markov_cells():
    for uc, uw, ut, um, uy in itertools.product(range(BINS), repeat=5):
        c = below(uc, 2)
        w = below(uw, 2)
        t = below(ut, 1 + 2 * c)
        m_of = {tt: below(um, MARKOV_M[(tt, w)]) for tt in (0, 1)}
        y_of = {key: below(uy, q) for key, q in MARKOV_Y.items()}
        yield {"c": c, "w": w, "t": t}, m_of, y_of


def structural_truths(cells) -> dict[str, F]:
    cells = list(cells)
    n = len(cells)

    def mean(t_y: int, t_m: int) -> F:
        return F(sum(y_of[(t_y, m_of[t_m])] for _, m_of, y_of in cells), n)

    base = mean(0, 0)
    return {
        "total_effect": mean(1, 1) - base,
        "natural_direct_effect": mean(1, 0) - base,
        "natural_indirect_effect": mean(0, 1) - base,
    }


def observed_table(cells, columns):
    counts: dict[tuple[int, ...], int] = {}
    for root, m_of, y_of in cells:
        t = root["t"]
        m = m_of[t]
        row = {**root, "m": m, "y": y_of[(t, m)]}
        key = tuple(row[name] for name in columns)
        counts[key] = counts.get(key, 0) + 1
    common = math.gcd(*counts.values())
    return [
        {**{name: float(v) for name, v in zip(columns, key, strict=True)}, "count": n // common}
        for key, n in sorted(counts.items())
    ]


def rows_of(table, columns):
    return [(tuple(int(c[name]) for name in columns), c["count"]) for c in table], {
        name: i for i, name in enumerate(columns)
    }


def mean_given(rows, index, target, given) -> F:
    num = den = 0
    for key, count in rows:
        if all(key[index[n]] == v for n, v in given.items()):
            den += count
            num += count * key[index[target]]
    return F(num, den)


def prob_given(rows, index, target, level, given) -> F:
    num = den = 0
    for key, count in rows:
        if all(key[index[n]] == v for n, v in given.items()):
            den += count
            num += count * (key[index[target]] == level)
    return F(num, den)


def table_functionals(table, columns, adjust: str | None) -> dict[str, F]:
    """Total, natural direct and pure natural indirect effect from the table alone.

    `adjust` names a covariate to condition on and average over (None: no covariate)."""
    rows, index = rows_of(table, columns)
    levels = (0, 1) if adjust else (None,)
    te = nde = nie = F(0)
    for w in levels:
        given = {adjust: w} if adjust else {}
        pw = prob_given(rows, index, adjust, w, {}) if adjust else F(1)
        te += pw * (
            mean_given(rows, index, "y", {**given, "t": 1})
            - mean_given(rows, index, "y", {**given, "t": 0})
        )
        for m in (0, 1):
            p0 = prob_given(rows, index, "m", m, {**given, "t": 0})
            p1 = prob_given(rows, index, "m", m, {**given, "t": 1})
            y0 = mean_given(rows, index, "y", {**given, "t": 0, "m": m})
            y1 = mean_given(rows, index, "y", {**given, "t": 1, "m": m})
            nde += pw * p0 * (y1 - y0)
            nie += pw * y0 * (p1 - p0)
    return {"total_effect": te, "natural_direct_effect": nde, "natural_indirect_effect": nie}


def fnums(values: dict[str, F]) -> dict[str, float]:
    return {k: float(v) for k, v in values.items()}


def build() -> dict:
    covariate_columns = ("w", "t", "m", "y")
    markov_columns = ("c", "w", "t", "m", "y")
    covariate_table = observed_table(covariate_cells(), covariate_columns)
    markov_table = observed_table(markov_cells(), markov_columns)
    return {
        "schema_version": 1,
        "case": "mediation_mixed_graphs",
        "query": {"treatment": "t", "outcome": "y", "mediators": ["m"]},
        "covariate_latent": {
            "columns": list(covariate_columns),
            "graph": {
                "class": "Admg",
                "directed_edges": [
                    ["w", "t"], ["w", "m"], ["w", "y"], ["t", "m"], ["t", "y"], ["m", "y"],
                ],
                "bidirected_edges": [["w", "y"]],
            },
            "contingency_table": covariate_table,
            "truth": fnums(structural_truths(covariate_cells())),
        },
        "markov_parents": {
            "columns": list(markov_columns),
            "graph": {
                "class": "Dag",
                "directed_edges": [
                    ["c", "t"], ["w", "m"], ["t", "m"], ["t", "y"], ["m", "y"],
                ],
            },
            "contingency_table": markov_table,
            "truth": fnums(structural_truths(markov_cells())),
        },
        "frequentist": {"estimator": "functional.effect", "absolute_tolerance": 1e-9},
        "bayesian": {
            "estimator": "functional.effect",
            "posterior_mean_tolerance": 0.03,
            "note": "Bayesian-bootstrap posterior mean within tolerance of the truth; declared acceptance band",
        },
    }


def close(got, want, label, problems, tol=1e-12):
    if not math.isclose(float(got), float(want), rel_tol=0.0, abs_tol=tol):
        problems.append(f"{label}: derived {float(got)!r} != fixture {want!r}")


def check(document: dict) -> list[str]:
    problems: list[str] = []
    for name, cells, adjust in (
        ("covariate_latent", covariate_cells, "w"),
        ("markov_parents", markov_cells, None),
    ):
        case = document[name]
        columns = tuple(case["columns"])
        if case["contingency_table"] != observed_table(cells(), columns):
            problems.append(f"{name}: the table is not the exact law of the SCM")
        truth = structural_truths(cells())
        for key, want in case["truth"].items():
            close(truth[key], want, f"{name} {key}", problems)
        plug_in = table_functionals(case["contingency_table"], columns, adjust)
        for key, want in case["truth"].items():
            close(plug_in[key], want, f"{name} {key} from the table", problems)
        t = case["truth"]
        if min(abs(t["natural_direct_effect"]), abs(t["natural_indirect_effect"])) < 0.02:
            problems.append(f"{name}: the direct and indirect effects must be material")
        if abs(t["total_effect"] - t["natural_direct_effect"] - t["natural_indirect_effect"]) < 1e-3:
            problems.append(f"{name}: an interaction should separate the pure indirect effect")
    return problems


MUTATIONS = [
    ("a count changed", lambda d: d["covariate_latent"]["contingency_table"][0].__setitem__("count", d["covariate_latent"]["contingency_table"][0]["count"] + 7)),
    ("direct truth changed", lambda d: d["covariate_latent"]["truth"].__setitem__("natural_direct_effect", 0.3)),
    ("indirect truth changed", lambda d: d["markov_parents"]["truth"].__setitem__("natural_indirect_effect", 0.3)),
    ("total truth changed", lambda d: d["markov_parents"]["truth"].__setitem__("total_effect", 0.5)),
]


def self_test(document: dict) -> list[str]:
    failures = []
    if check(document):
        failures.append("the unmodified fixture is rejected")
    for label, mutate in MUTATIONS:
        wrong = copy.deepcopy(document)
        mutate(wrong)
        if not check(wrong):
            failures.append(f"accepted: {label}")
    return failures


def main() -> int:
    if "--write" in sys.argv:
        retained = json.loads(FIXTURE.read_text(encoding="utf-8")) if FIXTURE.is_file() else {}
        document = {"oracle": retained.get("oracle", {}), **build()}
        FIXTURE.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")
        return 0
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    problems = self_test(document) if "--self-test" in sys.argv else check(document)
    for problem in problems:
        print("FAIL", problem)
    if not problems:
        print("ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
