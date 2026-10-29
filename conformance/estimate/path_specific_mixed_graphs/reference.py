# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the path-specific effect fixture for mixed graphs and classes.

Two binary structural causal models, every noise independent and uniform on four equally likely
bins, every threshold a multiple of 1/4. Enumerating the noise cells gives, with exact rational
arithmetic and with no identification formula involved,

* the cross-world truth E[Y(A(1), B(0))] - E[Y(A(0), B(0))] of the effect of T on Y along the path
  T -> A -> Y only (T enters A at the active level and B at the control level), by evaluating the
  structural equations under that assignment on every noise cell; and
* the observed law, each noise cell contributing one row, so the contingency table is the law times
  the number of cells divided by the gcd of the counts.

district_admg (latent U, columns t, a, b, y; the ADMG has t -> a, t -> b, a -> y, b -> y, a <-> y):

    U = 1[Uu < 2/4]        T = 1[Ut < 2/4]
    A(t, u) = 1[Ua < (1 + 2t + u)/4]        B(t) = 1[Ub < (1 + 2t)/4]
    Y(a, b, u) = 1[Uy < (1 + a + b + u)/4]

routed_markov (no latent, columns w, t, a, b, y; the graph has w -> t, t -> a, t -> b, a -> y, b -> y):

    W = 1[Uw < 2/4]        T = 1[Ut < (1 + 2w)/4]
    A(t) = 1[Ua < (1 + 2t)/4]        B(t) = 1[Ub < (1 + 2t)/4]
    Y(a, b) = 1[Uy < (1, 2, 2, 4)/4 for (a, b) = (0,0), (1,0), (0,1), (1,1)]

The table-only cross checks apply the identified functionals to the table alone. On district_admg
A and Y share a district, B is its own, and T is exogenous, so the district formula reduces to

    sum_b P(b | T=0) (E[Y | T=1, b] - E[Y | T=0, b]);

on routed_markov the edge g-formula is sum_{a,b} P(b | T=0) (P(a | T=1) - P(a | T=0)) E[Y | a, b].
Both must reproduce the structural truth exactly.

collider_class takes the (t, a, b, y) marginal of routed_markov, which is Markov to T -> A,
T -> B, A -> Y, B -> Y, and the class with A -> Y <- B compelled and T - A, T - B undirected. Its
three completions are (T -> A, T -> B): the structural truth above; (B -> T -> A): B is a
non-descendant of T and so adjusted, sum_b P(b) sum_a (P(a | T=1) - P(a | T=0)) E[Y | a, b]; and
(A -> T -> B): no directed path T -> A -> Y, so that completion carries no effect and is
unidentified mass, one third of the equally weighted completions.

    uv run --frozen --script conformance/estimate/path_specific_mixed_graphs/reference.py --check
    uv run --frozen --script conformance/estimate/path_specific_mixed_graphs/reference.py --self-test

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


def below(bin_: int, quarters: int) -> int:
    """1[U < quarters/4] for U on the bin [bin/4, (bin+1)/4)."""
    return int(bin_ < quarters)


def district_cells():
    """Yield (u, t, a_of, b_of, y_of) for every equally likely noise cell."""
    for uu, ut, ua, ub, uy in itertools.product(range(BINS), repeat=5):
        u = below(uu, 2)
        t = below(ut, 2)
        a_of = {tt: below(ua, 1 + 2 * tt + u) for tt in (0, 1)}
        b_of = {tt: below(ub, 1 + 2 * tt) for tt in (0, 1)}
        y_of = {
            (aa, bb): below(uy, 1 + aa + bb + u) for aa in (0, 1) for bb in (0, 1)
        }
        yield u, t, a_of, b_of, y_of


def routed_cells():
    """Yield (w, t, a_of, b_of, y_of) for every equally likely noise cell."""
    y_quarters = {(0, 0): 1, (1, 0): 2, (0, 1): 2, (1, 1): 4}
    for uw, ut, ua, ub, uy in itertools.product(range(BINS), repeat=5):
        w = below(uw, 2)
        t = below(ut, 1 + 2 * w)
        a_of = {tt: below(ua, 1 + 2 * tt) for tt in (0, 1)}
        b_of = {tt: below(ub, 1 + 2 * tt) for tt in (0, 1)}
        y_of = {key: below(uy, q) for key, q in y_quarters.items()}
        yield w, t, a_of, b_of, y_of


def structural_truths(cells) -> dict[str, F]:
    n = len(cells)

    def mean(t_a: int, t_b: int) -> F:
        # Y(A(t_a), B(t_b)): A answers one arm and B the other, on the same noise cell.
        total = 0
        for _, _, a_of, b_of, y_of in cells:
            total += y_of[(a_of[t_a], b_of[t_b])]
        return F(total, n)

    base = mean(0, 0)
    return {
        "path_specific_effect": mean(1, 0) - base,
        "total_effect": mean(1, 1) - base,
        "complementary_path_effect": mean(0, 1) - base,
    }


def observed_table(cells, columns: tuple[str, ...]) -> list[dict]:
    counts: dict[tuple[int, ...], int] = {}
    for latent, t, a_of, b_of, y_of in cells:
        a, b = a_of[t], b_of[t]
        y = y_of[(a, b)]
        row = {"u": latent, "w": latent, "t": t, "a": a, "b": b, "y": y}
        key = tuple(row[name] for name in columns)
        counts[key] = counts.get(key, 0) + 1
    common = math.gcd(*counts.values())
    return [
        {**{name: float(v) for name, v in zip(columns, key, strict=True)}, "count": n // common}
        for key, n in sorted(counts.items())
    ]


def rows_of(table: list[dict], columns: tuple[str, ...]) -> list[tuple[tuple[int, ...], int]]:
    return [(tuple(int(cell[name]) for name in columns), cell["count"]) for cell in table]


def conditional_mean(rows, index: dict[str, int], target: str, given: dict[str, int]) -> F:
    num = den = 0
    for key, count in rows:
        if all(key[index[name]] == level for name, level in given.items()):
            den += count
            num += count * key[index[target]]
    return F(num, den)


def probability(rows, index: dict[str, int], target: str, level: int, given: dict[str, int]) -> F:
    num = den = 0
    for key, count in rows:
        if all(key[index[name]] == value for name, value in given.items()):
            den += count
            num += count * (key[index[target]] == level)
    return F(num, den)


def district_functional(table: list[dict], columns: tuple[str, ...]) -> F:
    """sum_b P(b | T=0) (E[Y | T=1, b] - E[Y | T=0, b]) from the table alone."""
    rows = rows_of(table, columns)
    index = {name: i for i, name in enumerate(columns)}
    total = F(0)
    for b in (0, 1):
        pb = probability(rows, index, "b", b, {"t": 0})
        total += pb * (
            conditional_mean(rows, index, "y", {"t": 1, "b": b})
            - conditional_mean(rows, index, "y", {"t": 0, "b": b})
        )
    return total


def edge_functional(table: list[dict], columns: tuple[str, ...], adjusted: bool) -> F:
    """sum P(b | T=0 or marginal) (P(a | T=1) - P(a | T=0)) E[Y | a, b] from the table alone."""
    rows = rows_of(table, columns)
    index = {name: i for i, name in enumerate(columns)}
    total = F(0)
    for a in (0, 1):
        for b in (0, 1):
            pb = (
                probability(rows, index, "b", b, {})
                if adjusted
                else probability(rows, index, "b", b, {"t": 0})
            )
            shift = probability(rows, index, "a", a, {"t": 1}) - probability(
                rows, index, "a", a, {"t": 0}
            )
            total += pb * shift * conditional_mean(rows, index, "y", {"a": a, "b": b})
    return total


def collapse_w(table: list[dict]) -> list[dict]:
    """Marginalise w out of the routed_markov table, keeping the exact law (counts rescaled)."""
    merged: dict[tuple[float, ...], int] = {}
    for cell in table:
        key = (cell["t"], cell["a"], cell["b"], cell["y"])
        merged[key] = merged.get(key, 0) + cell["count"]
    common = math.gcd(*merged.values())
    return [
        {"t": t, "a": a, "b": b, "y": y, "count": n // common}
        for (t, a, b, y), n in sorted(merged.items())
    ]


def fnum(value: F) -> float:
    return float(value)


def build() -> dict:
    district = list(district_cells())
    routed = list(routed_cells())
    district_columns = ("t", "a", "b", "y")
    routed_columns = ("w", "t", "a", "b", "y")
    district_table = observed_table(district, district_columns)
    routed_table = observed_table(routed, routed_columns)
    collider_table = collapse_w(routed_table)
    district_truth = structural_truths(district)
    routed_truth = structural_truths(routed)
    query = {
        "treatment": "t",
        "outcome": "y",
        "path_nodes": ["a"],
        "control_level": 0.0,
        "active_level": 1.0,
    }
    upper = edge_functional(collider_table, ("t", "a", "b", "y"), adjusted=True)
    return {
        "schema_version": 1,
        "case": "path_specific_mixed_graphs",
        "district_admg": {
            "columns": list(district_columns),
            "query": query,
            "graph": {
                "class": "Admg",
                "directed_edges": [["t", "a"], ["t", "b"], ["a", "y"], ["b", "y"]],
                "bidirected_edges": [["a", "y"]],
            },
            "contingency_table": district_table,
            "identification": {
                "identifier": "path_specific.natural",
                "rule": "path_specific.district_formula",
            },
            "truth": {k: fnum(v) for k, v in district_truth.items()},
        },
        "routed_markov": {
            "columns": list(routed_columns),
            "query": query,
            "graph": {
                "class": "Dag",
                "directed_edges": [
                    ["w", "t"],
                    ["t", "a"],
                    ["t", "b"],
                    ["a", "y"],
                    ["b", "y"],
                ],
            },
            "contingency_table": routed_table,
            "truth": {k: fnum(v) for k, v in routed_truth.items()},
        },
        "collider_class": {
            "columns": ["t", "a", "b", "y"],
            "query": query,
            "cpdag": {
                "directed_edges": [["a", "y"], ["b", "y"]],
                "undirected_edges": [["t", "a"], ["t", "b"]],
            },
            "contingency_table": collider_table,
            "completions": 3,
            "truth": {
                "identified_set_lower": fnum(routed_truth["path_specific_effect"]),
                "identified_set_upper": fnum(upper),
                "unidentified_mass": fnum(F(1, 3)),
            },
        },
        "frequentist": {"estimator": "functional.effect", "absolute_tolerance": 1e-9},
        "bayesian": {
            "estimator": "functional.effect",
            "posterior_mean_tolerance": 0.02,
            "note": "conjugate posterior mean within tolerance of the truth; declared acceptance band",
        },
    }


def close(got: F, want: float, label: str, problems: list[str], tol: float = 1e-12) -> None:
    if not math.isclose(float(got), float(want), rel_tol=0.0, abs_tol=tol):
        problems.append(f"{label}: derived {float(got)!r} != fixture {want!r}")


def check(document: dict) -> list[str]:
    problems: list[str] = []
    district = list(district_cells())
    routed = list(routed_cells())
    district_columns = tuple(document["district_admg"]["columns"])
    routed_columns = tuple(document["routed_markov"]["columns"])
    expected_district = observed_table(district, district_columns)
    expected_routed = observed_table(routed, routed_columns)
    if document["district_admg"]["contingency_table"] != expected_district:
        problems.append("district_admg contingency table is not the exact law of the SCM")
    if document["routed_markov"]["contingency_table"] != expected_routed:
        problems.append("routed_markov contingency table is not the exact law of the SCM")
    if document["collider_class"]["contingency_table"] != collapse_w(expected_routed):
        problems.append("collider_class contingency table is not the w-marginal of routed_markov")
    district_truth = structural_truths(district)
    routed_truth = structural_truths(routed)
    for key, want in document["district_admg"]["truth"].items():
        close(district_truth[key], want, f"district_admg {key}", problems)
    for key, want in document["routed_markov"]["truth"].items():
        close(routed_truth[key], want, f"routed_markov {key}", problems)
    # Table-only identified functionals must reproduce the structural truth.
    close(
        district_functional(document["district_admg"]["contingency_table"], district_columns),
        document["district_admg"]["truth"]["path_specific_effect"],
        "district formula from the table",
        problems,
    )
    close(
        edge_functional(document["routed_markov"]["contingency_table"], routed_columns, False),
        document["routed_markov"]["truth"]["path_specific_effect"],
        "edge g-formula from the table",
        problems,
    )
    collider = document["collider_class"]
    columns = tuple(collider["columns"])
    close(
        edge_functional(collider["contingency_table"], columns, False),
        collider["truth"]["identified_set_lower"],
        "collider completion T->A, T->B",
        problems,
    )
    close(
        edge_functional(collider["contingency_table"], columns, True),
        collider["truth"]["identified_set_upper"],
        "collider completion B->T->A (adjusted)",
        problems,
    )
    close(F(1, collider["completions"]), collider["truth"]["unidentified_mass"], "mass", problems)
    for name in ("district_admg", "routed_markov"):
        truth = document[name]["truth"]
        if abs(truth["path_specific_effect"]) < 0.03:
            problems.append(f"{name}: the path-specific effect must be material")
        if abs(truth["path_specific_effect"] - truth["total_effect"]) < 0.01:
            problems.append(f"{name}: the effect must differ from the total effect")
    return problems


MUTATIONS = [
    ("a count changed", lambda d: d["district_admg"]["contingency_table"][0].__setitem__("count", 99)),
    ("district truth changed", lambda d: d["district_admg"]["truth"].__setitem__("path_specific_effect", 0.2)),
    ("routed truth changed", lambda d: d["routed_markov"]["truth"].__setitem__("path_specific_effect", 0.3)),
    ("upper bound changed", lambda d: d["collider_class"]["truth"].__setitem__("identified_set_upper", 0.1)),
    ("mass changed", lambda d: d["collider_class"]["truth"].__setitem__("unidentified_mass", 0.5)),
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
