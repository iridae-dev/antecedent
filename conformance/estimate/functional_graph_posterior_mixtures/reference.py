# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the graph-posterior mixtures of a path-restricted effect, an
interventional distribution and a mediation decomposition.

Every posterior is a hand-weighted mixture over two or three atoms that share one count table:

* a path-specific effect (table `collider` for the Dag, ADMG and CPDAG posteriors, table `routed` for the PAG
  posterior);
* an interventional distribution P(Y | do(T = 1)) (table `class_disagree`, and `class_point` for
  the PAG posterior);
* a mediation decomposition (table `covariate_latent`; table `markov_parents` for the PAG
  posterior; the Dag atoms read the linear table `linear_mediation`, the product-of-coefficients
  model the Dag cell estimates).

The count tables are copied from the single-graph fixtures `path_specific_mixed_graphs`,
`interventional_distribution_classes` and `mediation_mixed_graphs`, whose own references derive
the observed law by enumerating structural noise cells; `--check` confirms the copies still equal
those tables. Each atom's value is an identified functional of the table alone, evaluated in exact
rational arithmetic (the edge g-formula with or without the adjusted sibling, the front-door
formula, the unadjusted and the covariate-adjusted mediation functionals, the Dag linear fit by
exact normal equations). The mixture is the posterior-weighted mean of the values of the atoms that
identify the query, not renormalised for the atoms that do not: those carry the unidentified mass.
A class atom's value is the enumeration-weighted mean of the completions that identify the query
(a completion that does not is unidentified inside the class and the class keeps its identified
set); a class every completion of which refuses is unidentified mass of the posterior.

    uv run --frozen --script conformance/estimate/functional_graph_posterior_mixtures/reference.py --check
    uv run --frozen --script conformance/estimate/functional_graph_posterior_mixtures/reference.py --self-test

Standard library only; imports no Antecedent code and no statistics package.

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import copy
import json
import sys
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "expected.json"
SIBLINGS = HERE.parent


def sibling(name: str) -> dict:
    return json.loads((SIBLINGS / name / "expected.json").read_text(encoding="utf-8"))


# --- table readers ----------------------------------------------------------------------------


def rows_of(table, columns):
    index = {name: i for i, name in enumerate(columns)}
    return [(tuple(int(c[name]) for name in columns), c["count"]) for c in table], index


def prob(rows, index, target, level, given) -> F:
    num = den = 0
    for key, count in rows:
        if all(key[index[n]] == v for n, v in given.items()):
            den += count
            num += count * (key[index[target]] == level)
    return F(num, den)


def mean(rows, index, target, given) -> F:
    return prob(rows, index, target, 1, given)


# --- path-specific functionals ----------------------------------------------------------------


def edge_functional(case, adjusted: bool) -> F:
    """sum P(b | T=0 or marginal) (P(a | T=1) - P(a | T=0)) E[Y | a, b]."""
    rows, idx = rows_of(case["contingency_table"], case["columns"])
    total = F(0)
    for a in (0, 1):
        for b in (0, 1):
            pb = prob(rows, idx, "b", b, {} if adjusted else {"t": 0})
            shift = prob(rows, idx, "a", a, {"t": 1}) - prob(rows, idx, "a", a, {"t": 0})
            total += pb * shift * mean(rows, idx, "y", {"a": a, "b": b})
    return total


def district_functional(case) -> F:
    """sum_b P(b | T=0) (E[Y | T=1, b] - E[Y | T=0, b])."""
    rows, idx = rows_of(case["contingency_table"], case["columns"])
    return sum(
        (
            prob(rows, idx, "b", b, {"t": 0})
            * (mean(rows, idx, "y", {"t": 1, "b": b}) - mean(rows, idx, "y", {"t": 0, "b": b}))
            for b in (0, 1)
        ),
        F(0),
    )


# --- interventional distribution functionals --------------------------------------------------


def distribution_functional(case, formula: str) -> dict[str, F]:
    rows, idx = rows_of(case["contingency_table"], case["columns"])
    out = {}
    for level in (0, 1):
        if formula == "unadjusted":  # sum_x P(x | t) P(y | t, x)
            value = sum(
                (
                    prob(rows, idx, "x", x, {"t": 1}) * prob(rows, idx, "y", level, {"t": 1, "x": x})
                    for x in (0, 1)
                ),
                F(0),
            )
        elif formula == "front_door":  # sum_x P(x | t=1) sum_t P(t) P(y | t, x)
            value = sum(
                (
                    prob(rows, idx, "x", x, {"t": 1})
                    * sum(
                        (
                            prob(rows, idx, "t", tt, {}) * prob(rows, idx, "y", level, {"t": tt, "x": x})
                            for tt in (0, 1)
                        ),
                        F(0),
                    )
                    for x in (0, 1)
                ),
                F(0),
            )
        elif formula == "adjusted":  # sum_x P(x) P(y | t, x)
            value = sum(
                (
                    prob(rows, idx, "x", x, {}) * prob(rows, idx, "y", level, {"t": 1, "x": x})
                    for x in (0, 1)
                ),
                F(0),
            )
        else:
            raise ValueError(formula)
        out[f"y{level}"] = value
    return out


# --- mediation functionals --------------------------------------------------------------------


def mediation_functionals(case, adjust: str | None) -> dict[str, F]:
    rows, idx = rows_of(case["contingency_table"], case["columns"])
    levels = (0, 1) if adjust else (None,)
    te = nde = nie = F(0)
    for w in levels:
        given = {adjust: w} if adjust else {}
        pw = prob(rows, idx, adjust, w, {}) if adjust else F(1)
        te += pw * (
            mean(rows, idx, "y", {**given, "t": 1}) - mean(rows, idx, "y", {**given, "t": 0})
        )
        for m in (0, 1):
            p0 = prob(rows, idx, "m", m, {**given, "t": 0})
            p1 = prob(rows, idx, "m", m, {**given, "t": 1})
            y0 = mean(rows, idx, "y", {**given, "t": 0, "m": m})
            y1 = mean(rows, idx, "y", {**given, "t": 1, "m": m})
            nde += pw * p0 * (y1 - y0)
            nie += pw * y0 * (p1 - p0)
    return {"total": te, "direct": nde, "indirect": nie}


# --- the linear mediation table and its exact least squares ------------------------------------


def linear_table() -> list[dict]:
    """z in {-1,0,1}, noises in {-1,0,1} crossed: t = z + e_t, m = 2 t + e_m, y = 1 + t + 3 m + z + e_y."""
    out = []
    for z in (-1, 0, 1):
        for et in (-1, 0, 1):
            for em in (-1, 0, 1):
                for ey in (-1, 0, 1):
                    t = z + et
                    m = 2 * t + em
                    y = 1 + t + 3 * m + z + ey
                    out.append({"z": float(z), "t": float(t), "m": float(m), "y": float(y), "count": 1})
    return out


def solve(matrix: list[list[F]], rhs: list[F]) -> list[F]:
    n = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = next(r for r in range(col, n) if aug[r][col] != 0)
        aug[col], aug[pivot] = aug[pivot], aug[col]
        p = aug[col][col]
        aug[col] = [v / p for v in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0:
                f = aug[r][col]
                aug[r] = [a - f * b for a, b in zip(aug[r], aug[col], strict=True)]
    return [aug[i][n] for i in range(n)]


def ols(rows: list[dict], target: str, parents: list[str]) -> dict[str, F]:
    """Coefficients of `target` on an intercept and `parents` by exact normal equations."""
    design = [[F(1)] + [F(r[p]) for p in parents] for r in rows]
    y = [F(r[target]) for r in rows]
    k = len(design[0])
    xtx = [[sum(d[i] * d[j] for d in design) for j in range(k)] for i in range(k)]
    xty = [sum(d[i] * v for d, v in zip(design, y, strict=True)) for i in range(k)]
    beta = solve(xtx, xty)
    return dict(zip(parents, beta[1:], strict=True))


def linear_mediation(rows: list[dict], with_z: bool) -> dict[str, F]:
    """Regress each node on its graph parents and propagate the t contrast along the paths."""
    m_on_t = ols(rows, "m", ["t"])["t"]
    y_coef = ols(rows, "y", ["t", "m"] + (["z"] if with_z else []))
    direct = y_coef["t"]
    indirect = m_on_t * y_coef["m"]
    return {"total": direct + indirect, "direct": direct, "indirect": indirect}


# --- the posteriors ---------------------------------------------------------------------------

def atom(name, weight, graph, kind, **extra):
    return {"name": name, "weight": weight, "graph": graph, "formula": kind, **extra}


def dag(*edges):
    return {"directed": [list(e) for e in edges]}


def admg(directed, bidirected=()):
    return {"directed": [list(e) for e in directed], "bidirected": [list(e) for e in bidirected]}


def cpdag(directed, undirected=()):
    return {"directed": [list(e) for e in directed], "undirected": [list(e) for e in undirected]}


PS_MARKOV = [("t", "a"), ("t", "b"), ("a", "y"), ("b", "y")]
PS_ROUTED = [("w", "t"), ("t", "a"), ("t", "b"), ("a", "y"), ("b", "y")]
DIST_MARKOV = [("t", "x"), ("x", "y"), ("t", "y")]
MED_FULL = [("w", "t"), ("w", "m"), ("w", "y"), ("t", "m"), ("t", "y"), ("m", "y")]
MED_RECANT = [("t", "w"), ("w", "m"), ("w", "y"), ("t", "m"), ("t", "y"), ("m", "y")]
MED_NO_W = [("t", "m"), ("t", "y"), ("m", "y")]
MED_MARKOV = [("c", "t"), ("w", "m"), ("t", "m"), ("t", "y"), ("m", "y")]


def posterior_specs() -> dict:
    return {
        "path_specific": {
            "dag": {
                "table": "collider",
                "atoms": [
                    atom("markov", "1/2", dag(*PS_MARKOV), "edge_unadjusted"),
                    atom("sibling_parent", "3/10", dag(("b", "t"), ("t", "a"), ("a", "y"), ("b", "y")), "edge_adjusted"),
                    atom("no_path", "1/5", dag(("a", "t"), ("t", "b"), ("a", "y"), ("b", "y")), "unidentified"),
                ],
            },
            "admg": {
                "table": "collider",
                "atoms": [
                    atom("markov", "1/2", admg(PS_MARKOV), "edge_unadjusted"),
                    atom("sibling_parent", "3/10", admg([("b", "t"), ("t", "a"), ("a", "y"), ("b", "y")]), "edge_adjusted"),
                    atom("recanting", "1/5", admg(PS_MARKOV, [("a", "b")]), "unidentified"),
                ],
            },
            "cpdag": {
                "table": "collider",
                "atoms": [
                    atom(
                        "collider_class",
                        "1/2",
                        cpdag([("a", "y"), ("b", "y")], [("t", "a"), ("t", "b")]),
                        "class",
                        completions=["edge_unadjusted", "edge_adjusted", "unidentified"],
                    ),
                    atom("compelled", "3/10", cpdag(PS_MARKOV), "edge_unadjusted"),
                    atom("no_path", "1/5", cpdag([("a", "t"), ("t", "b"), ("a", "y"), ("b", "y")]), "unidentified"),
                ],
            },
            "pag": {
                "table": "routed",
                "atoms": [
                    atom("visible", "7/10", dag(*PS_ROUTED), "edge_unadjusted"),
                    atom(
                        "invisible",
                        "3/10",
                        dag(("t", "w"), ("t", "a"), ("t", "b"), ("a", "y"), ("b", "y")),
                        "unidentified",
                    ),
                ],
            },
        },
        "distribution": {
            "dag": {
                "table": "class_disagree",
                "atoms": [
                    atom("forward", "1/2", dag(*DIST_MARKOV), "unadjusted"),
                    atom("sibling_parent", "3/10", dag(("x", "t"), ("x", "y"), ("t", "y")), "adjusted"),
                    atom("no_mediator", "1/5", dag(("t", "y")), "unadjusted_marginal"),
                ],
            },
            "admg": {
                "table": "class_disagree",
                "atoms": [
                    atom("markov", "2/5", admg(DIST_MARKOV), "unadjusted"),
                    atom("sibling_parent", "2/5", admg([("x", "t"), ("x", "y"), ("t", "y")]), "adjusted"),
                    atom("front_door", "1/5", admg([("t", "x"), ("x", "y")], [("t", "y")]), "front_door"),
                ],
            },
            "cpdag": {
                "table": "class_disagree",
                "atoms": [
                    atom(
                        "class",
                        "1/2",
                        cpdag([("x", "y"), ("t", "y")], [("t", "x")]),
                        "class",
                        completions=["unadjusted", "adjusted"],
                    ),
                    atom("compelled", "1/2", cpdag(DIST_MARKOV), "unadjusted"),
                ],
            },
            "pag": {
                "table": "class_point",
                "atoms": [
                    atom(
                        "visible",
                        "7/10",
                        dag(("w", "t"), ("t", "x"), ("x", "y"), ("t", "y")),
                        "unadjusted",
                    ),
                    atom(
                        "invisible",
                        "3/10",
                        dag(("t", "w"), ("t", "x"), ("x", "y"), ("t", "y")),
                        "unidentified",
                    ),
                ],
            },
        },
        "mediation": {
            "dag": {
                "table": "linear_mediation",
                "atoms": [
                    atom(
                        "confounder_parent",
                        "3/5",
                        dag(("z", "t"), ("z", "y"), ("t", "m"), ("t", "y"), ("m", "y")),
                        "linear_with_z",
                    ),
                    atom(
                        "confounder_omitted",
                        "2/5",
                        dag(("t", "m"), ("t", "y"), ("m", "y")),
                        "linear_without_z",
                    ),
                ],
            },
            "admg": {
                "table": "covariate_latent",
                "atoms": [
                    atom("covariate_adjusted", "1/2", admg(MED_FULL), "adjust_w"),
                    atom("covariate_omitted", "3/10", admg(MED_NO_W), "unadjusted"),
                    atom("recanting_witness", "1/5", admg(MED_RECANT), "unidentified_direct"),
                ],
            },
            "cpdag": {
                "table": "covariate_latent",
                "atoms": [
                    atom("adjusted", "3/5", cpdag(MED_FULL), "adjust_w"),
                    atom("covariate_omitted", "2/5", cpdag(MED_NO_W), "unadjusted"),
                ],
            },
            "pag": {
                "table": "markov_parents",
                "atoms": [
                    atom("visible", "7/10", dag(*MED_MARKOV), "unadjusted"),
                    atom(
                        "invisible",
                        "3/10",
                        dag(("t", "c"), ("w", "m"), ("t", "m"), ("t", "y"), ("m", "y")),
                        "unidentified",
                    ),
                ],
            },
        },
    }


def atom_values(family: str, table: dict, spec: dict, formula: str):
    """The identified value(s) of one completion or atom: a float, a dict of floats, or None."""
    if formula in ("unidentified", "unidentified_direct"):
        return None
    if family == "path_specific":
        if formula == "edge_unadjusted":
            return edge_functional(table, False)
        if formula == "edge_adjusted":
            return edge_functional(table, True)
        if formula == "district":
            return district_functional(table)
    if family == "distribution":
        if formula == "unadjusted_marginal":
            rows, idx = rows_of(table["contingency_table"], table["columns"])
            return {f"y{lv}": prob(rows, idx, "y", lv, {"t": 1}) for lv in (0, 1)}
        return distribution_functional(table, formula)
    if family == "mediation":
        if formula == "adjust_w":
            return mediation_functionals(table, "w")
        if formula == "unadjusted":
            return mediation_functionals(table, None)
        if formula in ("linear_with_z", "linear_without_z"):
            return linear_mediation(table["rows"], formula == "linear_with_z")
    raise ValueError((family, formula))


def combine(values):
    """Equal-weight mean of the completions that identify (a scalar or a dict of scalars)."""
    values = [v for v in values if v is not None]
    if not values:
        return None
    if isinstance(values[0], dict):
        return {k: sum(v[k] for v in values) / len(values) for k in values[0]}
    return sum(values) / len(values)


def fnum(x):
    if isinstance(x, dict):
        return {k: fnum(v) for k, v in x.items()}
    return None if x is None else float(x)


def evaluate(family: str, name: str, spec: dict, tables: dict) -> dict:
    table = tables[spec["table"]]
    atoms_out = []
    for a in spec["atoms"]:
        a = copy.deepcopy(a)
        if a["formula"] == "class":
            parts = [atom_values(family, table, spec, f) for f in a["completions"]]
            identified = [p for p in parts if p is not None]
            a["value"] = fnum(combine(parts))
            if identified and not isinstance(identified[0], dict):
                a["set"] = [float(min(identified)), float(max(identified))]
            elif identified:
                a["set"] = {k: [float(min(p[k] for p in identified)), float(max(p[k] for p in identified))] for k in identified[0]}
        else:
            a["value"] = fnum(atom_values(family, table, spec, a["formula"]))
        atoms_out.append(a)
    return {"table": spec["table"], "atoms": atoms_out}


def mix(family: str, atoms: list[dict], key: str | None = None) -> dict:
    """Posterior-weighted mean over the atoms that identify, unidentified mass and the set hull."""
    total = sum(F(a["weight"]) for a in atoms)

    def pick(a):
        v = a["value"]
        if key is not None and v is not None:
            return v[key]
        return v

    def pick_set(a):
        s = a.get("set")
        if s is None:
            v = pick(a)
            return None if v is None else [v, v]
        return s[key] if key is not None and isinstance(s, dict) else s

    live = [a for a in atoms if pick(a) is not None]
    unidentified = sum((F(a["weight"]) for a in atoms if pick(a) is None), F(0)) / total
    weight = sum(F(a["weight"]) for a in live)
    value = sum(F(a["weight"]) * F(pick(a)).limit_denominator(10**12) for a in live) / weight
    bounds = [pick_set(a) for a in live]
    return {
        "value": float(value),
        "unidentified_mass": float(unidentified),
        "set": [min(b[0] for b in bounds), max(b[1] for b in bounds)],
    }


def build() -> dict:
    ps = sibling("path_specific_mixed_graphs")
    dist = sibling("interventional_distribution_classes")
    med = sibling("mediation_mixed_graphs")
    linear = linear_table()
    tables = {
        "district_admg": ps["district_admg"],
        "collider": ps["collider_class"],
        "routed": ps["routed_markov"],
        "class_disagree": dist["class_disagree"],
        "class_point": dist["class_point"],
        "covariate_latent": med["covariate_latent"],
        "markov_parents": med["markov_parents"],
        "linear_mediation": {"columns": ["z", "t", "m", "y"], "contingency_table": linear, "rows": linear},
    }
    out_tables = {
        k: {"columns": v["columns"], "contingency_table": v["contingency_table"]}
        for k, v in tables.items()
    }
    specs = posterior_specs()
    posteriors = {}
    for family, by_class in specs.items():
        posteriors[family] = {}
        for klass, spec in by_class.items():
            ev = evaluate(family, klass, spec, tables)
            entry = {**ev}
            if family == "mediation":
                entry["mixture"] = {c: mix(family, ev["atoms"], c) for c in ("total", "direct", "indirect")}
            elif family == "distribution":
                entry["mixture"] = {c: mix(family, ev["atoms"], c) for c in ("y0", "y1")}
            else:
                entry["mixture"] = mix(family, ev["atoms"])
            posteriors[family][klass] = entry
    return {
        "schema_version": 1,
        "case": "functional_graph_posterior_mixtures",
        "queries": {
            "path_specific": {"treatment": "t", "outcome": "y", "path_nodes": ["a"]},
            "distribution": {"outcome": "y", "do": {"t": 1.0}},
            "mediation": {"treatment": "t", "outcome": "y", "mediators": ["m"]},
        },
        "tables": out_tables,
        "posteriors": posteriors,
        "bayesian": {
            "posterior_mean_tolerance": 0.04,
            "linear_mean_tolerance": 0.15,
            "note": "declared acceptance bands of the conjugate and Dirichlet row posteriors around the exact mixture",
        },
        "frequentist": {"absolute_tolerance": 1e-9},
    }


def close(got, want, label, problems, tol=1e-12):
    if abs(float(got) - float(want)) > tol:
        problems.append(f"{label}: derived {float(got)!r} != fixture {want!r}")


def check(document: dict) -> list[str]:
    problems: list[str] = []
    fresh = build()
    for key in ("tables", "queries", "posteriors"):
        if document[key] != fresh[key]:
            problems.append(f"{key}: the fixture is not what the derivation produces")
    # The tables are the single-graph fixtures' tables.
    for name, owner, case in [
        ("district_admg", "path_specific_mixed_graphs", "district_admg"),
        ("collider", "path_specific_mixed_graphs", "collider_class"),
        ("routed", "path_specific_mixed_graphs", "routed_markov"),
        ("class_disagree", "interventional_distribution_classes", "class_disagree"),
        ("class_point", "interventional_distribution_classes", "class_point"),
        ("covariate_latent", "mediation_mixed_graphs", "covariate_latent"),
        ("markov_parents", "mediation_mixed_graphs", "markov_parents"),
    ]:
        if document["tables"][name]["contingency_table"] != sibling(owner)[case]["contingency_table"]:
            problems.append(f"table {name} differs from {owner}")
    # Generating-graph atoms reproduce the structural truth of their own fixture.
    ps = sibling("path_specific_mixed_graphs")
    for atom_ in document["posteriors"]["path_specific"]["dag"]["atoms"]:
        if atom_["name"] == "markov":
            close(atom_["value"], ps["routed_markov"]["truth"]["path_specific_effect"], "collider generating truth", problems)
    med = sibling("mediation_mixed_graphs")
    for atom_ in document["posteriors"]["mediation"]["admg"]["atoms"]:
        if atom_["name"] == "covariate_adjusted":
            for k, truth in [("total", "total_effect"), ("direct", "natural_direct_effect"), ("indirect", "natural_indirect_effect")]:
                close(atom_["value"][k], med["covariate_latent"]["truth"][truth], f"mediation {k}", problems)
    lin = document["posteriors"]["mediation"]["dag"]["atoms"][0]["value"]
    close(lin["direct"], 1, "linear direct", problems)
    close(lin["indirect"], 6, "linear indirect", problems)
    close(lin["total"], 7, "linear total", problems)
    # Every mixture must differ from every identified atom it mixes (the weights matter).
    for family, by_class in document["posteriors"].items():
        for klass, entry in by_class.items():
            mixtures = entry["mixture"] if family != "path_specific" else {"effect": entry["mixture"]}
            for comp, mixed in mixtures.items():
                values = []
                for a in entry["atoms"]:
                    v = a["value"]
                    if v is None:
                        continue
                    values.append(v[comp] if isinstance(v, dict) else v)
                if len(values) > 1 and max(values) - min(values) > 1e-6:
                    if min(abs(mixed["value"] - v) for v in values) < 1e-4:
                        problems.append(f"{family} {klass} {comp}: the mixture equals one atom")
    return problems


MUTATIONS = [
    ("a count changed", lambda d: d["tables"]["collider"]["contingency_table"][0].__setitem__("count", 99)),
    ("a mixture changed", lambda d: d["posteriors"]["path_specific"]["dag"]["mixture"].__setitem__("value", 0.2)),
    ("a weight changed", lambda d: d["posteriors"]["mediation"]["admg"]["atoms"][0].__setitem__("weight", "9/10")),
    ("an atom value changed", lambda d: d["posteriors"]["distribution"]["dag"]["atoms"][0].__setitem__("value", {"y0": 0.1, "y1": 0.9})),
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
