# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Exact laws of composed acyclic mechanisms: path algebra, conditioning, delta method, enumeration.

Regenerate with:  uv run --script acyclic_joint_model_reference.py > acyclic_joint_model_reference.json

Linear-Gaussian family.  Every mechanism is X_v = a_v + sum_p b_vp X_p + sum_l g_vl U_l + s_v e_v
with independent standard-normal e_v and U_l.  Writing X = (I - B)^-1 (c + G eps) with the rows of
an intervened node replaced by X_v = x, the mean is (I - B)^-1 c and the covariance
(I - B)^-1 G G' (I - B)^-T, evaluated here by exact rational Gauss-Jordan inversion rather than the
forward substitution of the product.  Conditioning on an observed block o is the Schur complement
S_rr - S_ro S_oo^-1 S_or with the log density -(k ln 2 pi + ln det S_oo + r' S_oo^-1 r) / 2.  The
expected mean of an outcome under do() is multilinear in the intercepts and coefficients (each
directed path uses an edge at most once), so its exact partial derivative with respect to one
parameter is the difference f(theta + e_i) - f(theta); the first-order variance is g' Sigma g for
a declared parameter covariance Sigma = A A' built from an integer matrix A (positive semidefinite
by construction, including a rank-deficient one).

Finite family.  P(v) = prod_i P(v_i | pa_i) over all joint assignments with Fraction arithmetic,
do() pinning a variable and dropping its factor.

Coefficients, intercepts, sigmas and interventions are dyadic so that the floats the product sees
are exactly the rationals used here.  Repository-authored (MIT OR Apache-2.0).
"""
import itertools
import json
import math
import random
from fractions import Fraction as F


def inv(M):
    n = len(M)
    A = [row[:] + [F(int(i == j)) for j in range(n)] for i, row in enumerate(M)]
    for c in range(n):
        piv = next(r for r in range(c, n) if A[r][c] != 0)
        A[c], A[piv] = A[piv], A[c]
        d = A[c][c]
        A[c] = [x / d for x in A[c]]
        for r in range(n):
            if r != c and A[r][c] != 0:
                f = A[r][c]
                A[r] = [x - f * y for x, y in zip(A[r], A[c])]
    return [row[n:] for row in A]


def matmul(A, B):
    return [[sum(a * b for a, b in zip(row, col)) for col in zip(*B)] for row in A]


def det(M):
    n = len(M)
    A = [row[:] for row in M]
    d = F(1)
    for c in range(n):
        piv = next((r for r in range(c, n) if A[r][c] != 0), None)
        if piv is None:
            return F(0)
        if piv != c:
            A[c], A[piv] = A[piv], A[c]
            d = -d
        d *= A[c][c]
        for r in range(c + 1, n):
            f = A[r][c] / A[c][c]
            A[r] = [x - f * y for x, y in zip(A[r], A[c])]
    return d


def lg_law(nodes, latents, fixed):
    names = [nd["name"] for nd in nodes]
    n = len(names)
    L = len(latents)
    ix = {v: i for i, v in enumerate(names)}
    Mx = [[F(int(i == j)) for j in range(n)] for i in range(n)]
    c = [F(0)] * n
    G = [[F(0)] * (n + L) for _ in range(n)]
    for i, nd in enumerate(nodes):
        if nd["name"] in fixed:
            c[i] = F(fixed[nd["name"]])
            continue
        c[i] = F(nd["intercept"])
        for p, b in nd["parents"].items():
            Mx[i][ix[p]] -= F(b)
        G[i][i] = F(nd["sigma"])
        for u, g in nd["loadings"].items():
            G[i][n + latents.index(u)] = F(g)
    Minv = inv(Mx)
    mean = [sum(Minv[i][j] * c[j] for j in range(n)) for i in range(n)]
    T = matmul(Minv, G)
    cov = matmul(T, [list(col) for col in zip(*T)])
    return names, mean, cov


def condition(names, mean, cov, observed):
    n = len(names)
    o = [names.index(v) for v in sorted(observed, key=names.index)]
    vals = [F(observed[names[i]]) for i in o]
    r = [i for i in range(n) if i not in o]
    Soo = [[cov[a][b] for b in o] for a in o]
    Sinv = inv(Soo)
    resid = [v - mean[i] for v, i in zip(vals, o)]
    w = [sum(Sinv[a][b] * resid[b] for b in range(len(o))) for a in range(len(o))]
    cmean = [mean[i] + sum(cov[i][o[a]] * w[a] for a in range(len(o))) for i in r]
    Sro = [[cov[i][b] for b in o] for i in r]
    red = matmul(matmul(Sro, Sinv), [list(col) for col in zip(*Sro)])
    ccov = [[cov[r[a]][r[b]] - red[a][b] for b in range(len(r))] for a in range(len(r))]
    quad = sum(a * b for a, b in zip(resid, w))
    logd = -0.5 * (len(o) * math.log(2 * math.pi) + math.log(float(det(Soo))) + float(quad))
    return [names[i] for i in r], cmean, ccov, logd


def fl(x):
    return [float(v) for v in x]


def law_json(names, mean, cov):
    n = len(names)
    return {"names": names, "mean": fl(mean), "cov": [fl(row) for row in cov]}


def build_nodes(rng, spec):
    """spec: list of (name, n_parent_choices, sigma). Parents chosen among earlier nodes."""
    nodes = []
    for i, (name, sigma) in enumerate(spec):
        k = rng.randint(0, min(3, i))
        parents = {spec[j][0]: F(rng.randint(-12, 12), 8) for j in rng.sample(range(i), k)}
        nodes.append({"name": name, "intercept": F(rng.randint(-16, 16), 8), "sigma": F(sigma),
                      "parents": parents, "loadings": {}})
    return nodes


def dump_nodes(nodes):
    return [{"name": nd["name"], "intercept": float(nd["intercept"]), "sigma": float(nd["sigma"]),
             "parents": {p: float(b) for p, b in nd["parents"].items()},
             "loadings": {u: float(g) for u, g in nd["loadings"].items()}} for nd in nodes]


def params_of(nodes):
    out = []
    for nd in sorted(nodes, key=lambda x: x["name"]):
        out.append((nd["name"], "intercept"))
        for p in sorted(nd["parents"]):
            out.append((nd["name"], p))
    return out


def with_param(nodes, key, delta):
    import copy
    new = copy.deepcopy(nodes)
    for nd in new:
        if nd["name"] == key[0]:
            if key[1] == "intercept":
                nd["intercept"] += delta
            else:
                nd["parents"][key[1]] += delta
    return new


def contrast_value(nodes, latents, outcome, arms):
    total = F(0)
    for w, fixed in arms:
        names, mean, _ = lg_law(nodes, latents, fixed)
        total += F(w) * mean[names.index(outcome)]
    return total


def linear_case(seed, spec, latent_decl, scenarios, outcome, arms, cond_scenarios, name):
    rng = random.Random(seed)
    nodes = build_nodes(rng, spec)
    latents = sorted(latent_decl)
    for u, members in latent_decl.items():
        for m in members:
            next(nd for nd in nodes if nd["name"] == m)["loadings"][u] = F(rng.choice([-1, 1]) * rng.randint(1, 6), 4)
    laws = []
    for fixed in scenarios:
        names, mean, cov = lg_law(nodes, latents, fixed)
        laws.append({"do": {k: float(v) for k, v in fixed.items()}, **law_json(names, mean, cov)})
    conds = []
    for fixed, observed in cond_scenarios:
        names, mean, cov = lg_law(nodes, latents, fixed)
        rn, cm, cc, ld = condition(names, mean, cov, observed)
        conds.append({"do": {k: float(v) for k, v in fixed.items()},
                      "observed": {k: float(v) for k, v in observed.items()},
                      "names": rn, "mean": fl(cm), "cov": [fl(r) for r in cc], "log_density": ld})
    # delta method
    params = params_of(nodes)
    f0 = contrast_value(nodes, latents, outcome, arms)
    grad = [contrast_value(with_param(nodes, k, F(1)), latents, outcome, arms) - f0 for k in params]
    # Sigma = A A' with an integer A of rank 3 (rank-deficient for more than three parameters)
    k = len(params)
    A = [[rng.randint(-3, 3) for _ in range(3)] for _ in range(k)]
    Sigma = [[F(sum(A[i][t] * A[j][t] for t in range(3)), 64) for j in range(k)] for i in range(k)]
    var_joint = sum(grad[i] * Sigma[i][j] * grad[j] for i in range(k) for j in range(k))
    # independent by design: only the diagonal blocks of each mechanism
    mech_of = [p[0] for p in params]
    var_ind = sum(grad[i] * Sigma[i][j] * grad[j] for i in range(k) for j in range(k) if mech_of[i] == mech_of[j])
    return {"name": name, "latents": latents, "nodes": dump_nodes(nodes), "laws": laws, "conditionings": conds,
            "contrast": {"outcome": outcome, "arms": [{"weight": float(F(w)), "do": {a: float(F(b)) for a, b in fx.items()}} for w, fx in arms],
                         "point": float(f0)},
            "parameters": [{"mechanism": m, "term": t} for m, t in params],
            "gradient": fl(grad), "sigma": [fl(r) for r in Sigma],
            "variance_joint": float(var_joint), "variance_independent": float(var_ind)}


def finite_case(seed, vars_, level_names, parent_choices, scenarios, name, dyadic=True):
    rng = random.Random(seed)
    tables = {}
    for v in vars_:
        pa = parent_choices[v]
        configs = list(itertools.product(*[level_names[p] for p in pa]))
        rows = {}
        for cfg in configs:
            m = len(level_names[v])
            if dyadic:
                cuts = sorted(rng.randint(0, 64) for _ in range(m - 1))
                parts = [b - a for a, b in zip([0] + cuts, cuts + [64])]
                if rng.random() < 0.3:  # an exact zero cell where possible
                    parts[0], parts[-1] = 0, parts[0] + parts[-1]
                probs = [F(x, 64) for x in parts]
            else:
                w = [rng.choice([1, 1, 3, 997]) for _ in range(m)]
                probs = [F(x, sum(w)) for x in w]
            rows[cfg] = probs
        tables[v] = rows
    out_scen = []
    for fixed in scenarios:
        states = []
        for assign in itertools.product(*[level_names[v] for v in vars_]):
            a = dict(zip(vars_, assign))
            p = F(1)
            for v in vars_:
                if v in fixed:
                    if a[v] != fixed[v]:
                        p = F(0)
                        break
                else:
                    cfg = tuple(a[q] for q in parent_choices[v])
                    p *= tables[v][cfg][level_names[v].index(a[v])]
            states.append({"state": a, "p": float(p), "exact": [p.numerator, p.denominator]})
        out_scen.append({"do": fixed, "states": states})
    return {"name": name, "variables": vars_, "levels": level_names, "parents": parent_choices,
            "tables": {v: [{"parents": list(cfg), "probs": [float(x) for x in pr], "exact": [[x.numerator, x.denominator] for x in pr]} for cfg, pr in rows.items()]
                       for v, rows in tables.items()},
            "scenarios": out_scen}


def main():
    spec6 = [("q", 1), ("Zb", F(1, 2)), ("a1", 1), ("m", F(3, 2)), ("Y", F(1, 2)), ("w", 2)]
    c1 = linear_case(11, spec6, {"U": ["Zb", "w"], "W": ["q", "m", "Y"]},
                     [{}, {"q": F(1)}, {"m": F(-3, 2), "a1": F(1, 4)}, {"q": F(2), "Y": F(1, 8), "w": F(-1)}],
                     "Y", [(1, {"q": F(1)}), (F(-1, 2), {"q": F(0)}), (F(-1, 2), {"q": F(0), "m": F(1, 2)})],
                     [({}, {"Y": F(1, 2), "a1": F(-1)}), ({"q": F(1)}, {"m": F(2), "w": F(1, 4)}), ({}, {"q": F(1, 4)})],
                     "six_nodes_two_latents")
    # near-deterministic child: sigma = 1/256 of its parent, conditioning on parent and child
    spec_det = [("x", 2), ("child", F(1, 256)), ("z", F(1, 2)), ("out", F(1, 2))]
    rng = random.Random(5)
    c2 = linear_case(27, spec_det, {"U": ["z", "out"]},
                     [{}, {"x": F(1)}, {"z": F(-1)}],
                     "out", [(1, {"x": F(1)}), (-1, {"x": F(0)})],
                     [({}, {"x": F(1, 2), "child": F(3, 4)})],
                     "near_deterministic_child")
    # a long chain (path products) with a collider and extreme coefficients
    spec_big = [("a", 1), ("b", F(1, 2)), ("c", F(1, 2)), ("d", F(1, 2)), ("e", F(1, 2)), ("f", F(1, 2)), ("g", F(1, 2)), ("h", F(1, 2))]
    c3 = linear_case(31, spec_big, {"U": ["a", "h"]},
                     [{}, {"a": F(1), "d": F(-1)}, {"b": F(1, 2), "g": F(2), "c": F(0)}],
                     "h", [(1, {"a": F(1)}), (-1, {"a": F(0)}), (F(1, 2), {"d": F(1)})],
                     [({"b": F(1)}, {"h": F(1), "g": F(0)}), ({}, {"a": F(1), "c": F(2), "e": F(-1), "h": F(1, 8)})],
                     "eight_nodes")
    levels = {"B": ["10", "2", "9"], "a": ["no", "yes"], "C": ["lo", "mid", "hi"], "d": ["0", "1"]}
    parents = {"a": [], "B": ["a"], "C": ["B", "a"], "d": ["C", "B"]}
    f1 = finite_case(41, ["B", "C", "a", "d"], levels, parents,
                     [{}, {"B": "2"}, {"B": "10", "d": "1"}, {"a": "yes", "C": "hi"}], "dyadic_tables_with_zero_cells")
    f2 = finite_case(43, ["B", "C", "a", "d"], levels, parents, [{}, {"C": "mid"}, {"a": "no", "B": "9", "C": "lo", "d": "0"}],
                     "extreme_odds_tables", dyadic=False)
    print(json.dumps({"linear": [c1, c2, c3], "finite": [f1, f2]}))


main()
