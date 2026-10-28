# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""High-precision reference for the discretized linear-Gaussian bridge.

Regenerate with:  uv run --script discretized_scm_reference.py > discretized_scm_reference.json

Everything is recomputed from the definitions at 50 digits, without the product's code path:
  * marginal mean and variance of each variable by the linear recursion on the observational
    model (variances from path covariances, not from the product's composed law);
  * the equal-probability grid: edges mu + sigma z_i with z_i = Phi^-1(i/n) from mpmath's
    erfinv, and the cell value mu + sigma n (phi(z_j) - phi(z_{j+1})), the conditional mean;
  * each conditional table: the cell probabilities of N(a + sum b_p x_p, s^2) at the parents'
    cell values by differences of Phi, and delta = table mean - mechanism mean;
  * E[outcome | do(arm)] in the discretized model by enumerating the whole joint space under
    the factorization with the intervened variables fixed, and the closed-form value by the
    linear recursion at the same intervention values;
  * the computed bound of the module doc: e_v = E|delta_v| under the intervened discretized joint
    law and |err_v| <= e_v + sum_p |b_vp| |err_p| in topological order (intervened: zero).

Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import itertools
import json

import mpmath as mp

mp.mp.dps = 50


def Phiinv(p):
    return mp.sqrt(2) * mp.erfinv(2 * p - 1)


def build(mechs, levels):
    names = [m[0] for m in mechs]
    idx = {n: i for i, n in enumerate(names)}
    mean, cov = [], []
    for i, (n, a, s, ps) in enumerate(mechs):
        m = mp.mpf(a) + sum(mp.mpf(c) * mean[idx[p]] for p, c in ps)
        row = [sum(mp.mpf(c) * cov[idx[p]][j] for p, c in ps) for j in range(i)]
        v = mp.mpf(s) ** 2 + sum(mp.mpf(c) * row[idx[p]] for p, c in ps)
        cov.append(row + [v])
        for j in range(i):
            cov[j].append(row[j])
        mean.append(m)
    nodes = []
    for i, (n, a, s, ps) in enumerate(mechs):
        L = levels[n]
        sd = mp.sqrt(cov[i][i])
        z = [Phiinv(mp.mpf(k) / L) for k in range(1, L)]
        edges = [mean[i] + sd * zk for zk in z]

        def dens(k, z=z, L=L):
            return mp.mpf(0) if k in (0, L) else mp.npdf(z[k - 1])

        values = [mean[i] + sd * L * (dens(j) - dens(j + 1)) for j in range(L)]
        nodes.append({"name": n, "a": mp.mpf(a), "s": mp.mpf(s), "ps": [(idx[p], mp.mpf(c)) for p, c in ps],
                      "edges": edges, "values": values, "L": L})
    for nd in nodes:
        pl = [range(nodes[p]["L"]) for p, _ in nd["ps"]]
        table, delta = {}, {}
        worst = mp.mpf(0)
        for cfg in itertools.product(*pl):
            mu = nd["a"] + sum(c * nodes[p]["values"][l] for (p, c), l in zip(nd["ps"], cfg))
            cdf = [mp.ncdf((e - mu) / nd["s"]) for e in nd["edges"]]
            probs = [cdf[0]] + [cdf[k] - cdf[k - 1] for k in range(1, len(cdf))] + [1 - cdf[-1]]
            tm = sum(p * v for p, v in zip(probs, nd["values"]))
            table[cfg] = probs
            delta[cfg] = tm - mu
            worst = max(worst, abs(tm - mu))
        nd["table"], nd["delta"], nd["worst"] = table, delta, worst
    return nodes


def evaluate(nodes, outcome, arm):
    n = len(nodes)
    expected = [mp.mpf(0)] * n
    mean_out = mp.mpf(0)
    for state in itertools.product(*[range(nd["L"]) for nd in nodes]):
        pr = mp.mpf(1)
        for v, nd in enumerate(nodes):
            if v in arm:
                if state[v] != arm[v]:
                    pr = 0
                    break
                continue
            pr *= nd["table"][tuple(state[p] for p, _ in nd["ps"])][state[v]]
        if pr == 0:
            continue
        mean_out += pr * nodes[outcome]["values"][state[outcome]]
        for v, nd in enumerate(nodes):
            if v not in arm:
                expected[v] += pr * abs(nd["delta"][tuple(state[p] for p, _ in nd["ps"])])
    closed, bound = [], []
    for v, nd in enumerate(nodes):
        if v in arm:
            closed.append(nd["values"][arm[v]])
            bound.append(mp.mpf(0))
        else:
            closed.append(nd["a"] + sum(c * closed[p] for p, c in nd["ps"]))
            bound.append(expected[v] + sum(abs(c) * bound[p] for p, c in nd["ps"]))
    return mean_out, closed[outcome], bound[outcome]


CHAIN = [("X", 1, 2, []), ("M", 0.5, 1, [("X", 2)]), ("Y", -1, 0.5, [("X", 0.5), ("M", 3)])]
CASES = [
    ("one_edge_two_levels", [("X", 0, 1, []), ("Y", 0, 1, [("X", 1)])], {"X": 2, "Y": 2},
     [("Y", [(1, {"X": 1}), (-1, {"X": 0})]), ("Y", [(1, {})])]),
    ("chain_three_levels", CHAIN, {"X": 3, "M": 3, "Y": 3},
     [("Y", [(1, {"X": 2}), (-1, {"X": 0})]), ("Y", [(1, {"M": 1})]), ("M", [(1, {"X": 1})])]),
    ("chain_mixed_levels", CHAIN, {"X": 7, "M": 5, "Y": 4},
     [("Y", [(1, {"X": 5}), (-1, {"X": 1})]), ("Y", [(2, {"X": 3, "M": 0}), (-1, {"X": 6, "M": 4})]),
      ("Y", [(1, {"Y": 2})]), ("Y", [(1, {})])]),
    ("chain_eight_levels", CHAIN, {"X": 8, "M": 8, "Y": 8}, [("Y", [(1, {"X": 6}), (-1, {"X": 1})])]),
    ("collider_and_fork_negative_coefficients",
     [("Z", 0, 1.5, []), ("A", 1, 0.7, [("Z", -2)]), ("B", -0.5, 0.4, [("Z", 1.25)]),
      ("C", 2, 0.3, [("A", 1.5), ("B", -3)])],
     {"Z": 4, "A": 4, "B": 3, "C": 4},
     [("C", [(1, {"Z": 3}), (-1, {"Z": 0})]), ("C", [(1, {"A": 0, "B": 2})]), ("C", [(1, {})])]),
    ("tiny_noise_relative_to_signal", [("X", 0, 1, []), ("Y", 0, 0.01, [("X", 5)])], {"X": 8, "Y": 8},
     [("Y", [(1, {"X": 7}), (-1, {"X": 0})]), ("Y", [(1, {})])]),
    ("large_noise_relative_to_signal", [("X", 0, 0.05, []), ("Y", 3, 4, [("X", 1)])], {"X": 6, "Y": 6},
     [("Y", [(1, {"X": 5}), (-1, {"X": 0})])]),
    ("sixty_four_levels", [("X", 0, 1, []), ("Y", 1, 1, [("X", -1)])], {"X": 64, "Y": 64},
     [("Y", [(1, {"X": 48}), (-1, {"X": 15})]), ("Y", [(1, {})])]),
    ("independent_variables", [("X", 2, 1, []), ("Y", -2, 3, [])], {"X": 5, "Y": 5},
     [("Y", [(1, {"X": 2})]), ("X", [(1, {})])]),
]

out = []
for name, mechs, levels, contrasts in CASES:
    nodes = build(mechs, levels)
    names = [nd["name"] for nd in nodes]
    case = {
        "name": name,
        "mechanisms": [{"variable": m[0], "intercept": float(m[1]), "sigma": float(m[2]),
                        "parents": [[p, float(c)] for p, c in m[3]]} for m in mechs],
        "levels": levels,
        "grid": {nd["name"]: {"edges": [float(e) for e in nd["edges"]], "values": [float(v) for v in nd["values"]],
                              "mechanism_mean_error": float(nd["worst"])} for nd in nodes},
        "contrasts": [],
    }
    for outcome, arms in contrasts:
        oi = names.index(outcome)
        rec = {"outcome": outcome, "arms": []}
        value = closed_total = bound_total = mp.mpf(0)
        for w, iv in arms:
            fixed = {names.index(k): l for k, l in iv.items()}
            d, c, b = evaluate(nodes, oi, fixed)
            rec["arms"].append({"weight": float(w), "interventions": [[k, l] for k, l in iv.items()],
                                "discretized": float(d), "closed_form": float(c), "bound": float(b)})
            value += w * d
            closed_total += w * c
            bound_total += abs(w) * b
        rec["value"], rec["closed_form"], rec["bound"] = float(value), float(closed_total), float(bound_total)
        rec["absolute_error"] = float(abs(value - closed_total))
        case["contrasts"].append(rec)
    out.append(case)
print(json.dumps(out, indent=1))
