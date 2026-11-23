"""Linear cyclic systems observed under clamp experiments, in exact rational arithmetic.

For each case a direct-effect matrix B (x = B x + c + e) with spectral radius below one is fixed,
and for each regime that clamps the set J the clamped columns are drawn rows (independent or
deliberately correlated) while the passive columns follow exactly,

    x_U = (I - B_UU)^-1 (B_UJ x_J + c_U),

with no disturbance, so the experimental effect of a clamped unit on a passive unit is exactly
(I - B_UU)^-1 B_UJ and every constraint the product assembles must be satisfied by B. Cases add
a response offset of 1e9, columns scaled over twelve decades (B is rescaled consistently) and a
pair of clamped columns that are nearly collinear (condition number about 1e10, so about six
digits are expected from binary64 and the tolerance says so).

    python3 clamp_experiment_reference.py > clamp_experiment_reference.json
"""

import json
from fractions import Fraction as F


def solve(a, b):
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = next(r for r in range(col, n) if m[r][col] != 0)
        m[col], m[piv] = m[piv], m[col]
        p = m[col][col]
        m[col] = [v / p for v in m[col]]
        for r in range(n):
            if r != col and m[r][col] != 0:
                f = m[r][col]
                m[r] = [v - f * w for v, w in zip(m[r], m[col])]
    return [m[i][n] for i in range(n)]


class Lcg:
    def __init__(self, seed):
        self.s = seed

    def next(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) % 2**64
        return self.s >> 33

    def draw(self):
        return F(self.next() % 801 - 400, 64)


def case(name, b, regimes, rows, seed, offset=F(0), scales=None, nearly_collinear=None, direct_tolerance=1e-7, exact_tolerance=1e-8, refusal=None):
    n = len(b)
    rng = Lcg(seed)
    c = [offset * (u + 1) for u in range(n)] if offset else [F(0)] * n
    out_regimes = []
    for clamped in regimes:
        passive = [u for u in range(n) if u not in clamped]
        cols = {u: [] for u in range(n)}
        for _ in range(rows):
            xj = {}
            first = None
            for k, j in enumerate(clamped):
                r = rng.draw()
                if nearly_collinear and clamped == nearly_collinear[0] and k == 1:
                    xj[j] = first + r * nearly_collinear[1]
                elif len(clamped) > 1 and k == 1:
                    xj[j] = F(3, 5) * first + r
                else:
                    xj[j] = r
                if k == 0:
                    first = xj[j]
                xj[j] += c[j]
            a = [[(F(1) if i == j else F(0)) - b[passive[i]][passive[j]] for j in range(len(passive))] for i in range(len(passive))]
            rhs = [sum(b[passive[i]][j] * xj[j] for j in clamped) + c[passive[i]] for i in range(len(passive))]
            xu = solve(a, rhs)
            for j in clamped:
                cols[j].append(xj[j])
            for i, u in enumerate(passive):
                cols[u].append(xu[i])
        out_regimes.append({"clamped": list(clamped), "columns": [[cols[u]] for u in range(n)]})
    s = scales or [F(1)] * n
    # x'_u = s_u x_u, so b'_tu = b_tu s_t / s_u
    b_scaled = [[b[t][u] * s[t] / s[u] for u in range(n)] for t in range(n)]
    for r in out_regimes:
        r["columns"] = [[float(v * s[u]) for v in col[0]] for u, col in enumerate(r["columns"])]
    return {
        "name": name,
        "units": n,
        "direct": [[float(v) for v in row] for row in b_scaled],
        "regimes": out_regimes,
        "rows": [] if refusal else expected_rows(n, out_regimes),
        "direct_tolerance": direct_tolerance,
        "exact_tolerance": exact_tolerance,
        "refusal": refusal,
    }


def expected_rows(n, regimes):
    """Rows of the constraint system computed in exact arithmetic from the binary64 data."""
    out = []
    for k, r in enumerate(regimes):
        j = r["clamped"]
        p = [u for u in range(n) if u not in j]
        cols = [[F(v) for v in col] for col in r["columns"]]
        rows = len(cols[0])
        mean = [sum(c) / rows for c in cols]
        def cov(a, b):
            return sum((x - mean[a]) * (y - mean[b]) for x, y in zip(cols[a], cols[b]))
        sjj = [[cov(a, b) for b in j] for a in j]
        effect = {}
        for u in p:
            beta = solve(sjj, [cov(a, u) for a in j])
            for a, value in zip(j, beta):
                effect[(a, u)] = value
        for probe in j:
            for target in p:
                coefficients = []
                for s in range(n):
                    if s == target:
                        continue
                    if s == probe:
                        coefficients.append(F(1))
                    elif s in j:
                        coefficients.append(F(0))
                    else:
                        coefficients.append(effect[(probe, s)])
                out.append(
                    {
                        "regime": k,
                        "probe": probe,
                        "target": target,
                        "coefficients": [float(v) for v in coefficients],
                        "rhs": float(effect[(probe, target)]),
                    }
                )
    return out


B4 = [
    [F(0), F(1, 5), F(0), F(-1, 10)],
    [F(1, 2), F(0), F(-1, 4), F(0)],
    [F(0), F(1, 3), F(0), F(1, 5)],
    [F(1, 4), F(0), F(-1, 2), F(0)],
]
B5 = [
    [F(0), F(1, 4), F(0), F(0), F(-1, 5)],
    [F(1, 3), F(0), F(1, 5), F(0), F(0)],
    [F(0), F(-1, 4), F(0), F(2, 5), F(0)],
    [F(1, 5), F(0), F(0), F(0), F(1, 4)],
    [F(0), F(0), F(-1, 3), F(1, 5), F(0)],
]
SINGLE4 = [(0,), (1,), (2,), (3,)]
SINGLE5 = [(j,) for j in range(5)]

cases = [
    case("four_units_cyclic", B4, SINGLE4 + [(0, 1)], 14, 1),
    case("four_units_offset_one_billion", B4, SINGLE4 + [(1, 2)], 14, 2, offset=F(10**9), direct_tolerance=1e-3),
    case("five_units_scaled_over_twelve_decades", B5, SINGLE5 + [(2, 3)], 16, 3, scales=[F(1, 10**6), F(1), F(10**4), F(1, 10**3), F(10**6)]),
    case("four_units_nearly_collinear_clamps", B4, SINGLE4 + [(0, 1)], 14, 4, nearly_collinear=((0, 1), F(1, 10**5)), direct_tolerance=1e-4, exact_tolerance=1e-4),
    case("clamps_collinear_beyond_double_precision", B4, SINGLE4 + [(0, 1)], 14, 5, nearly_collinear=((0, 1), F(1, 10**9)), refusal=["IllConditionedIntervention", "DegenerateIntervention"]),
    case("clamps_exactly_collinear", B4, SINGLE4 + [(0, 1)], 14, 6, nearly_collinear=((0, 1), F(0)), refusal=["DegenerateIntervention"]),
]
print(json.dumps({"cases": cases}))
