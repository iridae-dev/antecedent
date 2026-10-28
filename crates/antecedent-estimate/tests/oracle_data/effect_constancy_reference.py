"""Effect-constancy statistics, p-values and Holm adjustments in 60-digit arithmetic.

Cochran's Q = sum_i ((e_i - pooled) / se_i)^2 with the inverse-variance pooled effect; the Wald
statistic is d' V^-1 d for d_i = e_i - e_0 and V the covariance of the contrasts against the
first partition in label order. Tails: chi-square survival function with k - 1 degrees of freedom
(regularized upper incomplete gamma) and the two-sided normal p-value erfc(|z| / sqrt(2)) of each
contrast. Holm: sort ascending, multiply the i-th smallest by (m - i), take the running maximum,
cap at one.

    uv run --with mpmath python effect_constancy_reference.py > effect_constancy_reference.json
"""

import json
from itertools import combinations

from mpmath import erfc, gammainc, inf, lu_solve, matrix, mp, mpf, sqrt

mp.dps = 60

# (name, [(label, effect, se)] in the supplied order, covariance rows in supplied order or None,
#  family: "all" or a reference label)
CASES = [
    ("two_close", [("p1", 1.0, 0.5), ("p2", 1.2, 0.5)], None, "all"),
    ("three_unequal_precision", [("a", 0.0, 1.0), ("b", 1.0, 0.5), ("c", 5.0, 2.0)], None, "all"),
    (
        "five_strongly_heterogeneous",
        [("a", 0.0, 1.0), ("b", 10.0, 1.0), ("c", 20.0, 1.0), ("d", -5.0, 1.0), ("e", 40.0, 1.0)],
        None,
        "c",
    ),
    (
        "eight_precisions_over_six_decades",
        [
            ("a", 0.1, 1e-3), ("b", 0.3, 1e3), ("c", -2.0, 0.5), ("d", 0.12, 2e-3),
            ("e", 7.0, 30.0), ("f", 0.0, 1.0), ("g", 1.5, 0.25), ("h", -0.4, 4.0),
        ],
        None,
        "all",
    ),
    ("near_null", [("a", 1.0, 1.0), ("b", 1.0 + 1e-6, 1.0), ("c", 1.0 - 1e-6, 1.0)], None, "a"),
    ("far_tail", [("a", 0.0, 1.0), ("b", 40.0, 1.0)], None, "all"),
    (
        "huge_scale",
        [("a", 1e150, 1e149), ("b", 2e150, 2e149), ("c", 1.5e150, 1e149)],
        None,
        "all",
    ),
    (
        "correlated_supplied_out_of_label_order",
        [("c", 0.5, 1.0), ("a", 0.0, 0.8), ("d", 2.0, 1.1), ("b", 1.0, 0.9)],
        [
            [1.0, 0.3, -0.2, 0.1],
            [0.3, 0.64, 0.25, 0.2],
            [-0.2, 0.25, 1.21, 0.15],
            [0.1, 0.2, 0.15, 0.81],
        ],
        "a",
    ),
]


def chi_square_sf(x, df):
    return gammainc(mpf(df) / 2, x / 2, inf, regularized=True)


def holm(p):
    m = len(p)
    order = sorted(range(m), key=lambda i: (p[i], i))
    out = [mpf(0)] * m
    running = mpf(0)
    for rank, i in enumerate(order):
        running = max(running, min(mpf(1), (m - rank) * p[i]))
        out[i] = running
    return out


def main():
    results = []
    for name, parts, cov, family in CASES:
        k = len(parts)
        order = sorted(range(k), key=lambda i: (parts[i][0], i))
        labels = [parts[i][0] for i in order]
        e = [mpf(parts[i][1]) for i in order]
        se = [mpf(parts[i][2]) for i in order]
        if cov is None:
            weights = [1 / s**2 for s in se]
            pooled = sum(w * x for w, x in zip(weights, e)) / sum(weights)
            statistic = sum(((x - pooled) / s) ** 2 for x, s in zip(e, se))
            pooled_out = float(pooled)
            cov_sorted = [[(se[i] ** 2 if i == j else mpf(0)) for j in range(k)] for i in range(k)]
        else:
            cov_sorted = [[mpf(cov[order[i]][order[j]]) for j in range(k)] for i in range(k)]
            m = k - 1
            v = matrix(m, m)
            for i in range(m):
                for j in range(m):
                    v[i, j] = (
                        cov_sorted[i + 1][j + 1] - cov_sorted[i + 1][0]
                        - cov_sorted[0][j + 1] + cov_sorted[0][0]
                    )
            d = matrix([e[i] - e[0] for i in range(1, k)])
            statistic = (d.T * lu_solve(v, d))[0]
            pooled_out = None
        if family == "all":
            pairs = list(combinations(range(k), 2))
        else:
            ref = labels.index(family)
            pairs = [(i, ref) for i in range(k) if i != ref]
        rows = []
        for a, b in pairs:
            var = cov_sorted[a][a] + cov_sorted[b][b] - 2 * cov_sorted[a][b]
            diff = e[a] - e[b]
            s = sqrt(var)
            z = diff / s
            rows.append((a, b, diff, s, z, erfc(abs(z) / sqrt(2))))
        adjusted = holm([r[5] for r in rows])
        results.append(
            {
                "name": name,
                "partitions": [{"label": l, "effect": p, "se": s} for l, p, s in parts],
                "covariance": None if cov is None else [x for row in cov for x in row],
                "family": family,
                "statistic": float(statistic),
                "df": k - 1,
                "p_value": float(chi_square_sf(statistic, k - 1)),
                "pooled": pooled_out,
                "contrasts": [
                    {
                        "left": labels[a],
                        "right": labels[b],
                        "difference": float(d),
                        "se": float(s),
                        "z": float(z),
                        "p": float(p),
                        "p_holm": float(h),
                    }
                    for (a, b, d, s, z, p), h in zip(rows, adjusted)
                ],
            }
        )
    print(json.dumps({"cases": results}, indent=1))


main()
