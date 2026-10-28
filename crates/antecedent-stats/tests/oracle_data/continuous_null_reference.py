# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""Distance correlation and kNN distance statistic: values and exact permutation p-values with ties.

Regenerate with:  uv run --script continuous_null_reference.py > continuous_null_reference.json

Series are small and tie-rich, with values on a decimal grid so that reflections and relabelings of
the grid give a permuted series with the same statistic as a real number.  Every distinct
arrangement of Y is enumerated (at most 2520) with its multiplicity, and the exact p-value is the
fraction of arrangements whose statistic is at least the observed one; statistics within 1e-40 of
each other at 60 digits are equal (they are equal as real numbers when the arrangements are related
by a symmetry of the grid; no other pair is that close for the sizes here).

dCor is the Szekely-Rizzo-Bakirov V-statistic dCov^2 / sqrt(dVar^2(x) dVar^2(y)), square-rooted, from the
double-centered distance matrices.  The kNN statistic is minus the mean Euclidean distance from each
row's (x, y) point to its k-th nearest other row.
Repository-authored (MIT OR Apache-2.0).
"""
import itertools, json
from collections import Counter
from mpmath import mp, mpf, sqrt

mp.dps = 60
TOL = mpf(10) ** -40


def grid(v):
    return [mpf(a) / 10 for a in v]


def dcor(x, y):
    n = len(x)
    def centered(v):
        a = [[abs(v[i] - v[j]) for j in range(n)] for i in range(n)]
        r = [sum(row) / n for row in a]; g = sum(r) / n
        return [[a[i][j] - r[i] - r[j] + g for j in range(n)] for i in range(n)]
    A, B = centered(x), centered(y)
    cov = sum(A[i][j] * B[i][j] for i in range(n) for j in range(n)) / n**2
    va = sum(A[i][j] ** 2 for i in range(n) for j in range(n)) / n**2
    vb = sum(B[i][j] ** 2 for i in range(n) for j in range(n)) / n**2
    return sqrt(max(cov, mpf(0)) / sqrt(va * vb))


def knn(x, y, k):
    n = len(x)
    tot = mpf(0)
    for i in range(n):
        d = sorted(sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2) for j in range(n) if j != i)
        tot += d[k - 1]
    return -tot / n


def exact(stat, x, y_int):
    """y_int: integer grid codes of y; returns (observed float, exact p)."""
    obs = stat(grid(x), grid(y_int))
    total = hit = 0
    for perm, mult in Counter(itertools.permutations(y_int)).items():
        total += mult
        if stat(grid(x), grid(perm)) >= obs - TOL:
            hit += mult
    return float(obs), hit / total


cases = []


def add(name, kind, x, y, k=0):
    stat = (lambda a, b: dcor(a, b)) if kind == "dcor" else (lambda a, b: knn(a, b, k))
    obs, p = exact(stat, x, y)
    cases.append({"name": name, "kind": kind, "k": k, "x": [v / 10 for v in x], "y": [v / 10 for v in y],
                  "statistic": obs, "exact_p": p})


add("dcor_reflection_symmetric_grid", "dcor", [0, 1, 2, 3, 0, 1, 2, 3], [0, 1, 2, 3, 3, 2, 1, 0])
add("dcor_ternary_pairs", "dcor", [0, 1, 2, 0, 1, 2, 0, 1, 2], [0, 1, 2, 1, 2, 0, 0, 1, 2])
add("dcor_binary_balanced", "dcor", [0, 0, 0, 0, 3, 3, 3, 3], [0, 0, 0, 3, 0, 3, 3, 3])
add("dcor_four_levels_paired", "dcor", [0, 0, 1, 1, 2, 2, 3, 3], [3, 0, 1, 2, 3, 0, 2, 1])
add("dcor_exactly_independent_table", "dcor", [0, 0, 0, 0, 3, 3, 3, 3], [0, 0, 3, 3, 0, 0, 3, 3])
add("knn1_binary_balanced", "knn", [0, 0, 0, 0, 3, 3, 3, 3], [0, 0, 0, 3, 0, 3, 3, 3], 1)
add("knn1_ternary", "knn", [0, 1, 2, 0, 1, 2, 0, 1, 2], [0, 1, 2, 1, 2, 0, 0, 1, 2], 1)
add("knn2_reflection_symmetric_grid", "knn", [0, 1, 2, 3, 0, 1, 2, 3], [0, 1, 2, 3, 3, 2, 1, 0], 2)
add("knn2_four_levels_paired", "knn", [0, 0, 1, 1, 2, 2, 3, 3], [3, 0, 1, 2, 3, 0, 2, 1], 2)
print(json.dumps({"cases": cases}))
