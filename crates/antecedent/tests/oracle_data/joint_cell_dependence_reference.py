"""Exact-rational definition-level reference for the joint-cell AIPW under sampling weights, cluster
dependence and serial dependence.

The design has no covariates, so the cell-saturated nuisances are the weighted cell frequencies
(multinomial propensity) and the weighted cell means (outcome model) of each cross-fit training
fold. Everything below is written from the definition with `fractions.Fraction`:

    e_c   = sum_train w 1{cell=c} / sum_train w
    mu_c  = sum_train w y 1{cell=c} / sum_train w 1{cell=c}
    phi_i = mu_c + 1{cell_i=c} (y_i - mu_c) / e_c            (validation rows of the fold)
    theta = sum w phi / sum w
    psi_i = (w_i / mean w) (phi_i - theta)

    independent rows:  Var = sum psi^2 / (n (n - 1))
    clusters:          Var = G/(G-1) sum_g (sum_{i in g} psi_i)^2 / n^2
    Newey-West (lag L):Var = (g0 + 2 sum_{l=1..L} (1 - l/(L+1)) g_l) / n^2,  g_l = sum_t psi_t psi_{t-l}

The fold assignment is the one fixed input: the cross-fit fold plan is the product's seeded,
cell-stratified plan, frozen here (FOLDS, over the retained rows in table order); the test refuses
if the product's plan differs, so a changed fold plan means regenerating this file.

Run: python3 joint_cell_dependence_reference.py > joint_cell_dependence_reference.json
"""

import json
from fractions import Fraction

N = 60
ZERO_WEIGHT_ROW = 13


def cell_of(i):
    return (i % 2, (i // 2) % 2)


def outcome_of(i):
    a, b = cell_of(i)
    base = Fraction(3, 2) + 2 * a + 3 * b + 4 * a * b
    return base + Fraction((7 * i) % 11 - 5, 10) + Fraction((i * i) % 5, 20)


def weight_of(i):
    if i == ZERO_WEIGHT_ROW:
        return Fraction(0)
    if i == 27:
        return Fraction(40)  # one extreme weight
    return Fraction(1 + (i * 5) % 7)


# unbalanced clusters over the source rows (sizes 1..9, 12 clusters)
SIZES = [1, 9, 2, 7, 3, 6, 4, 5, 8, 1, 9, 5]
assert sum(SIZES) == N
CLUSTERS = [g for g, size in enumerate(SIZES) for _ in range(size)]
LAG = 3

FOLDS = {"unweighted": [2, 0, 0, 0, 1, 2, 3, 3, 0, 0, 2, 4, 0, 2, 1, 1, 3, 3, 4, 2, 1, 0, 4, 2, 0, 1, 3, 1, 2, 4, 3, 1, 2, 4, 4, 3, 4, 3, 0, 4, 3, 3, 2, 0, 4, 2, 2, 3, 1, 4, 0, 4, 3, 1, 1, 0, 4, 1, 1, 2], "weighted": [2, 4, 4, 4, 1, 0, 2, 2, 0, 3, 1, 3, 0, 0, 0, 3, 1, 3, 1, 1, 2, 3, 1, 0, 3, 2, 0, 2, 0, 2, 0, 2, 3, 3, 2, 4, 2, 4, 3, 3, 2, 1, 4, 4, 1, 1, 2, 1, 1, 4, 3, 3, 4, 0, 4, 4, 0, 0, 1]}


def estimate(use_weights, dependence):
    rows = [i for i in range(N) if (not use_weights) or weight_of(i) > 0]
    folds = FOLDS["weighted" if use_weights else "unweighted"]
    assert len(rows) == len(folds), (len(rows), len(folds))
    w = {i: (weight_of(i) if use_weights else Fraction(1)) for i in rows}
    cell = {i: cell_of(i)[0] + 2 * cell_of(i)[1] for i in rows}
    y = {i: outcome_of(i) for i in rows}
    fold = dict(zip(rows, folds))
    target = 3  # cell (1, 1)
    phi = {}
    for v in sorted(set(folds)):
        train = [i for i in rows if fold[i] != v]
        total = sum(w[i] for i in train)
        in_cell = [i for i in train if cell[i] == target]
        e = sum(w[i] for i in in_cell) / total
        mu = sum(w[i] * y[i] for i in in_cell) / sum(w[i] for i in in_cell)
        for i in rows:
            if fold[i] == v:
                phi[i] = mu + (1 if cell[i] == target else 0) * (y[i] - mu) / e
    n = len(rows)
    wsum = sum(w[i] for i in rows)
    theta = sum(w[i] * phi[i] for i in rows) / wsum
    wbar = wsum / n
    psi = [(w[i] / wbar) * (phi[i] - theta) for i in rows]
    if dependence == "independent":
        var = sum(p * p for p in psi) / (n * (n - 1))
    elif dependence == "clusters":
        labels = [CLUSTERS[i] for i in rows]
        groups = {}
        for lab, p in zip(labels, psi):
            groups[lab] = groups.get(lab, 0) + p
        g = len(groups)
        var = Fraction(g, g - 1) * sum(s * s for s in groups.values()) / (n * n)
    elif dependence == "newey_west":
        meat = sum(p * p for p in psi)
        for lag in range(1, LAG + 1):
            gamma = sum(psi[t] * psi[t - lag] for t in range(lag, n))
            meat += 2 * (1 - Fraction(lag, LAG + 1)) * gamma
        var = meat / (n * n)
    else:
        raise ValueError(dependence)
    return {
        "mean": [theta.numerator, theta.denominator],
        "variance": [var.numerator, var.denominator],
        "mean_f64": float(theta),
        "variance_f64": float(var),
        "rows": n,
    }


def dec(x):
    # exact decimal string of a rational with denominator dividing 20
    scaled = x * 20
    assert scaled.denominator == 1
    return f"{int(scaled) / 20:.2f}"


out = {
    "rows": N,
    "zero_weight_row": ZERO_WEIGHT_ROW,
    "lag": LAG,
    "t1": [cell_of(i)[0] for i in range(N)],
    "t2": [cell_of(i)[1] for i in range(N)],
    "y_decimal": [dec(outcome_of(i)) for i in range(N)],
    "weights": [int(weight_of(i)) for i in range(N)],
    "clusters": CLUSTERS,
    "folds": FOLDS,
    "unweighted": {d: estimate(False, d) for d in ("independent", "clusters", "newey_west")},
    "weighted": {d: estimate(True, d) for d in ("independent", "clusters", "newey_west")},
}
print(json.dumps(out, indent=1))
