# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Single-treated-unit synthetic control with a declared serial or clustered design, exactly.

Regenerate with:  uv run --script synthetic_dependence_reference.py > synthetic_dependence_reference.json

Every weight is the exact solution of the definition-level quadratic program

    min_w  sum_t (y_t - sum_j w_j x_jt)^2   subject to  w >= 0, sum_j w_j = 1,

found by enumerating the supports and solving the KKT system in Fraction arithmetic (no
iteration, no tolerance). On that:

* Placebo variance (clustered_panel). The point effect is the post-period mean gap between the
  treated unit and its synthetic control. The placebo effects refit each donor j as the treated
  unit against the other donors; with adoptions reassigned uniformly over the D donors, the
  variance of the placebo estimate is the population variance of those D effects, which the
  product approaches with its seeded replicates (the case checks it to a Monte Carlo tolerance).
* Conformal (dependent_time_series). The p-value of a null effect theta removes theta from the
  post periods, refits the simplex on all periods, takes the absolute residuals and counts the
  cyclic shifts whose post-block sum is at least the observed one; theta is retained when that
  count over the number of periods exceeds alpha = 0.05. The interval is the hull of the retained
  set around the point estimate, located here by exact bisection on the exact p-value (the
  retained set is an interval for these fixtures, which the script asserts on a 400-point scan).
Repository-authored (MIT OR Apache-2.0).
"""
import itertools
import json
from fractions import Fraction as F


def solve(matrix, rhs):
    n = len(rhs)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for c in range(n):
        pivot = next((r for r in range(c, n) if a[r][c] != 0), None)
        if pivot is None:
            return None
        a[c], a[pivot] = a[pivot], a[c]
        inv = 1 / a[c][c]
        a[c] = [v * inv for v in a[c]]
        for r in range(n):
            if r != c and a[r][c] != 0:
                f = a[r][c]
                a[r] = [x - f * y for x, y in zip(a[r], a[c])]
    return [a[i][n] for i in range(n)]


def simplex_fit(target, donors):
    """Exact simplex-constrained least squares; donors is a list of series."""
    k, n = len(donors), len(target)
    gram = [[sum(donors[i][t] * donors[j][t] for t in range(n)) for j in range(k)] for i in range(k)]
    xty = [sum(donors[i][t] * target[t] for t in range(n)) for i in range(k)]
    best = None
    for size in range(1, k + 1):
        for support in itertools.combinations(range(k), size):
            m = size + 1
            matrix = [[F(0)] * m for _ in range(m)]
            rhs = [F(0)] * m
            for a, i in enumerate(support):
                for b, j in enumerate(support):
                    matrix[a][b] = 2 * gram[i][j]
                matrix[a][size] = F(1)
                matrix[size][a] = F(1)
                rhs[a] = 2 * xty[i]
            rhs[size] = F(1)
            sol = solve(matrix, rhs)
            if sol is None:
                continue
            w = [F(0)] * k
            for a, i in enumerate(support):
                w[i] = sol[a]
            if any(w[i] <= 0 for i in support):
                continue
            lam = sol[size]
            ok = True
            for j in range(k):
                if j in support:
                    continue
                grad = 2 * (sum(gram[j][i] * w[i] for i in range(k)) - xty[j])
                if grad + lam < 0:
                    ok = False
                    break
            if ok:
                best = w
                break
        if best is not None:
            break
    assert best is not None
    return best


def synth(donors, w, t):
    return sum(w[j] * donors[j][t] for j in range(len(donors)))


def effect(panel, treated, donors_idx, pre, post):
    donors = [[panel[u][t] for t in pre] for u in donors_idx]
    w = simplex_fit([panel[treated][t] for t in pre], donors)
    gap = sum(panel[treated][t] - sum(w[j] * panel[u][t] for j, u in enumerate(donors_idx)) for t in post)
    return gap / len(post), w


def lcg(seed):
    state = seed
    while True:
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        yield F((state >> 40) % 2001 - 1000, 1000)


def conformal_p(target, donors, post):
    n = len(target)
    w = simplex_fit(target, donors)
    resid = [abs(target[t] - synth(donors, w, t)) for t in range(n)]
    def stat(shift):
        return sum(resid[(pos + n - shift) % n] for pos in range(n - post, n))
    observed = stat(0)
    return F(sum(1 for s in range(n) if stat(s) >= observed), n)


def conformal_interval(series, donors, n_pre, post, center, alpha, scale):
    """Hull of {theta: p(theta) > alpha}: bisection outward from the center."""
    def accept(theta):
        adjusted = list(series)
        for t in range(n_pre, n_pre + post):
            adjusted[t] = adjusted[t] - theta
        return conformal_p(adjusted, donors, post) > alpha
    assert accept(center), "the point estimate must be retained"
    def edge(direction):
        inside, outside = center, center
        step = scale
        for _ in range(60):
            outside = center + direction * step
            if not accept(outside):
                break
            inside = outside
            step *= 2
        else:
            raise AssertionError("unbounded")
        lo, hi = inside, outside
        for _ in range(45):
            mid = (lo + hi) / 2
            if accept(mid):
                lo = mid
            else:
                hi = mid
        return float(lo)
    lo, hi = edge(-1), edge(1)
    scan = [float(center) + (hi - lo) * (i / 399.0 * 3 - 1.0) for i in range(400)]
    inside = [accept(F(x)) for x in scan]
    # The retained set is an interval: every scanned point inside [lo, hi] is retained and outside is not.
    for x, ok in zip(scan, inside):
        if lo + 1e-6 < x < hi - 1e-6:
            assert ok, "retained set is not an interval inside the hull"
        elif x < lo - 1e-6 or x > hi + 1e-6:
            assert not ok, "retained set is not an interval outside the hull"
    return [lo, hi]


def placebo_case():
    gen = lcg(7)
    donors_n, periods = 5, 9
    pre, post = list(range(6)), [6, 7, 8]
    base = [[F(j) + next(gen) for _ in range(periods)] for j in range(donors_n)]
    treated = [F(3, 10) * base[0][t] + F(7, 10) * base[2][t] + next(gen) / 4 for t in range(periods)]
    tau = F(5, 2)
    for t in post:
        treated[t] += tau
    panel = base + [treated]
    point, weights = effect(panel, donors_n, list(range(donors_n)), pre, post)
    placebo = []
    for j in range(donors_n):
        others = [u for u in range(donors_n) if u != j]
        p, _ = effect(panel, j, others, pre, post)
        placebo.append(p)
    mean = sum(placebo) / donors_n
    variance = sum((p - mean) ** 2 for p in placebo) / donors_n
    return {
        "name": "placebo_variance_five_donors",
        "panel": [[float(v) for v in row] for row in panel],
        "unit_names": [f"d{j}" for j in range(donors_n)] + ["treated"],
        "intervention_period": 7,
        "effect": float(point),
        "weights": [float(w) for w in weights],
        "placebo_effects": [float(p) for p in placebo],
        "placebo_se": float(variance) ** 0.5,
    }


def conformal_case():
    gen = lcg(23)
    donors_n, n_pre, n_post = 4, 24, 2
    periods = n_pre + n_post
    shared = [next(gen) for _ in range(periods)]
    base = [[F(j) + shared[t] + next(gen) for t in range(periods)] for j in range(donors_n)]
    treated = [
        F(2, 5) * base[0][t] + F(3, 5) * base[2][t] + next(gen) / 2 for t in range(periods)
    ]
    tau = F(3)
    for t in range(n_pre, periods):
        treated[t] += tau
    panel = base + [treated]
    pre, post = list(range(n_pre)), list(range(n_pre, periods))
    point, weights = effect(panel, donors_n, list(range(donors_n)), pre, post)
    series = treated
    donors = [base[j] for j in range(donors_n)]
    scale = F(1, 4)
    overall = conformal_interval(series, donors, n_pre, n_post, point, F(1, 20), scale)
    # Period 1: the series is truncated to the pre periods plus that period.
    per_period = []
    for s in post:
        times = pre + [s]
        w = simplex_fit([series[t] for t in pre], [[d[t] for t in pre] for d in donors])
        est = series[s] - sum(w[j] * donors[j][s] for j in range(donors_n))
        interval = conformal_interval(
            [series[t] for t in times], [[d[t] for t in times] for d in donors], n_pre, 1, est, F(1, 20), scale
        )
        per_period.append({"period": s + 1, "estimate": float(est), "interval": interval})
    return {
        "name": "conformal_four_donors_24_pre_periods",
        "panel": [[float(v) for v in row] for row in panel],
        "unit_names": [f"d{j}" for j in range(donors_n)] + ["treated"],
        "intervention_period": n_pre + 1,
        "effect": float(point),
        "weights": [float(w) for w in weights],
        "overall_interval": overall,
        "periods": per_period,
    }


if __name__ == "__main__":
    print(json.dumps({"placebo": placebo_case(), "conformal": conformal_case()}, indent=1))
