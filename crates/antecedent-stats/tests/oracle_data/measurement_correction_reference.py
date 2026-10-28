# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Reference values for binary and categorical misclassification corrections.

Regenerate with:  uv run --script measurement_correction_reference.py > measurement_correction_reference.json

Binary (exact rationals): pi = (p* + Sp - 1)/J, J = Se + Sp - 1, delta-method variance
  Var(p*)/J^2 + ((1 - pi)^2 Var(Sp) + pi^2 Var(Se))/J^2,  risk difference (p1* - p0*)/J with
  RD^2 (Var(Se) + Var(Sp))/J^2 for shared rates; the partial derivatives were derived by hand:
  d pi/d p* = 1/J, d pi/d Sp = (1 - pi)/J, d pi/d Se = -pi/J.

Categorical (50 digits): theta = (class means, observed proportions, confusion entries);
  pi = C^{-1} p*, Q[j][t] = C[j][t] pi_t / p*_j, mu = Q^{-1} ybar, with the covariance of theta
  (class variance / count; multinomial for p*; one multinomial per confusion column) pushed through
  the Jacobian of (pi, mu), which is evaluated here by Richardson-free central differences with a
  step of 1e-20 at 60 digits (truncation error ~1e-40).  The 1-norm condition number is exact.

Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json
import random
from fractions import Fraction as F

import mpmath as mp

mp.mp.dps = 60
mpf = mp.mpf


def txt(q, digits=30):
    if isinstance(q, F):
        return mp.nstr(mpf(q.numerator) / q.denominator, digits)
    return mp.nstr(q, digits)


def prevalence(events, total, se, sp, vse, vsp):
    se, sp, vse, vsp = F(se), F(sp), F(vse), F(vsp)
    p = F(events, total)
    j = se + sp - 1
    pi = (p + sp - 1) / j
    sampling = p * (1 - p) / total / j**2
    rates = ((1 - pi) ** 2 * vsp + pi**2 * vse) / j**2
    return {"corrected": pi, "youden": j, "variance": sampling + rates, "sampling": sampling, "rates": rates}


def binary_cases():
    out = []
    for events, total, se, sp, n_pos, n_neg in [
        (30, 100, 0.9, 0.95, 200, 300), (0, 50, 0.8, 0.9, 40, 60), (50, 50, 0.8, 0.9, 40, 60), (3, 1000, 0.99, 0.999, 1000, 1000),
        (500, 1000, 0.55, 0.6, 5000, 5000), (12, 80, 0.7, 0.7, 10, 10), (45, 100, 1.0, 1.0, 1, 1), (1, 3, 0.5625, 0.75, 16, 8),
    ]:
        se, sp = F(se), F(sp)
        vse = se * (1 - se) / n_pos
        vsp = sp * (1 - sp) / n_neg
        r = prevalence(events, total, se, sp, vse, vsp)
        out.append({"events": events, "total": total, "se": float(se), "sp": float(sp), "vse": float(vse), "vsp": float(vsp),
                    **{k: txt(v) for k, v in r.items()}})
    return out


def risk_difference_cases():
    out = []
    for e1, n1, e0, n0, se, sp, vse, vsp in [
        (60, 200, 30, 250, 0.9, 0.95, 0.001, 0.0005), (5, 40, 5, 40, 0.8, 0.9, 0.01, 0.02), (0, 100, 40, 100, 0.95, 0.8, 0.0, 0.0),
        (900, 1000, 100, 1000, 0.6, 0.6, 1e-4, 1e-4),
    ]:
        se, sp, vse, vsp = F(se), F(sp), F(vse), F(vsp)
        j = se + sp - 1
        p1, p0 = F(e1, n1), F(e0, n0)
        corrected = (p1 - p0) / j
        sampling = (p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0) / j**2
        rates = corrected**2 * (vse + vsp) / j**2
        out.append({"table": [e1, n1, e0, n0], "se": float(se), "sp": float(sp), "vse": float(vse), "vsp": float(vsp),
                    "corrected": txt(corrected), "variance": txt(sampling + rates), "sampling": txt(sampling), "rates": txt(rates),
                    "risk_exposed": txt((p1 + sp - 1) / j), "risk_unexposed": txt((p0 + sp - 1) / j)})
    # differential: independent rates per arm
    diff = []
    for e1, n1, e0, n0, a, b in [(60, 200, 30, 250, (0.9, 0.95, 0.001, 0.0005), (0.8, 0.9, 0.002, 0.001)), (10, 50, 4, 60, (0.7, 0.99, 0.01, 0.0), (0.95, 0.85, 0.0, 0.01))]:
        r1 = prevalence(e1, n1, *a)
        r0 = prevalence(e0, n0, *b)
        diff.append({"table": [e1, n1, e0, n0], "exposed": list(a), "unexposed": list(b), "corrected": txt(r1["corrected"] - r0["corrected"]),
                     "variance": txt(r1["variance"] + r0["variance"]), "sampling": txt(r1["sampling"] + r0["sampling"]), "rates": txt(r1["rates"] + r0["rates"])})
    return out, diff


def inverse(c):
    return mp.matrix(c) ** -1


def categorical(counts, means, variances, confusion_counts):
    """counts: observed class sizes; confusion_counts[j][t] validation counts of observed j, true t."""
    k = len(counts)
    n = sum(counts)
    col = [sum(confusion_counts[j][t] for j in range(k)) for t in range(k)]
    c0 = [[mpf(confusion_counts[j][t]) / col[t] for t in range(k)] for j in range(k)]
    p0 = [mpf(c) / n for c in counts]
    ybar0 = [mpf(m) for m in means]
    theta0 = ybar0 + p0 + [c0[j][t] for j in range(k) for t in range(k)]
    dim = len(theta0)

    def maps(theta):
        yb, ps = theta[:k], theta[k:2 * k]
        c = mp.matrix(k, k)
        for j in range(k):
            for t in range(k):
                c[j, t] = theta[2 * k + j * k + t]
        pi = mp.lu_solve(c, mp.matrix(ps))
        q = mp.matrix(k, k)
        for j in range(k):
            for t in range(k):
                q[j, t] = c[j, t] * pi[t] / ps[j]
        mu = mp.lu_solve(q, mp.matrix(yb))
        return [pi[i] for i in range(k)] + [mu[i] for i in range(k)]

    cov = mp.zeros(dim, dim)
    for j in range(k):
        cov[j, j] = mpf(variances[j]) / counts[j]
    for a in range(k):
        for b in range(k):
            cov[k + a, k + b] = (p0[a] * (1 - p0[a]) if a == b else -p0[a] * p0[b]) / n
    for t in range(k):
        for a in range(k):
            for b in range(k):
                ca, cb = c0[a][t], c0[b][t]
                cov[2 * k + a * k + t, 2 * k + b * k + t] = (ca * (1 - ca) if a == b else -ca * cb) / col[t]
    h = mpf(10) ** -20
    jac = mp.zeros(2 * k, dim)
    for idx in range(dim):
        up, dn = list(theta0), list(theta0)
        up[idx] += h
        dn[idx] -= h
        fu, fd = maps(up), maps(dn)
        for i in range(2 * k):
            jac[i, idx] = (fu[i] - fd[i]) / (2 * h)
    full = jac * cov * jac.T
    base = maps(theta0)
    cmat = mp.matrix(c0)
    cinv = cmat ** -1

    def one_norm(a):
        return max(sum(abs(a[i, j]) for i in range(a.rows)) for j in range(a.cols))
    return {
        "proportions": [txt(base[i]) for i in range(k)],
        "outcome_means": [txt(base[k + i]) for i in range(k)],
        "proportion_covariance": [txt(full[i, j]) for i in range(k) for j in range(k)],
        "outcome_mean_covariance": [txt(full[k + i, k + j]) for i in range(k) for j in range(k)],
        "condition": txt(one_norm(cmat) * one_norm(cinv)),
    }


def categorical_cases():
    out = []
    specs = [
        ([60, 40], [1.2, 3.4], [0.5, 0.9], [[45, 5], [5, 45]]),
        ([100, 80, 60], [0.5, 1.5, 2.5], [1.0, 1.2, 0.8], [[80, 10, 4], [12, 70, 6], [8, 20, 90]]),
        ([200, 30, 25, 45], [0.0, 1.0, -1.0, 2.0], [1.0, 2.0, 0.5, 1.5], [[90, 5, 3, 2], [4, 85, 6, 3], [3, 6, 80, 8], [3, 4, 11, 87]]),
        ([20, 380], [10.0, 20.0], [4.0, 9.0], [[60, 40], [20, 130]]),
        ([500, 500], [100.0, 100.5], [0.01, 0.01], [[25, 24], [24, 25]]),
    ]
    for counts, means, variances, validation in specs:
        k = len(counts)
        out.append({"levels": k, "counts": counts, "means": means, "variances": variances,
                    "validation": [v for row in validation for v in row], **categorical(counts, means, variances, validation)})
    return out


def main():
    rd, rd_diff = risk_difference_cases()
    print(json.dumps({"prevalence": binary_cases(), "risk_difference": rd, "risk_difference_differential": rd_diff,
                      "categorical": categorical_cases()}, separators=(",", ":")))


main()
