# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""Cross-fitted trial-to-nonparticipant AIPW transport with a joint outer bootstrap at 40 digits.

Regenerate with:  uv run --script learned_trial_reference.py [recorded.json] > learned_trial_reference.json

The data, the fold of every row and the row indices of every bootstrap replicate are part of the
case (the seeded fold plan and the bootstrap stream are separate kernels; the test asserts the
product's fold assignment equals the recorded one, that each recorded draw stays inside its
resampling group and keeps the group sizes).  Nuisances, per fold, on the rows outside the fold:
  membership  unpenalised logistic maximum likelihood of source membership on [1, X] over all rows
              (Newton), predicted on the fold;
  outcome     per treatment arm, ridge regression of the outcome on [1, X] over source rows in the
              arm, minimising |y - X b|^2 + lambda |b_slopes|^2 (intercept unpenalised), predicted on
              every row of the fold (source and target).
Score: trial row  sign (1 - s)/(s arm) (y - mu_arm) with sign = +1 treated, -1 control, arm = e or
1 - e; target row  mu1 - mu0.  Estimate = sum of scores / (target rows); the row score is the term
times n / target rows.  A replicate refits every nuisance on the resampled rows with each
duplicate keeping its original row's fold.  The interval is the equal-tail type-7 percentile
interval of the successful replicate estimates at the declared level.
Values are written as nearest binary64.
Repository-authored (MIT OR Apache-2.0).
"""
import json
import sys
from mpmath import mp, mpf, matrix, lu_solve, exp

mp.dps = 40
_state = 0xA54FF53A5F1D36F1


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def logistic_fit(X, y):
    p = len(X[0])
    beta = [mpf(0)] * p
    for _ in range(60):
        g = matrix(p, 1)
        H = matrix(p, p)
        for x, t in zip(X, y):
            pr = 1 / (1 + exp(-sum(beta[j] * x[j] for j in range(p))))
            for j in range(p):
                g[j] += x[j] * (t - pr)
                for l in range(p):
                    H[j, l] += x[j] * x[l] * pr * (1 - pr)
        step = lu_solve(H, g)
        beta = [beta[j] + step[j] for j in range(p)]
        if max(abs(step[j]) for j in range(p)) < mpf(10) ** -30:
            break
    return beta


def ridge_fit(X, y, lam):
    p = len(X[0])
    A = matrix(p, p)
    b = matrix(p, 1)
    for x, t in zip(X, y):
        for i in range(p):
            b[i] += x[i] * t
            for j in range(p):
                A[i, j] += x[i] * x[j]
    for i in range(1, p):
        A[i, i] += lam
    sol = lu_solve(A, b)
    return [sol[i] for i in range(p)]


def point(rows, case, folds_of_rows):
    """rows: list of original row indices (with repeats); folds_of_rows: fold of each."""
    n = len(rows)
    X = [[mpf(1)] + [mpf(c[r]) for c in case["covariates"]] for r in rows]
    src = [case["source"][r] for r in rows]
    trt = [case["treatment"][r] for r in rows]
    y = [mpf(case["outcome"][r]) for r in rows]
    e = [mpf(case["randomization"][r]) for r in rows]
    K = case["n_folds"]
    s_hat = [None] * n
    mu = [[None] * n, [None] * n]
    for f in range(K):
        valid = [i for i in range(n) if folds_of_rows[i] == f]
        train = [i for i in range(n) if folds_of_rows[i] != f]
        mb = logistic_fit([X[i] for i in train], [1 if src[i] else 0 for i in train])
        rb = []
        for arm in (False, True):
            tr = [i for i in train if src[i] and trt[i] == arm]
            rb.append(ridge_fit([X[i] for i in tr], [y[i] for i in tr], mpf(case["ridge_lambda"])))
        for i in valid:
            s_hat[i] = 1 / (1 + exp(-sum(mb[j] * X[i][j] for j in range(len(mb)))))
            for a in (0, 1):
                mu[a][i] = sum(rb[a][j] * X[i][j] for j in range(len(rb[a])))
    target_n = sum(1 for v in src if not v)
    terms = []
    for i in range(n):
        if src[i]:
            sign = 1 if trt[i] else -1
            arm = e[i] if trt[i] else 1 - e[i]
            m = mu[1][i] if trt[i] else mu[0][i]
            terms.append(sign * (1 - s_hat[i]) / s_hat[i] / arm * (y[i] - m))
        else:
            terms.append(mu[1][i] - mu[0][i])
    return sum(terms) / target_n, terms, s_hat, mu, target_n


def type7(sorted_vals, p):
    d = len(sorted_vals)
    h = p * (d - 1)
    lo = int(h // 1)
    hi = min(lo + 1, d - 1)
    return sorted_vals[lo] + (h - lo) * (sorted_vals[hi] - sorted_vals[lo])


def reference(case):
    n = len(case["outcome"])
    est, terms, s_hat, mu, target_n = point(list(range(n)), case, case["folds"])
    scale = mpf(n) / target_n
    out = {"estimate": float(est), "scores": [float(t * scale) for t in terms],
           "membership": [float(v) for v in s_hat], "mu0": [float(v) for v in mu[0]],
           "mu1": [float(v) for v in mu[1]]}
    reps = []
    for rows in case["replicate_rows"]:
        reps.append(point(rows, case, [case["folds"][r] for r in rows])[0])
    out["replicates"] = [float(v) for v in reps]
    srt = sorted(reps)
    lv = mpf(case["coverage_level"])
    out["interval"] = [float(type7(srt, (1 - lv) / 2)), float(type7(srt, (1 + lv) / 2))]
    return out


def make(name, sampling, n_src, n_tgt, p, offset, lam, boot):
    n = n_src + n_tgt
    cols = [[offset + (u01() * 2 - 1) for _ in range(n)] for _ in range(p)]
    source = [i % 5 != 4 for i in range(n)] if False else None
    # interleave so that every fold holds both roles: source rows first then target rows
    source = [i < n_src for i in range(n)]
    # sources are the ones with higher first covariate on average: shift the first covariate
    for i in range(n):
        if source[i]:
            cols[0][i] += 0.4
    treatment = [u01() < 0.5 for _ in range(n)]
    outcome = [1.0 + 1.5 * float(treatment[i]) * (1 + 0.5 * (cols[0][i] - offset)) + 0.8 * (cols[0][i] - offset)
               + (u01() - 0.5) for i in range(n)]
    return {"name": name, "sampling": sampling, "features": list(range(1, p + 1)),
            "covariates": cols, "source": source, "treatment": treatment, "outcome": outcome,
            "randomization": [0.5] * n, "n_folds": 3, "ridge_lambda": lam, "bootstrap": boot,
            "coverage_level": 0.9}


if __name__ == "__main__":
    recorded = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
    specs = [
        ("nested cohort, two covariates, ridge outcome", "nested_cohort", 60, 36, 2, 0.0, 2.0, 11),
        ("independent samples, covariate offset 100", "independent_samples", 54, 36, 1, 100.0, 0.5, 9),
    ]
    cases = []
    for spec in specs:
        case = make(*spec)
        rec = recorded.get(case["name"])
        if rec is not None:
            case["folds"] = rec["folds"]
            case["replicate_rows"] = rec["replicate_rows"]
            case["expected"] = reference(case)
        cases.append(case)
    print(json.dumps({"cases": cases}, indent=1))
