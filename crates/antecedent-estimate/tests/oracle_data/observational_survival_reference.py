# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""Cross-fitted doubly robust survival and restricted-mean-survival effects at 40 digits.

Regenerate with:  uv run --script observational_survival_reference.py [folds.json] > observational_survival_reference.json

The data and the fold of every row are part of the case (the seeded fold plan is a separate kernel;
the test asserts the product's plan equals the recorded folds).  For each fold the nuisances are
fitted on the training rows only:
  propensity  unpenalised logistic maximum likelihood of the active arm on [1, Z] (Newton),
  event       per arm, Cox partial likelihood with Breslow ties (Newton), baseline increment
              dLam0(t_j) = d_j / sum_{time_k >= t_j} exp(b z_k) at each distinct event time,
  censoring   per arm, the same Cox fit to the censoring indicator on the declared covariates
              (at random) or on no covariates (independent); no censoring model when the training
              arm has no censored row.
With h = exp(b z), S(u|z) = exp(-h Lam0(u)) (right continuous) and G(u-|z) = exp(-hc Lamc0(u-)):
  Q    = D 1{X <= t*} / (S(X) G(X-)) - sum_{u_j <= min(X, t*)} h dLam0_j / (S(u_j) G(u_j-))
  psi_S = S(t*) - 1{A = a} S(t*) Q / e_a
  R(z)  = integral_0^{t*} S(u) du, evaluated exactly on the step function
  psi_R = R(z) - 1{A = a}/e_a * { D 1{X <= t*} Rres(X) / G(X-) - sum_j h dLam0_j Rres(u_j) / G(u_j-) }
          with Rres(u) = (integral_u^{t*} S(s) ds) / S(u).
Arm means, contrast means and their standard deviations over sqrt(n) (n-1 divisor) are written, with
the smallest propensity per arm and the smallest G(t*|z) over the rows of each arm.  Values are
written as nearest binary64.
Repository-authored (MIT OR Apache-2.0).
"""
import json
import sys
from mpmath import mp, mpf, matrix, lu_solve, exp, sqrt

mp.dps = 40
_state = 0x6A09E667F3BCC908


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
            eta = sum(beta[j] * x[j] for j in range(p))
            pr = 1 / (1 + exp(-eta))
            for j in range(p):
                g[j] += x[j] * (t - pr)
                for l in range(p):
                    H[j, l] += x[j] * x[l] * pr * (1 - pr)
        step = lu_solve(H, g)
        beta = [beta[j] + step[j] for j in range(p)]
        if max(abs(step[j]) for j in range(p)) < mpf(10) ** -30:
            break
    return beta


def cox_fit(time, fail, Z):
    """Breslow partial likelihood MLE; Z rows may be empty (p = 0)."""
    n, p = len(time), (len(Z[0]) if Z and Z[0] else 0)
    beta = [mpf(0)] * p
    for _ in range(60):
        if p == 0:
            break
        g = matrix(p, 1)
        H = matrix(p, p)
        for i in range(n):
            if not fail[i]:
                continue
            risk = [k for k in range(n) if time[k] >= time[i]]
            w = [exp(sum(beta[j] * Z[k][j] for j in range(p))) for k in risk]
            s0 = sum(w)
            s1 = [sum(w[a] * Z[k][j] for a, k in enumerate(risk)) for j in range(p)]
            for j in range(p):
                g[j] += Z[i][j] - s1[j] / s0
                for l in range(p):
                    s2 = sum(w[a] * Z[k][j] * Z[k][l] for a, k in enumerate(risk))
                    H[j, l] += s2 / s0 - s1[j] * s1[l] / s0 ** 2
        step = lu_solve(H, g)
        beta = [beta[j] + step[j] for j in range(p)]
        if max(abs(step[j]) for j in range(p)) < mpf(10) ** -30:
            break
    jumps = sorted({time[i] for i in range(n) if fail[i]})
    inc = []
    for t in jumps:
        d = sum(1 for i in range(n) if fail[i] and time[i] == t)
        den = sum(exp(sum(beta[j] * Z[k][j] for j in range(p))) for k in range(n) if time[k] >= t)
        inc.append(d / den)
    return {"beta": beta, "jumps": jumps, "inc": inc}


def lam_at(m, t, strict=False):
    return sum(inc for u, inc in zip(m["jumps"], m["inc"]) if (u < t if strict else u <= t))


def lin(m, z):
    return sum(b * zz for b, zz in zip(m["beta"], z))


def mean_sd(vals):
    n = len(vals)
    mu = sum(vals) / n
    return mu, sqrt(sum((v - mu) ** 2 for v in vals) / (n - 1))


def reference(case):
    n = len(case["duration"])
    arm = [1 if t == case["active"] else 0 for t in case["treatment"]]
    dur = [mpf(x) for x in case["duration"]]
    ev = [bool(e) for e in case["event"]]
    Z = [[mpf(c[i]) for c in case["covariates"]] for i in range(n)]
    X = [[mpf(1)] + Z[i] for i in range(n)]
    ts = mpf(case["horizon"])
    at_random = case["censoring"] == "at_random_given_covariates"
    K = case["n_folds"]
    surv = [[None] * n, [None] * n]
    rmst = [[None] * n, [None] * n]
    emin = [mpf(10), mpf(10)]
    gmin = [mpf(10), mpf(10)]
    for f in range(K):
        valid = [i for i in range(n) if case["folds"][i] == f]
        train = [i for i in range(n) if case["folds"][i] != f]
        pb = logistic_fit([X[i] for i in train], [arm[i] for i in train])
        for a in (0, 1):
            rows = [i for i in train if arm[i] == a]
            em = cox_fit([dur[i] for i in rows], [ev[i] for i in rows], [Z[i] for i in rows])
            cens = [not ev[i] for i in rows]
            if any(cens):
                cz = [Z[i] for i in rows] if at_random else [[] for _ in rows]
                cm = cox_fit([dur[i] for i in rows], cens, cz)
            else:
                cm = None
            jumps = [(u, inc) for u, inc in zip(em["jumps"], em["inc"]) if u <= ts]
            for i in valid:
                z = Z[i]
                h = exp(lin(em, z))
                p1 = 1 / (1 + exp(-sum(pb[j] * X[i][j] for j in range(len(pb)))))
                e = p1 if a == 1 else 1 - p1
                emin[a] = min(emin[a], e)
                Lam = lambda u: h * sum(inc for uu, inc in jumps if uu <= u)
                S = lambda u: exp(-Lam(u))
                hc = exp(lin(cm, z if at_random else [])) if cm is not None else None
                Gm = (lambda u: exp(-hc * lam_at(cm, u, strict=True))) if cm is not None else (lambda u: mpf(1))
                # R(z) and Rres by exact integration of the step function S on [u, ts]
                knots = [u for u, _ in jumps]

                def integral(u):
                    pts = [u] + [k for k in knots if k > u] + [ts]
                    return sum(S(pts[q]) * (pts[q + 1] - pts[q]) for q in range(len(pts) - 1))

                s_ts = S(ts)
                r_z = integral(mpf(0))
                own = arm[i] == a
                if own:
                    gmin[a] = min(gmin[a], exp(-hc * lam_at(cm, ts)) if cm is not None else mpf(1))
                    x_i = dur[i]
                    q_term = mpf(0)
                    r_term = mpf(0)
                    if ev[i] and x_i <= ts:
                        q_term += 1 / (S(x_i) * Gm(x_i))
                        r_term += integral(x_i) / S(x_i) / Gm(x_i)
                    for u, inc in jumps:
                        if u <= x_i:
                            q_term -= h * inc / (S(u) * Gm(u))
                            r_term -= h * inc * integral(u) / S(u) / Gm(u)
                    surv[a][i] = s_ts - s_ts * q_term / e
                    rmst[a][i] = r_z - r_term / e
                else:
                    surv[a][i] = s_ts
                    rmst[a][i] = r_z
    out = {"arms": []}
    for a in (0, 1):
        ms, ss = mean_sd(surv[a])
        mr, sr = mean_sd(rmst[a])
        out["arms"].append({"survival": float(ms), "survival_se": float(ss / sqrt(n)),
                            "rmst": float(mr), "rmst_se": float(sr / sqrt(n)),
                            "propensity_min": float(emin[a]), "censoring_survival_min": float(gmin[a])})
    ds = [surv[1][i] - surv[0][i] for i in range(n)]
    dr = [rmst[1][i] - rmst[0][i] for i in range(n)]
    m1, s1 = mean_sd(ds)
    m2, s2 = mean_sd(dr)
    out["survival_difference"] = {"estimate": float(m1), "se": float(s1 / sqrt(n))}
    out["rmst_difference"] = {"estimate": float(m2), "se": float(s2 / sqrt(n))}
    out["survival_difference_scores"] = [float(v) for v in ds]
    out["rmst_difference_scores"] = [float(v) for v in dr]
    return out


def make_case(name, n, p, folds, censoring, horizon, grid=None, cov_offset=0.0, rate=0.5, cens_rate=0.3):
    Zc = [[cov_offset + (u01() * 2 - 1) for _ in range(n)] for _ in range(p)]
    treat, dur, ev = [], [], []
    for i in range(n):
        z0 = Zc[0][i] - cov_offset
        pr = 1 / (1 + float(exp(-(0.7 * z0))))
        a = 1.0 if u01() < pr else 0.0
        treat.append(a)
        lin_ = 0.6 * z0 - 0.5 * a + (0.3 * (Zc[1][i] - cov_offset) if p > 1 else 0.0)
        t_event = -float(mp.log(1 - u01())) / (rate * float(exp(lin_)))
        cr = cens_rate * (float(exp(0.5 * z0)) if censoring == "at_random_given_covariates" else 1.0)
        t_cens = -float(mp.log(1 - u01())) / cr
        t_cens = min(t_cens, 3.0)  # administrative end of follow-up keeps G(t*) away from zero
        t, d = (t_event, 1.0) if t_event <= t_cens else (t_cens, 0.0)
        if grid:
            t = max(round(t / grid) * grid, grid)
        dur.append(t)
        ev.append(d)
    return {"name": name, "active": 1.0, "control": 0.0, "horizon": horizon, "censoring": censoring,
            "n_folds": folds, "duration": dur, "event": ev, "treatment": treat, "covariates": Zc,
            "min_propensity": 0.01, "min_censoring_survival": 0.05}


if __name__ == "__main__":
    recorded = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
    specs = [
        ("independent censoring, one covariate, three folds", 100, 1, 3, "independent_within_arm", 1.5, None, 0.0),
        ("censoring at random, two covariates, four folds", 140, 2, 4, "at_random_given_covariates", 1.5, None, 0.0),
        ("tied event and censoring times, two folds", 100, 1, 2, "independent_within_arm", 1.0, 0.25, 0.0),
        ("covariate offset 50, horizon before most events", 100, 1, 3, "at_random_given_covariates", 0.1, None, 50.0),
    ]
    cases = []
    for name, n, p, k, cens, hz, grid, off in specs:
        case = make_case(name, n, p, k, cens, hz, grid, off)
        case["folds"] = recorded.get(name)
        if case["folds"] is not None:
            case["expected"] = reference(case)
        cases.append(case)
    print(json.dumps({"cases": cases}, indent=1))
